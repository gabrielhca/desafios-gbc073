# -*- coding: utf-8 -*-
"""
treino.py — GBC073 Desafio 3 (STGCN-Dengue RJ)

Por que existe: o Desafio 3 compara otimizadores e agendas de η com o MESMO orçamento
(920 passos, lote 32) e lê sinais do treino (perda, ‖g‖ antes do clip, ‖θ‖, razão de
atualização por camada, η_t). Uma execução = uma linha de comando. Escreve:
  - logs/<id>.csv           instrumentação a cada 20 passos
  - logs/<id>_val.csv       val loss / val MAE por época
  - experimentos.csv        append de 1 linha com a configuração e as métricas
Regras: sem early stopping (sempre roda os 920 passos); o checkpoint guardado é o da
época de menor val loss; teste só com --avaliar_teste (usado uma vez, no final).
Retomável: se logs/<id>.csv existir, sai.
"""
import os, sys, json, time, math, argparse
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ----------------------------------------------------------------------------- CLI
ap = argparse.ArgumentParser()
ap.add_argument('--id', required=True)
ap.add_argument('--etapa', required=True)
ap.add_argument('--otimizador', required=True,
                choices=['sgd', 'sgd_nesterov', 'adam', 'adamw', 'muon'])
ap.add_argument('--lr', type=float, required=True)
ap.add_argument('--lr_adamw', type=float, default=3e-4)
ap.add_argument('--agenda', required=True,
                choices=['const', 'cos', 'warmup_cos', 'multistep'])
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--epocas', type=int, default=40)
ap.add_argument('--lote', type=int, default=32)
ap.add_argument('--clip', type=float, default=5.0)
ap.add_argument('--wd_sgd_adam', type=float, default=0.0)
ap.add_argument('--wd_adamw', type=float, default=0.01)
ap.add_argument('--dir_dataset', required=True)
ap.add_argument('--out_dir', required=True)
ap.add_argument('--avaliar_teste', action='store_true')
ap.add_argument('--div_mult', type=float, default=10.0)
args = ap.parse_args()

OUT = args.out_dir
LOGS = os.path.join(OUT, 'logs'); os.makedirs(LOGS, exist_ok=True)
LOG_PATH = os.path.join(LOGS, f'{args.id}.csv')
EXP_CSV = os.path.join(OUT, 'experimentos.csv')

if os.path.exists(LOG_PATH):
    print(f'[skip] {args.id} já tem log.'); sys.exit(0)

# ----------------------------------------------------------------------------- seeds
import random
random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------------- dados
P = args.dir_dataset
PR = os.path.join(P, 'processed')
GR = os.path.join(P, 'graphs')
STATE = '33'; PRE = f'estado_{STATE}'

X      = np.load(os.path.join(PR, f'{PRE}_X.npy'))
y_raw  = np.load(os.path.join(PR, f'{PRE}_y_raw.npy'))
y_log  = np.load(os.path.join(PR, f'{PRE}_y_log1p.npy'))
L_tilde= np.load(os.path.join(GR, f'{PRE}_L_tilde_k5.npy'))
with open(os.path.join(PR, f'{PRE}_splits.json')) as f: splits = json.load(f)
with open(os.path.join(PR, f'{PRE}_feature_names.json')) as f: fm = json.load(f)

W_IN, H = fm['W'], fm['H']
CANAIS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
F_IN = len(CANAIS)
X_in = X.copy(); X_in[:, :, 0] = y_log
X_in = X_in[:, :, CANAIS].astype(np.float32)

t0tr, t1tr = splits['train']; t0va, t1va = splits['val']; t0te, t1te = splits['test']

def make_windows(t_start, t_end):
    Xs, yl, yr, ts = [], [], [], []
    for t in range(t_start + W_IN - 1, t_end - H + 1):
        w = X_in[t - W_IN + 1:t + 1].transpose(1, 0, 2)
        yr_ = y_raw[t + 1:t + H + 1].T
        yl_ = y_log[t + 1:t + H + 1].T
        Xs.append(w); yr.append(yr_); yl.append(yl_); ts.append(t)
    return np.stack(Xs), np.stack(yl), np.stack(yr), np.array(ts)

Xtr, ytr_l, ytr_r, t_tr = make_windows(t0tr, t1tr)
Xva, yva_l, yva_r, t_va = make_windows(t0va, t1va)
Xte, yte_l, yte_r, t_te = make_windows(t0te, t1te)

assert len(t_tr) == 709 and len(t_va) == 149 and len(t_te) == 305, 'amostras ≠ do 03/04'

# ----------------------------------------------------------------------------- modelo
class TemporalConvGLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__(); self.kernel = kernel
        self.conv = nn.Conv2d(in_ch, 2 * out_ch, kernel_size=(1, kernel))
    def forward(self, x):
        x = F.pad(x, (self.kernel - 1, 0))
        a, b = self.conv(x).chunk(2, dim=1)
        return a * torch.sigmoid(b)

class ChebConv(nn.Module):
    def __init__(self, in_ch, out_ch, K, Lt):
        super().__init__()
        L = torch.from_numpy(Lt).float()
        Ts = [torch.eye(L.shape[0])]
        if K > 1: Ts.append(L)
        for _ in range(2, K): Ts.append(2 * L @ Ts[-1] - Ts[-2])
        self.register_buffer('T', torch.stack(Ts, dim=0))
        self.theta = nn.Parameter(torch.empty(K, in_ch, out_ch))
        nn.init.xavier_uniform_(self.theta)
        self.bias = nn.Parameter(torch.zeros(out_ch))
    def forward(self, x):
        out = 0
        for k in range(self.T.shape[0]):
            xk = torch.einsum('bcn,nm->bcm', x, self.T[k])
            out = out + torch.einsum('bcn,co->bon', xk, self.theta[k])
        return out + self.bias.view(1, -1, 1)

class STConvBlock(nn.Module):
    def __init__(self, in_ch, hid, K, Lt, kernel=3, dropout=0.2):
        super().__init__()
        self.temp1 = TemporalConvGLU(in_ch, hid, kernel)
        self.graph = ChebConv(hid, hid, K, Lt)
        self.temp2 = TemporalConvGLU(hid, hid, kernel)
        self.norm  = nn.BatchNorm2d(hid); self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv2d(in_ch, hid, 1) if in_ch != hid else nn.Identity()
    def forward(self, x):
        h = self.temp1(x)
        B, C, N_, Wt = h.shape
        h = h.permute(0, 3, 1, 2).reshape(B * Wt, C, N_)
        h = self.graph(h)
        h = h.reshape(B, Wt, -1, N_).permute(0, 2, 3, 1)
        h = self.temp2(h)
        return self.dropout(self.norm(h + self.residual(x)))

class STGCN(nn.Module):
    def __init__(self, F_in, hid, H, K, Lt, kernel=3, dropout=0.2):
        super().__init__()
        self.block1 = STConvBlock(F_in, hid, K, Lt, kernel, dropout)
        self.block2 = STConvBlock(hid, hid, K, Lt, kernel, dropout)
        self.head   = nn.Linear(hid, H)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        h = self.block2(self.block1(x))
        return self.head(h[:, :, :, -1].permute(0, 2, 1))

# ----------------------------------------------------------------------------- Muon
def _flatten(p):
    if p.ndim == 4: return p.reshape(p.shape[0], -1), p.shape, 'conv4'
    if p.ndim == 3:
        m = p.permute(2, 0, 1).reshape(p.shape[2], -1); return m, p.shape, 'cheb'
    return p, p.shape, 'linear'
def _unflatten(m, shp, modo):
    if modo in ('conv4', 'linear'): return m.reshape(shp)
    K, iN, oN = shp
    return m.reshape(oN, K, iN).permute(1, 2, 0)
def _ort(G, passos=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    T = X.size(0) > X.size(1)
    if T: X = X.T
    for _ in range(passos):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if T else X

class MuonMatrix(torch.optim.Optimizer):
    """Muon para parâmetros 2-D (kernels achatados)."""
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._buf = {}
        for g in self.param_groups:
            for p in g['params']:
                m, _, _ = _flatten(p)
                self._buf[id(p)] = torch.zeros_like(m)
    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            lr, mu, wd = g['lr'], g['momentum'], g['weight_decay']
            for p in g['params']:
                if p.grad is None: continue
                gm, shp, modo = _flatten(p.grad)
                buf = self._buf[id(p)]
                buf.mul_(mu).add_(gm)
                upd = _ort(buf)
                rows, cols = upd.shape
                scale = math.sqrt(max(1.0, rows / cols))
                p.mul_(1 - lr * wd)
                p.add_(_unflatten(upd, shp, modo), alpha=-lr * scale)

# ----------------------------------------------------------------------------- modelo/optim
model = STGCN(F_IN, 32, H, 3, L_tilde, dropout=0.2).to(device)
MUON_NAMES = {'block1.temp1.conv.weight', 'block1.temp2.conv.weight',
              'block2.temp1.conv.weight', 'block2.temp2.conv.weight',
              'block1.residual.weight', 'block1.graph.theta', 'block2.graph.theta'}

def make_opts():
    if args.otimizador == 'muon':
        muon_p  = [p for n, p in model.named_parameters() if n in MUON_NAMES]
        adamw_p = [p for n, p in model.named_parameters() if n not in MUON_NAMES]
        o_muon  = MuonMatrix(muon_p, lr=args.lr, momentum=0.95, weight_decay=args.wd_adamw)
        o_adamw = torch.optim.AdamW(adamw_p, lr=args.lr_adamw, weight_decay=args.wd_adamw)
        return [o_muon, o_adamw]
    wd = args.wd_sgd_adam if args.otimizador in ('sgd', 'sgd_nesterov') else args.wd_sgd_adam
    if args.otimizador == 'adamw': wd = args.wd_adamw
    if args.otimizador == 'sgd':
        return [torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=wd)]
    if args.otimizador == 'sgd_nesterov':
        return [torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                 nesterov=True, weight_decay=wd)]
    if args.otimizador == 'adam':
        return [torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)]
    if args.otimizador == 'adamw':
        return [torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd)]

# ----------------------------------------------------------------------------- agenda
N_EP, PASSOS_EP = args.epocas, 23
TOTAL = N_EP * PASSOS_EP
WARM = int(0.05 * TOTAL)

def lr_mult(t):
    # fator que multiplica o η base no passo t (0..TOTAL); 1.0 = η cheio
    if args.agenda == 'const': return 1.0
    if args.agenda == 'cos':
        return 0.5 * (1 + math.cos(math.pi * t / TOTAL))
    if args.agenda == 'warmup_cos':
        if t < WARM: return t / max(1, WARM)
        return 0.5 * (1 + math.cos(math.pi * (t - WARM) / max(1, TOTAL - WARM)))
    if args.agenda == 'multistep':
        if t >= 0.75 * TOTAL: return 0.01
        if t >= 0.50 * TOTAL: return 0.1
        return 1.0
    raise ValueError(args.agenda)

def make_scheds(opts):
    scheds = []
    for o in opts:
        s = torch.optim.lr_scheduler.LambdaLR(o, lr_lambda=lr_mult)
        scheds.append(s)
    return scheds

opts = make_opts()
scheds = make_scheds(opts)

# ----------------------------------------------------------------------------- loaders
def make_loader(Xn, yn, tn, shuffle, seed):
    ds = TensorDataset(torch.from_numpy(Xn).float(),
                        torch.from_numpy(yn).float(),
                        torch.from_numpy(tn).long())
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=args.lote, shuffle=True, generator=g)
    return DataLoader(ds, batch_size=args.lote, shuffle=False)

tr_loader = make_loader(Xtr, ytr_l, t_tr, True, args.seed)
va_loader = make_loader(Xva, yva_l, t_va, False, args.seed)

criterion = nn.MSELoss()

# ----------------------------------------------------------------------------- instrumentação
CAMADAS = ['block1.temp1.conv.weight', 'block2.graph.theta', 'head.weight']
log_rows = []
step = 0
historico_val = []
best_val = float('inf'); best_state = None; best_val_mae = None
divergiu = False
t0 = time.time()
perda_inicial = None
clip_hits = 0; n_steps = 0

def val_metrics():
    model.eval()
    tot_l = 0.0; preds = []
    with torch.no_grad():
        for xb, yb, _ in va_loader:
            xb = xb.to(device); yb = yb.to(device)
            out = model(xb)
            tot_l += criterion(out, yb).item() * xb.shape[0]
            preds.append(out.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    mse_log = tot_l / len(va_loader.dataset)
    preds_bruto = np.expm1(preds).clip(min=0)
    mae_bruto = float(np.mean(np.abs(preds_bruto - yva_r)))
    return mse_log, mae_bruto, preds_bruto

# ----------------------------------------------------------------------------- treino
for ep in range(1, N_EP + 1):
    model.train()
    for xb, yb, _ in tr_loader:
        xb, yb = xb.to(device), yb.to(device)
        for o in opts: o.zero_grad(set_to_none=True)
        out = model(xb)
        loss = criterion(out, yb)
        if perda_inicial is None: perda_inicial = loss.item()
        loss.backward()

        # --- sinais do passo (Seção 1 do PDF) ---
        # g_norm e gn_pre são medidos ANTES do clipping; antes = parâmetros antes do passo (para ρ)
        g_norm = math.sqrt(sum(float((p.grad ** 2).sum())
                                for p in model.parameters() if p.grad is not None))
        th_norm = math.sqrt(sum(float((p ** 2).sum()) for p in model.parameters()))

        gn_pre = {n: float(p.grad.norm()) for n, p in model.named_parameters()
                  if n in CAMADAS and p.grad is not None}

        antes = {n: p.detach().clone() for n, p in model.named_parameters() if n in CAMADAS}

        # clipping só depois de medir; clip_hits conta quantos passos foram cortados
        norm_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if float(norm_clip) > args.clip: clip_hits += 1

        for o in opts: o.step()
        for s in scheds: s.step()
        step += 1; n_steps += 1

        if not math.isfinite(loss.item()) or not math.isfinite(g_norm):
            divergiu = True; break
        if perda_inicial > 0 and loss.item() > args.div_mult * perda_inicial:
            divergiu = True; break

        if step % 20 == 0:
            row = {'passo': step, 'epoca': ep, 'perda': loss.item(),
                    'grad': g_norm, 'theta': th_norm,
                    'lr_0': opts[0].param_groups[0]['lr']}
            if len(opts) == 2: row['lr_1'] = opts[1].param_groups[0]['lr']
            for n, p in model.named_parameters():
                if n in CAMADAS:
                    rho = float((p - antes[n]).norm() / (antes[n].norm() + 1e-12))
                    row[f'rho/{n}']  = rho
                    row[f'gn/{n}']   = gn_pre.get(n, 0.0)
                    row[f'thn/{n}']  = float(p.norm())
            log_rows.append(row)

    # --- fim da época: validação. Sem early stopping: só guardamos o checkpoint de menor val loss ---
    mse_log, mae_bruto, _ = val_metrics()
    historico_val.append({'epoca': ep, 'val_loss': mse_log, 'val_mae': mae_bruto})
    if mse_log < best_val:
        best_val = mse_log
        best_val_mae = mae_bruto
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if divergiu: break

test_mae = None; test_rmse = None
if args.avaliar_teste and best_state is not None:
    model.load_state_dict(best_state); model.eval()
    preds = []
    with torch.no_grad():
        for xb, yb, _ in make_loader(Xte, yte_l, t_te, False, args.seed):
            preds.append(model(xb.to(device)).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    preds_bruto = np.expm1(preds).clip(min=0)
    test_mae  = float(np.mean(np.abs(preds_bruto - yte_r)))
    test_rmse = float(np.sqrt(np.mean((preds_bruto - yte_r) ** 2)))

pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
val_df = pd.DataFrame(historico_val)
val_df.to_csv(LOG_PATH.replace('.csv', '_val.csv'), index=False)

comando = ' '.join(sys.argv)
row = {
    'id': args.id, 'etapa': args.etapa, 'otimizador': args.otimizador,
    'lr': args.lr, 'lr_adamw': args.lr_adamw if args.otimizador == 'muon' else None,
    'agenda': args.agenda, 'semente': args.seed,
    'passos': step, 'lote': args.lote,
    'weight_decay': args.wd_adamw if args.otimizador == 'adamw' else
                     args.wd_adamw if args.otimizador == 'muon' else args.wd_sgd_adam,
    'clip': args.clip,
    'melhor_val_loss': best_val,
    'val_loss_final': historico_val[-1]['val_loss'] if historico_val else None,
    'val_mae_bruto': best_val_mae,
    'test_mae': test_mae, 'test_rmse': test_rmse,
    'divergiu': divergiu,
    'tempo_s': time.time() - t0,
    'clip_hits': clip_hits, 'n_passos_exec': n_steps,
    'caminho_do_log': LOG_PATH, 'comando': comando,
}
hdr = not os.path.exists(EXP_CSV)
pd.DataFrame([row]).to_csv(EXP_CSV, mode='a', header=hdr, index=False)
print(json.dumps(row, indent=2, ensure_ascii=False))
