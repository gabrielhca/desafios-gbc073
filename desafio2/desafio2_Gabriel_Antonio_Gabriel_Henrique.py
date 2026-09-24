"""
Autores: Gabriel Antonio Gomes Moutinho
         Gabriel Henrique Carneiro Amorim
"""
"""
Desafio 2: Ativação e inicialização em redes profundas (sem normalização)
GBC073 — Inteligência Computacional (FACOM/UFU) — Prof. Marcelo Keese Albertini

Entrega oficial:

*   Copie para desafio2/desafio2_nomes.py no seu repositório.
*   Ideia: em vez de decorar Xavier/He, CALIBRAR o ganho numericamente para a ativação escolhida.
*   Se z ~ N(0, 1) e W ~ N(0, s^2), a pré-ativação da próxima camada tem variância
    fan_in * s^2 * E[f(z)^2]
Para mantê-la em 1 camada após camada:  s^2 = 1 / (fan_in * E[f(z)^2]).
* Troque `ativacao` por outra função e a inicialização se ajusta sozinha.
* Rode:  python harness_desafio2.py exemplo_submissao_d2.py --rapido

### Referências da API do PyTorch usadas aqui:
  
  * torch.nn.functional.gelu   https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html
  * torch.Tensor.normal_       https://docs.pytorch.org/docs/stable/generated/torch.Tensor.normal_.html
  * torch.Tensor.zero_         https://docs.pytorch.org/docs/stable/generated/torch.Tensor.zero_.html
  * torch.no_grad              https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
  * torch.Generator            https://docs.pytorch.org/docs/stable/generated/torch.Generator.html
  * torch.nn.init (Xavier, He, calculate_gain — para comparar com a conta feita à mão) https://docs.pytorch.org/docs/stable/nn.init.html
  * outras ativações: relu, leaky_relu, elu, selu, silu em https://docs.pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions
"""


import math
import torch

def ativacao(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x, negative_slope=0.03)
    #função de ativação leaky_relu. Alteramos o slope de 0.01 (padrão) pra 0.03 pois ao fazermos testes obtivemos melhor resultado
    

_g = torch.Generator().manual_seed(0)
_E_f2 = ativacao(torch.randn(1_000_000, generator=_g)).pow(2).mean().item()

@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                fan_in: int, fan_out: int,
                camada: int, n_camadas: int) -> None:
    
    #primeira camada sem penalização pois não recebe dados de nenhuma camada anterior (recebe x normalizado)
    if camada == 1:
        desvio = math.sqrt(1.0 / fan_in)
    else:
        #camadas 2..n_camadas compensam a energia consumida pela ativação
        #(a última também recebe, e depois ganha o ajuste de logits abaixo)
        desvio = math.sqrt(1.0 / (fan_in * _E_f2))

    # diminuição para ajustar para a última camada, evitando a saturação
    if camada == n_camadas:
        desvio *= 0.5

    W.normal_(0.0, desvio)
    b.zero_()
