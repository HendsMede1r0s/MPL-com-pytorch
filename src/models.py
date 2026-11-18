import torch
import torch.nn as nn


class MLP(nn.Module):
    """Rede Neural Perceptron Multicamadas (MLP) parametrizável."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: tuple[int, ...] | list[int] = (32, 32),
        activation: str = "ReLU",
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        if not hasattr(nn, activation):
            raise ValueError(f"Ativação '{activation}' não encontrada em torch.nn.")
        act = getattr(nn, activation)

        camadas: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            camadas.append(nn.Linear(d, h))
            if use_batchnorm:
                camadas.append(nn.BatchNorm1d(h))
            camadas.append(act())
            if dropout > 0:
                camadas.append(nn.Dropout(dropout))
            d = h

        # Camada de saída linear: escalar livre na regressão ou logits na classificação
        camadas.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*camadas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def contar_parametros(self) -> int:
        """Retorna o total de parâmetros treináveis da rede."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)