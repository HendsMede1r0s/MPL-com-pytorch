import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def fixar_sementes(seed: int = 42) -> None:
    """Fixa sementes aleatórias do Python, NumPy e PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def treinar(
    modelo: nn.Module,
    dl_tr: DataLoader,
    dl_val: DataLoader,
    loss_fn: nn.Module,
    otimizador: torch.optim.Optimizer,
    n_epocas: int = 200,
    paciencia: int = 20,
    device: str | torch.device = "cpu",
) -> dict[str, list[float]]:
    """Treina a MLP com validação e early stopping, restaurando o melhor estado."""
    modelo.to(device)
    hist: dict[str, list[float]] = {"treino": [], "val": []}
    melhor = float("inf")
    espera = 0
    melhor_estado = copy.deepcopy(modelo.state_dict())

    for _ in range(n_epocas):
        modelo.train()
        soma = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            if isinstance(loss_fn, (nn.BCEWithLogitsLoss, nn.MSELoss, nn.L1Loss)):
                if yb.ndim == 1:
                    yb = yb.unsqueeze(1)
                yb = yb.float()
            otimizador.zero_grad()
            perda = loss_fn(modelo(xb), yb)
            perda.backward()
            otimizador.step()
            soma += perda.item() * len(xb)
        hist["treino"].append(soma / len(dl_tr.dataset))

        modelo.eval()
        soma_val = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                if isinstance(loss_fn, (nn.BCEWithLogitsLoss, nn.MSELoss, nn.L1Loss)):
                    if yb.ndim == 1:
                        yb = yb.unsqueeze(1)
                    yb = yb.float()
                soma_val += loss_fn(modelo(xb), yb).item() * len(xb)
        val = soma_val / len(dl_val.dataset)
        hist["val"].append(val)

        if val < melhor - 1e-6:
            melhor = val
            espera = 0
            melhor_estado = copy.deepcopy(modelo.state_dict())
        else:
            espera += 1
            if espera >= paciencia:
                break

    modelo.load_state_dict(melhor_estado)
    return hist