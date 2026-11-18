from pathlib import Path
import pandas as pd

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED_DIVISAO = 42
SEEDS_TREINO = [42, 100, 2024, 7, 99]

def obter_raiz_do_projeto() -> Path:
    """Retorna o caminho da pasta principal do projeto."""
    return Path(__file__).resolve().parent.parent

def carregar_dados_brutos(nome_do_arquivo: str | Path) -> pd.DataFrame:
    """Carrega um CSV pelo nome ou por caminho relativo/absoluto.

    Também aceita o formato usado nos notebooks, como
    ``"../data/raw/college_sleep_and_gpa.csv"``.
    """
    raiz = obter_raiz_do_projeto()
    caminho_informado = Path(nome_do_arquivo)
    candidatos = [caminho_informado] if caminho_informado.is_absolute() else [
        Path.cwd() / caminho_informado,
        raiz / caminho_informado,
        raiz / 'data' / 'raw' / caminho_informado.name,
    ]
    for caminho in candidatos:
        if caminho.is_file():
            return pd.read_csv(caminho)
    raise FileNotFoundError(f"Arquivo bruto não encontrado: {nome_do_arquivo}")

def carregar_dados_processados(nome_do_arquivo: str) -> pd.DataFrame:
    """Carrega um arquivo CSV da pasta data/processed/."""
    raiz = obter_raiz_do_projeto()
    caminho_do_arquivo = raiz / 'data' / 'processed' / nome_do_arquivo
    return pd.read_csv(caminho_do_arquivo)

def salvar_dados_processados(tabela: pd.DataFrame, nome_do_arquivo: str) -> None:
    """Salva a sua tabela (DataFrame) na pasta data/processed/."""
    raiz = obter_raiz_do_projeto()
    pasta_de_destino = raiz / 'data' / 'processed'
    pasta_de_destino.mkdir(parents=True, exist_ok=True)

    tabela.to_csv(pasta_de_destino / nome_do_arquivo, index=False)

def fixar_sementes(seed: int = 42):
    """Garante reprodutibilidade fixando as sementes de random, NumPy e PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preparar_dados_pytorch(
    caminho_csv: str,
    target_col: str,
    tipo: str = "reg",
    batch_size: int = 32,
    seed_divisao: int = SEED_DIVISAO,
):
    """
    Carrega o dataset limpo, executa a divisão Treino/Validação/Teste sem data leakage,
    padroniza features (e o target na regressão) e retorna os DataLoaders PyTorch.
    """
    df = pd.read_csv(caminho_csv)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # 1. Mesma divisão 80% Treino Completo / 20% Teste da atividade anterior (500 / 126)
    stratify_full = y if tipo == "class" else None
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed_divisao, stratify=stratify_full
    )

    # 2. Subconjunto de Validação: 20% dentro do treino (400 Treino / 100 Validação)
    stratify_val = y_train_full if tipo == "class" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=seed_divisao, stratify=stratify_val
    )

    # 3. Padronização das features (ajustada APENAS no treino)
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_val_sc = scaler_X.transform(X_val)
    X_test_sc = scaler_X.transform(X_test)

    # 4. Padronização do target apenas na regressão (ajustada APENAS no treino)
    scaler_y = None
    if tipo == "reg":
        scaler_y = StandardScaler()
        y_train_proc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_proc = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        y_test_proc = y_test  # Mantido na escala original para avaliação final
        y_dtype = torch.float32
    else:
        y_train_proc, y_val_proc, y_test_proc = y_train, y_val, y_test
        y_dtype = torch.long

    # 5. Conversão para Tensores e TensorDataset
    ds_train = TensorDataset(
        torch.tensor(X_train_sc, dtype=torch.float32),
        torch.tensor(y_train_proc, dtype=y_dtype),
    )
    ds_val = TensorDataset(
        torch.tensor(X_val_sc, dtype=torch.float32),
        torch.tensor(y_val_proc, dtype=y_dtype),
    )
    ds_test = TensorDataset(
        torch.tensor(X_test_sc, dtype=torch.float32),
        torch.tensor(y_test_proc, dtype=y_dtype),
    )

    # 6. Criação dos DataLoaders
    loaders = {
        "train": DataLoader(ds_train, batch_size=batch_size, shuffle=True),
        "val": DataLoader(ds_val, batch_size=batch_size, shuffle=False),
        "test": DataLoader(ds_test, batch_size=batch_size, shuffle=False),
    }

    return loaders, scaler_X, scaler_y, X_train_sc.shape[1]