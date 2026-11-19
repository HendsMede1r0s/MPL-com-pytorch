#RODE NO TERMINAL: python src/inference.py --modelo models/modelo_classificacao.pt --entrada data/processed/dataset_classification.csv



import argparse
import sys
from pathlib import Path
import pandas as pd
import torch

raiz_projeto = str(Path(__file__).resolve().parent.parent)
if raiz_projeto not in sys.path:
    sys.path.append(raiz_projeto)

from src.models import MLP


def executar_inferencia(caminho_modelo: str, caminho_entrada: str, caminho_saida: str | None = None) -> pd.DataFrame:
    ckpt = torch.load(caminho_modelo, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]

    modelo = MLP(
        in_dim=cfg["in_dim"],
        out_dim=cfg["out_dim"],
        hidden=tuple(cfg["hidden"]),
        activation=cfg["activation"],
        dropout=cfg["dropout"],
    )
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()

    df = pd.read_csv(caminho_entrada)
    cols_features = ckpt["features"]
    X_tensor = torch.tensor(df[cols_features].to_numpy(), dtype=torch.float32)

    # Aplica exatamente a mesma padronização do treino salva no checkpoint
    X_scaled = (X_tensor - ckpt["x_media"]) / ckpt["x_desvio"]

    with torch.inference_mode():
        saida = modelo(X_scaled).ravel()

    df_resultado = df.copy()
    if ckpt["tarefa"] == "regressao":
        preds_gpa = saida * ckpt["y_desvio"] + ckpt["y_media"]
        df_resultado["pred_term_gpa"] = preds_gpa.numpy()
    else:
        limiar = ckpt.get("limiar", 0.5)
        classes = ckpt.get("classes", ["0", "1"])
        probs = torch.sigmoid(saida).numpy()
        preds_idx = (probs >= limiar).astype(int)
        df_resultado["prob_classe_1"] = probs
        df_resultado["pred_classe"] = preds_idx
        df_resultado["pred_rotulo"] = [classes[i] for i in preds_idx]

    if caminho_saida:
        df_resultado.to_csv(caminho_saida, index=False)

    return df_resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferência limpa com MLP PyTorch salva.")
    parser.add_argument("--modelo", required=True, help="Caminho para o arquivo .pt")
    parser.add_argument("--entrada", required=True, help="Caminho para o CSV com novas amostras")
    parser.add_argument("--saida", default=None, help="Caminho opcional para salvar as previsões")
    args = parser.parse_args()

    resultado = executar_inferencia(args.modelo, args.entrada, args.saida)
    print(resultado.head(10).to_string(index=False))