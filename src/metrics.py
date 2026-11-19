import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def calcular_metricas_regressao(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula MSE, RMSE, MAE, R2 e MAPE na escala original do target."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    mascara_valida = np.abs(y_true) > 1e-8
    if np.any(mascara_valida):
        mape = float(
            np.mean(np.abs((y_true[mascara_valida] - y_pred[mascara_valida]) / y_true[mascara_valida])) * 100.0
        )
    else:
        mape = float("nan")

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE (%)": mape,
    }


def calcular_metricas_classificacao(
    y_true: np.ndarray, y_prob: np.ndarray, limiar: float = 0.5
) -> dict[str, float]:
    """Calcula Acurácia, Precisão, Recall, F1, ROC-AUC e PR-AUC a partir das probabilidades."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= limiar).astype(int)

    return {
        "Acuracia": float(accuracy_score(y_true, y_pred)),
        "Precisao": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
    }