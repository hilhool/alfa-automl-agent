"""Метрики и перевод в баллы по приложению 3 кейса."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


METRICS = {"regression": rmse, "multiclass": macro_f1,
           "mae": lambda t, p: float(np.mean(np.abs(np.asarray(t, float) - np.asarray(p, float)))),
           "roc_auc": lambda t, p: float(roc_auc_score(t, p)),
           "accuracy": lambda t, p: float(accuracy_score(t, p))}


def score(kind: str, value: float, baseline: float) -> float:
    """S_reg = 100·max(0, 1 − RMSE/B), S_cls = 100·max(0, (F1 − B)/(1 − B))."""
    if kind == "regression":
        return 100 * max(0.0, 1 - value / baseline)
    return 100 * max(0.0, (value - baseline) / (1 - baseline))
