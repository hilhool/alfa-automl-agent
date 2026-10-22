"""Метрики стенда и перевод в баллы по приложению 3 кейса.

Считаются через sklearn напрямую, независимо от agent/metrics.py: ошибка в метрике
агента не должна прятаться за той же ошибкой в проверке.
"""

import numpy as np
from sklearn import metrics as skm


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


def macro_f1(y_true, y_pred) -> float:
    return float(skm.f1_score(y_true, y_pred, average="macro"))


def _f(y) -> np.ndarray:
    return np.asarray(y, dtype=float)


def _labels(t, p):
    return np.asarray(t).astype(str), np.asarray(p).astype(str)


METRICS = {
    "regression": rmse, "rmse": rmse, "multiclass": macro_f1,
    "mae": lambda t, p: float(np.mean(np.abs(_f(t) - _f(p)))),
    "r2": lambda t, p: float(skm.r2_score(_f(t), _f(p))),
    "rmsle": lambda t, p: float(np.sqrt(np.mean((np.log1p(np.maximum(_f(t), 0)) - np.log1p(np.maximum(_f(p), 0))) ** 2))),
    "mape": lambda t, p: float(np.mean(np.abs(_f(t) - _f(p)) / np.maximum(np.abs(_f(t)), 1e-9))),
    "roc_auc": lambda t, p: float(skm.roc_auc_score(t, p)),
    "gini": lambda t, p: float(2 * skm.roc_auc_score(t, p) - 1),
    "pr_auc": lambda t, p: float(skm.average_precision_score(t, p)),
    "accuracy": lambda t, p: float(skm.accuracy_score(*_labels(t, p))),
    "balanced_accuracy": lambda t, p: float(skm.balanced_accuracy_score(*_labels(t, p))),
    "mcc": lambda t, p: float(skm.matthews_corrcoef(*_labels(t, p))),
    "f1_macro": lambda t, p: float(skm.f1_score(*_labels(t, p), average="macro")),
    "f1_weighted": lambda t, p: float(skm.f1_score(*_labels(t, p), average="weighted")),
}
# Метрики-ошибки: балл как у регрессии в кейсе, 1 − value/B.
LOSSES = ("regression", "rmse", "mae", "rmsle", "mape")


def score(kind: str, value: float, baseline: float) -> float:
    """Ошибка: 100·max(0, 1 − value/B), как S_reg. Качество с максимумом 1:
    100·max(0, (value − B)/(1 − B)), как S_cls."""
    if kind in LOSSES:
        return 100 * max(0.0, 1 - value / baseline)
    return 100 * max(0.0, (value - baseline) / (1 - baseline))
