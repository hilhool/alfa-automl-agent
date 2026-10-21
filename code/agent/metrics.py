"""Метрики агента. Для классификации на вход идёт матрица вероятностей
в порядке spec.classes, в метку её переводит to_labels.

Модуль закрыт от LLM: метрику и её направление задаёт только код.
"""

import numpy as np
from sklearn import metrics as skm

GREATER_IS_BETTER = {"rmse": False, "mae": False, "logloss": False,
                     "r2": True, "accuracy": True, "f1_macro": True, "f1_weighted": True,
                     "f1_binary": True, "roc_auc": True}


def to_labels(proba: np.ndarray, classes: list, weights: np.ndarray | None = None) -> np.ndarray:
    scaled = proba if weights is None else proba * weights
    return np.asarray(classes, dtype=object)[scaled.argmax(axis=1)]


POSITIVE_NAMES = ("1", "1.0", "true", "yes", "да", "positive")


def positive_class(classes: list, y) -> object:
    """Для binary F1 положительный класс — 1/True/yes, если такой есть, иначе более редкий."""
    for c in classes:
        if str(c).strip().lower() in POSITIVE_NAMES:
            return c
    values, counts = np.unique(np.asarray(y, dtype=object).astype(str), return_counts=True)
    rare = values[counts.argmin()]
    return next(c for c in classes if str(c) == rare)


def evaluate(metric: str, y_true, pred: np.ndarray, classes: list | None = None) -> float:
    y_true = np.asarray(y_true)
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_true.astype(float) - pred) ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(y_true.astype(float) - pred)))
    if metric == "r2":
        return float(skm.r2_score(y_true, pred))
    if metric == "logloss":
        return float(skm.log_loss(y_true, np.clip(pred, 1e-15, 1), labels=classes))
    if metric == "roc_auc":
        if len(classes) == 2:
            positive = positive_class(classes, y_true)
            column = classes.index(positive)
            return float(skm.roc_auc_score(y_true == positive, pred[:, column]))
        return float(skm.roc_auc_score(y_true, pred, multi_class="ovr", labels=classes))
    labels = to_labels(pred, classes)
    if metric == "accuracy":
        return float(skm.accuracy_score(y_true, labels))
    if metric == "f1_binary":
        return float(skm.f1_score(y_true, labels, average="binary", pos_label=positive_class(classes, y_true)))
    average = "weighted" if metric == "f1_weighted" else "macro"
    return float(skm.f1_score(y_true, labels, average=average, labels=classes))


def safe_evaluate(metric: str, y_true, pred: np.ndarray, classes: list | None = None) -> float:
    """Метрика фолда; NaN, если на фолде её не посчитать (один класс в val для AUC)."""
    try:
        return evaluate(metric, y_true, pred, classes)
    except ValueError:
        return float("nan")


def better(metric: str, a: float, b: float) -> bool:
    return a > b if GREATER_IS_BETTER[metric] else a < b
