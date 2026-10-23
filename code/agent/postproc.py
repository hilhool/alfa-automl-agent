"""Подстройка под метрику на OOF.

Для F1 (macro, weighted) подбираем множители к вероятностям классов
покоординатным подъёмом. Чтобы не подогнать множители под шум OOF, сначала
учим их на половине фолдов и проверяем на другой, и наоборот. Принимаем,
только если обе половины выиграли. Финальные множители учим на всём OOF.
"""

import logging
import time

import numpy as np

from agent.metrics import evaluate

log = logging.getLogger("solution")

GRID = np.array([0.33, 0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0])
PASSES = 3
MAX_RATIO = 10.0
TUNED_METRICS = ("f1_macro", "f1_weighted", "f1_binary", "balanced_accuracy", "mcc", "kappa", "qwk")


def fit_weights(y, proba: np.ndarray, classes: list, metric: str, deadline: float | None = None) -> np.ndarray:
    """Покоординатный подъём. Множители перебираются от веса класса на начало шага: раньше база
    менялась внутри перебора, ×2 и следом ×3 давали ×6, за три прохода веса уходили до 3^k
    (Generic 0,06 при Analysis 0,72) и прыгали между прогонами. Вес держим в [1/MAX_RATIO, MAX_RATIO]."""
    w = np.ones(proba.shape[1])
    best = evaluate(metric, y, proba * w, classes)
    for _ in range(PASSES):
        improved = False
        for j in range(len(w)):
            start, chosen = w[j], w[j]
            for factor in GRID:
                candidate = start * factor
                if not 1 / MAX_RATIO <= candidate <= MAX_RATIO:
                    continue
                w[j] = candidate
                s = evaluate(metric, y, proba * w, classes)
                if s > best + 1e-6:
                    best, chosen, improved = s, candidate, True
            w[j] = chosen
            if deadline is not None and time.monotonic() > deadline:
                return w / w.max()
        if not improved:
            break
    return w / w.max()


def class_weights(y, oof: np.ndarray, mask: np.ndarray, fold_of: np.ndarray, classes: list,
                  metric: str, deadline: float | None = None) -> np.ndarray | None:
    """Множители классов или None, если на перекрёстной проверке они не помогли.
    deadline: на 60 тыс. строк подбор идёт около 100 с и растёт как n·k, поэтому он ограничен."""
    if metric not in TUNED_METRICS:
        return None
    y = np.asarray(y, dtype=object)
    folds = np.unique(fold_of[mask])
    half = np.isin(fold_of, folds[::2]) & mask
    other = np.isin(fold_of, folds[1::2]) & mask
    gains = []
    for a, b in ((half, other), (other, half)):
        w = fit_weights(y[a], oof[a], classes, metric, deadline)
        gains.append(evaluate(metric, y[b], oof[b] * w, classes) - evaluate(metric, y[b], oof[b], classes))
    log.info("postproc: прирост от весов классов на чужой половине фолдов %s", np.round(gains, 4))
    if min(gains) <= 0:
        return None
    w = fit_weights(y[mask], oof[mask], classes, metric, deadline)
    log.info("postproc: веса классов %s", dict(zip(map(str, classes), w.round(2))))
    return w
