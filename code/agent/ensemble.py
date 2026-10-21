"""Ансамбль кандидатов по OOF: жадный отбор Caruana с повторами.

Начинаем с лучшего одиночного кандидата и на каждом шаге добавляем того,
кто сильнее улучшает метрику среднего. Повторы дают веса без оптимизатора.
Все кандидаты обязаны быть посчитаны на одних фолдах, иначе OOF несравнимы.
"""

import logging

import numpy as np

from agent.core import CoreResult
from agent.metrics import better, evaluate

log = logging.getLogger("solution")

MAX_STEPS = 25


def caruana(candidates: dict[str, CoreResult], y, metric: str, classes: list,
            folds) -> tuple[CoreResult, dict[str, int]] | None:
    """Смесь кандидатов как новый CoreResult или None, если смешивать нечего."""
    names = list(candidates)
    if len(names) < 2:
        return None
    mask = candidates[names[0]].oof_mask
    if any((c.oof_mask != mask).any() for c in candidates.values()):
        log.warning("ensemble: у кандидатов разные строки OOF, не смешиваем")
        return None
    y_m = np.asarray(y)[mask]

    def score(counts: dict[str, int]) -> float:
        total = sum(counts.values())
        blend = sum(candidates[n].oof[mask] * c for n, c in counts.items()) / total
        return evaluate(metric, y_m, blend, classes)

    def argbest(scores: dict[str, float]) -> str:
        top = next(iter(scores))
        for n, v in scores.items():
            if better(metric, v, scores[top]):
                top = n
        return top

    first = argbest({n: c.score for n, c in candidates.items()})
    counts, best = {first: 1}, candidates[first].score
    for _ in range(MAX_STEPS):
        trials = {n: score({**counts, n: counts.get(n, 0) + 1}) for n in names}
        n_best = argbest(trials)
        if not better(metric, trials[n_best], best):
            break
        counts[n_best] = counts.get(n_best, 0) + 1
        best = trials[n_best]
    # Один участник тоже результат: лучший одиночный кандидат может быть не текущей моделью.
    total = sum(counts.values())
    oof = sum(candidates[n].oof * c for n, c in counts.items()) / total
    test_pred = sum(candidates[n].test_pred * c for n, c in counts.items()) / total
    y_all = np.asarray(y)
    fold_scores = [evaluate(metric, y_all[va], oof[va], classes) for _, va in folds.splits]
    log.info("ensemble: веса %s, %s OOF %.5f", counts, metric, best)
    return CoreResult(oof, mask, test_pred, best, fold_scores, 0), counts


def median_blend(candidates: dict[str, CoreResult], y, metric: str, folds) -> CoreResult | None:
    """Медиана кандидатов для регрессии. Отбор по OOF не видит сдвига, которого нет в train:
    на reg_holdout XGBoost был лучшим по OOF и худшим на test (новый штат), Caruana дал
    ему вес 5/7 и RMSE 53 против 39,6 у медианы. Медиана переживает одну сломанную модель."""
    if len(candidates) < 3:
        return None
    results = list(candidates.values())
    mask = results[0].oof_mask
    oof = np.median(np.column_stack([r.oof for r in results]), axis=1)
    test_pred = np.median(np.column_stack([r.test_pred for r in results]), axis=1)
    y = np.asarray(y)
    score = evaluate(metric, y[mask], oof[mask], [])
    fold_scores = [evaluate(metric, y[va], oof[va], []) for _, va in folds.splits]
    log.info("ensemble: медиана %d кандидатов %s, %s OOF %.5f", len(results), list(candidates), metric, score)
    return CoreResult(oof, mask, test_pred, score, fold_scores, 0)
