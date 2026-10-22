"""Ансамбль кандидатов по OOF: жадный отбор Caruana с повторами.

Начинаем с лучшего одиночного кандидата и на каждом шаге добавляем того,
кто сильнее улучшает метрику среднего. Повторы дают веса без оптимизатора.
Все кандидаты обязаны быть посчитаны на одних фолдах, иначе OOF несравнимы.
"""

import logging

import numpy as np

from agent.core import CoreResult
from agent.metrics import GREATER_IS_BETTER, better, evaluate, safe_evaluate

log = logging.getLogger("solution")

MAX_STEPS = 25
BROKEN_GAP = 0.05


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
    fold_scores = [safe_evaluate(metric, y_all[va], oof[va], classes) for _, va in folds.splits]
    log.info("ensemble: веса %s, %s OOF %.5f", counts, metric, best)
    return CoreResult(oof, mask, test_pred, best, fold_scores, 0), counts


def hedge_blend(a: CoreResult, b: CoreResult, y, metric: str, classes: list, folds) -> CoreResult:
    """Среднее двух смесей. На проверочном наборе OpenML (14 задач без сдвига) Caruana по OOF
    в среднем лучше простого усреднения (балл к RF 14,2 против 13,6), но на маленьких данных
    промахивается (house: 7,8 против 14,0). Их среднее дало 14,4 и не проваливается."""
    oof, test_pred = (a.oof + b.oof) / 2, (a.test_pred + b.test_pred) / 2
    mask, y = a.oof_mask, np.asarray(y)
    score = evaluate(metric, y[mask], oof[mask], classes)
    fold_scores = [safe_evaluate(metric, y[va], oof[va], classes) for _, va in folds.splits]
    log.info("ensemble: среднее Caruana и усреднения, %s OOF %.5f", metric, score)
    return CoreResult(oof, mask, test_pred, score, fold_scores, 0)


def mean_blend(candidates: dict[str, CoreResult], y, metric: str, classes: list, folds) -> CoreResult | None:
    """Среднее вероятностей всех кандидатов для классификации. На holdout со сдвигом классов
    (cls_shift) Caruana по OOF выбрал один xgb и после множителей классов дал F1 0,570, среднее —
    0,592; без сдвига (cls_holdout) 0,638 против 0,638. Сломанного кандидата (хуже лучшего
    по OOF больше чем на BROKEN_GAP) не берём."""
    best = max(c.score for c in candidates.values()) if GREATER_IS_BETTER[metric] else \
        min(c.score for c in candidates.values())
    keep = {n: c for n, c in candidates.items() if abs(c.score - best) <= BROKEN_GAP * abs(best)}
    if len(keep) < 2:
        return None
    results = list(keep.values())
    mask = results[0].oof_mask
    oof = np.mean([r.oof for r in results], axis=0)
    test_pred = np.mean([r.test_pred for r in results], axis=0)
    y = np.asarray(y)
    score = evaluate(metric, y[mask], oof[mask], classes)
    fold_scores = [safe_evaluate(metric, y[va], oof[va], classes) for _, va in folds.splits]
    log.info("ensemble: среднее %d кандидатов %s, %s OOF %.5f", len(results), list(keep), metric, score)
    return CoreResult(oof, mask, test_pred, score, fold_scores, 0)


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
    fold_scores = [safe_evaluate(metric, y[va], oof[va], []) for _, va in folds.splits]
    log.info("ensemble: медиана %d кандидатов %s, %s OOF %.5f", len(results), list(candidates), metric, score)
    return CoreResult(oof, mask, test_pred, score, fold_scores, 0)
