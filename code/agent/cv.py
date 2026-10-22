"""Выбор схемы валидации по тому, чем test отличается от train.

1. Колонка-период: целые значения, в test только периоды не раньше последнего в train.
   Тогда валидация вперёд: учимся на периодах до p, проверяем на p.
2. Много полных дублей по признакам: StratifiedGroupKFold по хешу строки,
   иначе одинаковые строки попадут и в train, и в val фолда.
3. Иначе StratifiedKFold для классификации и KFold для регрессии.

Фолды считаются один раз, все модели учатся на одних и тех же.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

SEED = 0
N_FOLDS = 5
MAX_TIME_FOLDS = 3
DUP_SHARE = 0.10
INNER_SHARE = 0.15


@dataclass
class Folds:
    scheme: str
    splits: list[tuple[np.ndarray, np.ndarray]]
    period_col: str | None = None
    note: str = ""
    groups: np.ndarray | None = None
    periods: np.ndarray | None = None


# Сколько периодов должно остаться в обучении, когда последний уходит на early stopping.
MIN_FIT_PERIODS = 2


def inner_split(folds: Folds, tr: np.ndarray, share: float = INNER_SHARE) -> tuple[np.ndarray, np.ndarray]:
    """Делим train-часть фолда на обучение и early stopping. Останавливаться по val
    фолда нельзя: OOF тогда оптимистичен (на регрессии на 2,6 RMSE).
    Режем так же, как устроен сам фолд: последний период, группы или случайно.

    Последний период отрезаем, только если для обучения остаётся минимум MIN_FIT_PERIODS.
    Иначе в обучении один самый ранний период, у которого нет прошлого, и признаки истории
    сущности на фолде пустые: на reg_h4 (3 года) история отклонялась как бесполезная, хотя
    одно «значение прошлого года» давало RMSE 23 против 40 у модели."""
    rng = np.random.default_rng(SEED)
    if folds.periods is not None:
        p = folds.periods[tr]
        if len(np.unique(p)) > MIN_FIT_PERIODS:
            last = p == p.max()
            return tr[~last], tr[last]
    if folds.groups is not None:
        g = np.unique(folds.groups[tr])
        stop_groups = rng.choice(g, max(1, int(len(g) * share)), replace=False)
        stop = np.isin(folds.groups[tr], stop_groups)
        return tr[~stop], tr[stop]
    stop = rng.random(len(tr)) < share
    return tr[~stop], tr[stop]


def find_period(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> str | None:
    for col in features:
        a, b = train[col], test[col]
        if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
            continue
        a, b = a.dropna(), b.dropna()
        if a.empty or b.empty or (a % 1 != 0).any() or not 2 <= a.nunique() <= 50:
            continue
        if b.min() >= a.max() and not set(b.unique()) <= set(a.unique()):
            return col
    return None


def row_groups(train: pd.DataFrame, features: list[str]) -> np.ndarray:
    return pd.util.hash_pandas_object(train[features], index=False).to_numpy()


def make_folds(train: pd.DataFrame, test: pd.DataFrame, features: list[str], y: pd.Series, kind: str) -> Folds:
    n = len(train)
    period = find_period(train, test, features)
    if period is not None:
        values = np.sort(train[period].dropna().unique())
        splits = []
        for p in values[-min(MAX_TIME_FOLDS, len(values) - 1):]:
            tr = np.flatnonzero((train[period] < p).to_numpy())
            va = np.flatnonzero((train[period] == p).to_numpy())
            splits.append((tr, va))
        return Folds("time_forward", splits, period, f"валидация вперёд по {period}, периоды {list(values[-len(splits):])}",
                     periods=train[period].to_numpy())

    groups = row_groups(train, features)
    dup_share = 1 - len(np.unique(groups)) / n
    if kind != "regression" and dup_share > DUP_SHARE:
        cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(cv.split(np.zeros(n), y.astype(str), groups))
        return Folds("group_dups", splits, note=f"дубли по признакам {dup_share:.0%}, группы по хешу строки",
                     groups=groups)
    if dup_share > DUP_SHARE:
        splits = list(GroupKFold(N_FOLDS).split(np.zeros(n), y, groups))
        return Folds("group_dups", splits, note=f"дубли по признакам {dup_share:.0%}, группы по хешу строки",
                     groups=groups)

    if kind == "regression":
        splits = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(n)))
        return Folds("kfold", splits)
    # Классы реже числа фолдов StratifiedKFold не разложит, такие оставляем как есть: sklearn предупредит.
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(n), y.astype(str)))
    return Folds("stratified", splits)
