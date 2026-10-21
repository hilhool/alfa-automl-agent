"""Повторяющиеся сущности между периодами и признаки по их прошлому.

Ключ сущности ищем жадно: на каждом шаге добавляем колонку, после которой
больше строк test находят свою сущность в train, а внутри одного периода
ключ всё реже повторяется (одна сущность — одна строка за период).
Погодные и прочие колонки, которые меняются у сущности между периодами,
рвут совпадения с train и сами не проходят.

Признаки считаем строго по периодам раньше текущей строки, на train и на test
одинаково. Таргет своей строки и будущих периодов в признак не попадает.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger("solution")

MAX_KEY_COLS = 6
MIN_TEST_COVERAGE = 0.2
MAX_STD_RATIO = 0.5      # std таргета внутри ключа против общего
MAX_CARDINALITY = 0.9    # колонка почти уникальна в каждой строке — это не свойство сущности
LAGS = (1, 2, 3)
N_CHANGING = 3
MIN_CHANGE_ROWS = 100
MIN_CHANGE_RATE, MAX_CHANGE_RATE = 0.01, 0.99


@dataclass
class EntityKey:
    cols: list[str]
    test_coverage: float
    rows_per_key_period: float
    std_ratio: float


def _key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return pd.Series(pd.util.hash_pandas_object(df[cols].astype(str), index=False).to_numpy(), index=df.index)


def _stats(train, test, cols, period) -> tuple[float, float]:
    k_tr, k_te = _key(train, cols), _key(test, cols)
    coverage = float(k_te.isin(set(k_tr)).mean())
    per_period = float(pd.DataFrame({"k": k_tr, "p": train[period]}).groupby(["k", "p"]).size().mean())
    return coverage, per_period


def find_key(train: pd.DataFrame, test: pd.DataFrame, features: list[str], period: str,
             target: str) -> EntityKey | None:
    candidates = [c for c in features if c != period and train[c].nunique() < MAX_CARDINALITY * len(train)]
    chosen: list[str] = []
    best_score = 0.0
    for _ in range(MAX_KEY_COLS):
        step = None
        for col in candidates:
            if col in chosen:
                continue
            coverage, per_period = _stats(train, test, chosen + [col], period)
            score = coverage / per_period
            if score > best_score * 1.01:
                best_score, step = score, (col, coverage, per_period)
        if step is None:
            break
        chosen.append(step[0])
        log.info("entity: +%s, покрытие test %.3f, строк на ключ в периоде %.3f", *step)
    if not chosen:
        return None

    coverage, per_period = _stats(train, test, chosen, period)
    y = train[target]
    if not pd.api.types.is_numeric_dtype(y):
        return None
    within = y.groupby(_key(train, chosen)).std().dropna()
    std_ratio = float(within.mean() / y.std()) if len(within) else 1.0
    key = EntityKey(chosen, coverage, per_period, std_ratio)
    if coverage < MIN_TEST_COVERAGE or std_ratio > MAX_STD_RATIO:
        log.info("entity: ключ %s отклонён: %s", chosen, key)
        return None
    log.info("entity: ключ %s", key)
    return key


def add_history(train: pd.DataFrame, test: pd.DataFrame, key: EntityKey, period: str,
                target: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Для каждой строки: последнее и среднее значение таргета сущности в прошлых периодах,
    число таких периодов, разрыв до последнего и наклон по двум последним."""
    hist = pd.DataFrame({"k": _key(train, key.cols), "p": train[period], "y": train[target]})
    hist = hist.groupby(["k", "p"], as_index=False)["y"].mean().sort_values(["k", "p"])

    def build(df: pd.DataFrame) -> pd.DataFrame:
        rows = pd.DataFrame({"k": _key(df, key.cols).to_numpy(), "p": df[period].to_numpy(), "_i": np.arange(len(df))})
        merged = rows.merge(hist, on="k", suffixes=("", "_h"))
        merged = merged[merged["p_h"] < merged["p"]].sort_values(["_i", "p_h"])
        g = merged.groupby("_i")
        out = pd.DataFrame(index=np.arange(len(df)))
        out["ent_last"] = g["y"].last()
        out["ent_mean"] = g["y"].mean()
        out["ent_count"] = g.size()
        out["ent_gap"] = rows["p"] - g["p_h"].last()
        prev = g["y"].nth(-2)
        prev.index = merged.loc[prev.index, "_i"].to_numpy()
        out["ent_trend"] = out["ent_last"] - prev
        out["ent_count"] = out["ent_count"].fillna(0)
        return out

    names = ["ent_last", "ent_mean", "ent_count", "ent_gap", "ent_trend"]
    f_tr, f_te = build(train), build(test)
    train = pd.concat([train.reset_index(drop=True), f_tr[names]], axis=1)
    test = pd.concat([test.reset_index(drop=True), f_te[names]], axis=1)
    log.info("entity: история есть у %.1f%% train и %.1f%% test", 100 * f_tr["ent_count"].gt(0).mean(),
             100 * f_te["ent_count"].gt(0).mean())
    return train, test, names


def changing_columns(train: pd.DataFrame, key: EntityKey, period: str, target: str,
                     features: list[str]) -> list[str]:
    """Числовые колонки, которые у сущности меняются между периодами, по силе связи с таргетом.
    Их прошлое значение и изменение — то, чего бустинг сам из строки не увидит."""
    k = _key(train, key.cols)
    order = np.lexsort((train[period].to_numpy(), k.to_numpy()))
    df = train.iloc[order]
    same = (k.iloc[order].to_numpy()[1:] == k.iloc[order].to_numpy()[:-1])
    scored = []
    for col in features:
        if col in key.cols or col == period or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        v = df[col].to_numpy(dtype=float)
        cur, prev = v[1:][same], v[:-1][same]
        ok = ~np.isnan(cur) & ~np.isnan(prev)
        if ok.sum() < MIN_CHANGE_ROWS:
            continue
        rate = float((cur[ok] != prev[ok]).mean())
        if not MIN_CHANGE_RATE <= rate <= MAX_CHANGE_RATE:
            continue
        corr = abs(df[col].corr(df[target], method="spearman"))
        if np.isfinite(corr):
            scored.append((corr, col))
    return [c for _, c in sorted(scored, reverse=True)[:N_CHANGING]]


def add_lags(train: pd.DataFrame, test: pd.DataFrame, key: EntityKey, period: str, target: str,
             features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Таргет сущности ровно 1, 2, 3 периода назад и прошлое значение / изменение
    меняющихся колонок. Берём только из train: test — всегда последний период."""
    cols = changing_columns(train, key, period, target, features)
    table = train[cols + [target]].copy()
    table["k"], table["p"] = _key(train, key.cols).to_numpy(), train[period].to_numpy()
    table = table.groupby(["k", "p"], as_index=False).mean()

    def build(df: pd.DataFrame) -> pd.DataFrame:
        rows = pd.DataFrame({"k": _key(df, key.cols).to_numpy(), "p": df[period].to_numpy()})
        out = pd.DataFrame(index=np.arange(len(df)))
        for lag in LAGS:
            shifted = table[["k", "p", target] + (cols if lag == 1 else [])].copy()
            shifted["p"] = shifted["p"] + lag
            m = rows.merge(shifted, on=["k", "p"], how="left")
            out[f"ent_lag{lag}"] = m[target].to_numpy()
            if lag == 1:
                for c in cols:
                    out[f"{c}_prev"] = m[c].to_numpy()
                    out[f"{c}_diff"] = df[c].to_numpy(dtype=float) - m[c].to_numpy()
        return out

    f_tr, f_te = build(train), build(test)
    names = list(f_tr.columns)
    log.info("entity: лаги %s, меняющиеся колонки %s", LAGS, cols)
    return (pd.concat([train.reset_index(drop=True), f_tr], axis=1),
            pd.concat([test.reset_index(drop=True), f_te], axis=1), names)
