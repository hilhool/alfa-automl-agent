"""Ядро без LLM: предобработка, LightGBM на зафиксированных фолдах, рефит на всём train.

Это нижняя ступень над константой. Всё остальное (признаки, цикл гипотез,
тяжёлые модели) сравнивается с ним на тех же фолдах.
"""

import logging
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from agent.cv import Folds, inner_split
from agent.metrics import evaluate
from agent.spec import TaskSpec

log = logging.getLogger("solution")

N_JOBS = 8
LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=20, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=N_JOBS, seed=0)
MAX_ROUNDS, EARLY_STOP = 5000, 100
DATE_SAMPLE, DATE_SHARE = 500, 0.95


@dataclass
class CoreResult:
    oof: np.ndarray          # (n,) для регрессии, (n, k) вероятности для классификации
    oof_mask: np.ndarray     # строки, у которых есть OOF
    test_pred: np.ndarray
    score: float
    fold_scores: list[float]
    rounds: int


def feature_columns(train: pd.DataFrame, spec: TaskSpec) -> list[str]:
    return [c for c in train.columns if c not in (spec.id_col, spec.target) and train[c].nunique(dropna=False) > 1]


def parse_dates(s: pd.Series) -> pd.Series | None:
    """Строковая колонка, в которой почти все значения читаются как дата, иначе None."""
    sample = s.dropna().astype(str).head(DATE_SAMPLE)
    if sample.empty or sample.str.len().median() < 6 or not sample.str.contains(r"\d", regex=True).all():
        return None
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    if parsed.notna().mean() < DATE_SHARE:
        return None
    return pd.to_datetime(s, errors="coerce", format="mixed")


def expand_dates(x_tr: pd.DataFrame, x_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Дату раскладываем на части: бустинг не видит в строке ни сезон, ни день недели."""
    for col in [c for c in x_tr.columns if not pd.api.types.is_numeric_dtype(x_tr[c])]:
        d_tr = parse_dates(x_tr[col])
        if d_tr is None:
            continue
        d_te = pd.to_datetime(x_te[col], errors="coerce", format="mixed")
        for frame, d in ((x_tr, d_tr), (x_te, d_te)):
            frame[f"{col}_year"], frame[f"{col}_month"] = d.dt.year, d.dt.month
            frame[f"{col}_dow"], frame[f"{col}_doy"] = d.dt.dayofweek, d.dt.dayofyear
            frame[f"{col}_days"] = (d - pd.Timestamp("1970-01-01")).dt.days
            frame.drop(columns=col, inplace=True)
        log.info("dates: %s разложена на части", col)
    return x_tr, x_te


def prepare(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Даты в части, строковые колонки в category с общим словарём train+test, bool в int.
    Пропуски LightGBM разберёт сам."""
    x_tr, x_te = expand_dates(train[features].copy(), test[features].copy())
    for col in x_tr.columns:
        if pd.api.types.is_bool_dtype(x_tr[col]):
            x_tr[col], x_te[col] = x_tr[col].astype(int), x_te[col].astype(int)
        elif not pd.api.types.is_numeric_dtype(x_tr[col]):
            cats = pd.Index(pd.concat([x_tr[col], x_te[col]]).dropna().astype(str).unique())
            x_tr[col] = pd.Categorical(x_tr[col].astype("string"), categories=cats)
            x_te[col] = pd.Categorical(x_te[col].astype("string"), categories=cats)
    return x_tr, x_te


def lgb_objective(spec: TaskSpec) -> dict:
    if spec.kind == "regression":
        return {"objective": "l1" if spec.metric == "mae" else "l2"}
    if spec.kind == "binary":
        return {"objective": "binary"}
    return {"objective": "multiclass", "num_class": len(spec.classes)}


def encode_target(y: pd.Series, spec: TaskSpec) -> np.ndarray:
    if spec.kind == "regression":
        return y.to_numpy(dtype=float)
    index = {str(c): i for i, c in enumerate(spec.classes)}
    return y.astype(str).map(index).to_numpy()


def predict(model: lgb.Booster, x: pd.DataFrame, spec: TaskSpec) -> np.ndarray:
    raw = model.predict(x, num_threads=N_JOBS)
    if spec.kind == "binary":
        return np.column_stack([1 - raw, raw])
    return raw


def run_core(train: pd.DataFrame, test: pd.DataFrame, spec: TaskSpec, folds: Folds,
             params_override: dict | None = None) -> CoreResult:
    features = feature_columns(train, spec)
    x, x_test = prepare(train, test, features)
    y = encode_target(train[spec.target], spec)
    params = {**LGB_PARAMS, **lgb_objective(spec), **(params_override or {})}

    n, k = len(train), len(spec.classes)
    oof = np.zeros(n) if spec.kind == "regression" else np.zeros((n, k))
    mask = np.zeros(n, dtype=bool)
    fold_scores, rounds = [], []
    for i, (tr, va) in enumerate(folds.splits):
        t0 = time.monotonic()
        fit, stop = inner_split(folds, tr)
        model = lgb.train(params, lgb.Dataset(x.iloc[fit], y[fit]), MAX_ROUNDS,
                          valid_sets=[lgb.Dataset(x.iloc[stop], y[stop])],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        oof[va] = predict(model, x.iloc[va], spec)
        mask[va] = True
        fold_scores.append(evaluate(spec.metric, train[spec.target].iloc[va], oof[va], spec.classes))
        rounds.append(model.best_iteration or MAX_ROUNDS)
        log.info("core fold %d: %s=%.5f, rounds=%d, %.0f с", i, spec.metric, fold_scores[-1], rounds[-1],
                 time.monotonic() - t0)

    score = evaluate(spec.metric, train[spec.target][mask], oof[mask], spec.classes)
    # На всём train данных больше, чем в любом фолде, поэтому раундов берём чуть больше среднего.
    full_rounds = max(50, int(np.mean(rounds) * 1.1))
    model = lgb.train(params, lgb.Dataset(x, y), full_rounds)
    test_pred = predict(model, x_test, spec)
    log.info("core: %s OOF=%.5f по фолдам %s, рефит %d раундов", spec.metric, score,
             [round(s, 5) for s in fold_scores], full_rounds)
    return CoreResult(oof, mask, test_pred, score, fold_scores, full_rounds)
