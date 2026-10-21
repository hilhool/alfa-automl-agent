"""Ядро без LLM: предобработка, бустинг (LightGBM, CatBoost, XGBoost) на зафиксированных
фолдах, рефит на всём train.

Это нижняя ступень над константой. Всё остальное (признаки, цикл гипотез,
тяжёлые модели) сравнивается с ним на тех же фолдах.
"""

import logging
import re
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from agent.cv import Folds, inner_split
from agent.metrics import evaluate, safe_evaluate
from agent.spec import TaskSpec

log = logging.getLogger("solution")

N_JOBS = 8
LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=20, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=N_JOBS, seed=0)
MAX_ROUNDS, EARLY_STOP = 5000, 100
CAT_MAX_ROUNDS = 3000
TRANSFORMS = {"log1p": (np.log1p, np.expm1)}
DATE_SAMPLE, DATE_SHARE = 500, 0.95
MAX_CAT_LEVELS, MAX_CAT_SHARE = 1000, 0.5
L1_METRICS = ("mae", "medae", "smape")
# На всём train данных больше, чем в любом фолде, поэтому раундов при рефите больше среднего.
REFIT_MULT = 1.1
MIN_FIT_ROWS, MIN_STOP_ROWS = 200, 50


@dataclass
class CoreResult:
    oof: np.ndarray          # (n,) для регрессии, (n, k) вероятности для классификации
    oof_mask: np.ndarray     # строки, у которых есть OOF
    test_pred: np.ndarray
    score: float
    fold_scores: list[float]
    rounds: int


def feature_columns(train: pd.DataFrame, spec: TaskSpec, test: pd.DataFrame | None = None) -> list[str]:
    """Колонки, которые есть и в test (колонка только в train — утечка или служебная)."""
    return [c for c in train.columns if c not in (spec.id_col, spec.target) and train[c].nunique(dropna=False) > 1
            and (test is None or c in test.columns)]


def safe_names(columns) -> list[str]:
    """LightGBM не принимает в именах JSON-символы, XGBoost — [ ] <."""
    out, seen = [], set()
    for i, c in enumerate(columns):
        name = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_]", "_", str(c)) or f"f{i}"
        while name in seen:
            name += f"_{i}"
        seen.add(name)
        out.append(name)
    return out


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
    for col in list(x_tr.columns):
        if pd.api.types.is_numeric_dtype(x_tr[col]) and not pd.api.types.is_numeric_dtype(x_te[col]):
            x_te[col] = pd.to_numeric(x_te[col], errors="coerce")
        if (not pd.api.types.is_numeric_dtype(x_tr[col]) and not pd.api.types.is_bool_dtype(x_tr[col])
                and x_tr[col].nunique() > max(MAX_CAT_LEVELS, MAX_CAT_SHARE * len(x_tr))):
            # Почти уникальная строка (id, хеш) как category тормозит бустинг и переобучает его.
            freq = x_tr[col].astype(str).value_counts(normalize=True)
            x_tr[col] = x_tr[col].astype(str).map(freq).astype(float)
            x_te[col] = x_te[col].astype(str).map(freq).astype(float)
            log.info("prepare: %s — %d уровней, заменили на частоту", col, len(freq))
            continue
        if pd.api.types.is_bool_dtype(x_tr[col]):
            x_tr[col], x_te[col] = x_tr[col].astype(int), x_te[col].astype(int)
        elif not pd.api.types.is_numeric_dtype(x_tr[col]):
            # Словарь только по train: значение, которого модель не видела, честнее отдать
            # как пропуск. XGBoost отправлял невиданный State_11 в случайную ветку (RMSE 64 вместо 41).
            cats = pd.Index(x_tr[col].dropna().astype(str).unique())
            unseen = x_te[col].notna() & ~x_te[col].astype(str).isin(cats)
            if unseen.any():
                log.info("prepare: %s — %d строк test с невиданными значениями, считаем пропуском",
                         col, int(unseen.sum()))
            x_tr[col] = pd.Categorical(x_tr[col].astype("string"), categories=cats)
            x_te[col] = pd.Categorical(x_te[col].astype("string"), categories=cats)
    x_tr.columns = x_te.columns = safe_names(x_tr.columns)
    return x_tr, x_te


def lgb_objective(spec: TaskSpec) -> dict:
    if spec.kind == "regression":
        # Для MAE-подобных метрик оптимум — медиана, для MAPE — взвешенная медиана.
        if spec.metric == "mape":
            return {"objective": "mape"}
        return {"objective": "l1" if spec.metric in L1_METRICS else "l2"}
    if spec.kind == "binary":
        return {"objective": "binary"}
    return {"objective": "multiclass", "num_class": len(spec.classes)}


def encode_target(y: pd.Series, spec: TaskSpec) -> np.ndarray:
    if spec.kind == "regression":
        return y.to_numpy(dtype=float)
    index = {str(c): i for i, c in enumerate(spec.classes)}
    return y.astype(str).map(index).to_numpy()


class OutOfTime(Exception):
    """Модель не успевает до дедлайна: бросаем её, сабмит остаётся прежним."""


def _cat_frame(x: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """CatBoost хочет категории строками без NaN."""
    x = x.copy()
    cats = [c for c in x.columns if isinstance(x[c].dtype, pd.CategoricalDtype)]
    for c in cats:
        x[c] = x[c].astype("string").fillna("__nan__").astype(str)
    return x, cats


def _to_proba(raw: np.ndarray, seen: np.ndarray, spec: TaskSpec) -> np.ndarray:
    """Столбцы вероятностей в порядке spec.classes, даже если в fit был не каждый класс."""
    out = np.zeros((raw.shape[0], max(len(spec.classes), 2)))
    out[:, np.asarray(seen).astype(int)] = raw
    return out


def _fit_lgb(params, spec, x_fit, y_fit, x_stop, y_stop, to_predict, rounds):
    full = {**LGB_PARAMS, **lgb_objective(spec), **params}
    if x_stop is not None:
        model = lgb.train(full, lgb.Dataset(x_fit, y_fit), MAX_ROUNDS, valid_sets=[lgb.Dataset(x_stop, y_stop)],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        rounds = model.best_iteration or MAX_ROUNDS
    else:
        model = lgb.train(full, lgb.Dataset(x_fit, y_fit), rounds)
    preds = []
    for x in to_predict:
        raw = model.predict(x, num_threads=N_JOBS)
        preds.append(np.column_stack([1 - raw, raw]) if spec.kind == "binary" else raw)
    return preds, rounds


def _fit_cat(params, spec, x_fit, y_fit, x_stop, y_stop, to_predict, rounds):
    from catboost import CatBoostClassifier, CatBoostRegressor
    loss = {"regression": "MAPE" if spec.metric == "mape" else "MAE" if spec.metric in L1_METRICS else "RMSE",
            "binary": "Logloss",
            "multiclass": "MultiClass"}[spec.kind]
    cls = CatBoostRegressor if spec.kind == "regression" else CatBoostClassifier
    base = dict(loss_function=loss, learning_rate=0.08, depth=6, thread_count=N_JOBS, random_seed=0,
                verbose=False, allow_writing_files=False, **params)
    x_fit, cats = _cat_frame(x_fit)
    if x_stop is not None:
        model = cls(iterations=CAT_MAX_ROUNDS, od_type="Iter", od_wait=EARLY_STOP, **base)
        model.fit(x_fit, y_fit, cat_features=cats, eval_set=(_cat_frame(x_stop)[0], y_stop), use_best_model=True)
        rounds = max(1, model.get_best_iteration() or CAT_MAX_ROUNDS)
    else:
        model = cls(iterations=rounds, **base)
        model.fit(x_fit, y_fit, cat_features=cats)
    if spec.kind == "regression":
        return [model.predict(_cat_frame(x)[0]) for x in to_predict], rounds
    return [_to_proba(model.predict_proba(_cat_frame(x)[0]), model.classes_, spec) for x in to_predict], rounds


def _fit_xgb(params, spec, x_fit, y_fit, x_stop, y_stop, to_predict, rounds):
    import xgboost as xgb
    objective = {"regression": "reg:absoluteerror" if spec.metric in L1_METRICS + ("mape",) else "reg:squarederror",
                 "binary": "binary:logistic", "multiclass": "multi:softprob"}[spec.kind]
    base = dict(objective=objective, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=3, tree_method="hist", max_cat_to_onehot=1, nthread=N_JOBS, seed=0, **params)
    seen = None
    if spec.kind != "regression":
        # XGBoost требует метки 0..m-1, поэтому перекодируем классы, которые есть в fit.
        seen = np.unique(y_fit)
        remap = np.full(int(seen.max()) + 1, -1)
        remap[seen] = np.arange(len(seen))
        y_fit = remap[y_fit]
        if spec.kind == "multiclass":
            base["num_class"] = len(seen)
    d_fit = xgb.DMatrix(x_fit, y_fit, enable_categorical=True)
    if x_stop is not None:
        if seen is not None:
            keep = np.isin(y_stop, seen)
            x_stop, y_stop = x_stop[keep], remap[y_stop[keep]]
        model = xgb.train(base, d_fit, MAX_ROUNDS, evals=[(xgb.DMatrix(x_stop, y_stop, enable_categorical=True), "s")],
                          early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        rounds = model.best_iteration + 1
    else:
        model = xgb.train(base, d_fit, rounds)
    preds = []
    for x in to_predict:
        raw = model.predict(xgb.DMatrix(x, enable_categorical=True), iteration_range=(0, rounds))
        if spec.kind == "binary":
            preds.append(_to_proba(np.column_stack([1 - raw, raw]), seen, spec))
        elif spec.kind == "multiclass":
            preds.append(_to_proba(raw, seen, spec))
        else:
            preds.append(raw)
    return preds, rounds


FITTERS = {"lgb": _fit_lgb, "cat": _fit_cat, "xgb": _fit_xgb}


def run_core(train: pd.DataFrame, test: pd.DataFrame, spec: TaskSpec, folds: Folds,
             params_override: dict | None = None, family: str = "lgb", transform: str | None = None,
             deadline: float | None = None, name: str = "core", refit_mult: float = REFIT_MULT,
             repeats: np.ndarray | None = None, residual_of: str | None = None,
             fallback: CoreResult | None = None) -> CoreResult:
    """OOF на фолдах и предсказание test. transform="log1p" учит модель на log1p(y),
    но OOF и метрика всегда в исходной шкале. deadline — момент time.monotonic(), после
    которого модель не доучиваем: по первым фолдам видно, успеет ли она вместе с рефитом.

    repeats — сколько раз повторить строку в обучении (вес). residual_of — колонка-база
    (прошлое значение сущности): модель учится на остатке y − база только на строках, где база
    есть, а строки без базы берут предсказания fallback."""
    features = feature_columns(train, spec, test)
    x, x_test = prepare(train, test, features)
    y = encode_target(train[spec.target], spec)
    forward, back = TRANSFORMS.get(transform, (None, None))
    to_model = forward or (lambda v: v)
    from_model = back or (lambda v: v)
    y_model = to_model(y)
    base_tr = base_te = None
    rows = np.ones(len(train), dtype=bool)
    if residual_of is not None:
        base_tr = train[residual_of].to_numpy(dtype=float)
        base_te = test[residual_of].to_numpy(dtype=float)
        rows = ~np.isnan(base_tr)
        y_model = np.where(rows, y_model - to_model(np.nan_to_num(base_tr)), np.nan)
    fitter, params = FITTERS[family], params_override or {}

    def expand(idx: np.ndarray) -> np.ndarray:
        idx = idx[rows[idx]]
        return idx if repeats is None else np.repeat(idx, repeats[idx])

    def restore(pred: np.ndarray, base: np.ndarray | None) -> np.ndarray:
        if base is None:
            return from_model(pred)
        return from_model(pred + to_model(np.nan_to_num(base)))

    n, k = len(train), len(spec.classes)
    oof = np.zeros(n) if spec.kind == "regression" else np.zeros((n, max(k, 2)))
    mask = np.zeros(n, dtype=bool)
    fold_scores, rounds = [], []
    started = time.monotonic()
    for i, (tr, va) in enumerate(folds.splits):
        if deadline is not None and i > 0:
            per_fold = (time.monotonic() - started) / i
            # Рефит на всём train стоит примерно как два фолда.
            if time.monotonic() + per_fold * (len(folds.splits) - i + 2) > deadline:
                raise OutOfTime(f"{name}: не успеваем, {per_fold:.0f} с на фолд")
        t0 = time.monotonic()
        fit, stop = inner_split(folds, tr)
        fit, stop = expand(fit), stop[rows[stop]]
        # Остаток учится на подмножестве строк с базой: в раннем фолде их может почти не быть.
        if residual_of is not None and (len(fit) < MIN_FIT_ROWS or len(stop) < MIN_STOP_ROWS):
            oof[va] = fallback.oof[va]
        else:
            (pred,), r = fitter(params, spec, x.iloc[fit], y_model[fit], x.iloc[stop], y_model[stop],
                                [x.iloc[va]], None)
            oof[va] = restore(pred, None if base_tr is None else base_tr[va])
            if fallback is not None:
                no_base = ~rows[va]
                oof[va[no_base]] = fallback.oof[va[no_base]]
            rounds.append(r)
        mask[va] = True
        fold_scores.append(safe_evaluate(spec.metric, train[spec.target].iloc[va], oof[va], spec.classes))
        log.info("%s fold %d: %s=%.5f, %.0f с", name, i, spec.metric, fold_scores[-1], time.monotonic() - t0)

    if not rounds:
        raise ValueError(f"{name}: ни один фолд не обучился")
    score = evaluate(spec.metric, train[spec.target][mask], oof[mask], spec.classes)
    full_rounds = max(50, int(np.mean(rounds) * refit_mult))
    (test_pred,), _ = fitter(params, spec, x.iloc[expand(np.arange(n))], y_model[expand(np.arange(n))],
                             None, None, [x_test], full_rounds)
    test_pred = restore(test_pred, base_te)
    if fallback is not None and base_te is not None:
        no_base = np.isnan(base_te)
        test_pred[no_base] = fallback.test_pred[no_base]
    log.info("%s: %s OOF=%.5f по фолдам %s, рефит %d раундов, %.0f с", name, spec.metric, score,
             [round(s, 5) for s in fold_scores], full_rounds, time.monotonic() - started)
    return CoreResult(oof, mask, test_pred, score, fold_scores, full_rounds)
