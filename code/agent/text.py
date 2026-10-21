"""Текстовые колонки: TF-IDF (слова и char n-gram) + линейная модель, OOF как признак бустинга.

Текст в category превращать нельзя: у LightGBM будет десятки тысяч уровней.
Колонку считаем текстовой, если data_dictionary называет её текстом, или если
строки длинные и почти все уникальны.
"""

import logging

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

from agent.cv import Folds
from agent.spec import TaskSpec

log = logging.getLogger("solution")

MIN_MEAN_LEN = 30
MIN_WORDS = 4
MIN_UNIQUE_SHARE = 0.3
MAX_FEATURES = 100_000
TEXT_DTYPES = ("text", "str_text", "free_text", "текст")


def find_text_columns(train: pd.DataFrame, features: list[str], dictionary: pd.DataFrame | None = None) -> list[str]:
    declared = set()
    if dictionary is not None and {"feature", "dtype"} <= set(dictionary.columns):
        declared = set(dictionary.loc[dictionary["dtype"].astype(str).str.lower().isin(TEXT_DTYPES), "feature"])
    found = []
    for col in features:
        if pd.api.types.is_numeric_dtype(train[col]):
            continue
        s = train[col].dropna().astype(str)
        if s.empty:
            continue
        long_text = (s.str.len().mean() >= MIN_MEAN_LEN and s.str.split().str.len().mean() >= MIN_WORDS
                     and s.nunique() >= MIN_UNIQUE_SHARE * len(s))
        if col in declared or long_text:
            found.append(col)
    return found


def _vectorize(fit_text: pd.Series, other: list[pd.Series]):
    words = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    chars = TfidfVectorizer(max_features=MAX_FEATURES, analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                            sublinear_tf=True)
    fitted = sparse.hstack([words.fit_transform(fit_text), chars.fit_transform(fit_text)]).tocsr()
    return fitted, [sparse.hstack([words.transform(t), chars.transform(t)]).tocsr() for t in other]


def _model(spec: TaskSpec):
    if spec.kind == "regression":
        return Ridge(alpha=1.0)
    return LogisticRegression(C=4.0, max_iter=300)


def _predict(model, x, spec: TaskSpec) -> np.ndarray:
    if spec.kind == "regression":
        return model.predict(x)[:, None]
    proba = np.zeros((x.shape[0], len(spec.classes)))
    proba[:, [spec.classes.index(c) for c in model.classes_]] = model.predict_proba(x)
    return proba[:, 1:] if spec.kind == "binary" else proba


def text_features(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], spec: TaskSpec,
                  folds: Folds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OOF-предсказания текстовой модели по каждой колонке. Строки без OOF (ранние периоды
    при валидации вперёд) получают NaN, LightGBM разберёт."""
    y = train[spec.target].to_numpy()
    out_tr, out_te = {}, {}
    for col in cols:
        t_tr, t_te = train[col].fillna("").astype(str), test[col].fillna("").astype(str)
        width = 1 if spec.kind in ("regression", "binary") else len(spec.classes)
        oof = np.full((len(train), width), np.nan)
        for tr, va in folds.splits:
            x_fit, (x_val,) = _vectorize(t_tr.iloc[tr], [t_tr.iloc[va]])
            oof[va] = _predict(_model(spec).fit(x_fit, y[tr]), x_val, spec)
        x_all, (x_test,) = _vectorize(t_tr, [t_te])
        pred_test = _predict(_model(spec).fit(x_all, y), x_test, spec)
        for j in range(width):
            out_tr[f"txt_{col}_{j}"], out_te[f"txt_{col}_{j}"] = oof[:, j], pred_test[:, j]
        log.info("text: %s даёт %d признаков", col, width)
    return pd.DataFrame(out_tr), pd.DataFrame(out_te)
