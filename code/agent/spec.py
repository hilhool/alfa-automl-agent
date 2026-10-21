"""Разбор задачи правилами: тип, таргет, id, метрика.

Таргет и id берём из sample_submission, тип по таргету в train, метрику по
тексту task.md. LLM потом может уточнить, но при расхождении верим этим правилам.
"""

import re
from dataclasses import dataclass, field

import pandas as pd

# Порядок важен: более специфичные шаблоны раньше общих.
METRIC_PATTERNS = [
    ("f1_macro", r"macro[\s_-]*f1|f1[\s_-]*macro"),
    ("f1_weighted", r"weighted[\s_-]*f1|f1[\s_-]*weighted"),
    ("roc_auc", r"roc[\s_-]*auc|\bauc\b"),
    ("logloss", r"log[\s_-]*loss|кросс-энтропи|cross[\s_-]*entropy"),
    ("rmse", r"\brmse\b"),
    ("mae", r"\bmae\b"),
    ("r2", r"\br2\b|\br²"),
    ("f1_binary", r"\bf1\b"),
    ("accuracy", r"\baccuracy\b|доля верных"),
]
DEFAULT_METRIC = {"regression": "rmse", "binary": "f1_binary", "multiclass": "f1_macro"}
KIND_OF_METRIC = {"rmse": "regression", "mae": "regression", "r2": "regression"}


@dataclass
class TaskSpec:
    kind: str                # regression | binary | multiclass
    metric: str
    target: str
    id_col: str
    classes: list = field(default_factory=list)
    proba_output: bool = False   # в сабмит идёт вероятность, а не метка


def find_metric(text: str) -> str | None:
    """Метрика из предложения со словами «основная метрика» / «метрика оценки», иначе первое упоминание."""
    sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    ranked = ([s for s in sentences if re.search(r"основн\w* метрик|метрик\w* оценки|main metric", s, re.I)]
              + [s for s in sentences if re.search(r"метрик|metric", s, re.I)] + [text])
    for sentence in ranked:
        hits = [(m.start(), name) for name, pat in METRIC_PATTERNS for m in [re.search(pat, sentence, re.I)] if m]
        if hits:
            return min(hits)[1]
    return None


def parse(task_text: str, train: pd.DataFrame, sample: pd.DataFrame) -> TaskSpec:
    id_col, target = sample.columns[0], sample.columns[1]
    y = train[target].dropna()
    numeric = pd.api.types.is_numeric_dtype(y)
    found = find_metric(task_text)
    # Целочисленный таргет с малым числом значений (счётчик, рейтинг) — регрессия,
    # если в task.md названа регрессионная метрика.
    if numeric and (pd.api.types.is_float_dtype(y) and (y % 1 != 0).any() or y.nunique() > 50
                    or KIND_OF_METRIC.get(found) == "regression"):
        kind = "regression"
    else:
        kind = "binary" if y.nunique() == 2 else "multiclass"

    metric = found or DEFAULT_METRIC[kind]
    if (KIND_OF_METRIC.get(metric) == "regression") != (kind == "regression"):
        metric = DEFAULT_METRIC[kind]
    if kind == "multiclass" and metric == "f1_binary":
        metric = "f1_macro"

    classes = [] if kind == "regression" else sorted(y.unique().tolist(), key=str)
    # Вероятность пишем для binary, если метрика её требует и шаблон числовой (0/0.0).
    # Мультикласс с вероятностями в одну колонку не записать, там остаются метки.
    proba = (kind == "binary" and metric in ("roc_auc", "logloss")
             and pd.api.types.is_numeric_dtype(sample[target]))
    return TaskSpec(kind=kind, metric=metric, target=target, id_col=id_col, classes=classes, proba_output=proba)
