"""Разбор задачи правилами: тип, таргет, id, метрика.

Таргет и id берём из sample_submission, тип по таргету в train, метрику по
тексту task.md. LLM потом может уточнить, но при расхождении верим этим правилам.
"""

import re
from dataclasses import dataclass, field

import pandas as pd

# Порядок важен: более специфичные шаблоны раньше общих (rmsle раньше rmse, qwk раньше kappa).
METRIC_PATTERNS = [
    ("f1_macro", r"macro[\s_-]*f1|f1[\s_-]*macro|макро[\s_-]*f1|f1[\s_-]*макро|макро-?усредн"),
    ("f1_weighted", r"weighted[\s_-]*f1|f1[\s_-]*weighted|взвешенн\w*[\s_-]*f1|f1[\s_-]*взвешенн"),
    ("pr_auc", r"pr[\s_-]*auc|average[\s_-]*precision|precision[\s_-]*recall[\s_-]*curve|площад\w* под pr"),
    ("gini", r"\bgini\b|джини"),
    ("roc_auc", r"roc[\s_-]*auc|\bauc\b|площад\w* под roc|roc-крив"),
    ("logloss", r"log[\s_-]*loss|кросс-?энтропи|cross[\s_-]*entropy|логистическ\w* функци\w* потерь"),
    ("rmsle", r"\brmsle\b|root[\s_-]*mean[\s_-]*squared[\s_-]*log|среднеквадратичн\w* логарифм"),
    ("msle", r"\bmsle\b|mean[\s_-]*squared[\s_-]*log"),
    ("smape", r"\bsmape\b|symmetric[\s_-]*mean[\s_-]*absolute[\s_-]*percent|симметричн\w*.{0,20}процентн"),
    ("mape", r"\bmape\b|mean[\s_-]*absolute[\s_-]*percent|процентн\w* ошибк"),
    ("medae", r"median[\s_-]*absolute[\s_-]*error|медианн\w* абсолютн"),
    ("mae", r"\bmae\b|mean[\s_-]*absolute[\s_-]*error|средн\w* абсолютн\w* (ошибк|отклонен)"),
    ("rmse", r"\brmse\b|\bmse\b|root[\s_-]*mean[\s_-]*squared|mean[\s_-]*squared[\s_-]*error|"
             r"среднеквадратичн\w* (ошибк|отклонен)|корень из средн"),
    ("r2", r"\br2\b|\br²|r[\s_-]*squared|r\^2|коэффициент\w* детерминац"),
    ("qwk", r"quadratic\w*[\s_-]*weighted[\s_-]*kappa|\bqwk\b|квадратичн\w* взвешенн\w* капп"),
    ("kappa", r"cohen\W{0,2}s[\s_-]*kappa|\bkappa\b|капп[аы]"),
    ("mcc", r"\bmcc\b|matthews|мэттьюс"),
    ("balanced_accuracy", r"balanced[\s_-]*accuracy|сбалансированн\w* точност"),
    ("f1_binary", r"\bf1\b|f-?мер|f1-?мер|f[\s_-]*score"),
    ("accuracy", r"\baccuracy\b|доля верных|доля правильн|\bточност[ьи]\b"),
]
DEFAULT_METRIC = {"regression": "rmse", "binary": "f1_binary", "multiclass": "f1_macro"}
REGRESSION_METRICS = ("rmse", "mae", "r2", "rmsle", "msle", "mape", "smape", "medae")
KIND_OF_METRIC = {m: "regression" for m in REGRESSION_METRICS}
# Метрики, которым нужны вероятности, а не метки (для binary).
PROBA_METRICS = ("roc_auc", "gini", "pr_auc", "logloss")
# Метрики только для binary: на мультиклассе заменяем ближайшей.
BINARY_TO_MULTI = {"f1_binary": "f1_macro", "gini": "roc_auc", "pr_auc": "roc_auc"}


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
    if kind == "multiclass":
        metric = BINARY_TO_MULTI.get(metric, metric)

    # Числовые классы сортируем по значению: строковая сортировка ставит 10 между 1 и 2, и
    # roc_auc(labels=classes) падает с «labels must be ordered», а QWK считает порядок неверно.
    values = y.dropna().unique().tolist()
    numeric = pd.api.types.is_numeric_dtype(y)
    classes = [] if kind == "regression" else sorted(values) if numeric else sorted(values, key=str)
    # Вероятность пишем для binary, если метрика её требует и шаблон числовой (0/0.0).
    # Мультикласс с вероятностями в одну колонку не записать, там остаются метки.
    proba = (kind == "binary" and metric in PROBA_METRICS
             and pd.api.types.is_numeric_dtype(sample[target]))
    return TaskSpec(kind=kind, metric=metric, target=target, id_col=id_col, classes=classes, proba_output=proba)
