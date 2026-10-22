"""Проверочный набор: 14 разных задач OpenML в шаблоне платформы.

Две публичные задачи хакатона — слишком узкая база, чтобы судить, хорош ли агент вообще.
Здесь бинарная и многоклассовая классификация, регрессия, от 1 тыс. до 50 тыс. строк,
с категориями и без, с пропусками, с дисбалансом, и метрики из всего словаря агента.
Сплит случайный 70/30 (стратифицированный для классификации), seed 0.

Для каждой задачи считаем два простых бейзлайна, как «самую простую модель» Qwen на
платформе: RandomForest и линейную модель. Балл стенд считает по формулам кейса.
Датасеты кешируются в bench/openml_cache (в .gitignore). Нужен интернет (через прокси).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from metrics import METRICS

BENCH = Path(__file__).resolve().parent
TASKS, ANSWERS, CACHE = BENCH / "tasks", BENCH / "answers", BENCH / "openml_cache"
MAX_ROWS = 50_000
SEED = 0

# имя задачи, id OpenML, метрика (как в task.md), ключ метрики стенда, тип
SUITE = [
    ("s_credit", 31, "ROC AUC", "roc_auc", "binary"),
    ("s_adult", 1590, "MCC (коэффициент корреляции Мэттьюса)", "mcc", "binary"),
    ("s_bank", 1461, "PR AUC (average precision)", "pr_auc", "binary"),
    ("s_phoneme", 1489, "Balanced accuracy", "balanced_accuracy", "binary"),
    ("s_kick", 41162, "Gini", "gini", "binary"),
    ("s_segment", 40984, "Accuracy", "accuracy", "multiclass"),
    ("s_letter", 6, "Macro F1", "f1_macro", "multiclass"),
    ("s_chess", 41027, "Weighted F1", "f1_weighted", "multiclass"),
    ("s_wine", 287, "RMSE", "rmse", "regression"),
    ("s_cpu", 197, "R2 (коэффициент детерминации)", "r2", "regression"),
    ("s_house", 42563, "RMSLE", "rmsle", "regression"),
    ("s_elevators", 216, "MAE", "mae", "regression"),
    ("s_diamonds", 42225, "MAPE", "mape", "regression"),
    ("s_moneyball", 41021, "RMSE", "rmse", "regression"),
]
PROBA = ("roc_auc", "pr_auc", "gini")

TEMPLATE = """# {title}

## Задача

{about}

## Данные

Пути указаны относительно каталога этого Markdown-файла:

- `data/participants/train.csv` — {n_train} строк с признаками и таргетом;
- `data/participants/test.csv` — {n_test} строк с признаками без таргета;
- `data/participants/data_dictionary.csv` — описание всех полей;
- `data/participants/sample_submission.csv` — шаблон ответа.

## Что предсказать

Для каждого `id` из `test.csv` предскажите `{target}`{proba}.

Метрика оценки — **{metric}**.

## Формат результата

Файл `submission.csv` в каталоге этого Markdown-файла, ровно два столбца:

```csv
id,{target}
```
"""


def baselines(train: pd.DataFrame, test: pd.DataFrame, target: str, kind: str, key: str,
              y_test: pd.Series) -> dict:
    """RandomForest и линейная модель по умолчанию — грубые аналоги бейзлайна платформы."""
    features = [c for c in train.columns if c not in ("id", target)]
    cats = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
    nums = [c for c in features if c not in cats]
    tree_prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), nums),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), cats)])
    lin_prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=20)), cats)])
    if kind == "regression":
        models = {"random_forest": RandomForestRegressor(200, n_jobs=8, random_state=SEED),
                  "linear": Ridge(1.0)}
    else:
        models = {"random_forest": RandomForestClassifier(200, n_jobs=8, random_state=SEED),
                  "linear": LogisticRegression(max_iter=2000)}
    out = {}
    for name, model in models.items():
        pipe = make_pipeline(tree_prep if name == "random_forest" else lin_prep, model)
        x_tr, x_te = train[features].astype({c: str for c in cats}), test[features].astype({c: str for c in cats})
        pipe.fit(x_tr, train[target])
        if key in PROBA:
            pred = pipe.predict_proba(x_te)[:, list(pipe.classes_).index(1)]
        else:
            pred = pipe.predict(x_te)
            if key in ("rmsle", "mape"):
                pred = np.maximum(pred, 0)
        out[name] = float(METRICS[key](y_test, pred))
    return out


def build(name: str, data_id: int, metric_text: str, key: str, kind: str) -> None:
    d = fetch_openml(data_id=data_id, as_frame=True, parser="auto", data_home=str(CACHE))
    df, target = d.frame.copy(), d.target.name
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=SEED)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(object)
    if kind == "binary" and key in PROBA:
        # Для метрик по вероятности таргет 0/1, единица — более редкий класс.
        minority = df[target].value_counts().index[-1]
        df[target] = (df[target] == minority).astype(int)
    elif kind == "regression":
        df[target] = df[target].astype(float)
    df.insert(0, "id", np.arange(len(df)))
    strat = df[target] if kind != "regression" else None
    train, test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=strat)

    data = TASKS / name / "data" / "participants"
    data.mkdir(parents=True, exist_ok=True)
    train.to_csv(data / "train.csv", index=False)
    test.drop(columns=[target]).to_csv(data / "test.csv", index=False)
    placeholder = 0.0 if kind == "regression" or key in PROBA else train[target].mode().iloc[0]
    pd.DataFrame({"id": test["id"], target: placeholder}).to_csv(data / "sample_submission.csv", index=False)
    rows = [{"feature": c, "role": "feature",
             "dtype": "numeric" if pd.api.types.is_numeric_dtype(train[c]) else "category",
             "description_ru": f"Признак {c} из набора OpenML {d.details['name']}"}
            for c in train.columns if c not in ("id", target)]
    rows.append({"feature": target, "role": "target", "dtype": "numeric" if kind == "regression" else "category",
                 "description_ru": "Таргет"})
    pd.DataFrame(rows).to_csv(data / "data_dictionary.csv", index=False)
    descr = " ".join((d.DESCR or "").split())[:1200]
    about = (f"Набор OpenML «{d.details['name']}» (id {data_id}). Постройте модель для `{target}`.\n\n"
             f"Описание из OpenML: {descr}")
    proba = " — вероятность класса 1" if key in PROBA else ""
    (TASKS / name / "task.md").write_text(TEMPLATE.format(
        title=name, about=about, n_train=len(train), n_test=len(test), target=target, proba=proba,
        metric=metric_text), encoding="utf-8")
    test[["id", target]].to_csv(ANSWERS / f"{name}.csv", index=False)
    base = baselines(train, test, target, kind, key, test[target])
    (ANSWERS / f"{name}.json").write_text(json.dumps(
        {"kind": key, "target": target, "id": "id", "proxy_baseline": base}, ensure_ascii=False), encoding="utf-8")
    print(f"{name}: {d.details['name']}, train {len(train)}, test {len(test)}, {key}, baselines {base}")


def main() -> None:
    ANSWERS.mkdir(exist_ok=True)
    for spec in SUITE:
        build(*spec)


if __name__ == "__main__":
    main()
