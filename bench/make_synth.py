"""Синтетические псевдо-приватные задачи для проверки веток, которых нет в public.

  syn_text_auc — binary, текстовая колонка + числа, ROC AUC, в сабмит идёт вероятность;
  syn_mae      — регрессия с датой, пропусками и выбросами, MAE;
  syn_small    — маленький мультикласс (600 строк), accuracy.

task.md повторяет шаблон публичных задач. Ответы кладём в bench/answers/,
прокси-бейзлайн для них не считаем: тут проверяем, что ветки работают и не падают.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parent
TASKS, ANSWERS = BENCH / "tasks", BENCH / "answers"
RNG = np.random.default_rng(0)

TEMPLATE = """# {title}

## Задача

{about}

## Данные

Пути указаны относительно каталога этого Markdown-файла:

- `{sub}/participants/train.csv` — {n_train} строк с признаками и таргетом;
- `{sub}/participants/test.csv` — {n_test} строк с признаками без таргета;
- `{sub}/participants/data_dictionary.csv` — описание всех полей;
- `{sub}/participants/sample_submission.csv` — шаблон ответа.

## Что предсказать

Для каждого `id` из `test.csv` предскажите `{target}`.

Метрика оценки — **{metric}**.

## Формат результата

Файл `submission.csv` в каталоге этого Markdown-файла, ровно два столбца:

```csv
id,{target}
```
"""

GOOD = "отличный быстрый удобный надёжный приятный вежливый качественный".split()
BAD = "ужасный медленный сломанный грубый дорогой неудобный задержка".split()
FILLER = "заказ доставка товар курьер магазин упаковка цена сайт приложение поддержка".split()


def review(positive: bool) -> str:
    words = list(RNG.choice(FILLER, RNG.integers(5, 12)))
    tone = GOOD if positive else BAD
    for _ in range(RNG.integers(1, 3)):
        words.insert(RNG.integers(0, len(words)), RNG.choice(tone))
    if RNG.random() < 0.2:   # шум: слово противоположного тона
        words.insert(RNG.integers(0, len(words)), RNG.choice(BAD if positive else GOOD))
    return " ".join(words)


def text_auc(n: int) -> tuple:
    y = RNG.random(n) < 0.3
    df = pd.DataFrame({"id": np.arange(n), "review": [review(not v) for v in y],
                       "order_sum": RNG.lognormal(7, 1, n).round(2), "items": RNG.integers(1, 10, n),
                       "complaint": y.astype(int)})
    df.loc[df["complaint"] == 1, "order_sum"] *= 1.3
    about = "По тексту отзыва и параметрам заказа определите, подаст ли клиент жалобу (1) или нет (0)."
    return df, "complaint", "ROC AUC", {"review": ("text", "Текст отзыва"),
                                        "order_sum": ("numeric", "Сумма заказа"),
                                        "items": ("numeric", "Число позиций")}, about


def mae_dates(n: int) -> tuple:
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(RNG.integers(0, 365, n), unit="D")
    area = RNG.lognormal(4, 0.5, n)
    region = RNG.choice(list("ABCDE"), n)
    y = 2 * area + 15 * (pd.Series(dates).dt.dayofweek >= 5).to_numpy() + pd.Series(region).map(
        dict(zip("ABCDE", [0, 10, 20, 30, 40]))).to_numpy() + RNG.normal(0, 10, n)
    y[RNG.random(n) < 0.02] *= 8    # выбросы: под MAE их нельзя тянуть средним
    df = pd.DataFrame({"id": np.arange(n), "date": dates.strftime("%Y-%m-%d"), "area": area.round(1),
                       "region": region, "price": y.round(2)})
    df.loc[RNG.random(n) < 0.15, "area"] = np.nan
    about = "По дате, площади и региону оцените цену объявления."
    return df, "price", "MAE", {"date": ("date", "Дата публикации"), "area": ("numeric", "Площадь"),
                                "region": ("category", "Регион")}, about


def small_multiclass(n: int) -> tuple:
    x = RNG.normal(size=(n, 6))
    y = np.array(["low", "mid", "high"])[np.digitize(x[:, 0] + 0.5 * x[:, 1] ** 2, [-0.3, 1.0])]
    df = pd.DataFrame(x.round(4), columns=[f"f{i}" for i in range(6)])
    df.insert(0, "id", np.arange(n))
    df["grade"] = y
    about = "Маленький набор: по шести числовым признакам определите класс (low, mid, high)."
    return df, "grade", "Accuracy", {f"f{i}": ("numeric", f"Признак {i}") for i in range(6)}, about


def write(name: str, df: pd.DataFrame, target: str, metric: str, fields: dict, about: str) -> None:
    test_mask = RNG.random(len(df)) < 0.3
    train, test = df[~test_mask], df[test_mask]
    sub = name.removeprefix("syn_")
    data = TASKS / name / sub / "participants"
    data.mkdir(parents=True, exist_ok=True)
    train.to_csv(data / "train.csv", index=False)
    test.drop(columns=[target]).to_csv(data / "test.csv", index=False)
    placeholder = 0.0 if metric in ("MAE", "ROC AUC") else train[target].mode().iloc[0]
    pd.DataFrame({"id": test["id"], target: placeholder}).to_csv(data / "sample_submission.csv", index=False)
    pd.DataFrame([{"feature": k, "role": "feature", "dtype": d, "description_ru": t} for k, (d, t) in fields.items()]
                 ).to_csv(data / "data_dictionary.csv", index=False)
    (TASKS / name / "task.md").write_text(TEMPLATE.format(
        title=name, about=about, sub=sub, n_train=len(train), n_test=len(test), target=target, metric=metric),
        encoding="utf-8")
    test[["id", target]].to_csv(ANSWERS / f"{name}.csv", index=False)
    kind = {"MAE": "mae", "ROC AUC": "roc_auc", "Accuracy": "accuracy"}[metric]
    (ANSWERS / f"{name}.json").write_text(json.dumps({"kind": kind, "target": target, "id": "id"}), encoding="utf-8")
    print(f"{name}: train {len(train)}, test {len(test)}")


def main() -> None:
    ANSWERS.mkdir(exist_ok=True)
    for name, maker, n in (("syn_text_auc", text_auc, 6000), ("syn_mae", mae_dates, 8000),
                           ("syn_small", small_multiclass, 600)):
        write(name, *maker(n))


if __name__ == "__main__":
    main()
