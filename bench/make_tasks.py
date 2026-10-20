"""Собирает локальные задачи в bench/tasks/ в раскладке платформы.

<name>/task.md и <name>/<подкаталог из task.md>/{train,test,data_dictionary,sample_submission}.csv.
Ответы holdout-задач лежат в bench/answers/, вне каталога задачи: агент их не видит.

  *_public   — исходные данные, ответов нет, проверяем только формат и время;
  reg_holdout — train из Year_Factor 1–4, test из Year_Factor 5 (повторяет сдвиг по годам);
  cls_holdout — стратифицированные 30% train в test. Сдвига train/test тут нет,
                поэтому скор будет оптимистичнее платформенного.
  cls_shift   — в test 30% строк train, которые adversarial-классификатор считает
                самыми похожими на настоящий test. Сдвиг ближе к платформенному.

Для holdout считаем прокси-бейзлайн (линейная модель и RandomForest), настоящий
бейзлайн платформы неизвестен.
"""

import json
import sys
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metrics import METRICS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.adversarial import adversarial  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent
TASKS, ANSWERS = BENCH / "tasks", BENCH / "answers"

SOURCES = {
    "reg": ROOT / "Регрессия",
    "cls": ROOT / "Многоклассовая классификация",
}


def data_subdir(task_md: str) -> str:
    """Подкаталог данных, как он записан в task.md (regression/participants)."""
    rel = re.search(r"([\w./-]+)/train\.csv", task_md)
    return rel.group(1) if rel else "."


def write_task(name: str, src: Path, train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> None:
    task_md = (src / "task.md").read_text(encoding="utf-8")
    dst = TASKS / name
    if dst.exists():
        shutil.rmtree(dst)
    data = dst / data_subdir(task_md)
    data.mkdir(parents=True)
    (dst / "task.md").write_text(task_md, encoding="utf-8")
    shutil.copy(src / "data_dictionary.csv", data / "data_dictionary.csv")
    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    sample.to_csv(data / "sample_submission.csv", index=False)


def encode(train: pd.DataFrame, test: pd.DataFrame, drop: list[str]):
    """Простое кодирование для прокси-бейзлайна: категории в коды, пропуски в медиану."""
    x_tr, x_te = train.drop(columns=drop), test.drop(columns=[c for c in drop if c in test])
    for col in x_tr.columns:
        if not pd.api.types.is_numeric_dtype(x_tr[col]):
            cats = pd.Categorical(pd.concat([x_tr[col], x_te[col]]).astype(str)).categories
            x_tr[col] = pd.Categorical(x_tr[col].astype(str), categories=cats).codes
            x_te[col] = pd.Categorical(x_te[col].astype(str), categories=cats).codes
    med = x_tr.median()
    return x_tr.fillna(med), x_te.fillna(med)


def proxy_baselines(train, test, y_true, id_col, target, kind) -> dict:
    x_tr, x_te = encode(train, test, [id_col, target])
    y = train[target]
    if kind == "regression":
        models = {"linear": make_pipeline(StandardScaler(), LinearRegression()),
                  "random_forest": RandomForestRegressor(n_estimators=100, n_jobs=8, random_state=0)}
    else:
        models = {"linear": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
                  "random_forest": RandomForestClassifier(n_estimators=100, n_jobs=8, random_state=0)}
    metric = METRICS[kind]
    out = {}
    for name, model in models.items():
        model.fit(x_tr, y)
        out[name] = float(metric(y_true, model.predict(x_te)))
        print(f"  {name}: {out[name]:.4f}")
    return out


def holdout(name: str, kind: str, train: pd.DataFrame, mask_test: np.ndarray, src: Path) -> None:
    id_col, target = "id", pd.read_csv(src / "sample_submission.csv", nrows=1).columns[1]
    tr, te = train[~mask_test].reset_index(drop=True), train[mask_test].reset_index(drop=True)
    placeholder = 0.0 if kind == "regression" else tr[target].mode().iloc[0]
    sample = pd.DataFrame({id_col: te[id_col], target: placeholder})
    write_task(name, src, tr, te.drop(columns=[target]), sample)
    te[[id_col, target]].to_csv(ANSWERS / f"{name}.csv", index=False)
    print(f"{name}: train {len(tr)}, test {len(te)}")
    meta = {"kind": kind, "target": target, "id": id_col,
            "proxy_baseline": proxy_baselines(tr, te.drop(columns=[target]), te[target], id_col, target, kind)}
    (ANSWERS / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    TASKS.mkdir(exist_ok=True)
    ANSWERS.mkdir(exist_ok=True)
    for short, src in SOURCES.items():
        write_task(f"{short}_public", src, pd.read_csv(src / "train.csv"), pd.read_csv(src / "test.csv"),
                   pd.read_csv(src / "sample_submission.csv"))
        print(f"{short}_public: исходные данные")

    reg = pd.read_csv(SOURCES["reg"] / "train.csv")
    holdout("reg_holdout", "regression", reg, (reg["Year_Factor"] == 5).to_numpy(), SOURCES["reg"])

    cls = pd.read_csv(SOURCES["cls"] / "train.csv")
    _, test_idx = train_test_split(np.arange(len(cls)), test_size=0.3, stratify=cls["attack_cat"], random_state=0)
    mask = np.zeros(len(cls), dtype=bool)
    mask[test_idx] = True
    holdout("cls_holdout", "multiclass", cls, mask, SOURCES["cls"])

    cls_test = pd.read_csv(SOURCES["cls"] / "test.csv")
    features = [c for c in cls.columns if c not in ("id", "attack_cat")]
    likeness = adversarial(cls, cls_test, features).train_test_likeness
    shift = likeness >= np.quantile(likeness, 0.7)
    print("cls_shift: доли классов в test", cls.loc[shift, "attack_cat"].value_counts(normalize=True).round(3).to_dict())
    holdout("cls_shift", "multiclass", cls, shift, SOURCES["cls"])


if __name__ == "__main__":
    main()
