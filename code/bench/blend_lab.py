"""Сравнение стратегий ансамбля на holdout по сохранённым кандидатам.

Прогон с AGENT_DUMP_CANDIDATES=1 кладёт candidates.pkl в каталог прогона.
Здесь для каждой стратегии считаем OOF и метрику на ответах holdout, без
перезапуска моделей. Запуск: python blend_lab.py runs/cls_shift_... [runs/...]
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH.parent))

from agent.core import CoreResult  # noqa: E402
from agent.ensemble import caruana  # noqa: E402
from agent.metrics import evaluate, to_labels  # noqa: E402
from agent.postproc import class_weights  # noqa: E402
import agent.core  # noqa: E402

# В ZIP модули плоские (agent_core), pickle ссылается на них.
sys.modules["agent_core"] = agent.core


class FakeFolds:
    def __init__(self, fold_of: np.ndarray):
        self.splits = [(np.where(fold_of != i)[0], np.where(fold_of == i)[0]) for i in np.unique(fold_of[fold_of >= 0])]


def mean_of(cands: dict, names: list, stat=np.mean) -> CoreResult:
    oof = stat(np.stack([cands[n].oof for n in names]), axis=0)
    test = stat(np.stack([cands[n].test_pred for n in names]), axis=0)
    return CoreResult(oof, cands[names[0]].oof_mask, test, 0.0, [], 0)


def lab(run_dir: Path) -> None:
    data = pickle.loads((run_dir / "candidates.pkl").read_bytes())
    task = run_dir.name.rsplit("_", 2)[0]
    truth = pd.read_csv(BENCH / "answers" / f"{task}.csv")
    meta = json.loads((BENCH / "answers" / f"{task}.json").read_text(encoding="utf-8"))
    test = pd.read_csv(next(run_dir.rglob("test.csv")), usecols=[meta["id"]])
    y_test = test.merge(truth, on=meta["id"], how="left")[meta["target"]].to_numpy()
    cands, y, fold_of = data["candidates"], data["y"], data["fold_of"]
    metric, classes, kind = data["metric"], data["classes"], data["kind"]
    mask = next(iter(cands.values())).oof_mask
    folds = FakeFolds(fold_of)

    def report(name: str, r: CoreResult, weights=None) -> None:
        oof = evaluate(metric, y[mask], r.oof[mask] * (weights if weights is not None else 1), classes)
        if kind == "regression":
            hold = float(np.sqrt(np.mean((y_test - r.test_pred) ** 2)))
        else:
            labels = to_labels(r.test_pred, classes, weights)
            from sklearn.metrics import f1_score
            hold = f1_score(y_test.astype(str), labels.astype(str), average="macro")
        print(f"  {name:40s} OOF {oof:.5f}  holdout {hold:.5f}")

    print(f"== {run_dir.name} ({metric})")
    for n, r in cands.items():
        report(n, r)
    names = list(cands)
    if kind == "regression":
        report("median all", mean_of(cands, names, np.median))
        report("mean all", mean_of(cands, names))
        plain = [n for n in names if "resid" not in n]
        report("median no-resid", mean_of(cands, plain, np.median))
        resid = [n for n in names if "resid" in n]
        if resid:
            report("median resid only", mean_of(cands, resid, np.median))
        return
    mixed = caruana(cands, y, metric, classes, folds)
    if mixed:
        report(f"caruana {mixed[1]}", mixed[0])
    report("mean all", mean_of(cands, names))
    pair = [n for n in names if n in ("lgb", "xgb")]
    if len(pair) == 2:
        report("mean lgb+xgb", mean_of(cands, pair))
    for label, r in [("caruana", mixed[0] if mixed else None), ("mean all", mean_of(cands, names)),
                     ("xgb", cands.get("xgb")), ("lgb", cands.get("lgb"))]:
        if r is None:
            continue
        w = class_weights(y, r.oof, mask, fold_of, classes, metric)
        if w is not None:
            report(label + " + class weights", r, w)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        lab(Path(arg))
