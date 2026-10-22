"""Сводка по проверочному набору: последний прогон каждой задачи против бейзлайнов.

Балл считается по формулам кейса к RandomForest (как к «самой простой модели» платформы).
Главная цифра — средний балл по набору и число задач, где агент хуже бейзлайна:
изменение агента принимаем, если среднее растёт и таких задач не прибавилось.

Запуск: python suite_report.py [префикс задач, по умолчанию s_] [--before ГГГГ-ММ-ДД ЧЧ:ММ]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", nargs="?", default="s_")
    parser.add_argument("--before", help="брать прогоны, закончившиеся до этого момента")
    args = parser.parse_args()
    runs = pd.read_csv(BENCH / "results.csv")
    runs = runs[runs["task"].astype(str).str.startswith(args.prefix)]
    if args.before:
        runs = runs[runs["finished"] < args.before]
    last = runs.groupby("task").tail(1).set_index("task")
    rows = []
    for task, r in last.iterrows():
        meta = json.loads((BENCH / "answers" / f"{task}.json").read_text(encoding="utf-8"))
        base = meta.get("proxy_baseline", {})
        rows.append({"task": task, "metric": meta["kind"], "agent": r["metric"],
                     "rf": round(base.get("random_forest", float("nan")), 5),
                     "linear": round(base.get("linear", float("nan")), 5),
                     "S_rf": r["score_vs_rf"], "S_lin": r["score_vs_linear"], "min": round(r["seconds"] / 60, 1),
                     "status": r["status"], "format": r["format"]})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    s_rf = pd.to_numeric(table["S_rf"], errors="coerce")
    s_lin = pd.to_numeric(table["S_lin"], errors="coerce")
    print(f"\nзадач {len(table)}, средний балл к RF {s_rf.mean():.2f}, к линейной {s_lin.mean():.2f}; "
          f"хуже RF: {int((s_rf <= 0).sum())}, хуже линейной: {int((s_lin <= 0).sum())}; "
          f"время всего {table['min'].sum():.0f} мин")


if __name__ == "__main__":
    main()
