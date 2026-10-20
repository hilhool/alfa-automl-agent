"""Прогон solution.py на локальной задаче в условиях, близких к платформе.

  python bench/run.py reg_holdout [--timeout 10800] [--cpus 8] [--zip dist/solution.zip]

Что эмулируем: свежая копия задачи, ZIP распакован в каталог задачи, запуск
`python solution.py task.md` оттуда же (так делает платформа), привязка к 8 логическим
ядрам, kill всего дерева процессов по таймауту. Без --zip архив собирается заново из code/.
Что не эмулируем без Docker: лимит памяти (только меряем пик), отключение сети
(агент обязан сам не ходить никуда, кроме OPENAI_BASE_URL).

Переменные OPENAI_* берутся из окружения или из code/.env.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil

from build_zip import build
from metrics import METRICS, score

CODE = Path(__file__).resolve().parents[1]
BENCH = CODE / "bench"
sys.path.insert(0, str(CODE))
from agent.taskio import check_submission, locate  # noqa: E402

PASS_ENV = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "HOME",
            "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL")


def load_env() -> dict:
    env = {k: os.environ[k] for k in PASS_ENV if k in os.environ}
    dotenv = CODE / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                env.setdefault(key.strip(), value.strip().strip('"'))
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def tree(proc: psutil.Process) -> list[psutil.Process]:
    try:
        return [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def run(name: str, timeout: float, cpus: int, archive: Path) -> dict:
    run_dir = BENCH / "runs" / f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copytree(BENCH / "tasks" / name, run_dir)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(run_dir)
    task_md = run_dir / "task.md"
    log_path = run_dir / "run.log"

    started = time.monotonic()
    peak_rss, status = 0, "ok"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "solution.py", "task.md"], cwd=run_dir,
                                env=load_env(), stdout=log, stderr=subprocess.STDOUT)
        ps = psutil.Process(proc.pid)
        try:
            ps.cpu_affinity(list(range(min(cpus, psutil.cpu_count()))))
        except (psutil.Error, AttributeError):
            pass
        while proc.poll() is None:
            members = tree(ps)
            rss = 0
            for p in members:
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            peak_rss = max(peak_rss, rss)
            if time.monotonic() - started > timeout:
                status = "timeout"
                for p in reversed(members):
                    try:
                        p.kill()
                    except psutil.Error:
                        pass
                break
            time.sleep(2)
        proc.wait()
    elapsed = time.monotonic() - started
    if status == "ok" and proc.returncode != 0:
        status = f"exit {proc.returncode}"

    result = {"task": name, "run_dir": run_dir.name, "status": status, "seconds": round(elapsed, 1),
              "peak_rss_gb": round(peak_rss / 2**30, 2), "format": "", "metric": "",
              "score_vs_rf": "", "score_vs_linear": ""}
    files = locate(task_md)
    if not files.submission.exists():
        result["format"] = "нет submission.csv"
        return result
    sub, test, sample = (pd.read_csv(p) for p in (files.submission, files.test, files.sample_submission))
    train_target = pd.read_csv(files.train, usecols=[sample.columns[1]]).iloc[:, 0]
    classes = None if pd.api.types.is_numeric_dtype(train_target) else set(train_target.dropna().unique())
    problems = check_submission(sub, test, sample, classes)
    result["format"] = "ok" if not problems else "; ".join(problems)

    answers = BENCH / "answers" / f"{name}.csv"
    if answers.exists() and not problems:
        meta = json.loads((BENCH / "answers" / f"{name}.json").read_text(encoding="utf-8"))
        truth = pd.read_csv(answers)
        merged = truth.merge(sub, on=meta["id"], suffixes=("_true", "_pred"))
        value = METRICS[meta["kind"]](merged[f"{meta['target']}_true"], merged[f"{meta['target']}_pred"])
        result["metric"] = round(value, 5)
        for base in meta.get("proxy_baseline", {}):
            key = "score_vs_rf" if base == "random_forest" else "score_vs_linear"
            result[key] = round(score(meta["kind"], value, meta["proxy_baseline"][base]), 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--timeout", type=float, default=3 * 3600)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--zip", type=Path, help="готовый архив; по умолчанию собираем из code/")
    args = parser.parse_args()
    archive = args.zip.resolve() if args.zip else build(CODE / "dist" / "solution.zip")

    results_path = BENCH / "results.csv"
    for name in args.tasks:
        result = run(name, args.timeout, args.cpus, archive)
        print(json.dumps(result, ensure_ascii=False))
        new = not results_path.exists()
        with open(results_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["finished", *result])
            if new:
                writer.writeheader()
            writer.writerow({"finished": f"{datetime.now():%Y-%m-%d %H:%M}", **result})


if __name__ == "__main__":
    main()
