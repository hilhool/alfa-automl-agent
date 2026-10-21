"""Поиск файлов задачи, запись и проверка сабмита.

Пути к CSV берём из task.md, как в условии. Если по ним файла нет,
ищем по имени во всём каталоге задачи: на платформе раскладка может
отличаться от локальной.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = ("train", "test", "sample_submission", "data_dictionary")
CSV_IN_TEXT = re.compile(r"[\w./\\-]+\.csv")


@dataclass
class TaskFiles:
    task_dir: Path
    task_text: str
    train: Path
    test: Path
    sample_submission: Path
    data_dictionary: Path | None
    submission: Path


def _find(task_dir: Path, mentioned: list[str], role: str) -> Path | None:
    name = f"{role}.csv"
    for rel in mentioned:
        if Path(rel).name == name:
            path = (task_dir / rel).resolve()
            if path.is_file():
                return path
    direct = task_dir / name
    if direct.is_file():
        return direct
    # Самый мелкий по глубине кандидат, submission.csv агента не попадёт: имя другое.
    found = sorted(task_dir.rglob(name), key=lambda p: len(p.parts))
    return found[0] if found else None


def locate(task_path: Path) -> TaskFiles:
    task_path = task_path.resolve()
    task_dir = task_path.parent
    text = task_path.read_text(encoding="utf-8")
    mentioned = [m.replace("\\", "/") for m in CSV_IN_TEXT.findall(text)]
    paths = {role: _find(task_dir, mentioned, role) for role in ROLES}
    missing = [r for r in ("train", "test", "sample_submission") if paths[r] is None]
    if missing:
        raise FileNotFoundError(f"Не нашли {missing} в {task_dir}")
    return TaskFiles(task_dir=task_dir, task_text=text, submission=task_dir / "submission.csv", **paths)


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    """Пишем во временный файл и переименовываем: kill посреди записи не оставит битый CSV."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def check_submission(sub: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame,
                     classes: set | None = None) -> list[str]:
    """Пустой список, если сабмит корректен, иначе список проблем."""
    problems = []
    if list(sub.columns) != list(sample.columns):
        problems.append(f"колонки {list(sub.columns)} вместо {list(sample.columns)}")
        return problems
    id_col, target = sample.columns[0], sample.columns[1]
    if len(sub) != len(test):
        problems.append(f"{len(sub)} строк вместо {len(test)}")
        return problems
    if id_col in test.columns and not (sub[id_col].to_numpy() == test[id_col].to_numpy()).all():
        problems.append("id не совпадают с test по значениям или порядку")
    if sub[id_col].duplicated().any():
        problems.append("повторяющиеся id")
    values = sub[target]
    if values.isna().any():
        problems.append(f"{int(values.isna().sum())} пропусков в {target}")
    if pd.api.types.is_numeric_dtype(values) and not np.isfinite(values.to_numpy(dtype=float)).all():
        problems.append(f"inf в {target}")
    if classes is not None:
        extra = set(values.dropna().unique()) - classes
        if extra:
            problems.append(f"классы вне train: {sorted(map(str, extra))[:5]}")
    return problems
