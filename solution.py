"""Запуск: OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... uv run solution.py task.md

Цепочка fallback: сначала на диск ложится константа (среднее или мода),
потом поверх неё ядро без LLM: разбор задачи правилами, схема CV, LightGBM.
Каждый следующий шаг (история сущностей, веса классов) принимается, только если
выиграл на тех же фолдах, и сразу перезаписывает сабмит. Упал шаг — остаётся прошлый файл.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.adversarial import adversarial  # noqa: E402
from agent.core import CoreResult, run_core  # noqa: E402
from agent.cv import Folds, make_folds  # noqa: E402
from agent.entity import add_history, find_key  # noqa: E402
from agent.metrics import better, to_labels  # noqa: E402
from agent.postproc import class_weights  # noqa: E402
from agent.spec import TaskSpec, parse  # noqa: E402
from agent.taskio import check_submission, locate, write_atomic  # noqa: E402
from agent.text import find_text_columns, text_features  # noqa: E402

log = logging.getLogger("solution")

MIN_FOLD_WIN_SHARE = 0.7


def constant_submission(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    id_col, target = sample.columns[0], sample.columns[1]
    y = train[target]
    is_regression = pd.api.types.is_float_dtype(y) or (
        pd.api.types.is_numeric_dtype(y) and y.nunique() > 50)
    value = y.mean() if is_regression else y.mode().iloc[0]
    return pd.DataFrame({id_col: test[id_col].to_numpy(), target: value})


def to_submission(pred: np.ndarray, test: pd.DataFrame, spec: TaskSpec,
                  weights: np.ndarray | None = None) -> pd.DataFrame:
    if spec.kind == "regression":
        values = pred
    elif spec.proba_output:
        values = pred[:, 1]
    else:
        values = to_labels(pred, spec.classes, weights)
    return pd.DataFrame({spec.id_col: test[spec.id_col].to_numpy(), spec.target: values})


def save(sub: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, classes: set | None, path: Path) -> bool:
    problems = check_submission(sub, test, sample, classes)
    if problems:
        log.error("Сабмит не прошёл проверку, не пишем: %s", problems)
        return False
    write_atomic(sub, path)
    return True


def wins(metric: str, new: CoreResult, old: CoreResult) -> bool:
    """Новый вариант лучше в среднем и не хуже минимум на 70% фолдов. Ничья не проигрыш:
    у признаков по прошлому на самом раннем периоде пусто, и там фолд совпадает до знака."""
    worse = [better(metric, b, a) for a, b in zip(new.fold_scores, old.fold_scores)]
    return better(metric, new.score, old.score) and 1 - np.mean(worse) >= MIN_FOLD_WIN_SHARE


def fold_index(folds: Folds, n: int) -> np.ndarray:
    fold_of = np.full(n, -1)
    for i, (_, val) in enumerate(folds.splits):
        fold_of[val] = i
    return fold_of


def with_text(train: pd.DataFrame, test: pd.DataFrame, features: list[str], spec: TaskSpec, folds: Folds,
              dictionary_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Текстовые колонки заменяем OOF текстовой модели. Не вышло — просто выбрасываем их:
    как category они только тормозят бустинг."""
    dictionary = pd.read_csv(dictionary_path) if dictionary_path else None
    text_cols = find_text_columns(train, features, dictionary)
    if not text_cols:
        return train, test, features
    log.info("text: колонки %s", text_cols)
    rest = [c for c in features if c not in text_cols]
    try:
        f_tr, f_te = text_features(train, test, text_cols, spec, folds)
    except Exception:
        log.exception("Текстовая модель упала, текстовые колонки выбрасываем")
        f_tr, f_te = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    train = pd.concat([train.drop(columns=text_cols), f_tr], axis=1)
    test = pd.concat([test.drop(columns=text_cols), f_te], axis=1)
    return train, test, rest


def main(task_path: Path) -> None:
    started = time.monotonic()
    files = locate(task_path)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(files.task_dir / "agent.log", encoding="utf-8")],
    )
    log.info("train=%s test=%s sample=%s", files.train, files.test, files.sample_submission)

    train = pd.read_csv(files.train)
    test = pd.read_csv(files.test)
    sample = pd.read_csv(files.sample_submission)

    target = sample.columns[1]
    classes = None if pd.api.types.is_numeric_dtype(train[target]) else set(train[target].dropna().unique())
    if not save(constant_submission(train, test, sample), test, sample, classes, files.submission):
        raise RuntimeError("Даже константный сабмит не прошёл проверку")
    log.info("Константа записана за %.1f с", time.monotonic() - started)

    try:
        spec = parse(files.task_text, train.dropna(subset=[target]), sample)
        train = train.dropna(subset=[target]).reset_index(drop=True)
        log.info("spec: %s", spec)
        if spec.proba_output:
            classes = None
        features = [c for c in train.columns if c not in (spec.id_col, spec.target)]
        folds = make_folds(train, test, features, train[spec.target], spec.kind)
        log.info("cv: %s, %d фолдов. %s", folds.scheme, len(folds.splits), folds.note)
        train, test, features = with_text(train, test, features, spec, folds, files.data_dictionary)
        best = run_core(train, test, spec, folds)
        if save(to_submission(best.test_pred, test, spec), test, sample, classes, files.submission):
            log.info("Ядро записано за %.1f с", time.monotonic() - started)
    except Exception:
        log.exception("Ядро упало, остаётся константа")
        return

    try:
        adversarial(train, test, features)
    except Exception:
        log.exception("adversarial validation упала, идём дальше")

    if folds.period_col is not None and spec.kind == "regression":
        try:
            key = find_key(train, test, features, folds.period_col, spec.target)
            if key is not None:
                tr_e, te_e, _ = add_history(train, test, key, folds.period_col, spec.target)
                candidate = run_core(tr_e, te_e, spec, folds)
                if wins(spec.metric, candidate, best):
                    best = candidate
                    log.info("entity: принято, %s %.5f", spec.metric, best.score)
                    if save(to_submission(best.test_pred, test, spec), test, sample, classes, files.submission):
                        log.info("Сабмит с историей сущностей записан за %.1f с", time.monotonic() - started)
                else:
                    log.info("entity: отклонено, %s %.5f против %.5f", spec.metric, candidate.score, best.score)
        except Exception:
            log.exception("История сущностей упала, остаётся прошлый сабмит")

    if spec.kind != "regression" and not spec.proba_output:
        try:
            weights = class_weights(train[spec.target], best.oof, best.oof_mask,
                                    fold_index(folds, len(train)), spec.classes, spec.metric)
            if weights is not None and save(to_submission(best.test_pred, test, spec, weights), test, sample,
                                            classes, files.submission):
                log.info("Сабмит с весами классов записан за %.1f с", time.monotonic() - started)
        except Exception:
            log.exception("Подстройка весов классов упала, остаётся прошлый сабмит")
    log.info("Готово за %.1f с", time.monotonic() - started)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "task.md"))
