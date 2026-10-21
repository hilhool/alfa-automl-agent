"""Запуск: OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... uv run solution.py task.md

Цепочка fallback: сначала на диск ложится константа (среднее или мода),
потом поверх неё ядро без LLM: разбор задачи правилами, схема CV, LightGBM.
Каждый следующий шаг (история сущностей, лаги, log-таргет, ансамбль, веса классов) принимается, только если
выиграл на тех же фолдах, и сразу перезаписывает сабмит. Упал шаг — остаётся прошлый файл.
"""

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.adversarial import adversarial  # noqa: E402
from agent.core import CoreResult, OutOfTime, run_core  # noqa: E402
from agent.cv import Folds, make_folds  # noqa: E402
from agent.ensemble import caruana, median_blend  # noqa: E402
from agent.entity import add_history, add_lags, find_key  # noqa: E402
from agent.metrics import GREATER_IS_BETTER, NONNEG_METRICS, POSITIVE_NAMES, better, to_labels  # noqa: E402
from agent.llm import LLM  # noqa: E402
from agent.postproc import class_weights  # noqa: E402
from agent.research import ask_metric, research_loop  # noqa: E402
from agent.spec import KIND_OF_METRIC, PROBA_METRICS, TaskSpec, find_metric, parse  # noqa: E402
from agent.taskio import check_submission, locate, write_atomic  # noqa: E402
from agent.text import find_text_columns, text_features  # noqa: E402

log = logging.getLogger("solution")

MIN_FOLD_WIN_SHARE = 0.7
MIN_REL_GAIN = 0.002
# Новые модели зоопарка не запускаем позже этого срока от старта. Лимит платформы 3 ч,
# запас на случай, если её CPU медленнее нашего стенда.
ZOO_DEADLINE_S = 100 * 60
# Цикл гипотез LLM не дольше этого срока от старта: дальше время нужно зоопарку.
RESEARCH_DEADLINE_S = 45 * 60
# Потолок на одну модель: CatBoost на 10 классах шёл 28 минут и смесь не улучшил.
MODEL_MAX_S = 15 * 60
# CatBoost MultiClass на 10 классах и 87 тыс. строк шёл 15 минут на фолд: потолок срабатывает
# только после первого фолда, поэтому при многих классах его не запускаем вовсе.
MAX_CAT_CLASSES = 5
FORWARD_REFIT_MULT = 2.5
LAST_PERIOD_REPEATS = 3
ZOO = [
    ("lgb_slow", "lgb", dict(learning_rate=0.03, num_leaves=15, min_child_samples=50, feature_fraction=0.6)),
    ("cat", "cat", {}),
    ("xgb", "xgb", {}),
    ("lgb_deep", "lgb", dict(learning_rate=0.03, num_leaves=255, min_child_samples=10, lambda_l2=5.0,
                              feature_fraction=0.5)),
]


@dataclass
class Stage:
    """Текущее лучшее: признаки, результат на фолдах и преобразование таргета."""
    train: pd.DataFrame
    test: pd.DataFrame
    result: CoreResult
    transform: str | None


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
        values = np.maximum(pred, 0) if spec.metric in NONNEG_METRICS else pred
    elif spec.proba_output:
        named = [i for i, c in enumerate(spec.classes) if str(c).strip().lower() in POSITIVE_NAMES]
        values = pred[:, named[0] if named else 1]
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
    """Новый вариант лучше в среднем хотя бы на MIN_REL_GAIN и не хуже минимум на 70% фолдов.
    Ничья не проигрыш: у признаков по прошлому на самом раннем периоде пусто, и там фолд
    совпадает до знака. Порог нужен против шума: лаги дали +0,05% по OOF и −2,5 RMSE на holdout."""
    pairs = [(a, b) for a, b in zip(new.fold_scores, old.fold_scores) if np.isfinite(a) and np.isfinite(b)]
    if not pairs:
        return False
    worse = [better(metric, b, a) for a, b in pairs]
    margin = MIN_REL_GAIN * abs(old.score)
    # Штрафуем новый вариант на margin: он должен выиграть с запасом.
    shifted = new.score - margin if GREATER_IS_BETTER[metric] else new.score + margin
    return better(metric, shifted, old.score) and 1 - np.mean(worse) >= MIN_FOLD_WIN_SHARE


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
    try:
        classes = None if pd.api.types.is_numeric_dtype(train[target]) else set(train[target].dropna().unique())
        if not save(constant_submission(train, test, sample), test, sample, classes, files.submission):
            raise RuntimeError("константный сабмит не прошёл проверку")
        log.info("Константа записана за %.1f с", time.monotonic() - started)
    except Exception:
        # Последний рубеж: таргет или id называются не как в sample. Шаблон лучше, чем ничего.
        log.exception("Константа не вышла, пишем копию sample_submission")
        write_atomic(sample, files.submission)
        return

    try:
        spec = parse(files.task_text, train.dropna(subset=[target]), sample)
        train = train.dropna(subset=[target]).reset_index(drop=True)
        llm = LLM()
        if find_metric(files.task_text) is None and llm.enabled:
            allowed = [m for m in GREATER_IS_BETTER
                       if (KIND_OF_METRIC.get(m) == "regression") == (spec.kind == "regression")]
            metric = ask_metric(llm, files.task_text, spec.kind, allowed)
            if metric:
                spec.metric = metric
                spec.proba_output = (spec.kind == "binary" and metric in PROBA_METRICS
                                     and pd.api.types.is_numeric_dtype(sample[target]))
        log.info("spec: %s", spec)
        if spec.proba_output:
            classes = None
        features = [c for c in train.columns if c not in (spec.id_col, spec.target) and c in test.columns]
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

    state = Stage(train, test, best, None)

    def submit(result: CoreResult, what: str, weights: np.ndarray | None = None) -> None:
        if save(to_submission(result.test_pred, test, spec, weights), test, sample, classes, files.submission):
            log.info("Сабмит (%s) записан за %.1f с", what, time.monotonic() - started)

    def attempt(what: str, make) -> tuple[bool, float | None]:
        """Пробуем шаг. Выиграл по фолдам — он становится текущим и пишет сабмит.
        Возвращает (принят ли, скор кандидата)."""
        try:
            new = make()
        except OutOfTime as error:
            log.info("%s: пропущено, %s", what, error)
            return False, None
        except Exception:
            log.exception("%s упало, остаётся прошлый сабмит", what)
            return False, None
        if new is None:
            return False, None
        if wins(spec.metric, new.result, state.result):
            log.info("%s: принято, %s %.5f против %.5f", what, spec.metric, new.result.score, state.result.score)
            state.__dict__.update(new.__dict__)
            submit(state.result, what)
            return True, new.result.score
        log.info("%s: отклонено, %s %.5f против %.5f", what, spec.metric, new.result.score, state.result.score)
        return False, new.result.score

    if folds.period_col is not None and spec.kind == "regression":
        key = None
        try:
            key = find_key(train, test, features, folds.period_col, spec.target)
        except Exception:
            log.exception("Поиск ключа сущности упал")
        if key is not None:
            def history() -> Stage:
                tr, te, _ = add_history(train, test, key, folds.period_col, spec.target)
                return Stage(tr, te, run_core(tr, te, spec, folds, name="entity"), None)
            attempt("entity", history)

            def lags() -> Stage:
                tr, te, _ = add_lags(state.train, state.test, key, folds.period_col, spec.target, features)
                return Stage(tr, te, run_core(tr, te, spec, folds, name="lags"), None)
            attempt("lags", lags)

    # AutoResearch: LLM предлагает гипотезы по смыслу колонок и по ошибкам модели,
    # проверяем быстрым LightGBM на тех же фолдах. Без ключа LLM цикл не запускается.
    if llm.enabled:
        # Дедлайн с запасом: начатую оценку гипотезы на большой таблице не бросаем на середине,
        # но и затянуть её на время зоопарка не даём.
        research_deadline = started + RESEARCH_DEADLINE_S

        def evaluate_stage(what: str, tr: pd.DataFrame, te: pd.DataFrame):
            return attempt(what, lambda: Stage(tr, te, run_core(tr, te, spec, folds, name=what,
                                                               deadline=research_deadline + 10 * 60), None))
        try:
            dictionary = pd.read_csv(files.data_dictionary) if files.data_dictionary else None
            research_loop(llm, state, spec, folds, files.task_text, dictionary, evaluate_stage,
                          research_deadline, files.task_dir / "agent_journal.jsonl")
        except Exception:
            log.exception("research упал, идём дальше с тем, что есть")

    # Зоопарк: те же признаки и фолды, разные модели. Для неотрицательного таргета регрессии
    # каждая модель ещё и на log1p(y): выбирать по OOF, логарифмировать ли, ненадёжно
    # (на reg_holdout OOF почти равны, а test 38,9 против 41,2), медиана решит без выбора.
    #
    # При валидации вперёд по времени (проверено на двух holdout регрессии, H4 и H5):
    # early stopping по одному следующему периоду останавливается рано, рефит ×2,5 раунда
    # дал −0,9 RMSE; вес ×3 последнему периоду ещё −0,3; модель на остатке к прошлому
    # значению сущности как второй вариант каждого кандидата — ещё −0,6 сверху.
    candidates = {"lgb": state.result}
    transforms = [None]
    if spec.kind == "regression" and train[spec.target].min() >= 0:
        transforms.append("log1p")
    forward = folds.scheme == "time_forward"
    extra = {}
    if forward:
        last = state.train[folds.period_col] == state.train[folds.period_col].max()
        extra = dict(refit_mult=FORWARD_REFIT_MULT, repeats=np.where(last, LAST_PERIOD_REPEATS, 1))
    residual = "ent_last" if spec.kind == "regression" and "ent_last" in state.train.columns else None
    deadline = started + ZOO_DEADLINE_S
    plan = [(f"{name}+{t}" if t else name, family, params, t) for t in transforms
            for name, family, params in [("lgb", "lgb", {})] + ZOO]
    for name, family, params, transform in plan:
        if name == "lgb" and not forward:
            continue
        if family == "cat" and spec.kind == "multiclass" and len(spec.classes) > MAX_CAT_CLASSES:
            log.info("zoo: %s пропускаем, %d классов", name, len(spec.classes))
            continue
        if time.monotonic() > deadline:
            log.info("zoo: время вышло, %s и дальше не запускаем", name)
            break
        try:
            candidates[name] = run_core(state.train, state.test, spec, folds, params, family, transform,
                                        min(deadline, time.monotonic() + MODEL_MAX_S), name, **extra)
            if residual:
                candidates[name + "+resid"] = run_core(
                    state.train, state.test, spec, folds, params, family, transform,
                    min(deadline, time.monotonic() + MODEL_MAX_S), name + "+resid", residual_of=residual,
                    fallback=candidates[name], **extra)
        except OutOfTime as error:
            log.info("zoo: %s", error)
        except Exception:
            log.exception("zoo: %s упала", name)

    def blend() -> Stage | None:
        y = state.train[spec.target]
        if spec.kind == "regression":
            mixed = median_blend(candidates, y, spec.metric, folds)
        else:
            mixed = caruana(candidates, y, spec.metric, spec.classes, folds)
            mixed = mixed and mixed[0]
        return None if mixed is None else Stage(state.train, state.test, mixed, None)
    attempt("ensemble", blend)

    if spec.kind != "regression" and not spec.proba_output:
        try:
            weights = class_weights(train[spec.target], state.result.oof, state.result.oof_mask,
                                    fold_index(folds, len(train)), spec.classes, spec.metric)
            if weights is not None:
                submit(state.result, "веса классов", weights)
        except Exception:
            log.exception("Подстройка весов классов упала, остаётся прошлый сабмит")
    log.info("Готово за %.1f с", time.monotonic() - started)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "task.md"))
