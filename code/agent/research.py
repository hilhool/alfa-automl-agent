"""Цикл AutoResearch: LLM формулирует гипотезу, код её проверяет.

Каждую итерацию LLM получает сводку: задачу, колонки с описаниями, текущий
скор по фолдам, срезы с наибольшей ошибкой и журнал уже проверенных гипотез.
Она возвращает одну гипотезу: операцию из меню (отношение, разность,
групповой агрегат, удаление колонки) или короткую функцию на pandas.

Признак считается на train+test без таргета, проверяется на утечку и
оценивается на тех же фолдах тем же правилом принятия, что и остальные шаги.
Фолды, метрику и таргет LLM не видит и не трогает. Журнал пишется на диск.
"""

import builtins
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agent.llm import LLM
from agent.metrics import GREATER_IS_BETTER, to_labels
from agent.spec import TaskSpec

log = logging.getLogger("solution")

MAX_ITERS = 12
MAX_PROMPT_COLUMNS = 80
MAX_TASK_CHARS = 2500
CODE_TIMEOUT_S = 60
LEAK_CORR = 0.995
MIN_SLICE_ROWS = 30
ACTIONS = ("ratio", "diff", "product", "group_mean", "group_std", "group_count", "freq", "drop", "code")
FORBIDDEN_CODE = ("import", "__", "open(", "exec", "eval", "compile", "globals", "locals", "getattr",
                  "setattr", "os.", "sys.", "subprocess", "read_csv", "to_csv")
SAFE_BUILTINS = {k: getattr(builtins, k) for k in (
    "abs", "min", "max", "len", "range", "float", "int", "str", "round", "sum", "zip", "enumerate",
    "list", "dict", "set", "tuple", "isinstance", "bool", "sorted", "any", "all", "map", "filter",
    "ValueError", "KeyError", "Exception")}

SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "action": {"type": "string", "enum": list(ACTIONS)},
        "columns": {"type": "array", "items": {"type": "string"}},
        "group_by": {"type": "string"},
        "code": {"type": "string"},
    },
    # code и group_by обязательны (пустая строка, если не нужны): необязательные поля модель опускает,
    # и все гипотезы с кодом приходили без кода.
    "required": ["hypothesis", "action", "columns", "group_by", "code"],
}

SYSTEM = """You are a senior data scientist inside an automated ML agent.
Each turn you propose ONE hypothesis that may improve the cross-validation score of a gradient boosting model
(LightGBM) on a tabular task. You never see test labels; the agent evaluates your idea on fixed folds and keeps it
only if the score improves. Prefer ideas grounded in the meaning of the columns and in where the model errs.
Do not repeat hypotheses from the journal. Use exact column names. Answer with JSON only."""

ACTION_HELP = """Allowed actions:
- ratio: columns=[a, b] -> a / b
- diff: columns=[a, b] -> a - b
- product: columns=[a, b] -> a * b
- group_mean / group_std: columns=[a], group_by=g -> mean/std of numeric a within each value of g
- group_count: columns=[], group_by=g -> number of rows with the same g
- freq: columns=[a] -> share of rows with the same value of a
- drop: columns=[a, ...] -> remove noisy or shifted columns
- code: columns=[columns you use], code = Python defining `def feature(df):` that returns a pandas Series of
  len(df). df holds train and test feature rows together WITHOUT the target. Only `np` and `pd` are available,
  no imports, no file access. Keep it vectorized and short. The "code" field is REQUIRED for this action, e.g.
  "def feature(df):
    return df['a'] / df.groupby('g')['a'].transform('median')"
Monotone transforms of a single column (log, sqrt, scaling) do not help tree models: do not propose them."""


@dataclass
class Trial:
    iteration: int
    hypothesis: str
    action: str
    columns: list
    group_by: str | None
    code: str | None
    status: str          # accepted | rejected | invalid | error
    score: float | None
    before: float
    note: str = ""


def describe_columns(train: pd.DataFrame, features: list[str], dictionary: pd.DataFrame | None) -> str:
    notes = {}
    if dictionary is not None and "feature" in dictionary.columns:
        text_col = next((c for c in dictionary.columns if "desc" in c.lower()), None)
        if text_col:
            notes = dict(zip(dictionary["feature"].astype(str), dictionary[text_col].astype(str)))
    rows = []
    for col in features[:MAX_PROMPT_COLUMNS]:
        s = train[col]
        rows.append(f"{col} | {s.dtype} | uniq {s.nunique()} | miss {s.isna().mean():.0%} | "
                    f"{notes.get(col, '')[:100]}")
    more = len(features) - MAX_PROMPT_COLUMNS
    return "\n".join(rows) + (f"\n... and {more} more columns" if more > 0 else "")


def error_slices(train: pd.DataFrame, spec: TaskSpec, result) -> str:
    """Где модель ошибается сильнее всего: по значениям категорий для регрессии,
    по классам и путаницам для классификации."""
    mask = result.oof_mask
    y = train[spec.target].to_numpy()[mask]
    oof = result.oof[mask]
    lines = []
    if spec.kind == "regression":
        err = (y.astype(float) - oof) ** 2
        overall = float(np.sqrt(err.mean()))
        lines.append(f"overall OOF RMSE {overall:.4g}")
        cats = [c for c in train.columns if c != spec.target and not pd.api.types.is_numeric_dtype(train[c])
                and train[c].nunique() <= 60]
        worst = []
        for col in cats:
            groups = pd.Series(err).groupby(train[col].to_numpy()[mask])
            for value, v in groups:
                if len(v) >= MIN_SLICE_ROWS:
                    worst.append((float(np.sqrt(v.mean())), col, value, len(v)))
        for rmse, col, value, n in sorted(worst, reverse=True)[:8]:
            lines.append(f"{col}={value}: RMSE {rmse:.4g} on {n} rows")
        resid = y.astype(float) - oof
        lines.append(f"mean residual {resid.mean():.4g} (positive = model underestimates)")
    else:
        labels = to_labels(oof, spec.classes)
        from sklearn.metrics import confusion_matrix, f1_score
        f1 = f1_score(y.astype(str), labels.astype(str), average=None, labels=[str(c) for c in spec.classes])
        for c, v in sorted(zip(spec.classes, f1), key=lambda t: t[1])[:6]:
            lines.append(f"class {c}: F1 {v:.3f}, share {np.mean(y.astype(str) == str(c)):.3f}")
        cm = confusion_matrix(y.astype(str), labels.astype(str), labels=[str(c) for c in spec.classes])
        np.fill_diagonal(cm, 0)
        pairs = sorted(((cm[i, j], spec.classes[i], spec.classes[j]) for i in range(len(cm)) for j in range(len(cm))
                        if cm[i, j] > 0), reverse=True)[:5]
        for n, a, b in pairs:
            lines.append(f"true {a} predicted as {b}: {n} rows")
    return "\n".join(lines)


def build_prompt(task_text: str, spec: TaskSpec, train: pd.DataFrame, features: list[str],
                 dictionary: pd.DataFrame | None, result, scheme: str, journal: list[Trial]) -> str:
    direction = "higher is better" if GREATER_IS_BETTER[spec.metric] else "lower is better"
    history = "\n".join(
        f"- [{t.status}] {t.action} {t.columns} {t.group_by or ''}: {t.hypothesis[:140]}"
        + (f" -> {t.score:.5g} vs {t.before:.5g}" if t.score is not None else f" ({t.note[:80]})")
        for t in journal) or "- nothing yet"
    return f"""Task description (from task.md):
{task_text[:MAX_TASK_CHARS]}

Target: {spec.target}; task type: {spec.kind}; metric: {spec.metric} ({direction}); validation: {scheme}.
Current CV score: {result.score:.5g}; per fold: {[round(s, 5) for s in result.fold_scores]}

Columns (name | dtype | unique | missing | description):
{describe_columns(train, features, dictionary)}

Where the model errs:
{error_slices(train, spec, result)}

Journal of tried hypotheses:
{history}

{ACTION_HELP}

The target column `{spec.target}` is NOT available to features (no target encoding).
Propose the single most promising NEW hypothesis."""


def run_code(code: str, frame: pd.DataFrame) -> pd.Series:
    lowered = code.lower()
    bad = [w for w in FORBIDDEN_CODE if w in lowered]
    if bad:
        raise ValueError(f"запрещённые конструкции в коде: {bad}")
    namespace = {"np": np, "pd": pd, "__builtins__": SAFE_BUILTINS}
    exec(compile(code, "<llm_feature>", "exec"), namespace)  # noqa: S102 — песочница без импортов и builtins
    fn = namespace.get("feature")
    if not callable(fn):
        raise ValueError("в коде нет функции feature(df)")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, frame.copy())
        try:
            out = future.result(timeout=CODE_TIMEOUT_S)
        except FutureTimeout:
            raise ValueError(f"код не уложился в {CODE_TIMEOUT_S} с")
    return pd.Series(np.asarray(out).ravel() if not isinstance(out, pd.Series) else out.to_numpy())


def make_feature(action: str, cols: list[str], group_by: str | None, code: str | None,
                 frame: pd.DataFrame) -> pd.Series:
    def num(c: str) -> pd.Series:
        return pd.to_numeric(frame[c], errors="coerce")

    if action == "code":
        return run_code(code or "", frame)
    if action in ("ratio", "diff", "product"):
        a, b = num(cols[0]), num(cols[1])
        if action == "ratio":
            return a / b.replace(0, np.nan)
        return a - b if action == "diff" else a * b
    if action == "freq":
        return frame[cols[0]].map(frame[cols[0]].value_counts(normalize=True)).astype(float)
    key = frame[group_by].astype(str)
    if action == "group_count":
        return key.map(key.value_counts()).astype(float)
    stat = "mean" if action == "group_mean" else "std"
    return num(cols[0]).groupby(key).transform(stat)


def validate(action: str, cols: list[str], group_by: str | None, code: str | None, columns: set[str]) -> str | None:
    """Текст ошибки или None. Проверяем имена до того, как что-то считать."""
    unknown = [c for c in cols + ([group_by] if group_by else []) if c not in columns]
    if action != "code" and unknown:
        return f"нет колонок {unknown}"
    need = {"ratio": 2, "diff": 2, "product": 2, "group_mean": 1, "group_std": 1, "freq": 1}
    if action in need and len(cols) < need[action]:
        return f"для {action} нужно {need[action]} колонки"
    if action.startswith("group_") and not group_by:
        return "нет group_by"
    if action == "drop" and not cols:
        return "нечего удалять"
    if action == "code" and not code:
        return "для code нужно поле code с функцией feature(df)"
    return None


METRIC_SCHEMA = {"type": "object", "properties": {"metric": {"type": "string"}, "quote": {"type": "string"}},
                 "required": ["metric"]}


def ask_metric(llm: LLM, task_text: str, kind: str, allowed: list[str]) -> str | None:
    """Метрика, когда регулярки её не нашли. Ответ принимаем, только если он из списка."""
    answer = llm.ask_json(
        "You read ML task descriptions and identify the evaluation metric. Answer with JSON only.",
        f"Task type: {kind}. Allowed metric ids: {allowed}.\n"
        f"Which metric will be used to score predictions? Quote the sentence you rely on.\n\n{task_text[:4000]}",
        METRIC_SCHEMA, "metric")
    metric = str((answer or {}).get("metric", "")).strip().lower()
    if metric in allowed:
        log.info("llm: метрика %s по фразе «%s»", metric, str(answer.get("quote", ""))[:150])
        return metric
    return None


def research_loop(llm: LLM, state, spec: TaskSpec, folds, task_text: str, dictionary: pd.DataFrame | None,
                  evaluate_stage, deadline: float, journal_path: Path) -> list[Trial]:
    """evaluate_stage(name, train, test) -> (принят ли, скор кандидата): оценивает вариант
    признаков и, если он выиграл по правилу принятия, делает его текущим (и пишет сабмит)."""
    journal: list[Trial] = []
    for it in range(MAX_ITERS):
        if not llm.enabled or time.monotonic() > deadline:
            log.info("research: стоп на итерации %d (llm %s, время %s)", it, llm.enabled,
                     time.monotonic() > deadline)
            break
        features = [c for c in state.train.columns if c not in (spec.id_col, spec.target)]
        before = state.result.score
        prompt = build_prompt(task_text, spec, state.train, features, dictionary, state.result,
                              folds.scheme, journal)
        answer = llm.ask_json(SYSTEM, prompt, SCHEMA, "hypothesis")
        if answer is None:
            continue
        action = str(answer.get("action", ""))
        cols = [str(c) for c in answer.get("columns") or []]
        group_by = answer.get("group_by") or None
        if isinstance(group_by, list):
            group_by = group_by[0] if group_by else None
        group_by = str(group_by) if group_by else None
        code = answer.get("code") or None
        code = str(code) if code else None
        trial = Trial(it, str(answer.get("hypothesis", ""))[:500], action, cols, group_by, code, "invalid", None,
                      before)
        error = "неизвестное действие" if action not in ACTIONS else validate(action, cols, group_by, code,
                                                                                  set(features))
        if error is None:
            try:
                accepted, score, note = try_hypothesis(trial, state, spec, features, evaluate_stage)
                trial.status, trial.score, trial.note = ("accepted" if accepted else "rejected"), score, note
            except Exception as exc:  # одна кривая гипотеза не должна останавливать цикл
                trial.status, trial.note = "error", f"{type(exc).__name__}: {str(exc)[:200]}"
        else:
            trial.note = error
        journal.append(trial)
        log.info("research %d: [%s] %s %s %s — %s", it, trial.status, action, cols, group_by or "",
                 trial.hypothesis[:150])
        try:
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(trial), ensure_ascii=False) + "\n")
        except OSError:
            pass
    accepted = sum(t.status == "accepted" for t in journal)
    log.info("research: %d гипотез, принято %d, вызовов LLM %d, токенов %d", len(journal), accepted,
             llm.calls, llm.tokens)
    return journal


def try_hypothesis(trial: Trial, state, spec: TaskSpec, features: list[str], evaluate_stage):
    train, test = state.train, state.test
    if trial.action == "drop":
        keep = [c for c in train.columns if c not in trial.columns]
        ok, score = evaluate_stage(f"research drop {trial.columns}", train[keep],
                                   test[[c for c in keep if c in test]])
        return ok, score, ""

    frame = pd.concat([train[features], test[features]], ignore_index=True)
    values = make_feature(trial.action, trial.columns, trial.group_by, trial.code, frame)
    values = pd.to_numeric(pd.Series(np.asarray(values).ravel()), errors="coerce").replace([np.inf, -np.inf], np.nan)
    if len(values) != len(frame):
        raise ValueError(f"длина признака {len(values)} вместо {len(frame)}")
    if values.notna().sum() < 10 or values.nunique() < 2:
        raise ValueError("признак пустой или константный")
    v_tr, v_te = values.iloc[:len(train)].to_numpy(), values.iloc[len(train):].to_numpy()
    if spec.kind == "regression" or len(spec.classes) == 2:
        target = train[spec.target]
        if spec.kind != "regression":
            target = (target.astype(str) == str(spec.classes[1])).astype(float)
        corr = pd.Series(v_tr).corr(target.reset_index(drop=True), method="spearman")
        if np.isfinite(corr) and abs(corr) > LEAK_CORR:
            raise ValueError(f"подозрение на утечку: корреляция с таргетом {corr:.4f}")
    name = f"llm_{trial.iteration}_{trial.action}"
    new_train = train.assign(**{name: v_tr})
    new_test = test.assign(**{name: v_te})
    ok, score = evaluate_stage(f"research {name}", new_train, new_test)
    return ok, score, ""
