"""Цикл AutoResearch: LLM формулирует гипотезу, код её проверяет.

Каждую итерацию LLM получает сводку: задачу, план аналитика, колонки с описаниями,
текущий скор по фолдам, срезы с наибольшей ошибкой и журнал уже проверенных гипотез.
Она возвращает одну гипотезу: операцию из меню (отношение, разность, групповой
агрегат, удаление колонки, колонка как категория, гиперпараметры LightGBM) или
короткую функцию на pandas. Перед запуском гипотезу смотрит критик (второй вызов LLM
без рассуждения): повтор журнала, таргет в признаке, бесполезное для деревьев
преобразование — гипотеза не запускается, время уходит на следующую.

Признак считается на train+test без таргета, проверяется на утечку и
оценивается на тех же фолдах тем же правилом принятия, что и остальные шаги.
Фолды, метрику и таргет LLM не видит и не трогает. Журнал пишется на диск.
"""

import builtins
import io
import json
import logging
import multiprocessing as mp
import os
import tempfile
import time
import tokenize
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agent.llm import LLM
from agent.metrics import GREATER_IS_BETTER, to_labels
from agent.spec import TaskSpec

log = logging.getLogger("solution")

# Число гипотез ограничивает в первую очередь дедлайн цикла. 12 гипотез на регрессии шли 5 минут,
# на классификации 17, при лимите цикла 45 минут.
MAX_ITERS = 25
MAX_PROMPT_COLUMNS = 80
MAX_TASK_CHARS = 2500
CODE_TIMEOUT_S = 60
LEAK_CORR = 0.995
MIN_SLICE_ROWS = 30
MISSING = -1e30   # пропуск для дерева по остаткам: порог ниже половины значит «пропущено»
# Действия, которые смотрят на другие строки: с производными колонками истории они дают утечку.
CROSS_ROW_ACTIONS = ("group_mean", "group_std", "group_count", "freq", "code")
ACTIONS = ("ratio", "diff", "product", "group_mean", "group_std", "group_count", "freq", "drop", "as_category",
           "params", "code")
# Гиперпараметры LightGBM, которые LLM может менять, и их допустимые границы.
PARAM_RANGES = {"learning_rate": (0.01, 0.2), "num_leaves": (4, 512), "min_child_samples": (2, 1000),
                "feature_fraction": (0.1, 1.0), "bagging_fraction": (0.3, 1.0), "lambda_l1": (0.0, 100.0),
                "lambda_l2": (0.0, 100.0), "max_depth": (-1, 32), "min_split_gain": (0.0, 10.0),
                "max_bin": (15, 1023), "cat_smooth": (1.0, 100.0), "min_data_per_group": (1, 1000),
                "path_smooth": (0.0, 100.0), "extra_trees": (0, 1)}
# Подбор параметров по CV переобучается под фолды быстрее, чем признаки: на синтетике LLM
# тратила на него 18 гипотез из 25. Ограничиваем число попыток.
MAX_PARAM_TRIALS = 6
INT_PARAMS = ("num_leaves", "min_child_samples", "max_depth", "max_bin", "min_data_per_group")
FORBIDDEN_NAMES = {"import", "exec", "eval", "compile", "globals", "locals", "vars", "getattr", "setattr",
                   "delattr", "open", "os", "sys", "subprocess", "input", "breakpoint", "memoryview"}
# Методы pandas и numpy, которые читают или пишут файлы (read_* ловится по префиксу).
FILE_METHODS = {"to_csv", "to_json", "to_pickle", "to_parquet", "to_excel", "to_sql", "to_hdf", "to_feather",
                "to_stata", "to_html", "to_latex", "to_clipboard", "load", "save", "savez", "savez_compressed",
                "savetxt", "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap", "pickle", "dump"}
SEED = 0
SAFE_BUILTINS = {k: getattr(builtins, k) for k in (
    "abs", "min", "max", "len", "range", "float", "int", "str", "round", "sum", "zip", "enumerate",
    "list", "dict", "set", "tuple", "isinstance", "bool", "sorted", "any", "all", "map", "filter",
    "ValueError", "KeyError", "Exception")}

SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string"},
        "hypothesis": {"type": "string"},
        "action": {"type": "string", "enum": list(ACTIONS)},
        "columns": {"type": "array", "items": {"type": "string"}},
        "group_by": {"type": "string"},
        "code": {"type": "string"},
        "params": {"type": "string"},
    },
    # code, group_by и params обязательны (пустая строка, если не нужны): необязательные поля модель
    # опускает, и все гипотезы с кодом приходили без кода.
    "required": ["observation", "hypothesis", "action", "columns", "group_by", "code", "params"],
}
CRITIC_SCHEMA = {"type": "object", "properties": {"run": {"type": "boolean"}, "reason": {"type": "string"}},
                 "required": ["run", "reason"]}
CRITIC_SYSTEM = """You review hypotheses of an automated ML agent before they are run. Reject a hypothesis
(run=false) ONLY if: it repeats or nearly repeats a journal entry; it uses the target column or information
unavailable at prediction time; it is a monotone transform of a single column (useless for trees); or it is
obviously meaningless. Otherwise approve (run=true). Answer with JSON only."""

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
- as_category: columns=[a, ...] -> treat numeric codes as categories instead of numbers
- params: params = JSON object with LightGBM parameters to change, e.g. "{\"num_leaves\": 15,
  \"min_child_samples\": 100}". Allowed keys: {PARAMS}. Use it when the journal shows
  overfitting or underfitting rather than a missing feature. At most {MAX_PARAM} params trials per run:
  features usually matter more.
- code: columns=[columns you use], code = Python defining `def feature(df):` that returns a pandas Series of
  len(df). df holds train and test feature rows together WITHOUT the target. Only `np` and `pd` are available,
  no imports, no file access. Keep it vectorized and short. The "code" field is REQUIRED for this action, e.g.
  "def feature(df):
    return df['a'] / df.groupby('g')['a'].transform('median')"
Monotone transforms of a single column (log, sqrt, scaling) do not help tree models: do not propose them.
Fields that the action does not use must be empty strings."""
ACTION_HELP = ACTION_HELP.replace("{PARAMS}", ", ".join(PARAM_RANGES)).replace("{MAX_PARAM}", str(MAX_PARAM_TRIALS))


@dataclass
class Trial:
    iteration: int
    hypothesis: str
    action: str
    columns: list
    group_by: str | None
    code: str | None
    status: str          # accepted | rejected | invalid | error | vetoed | duplicate
    score: float | None
    before: float
    note: str = ""
    params: dict | None = None
    observation: str = ""   # что LLM заметила в ошибках или журнале и какой эффект ждёт


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


def residual_rules(train: pd.DataFrame, spec: TaskSpec, result, columns: list[str]) -> list[str]:
    """Неглубокое дерево на величине ошибки OOF: правила вида «a > 5 и b пропущен: ошибка X на N строках».
    Срезы по одной категории не видят числовых колонок и сочетаний."""
    from sklearn.tree import DecisionTreeRegressor
    mask = result.oof_mask
    if mask.sum() < 10 * MIN_SLICE_ROWS:
        return []
    y = train[spec.target].to_numpy()[mask]
    if spec.kind == "regression":
        err = (y.astype(float) - result.oof[mask]) ** 2
    else:
        err = (to_labels(result.oof[mask], spec.classes).astype(str) != y.astype(str)).astype(float)
    x = pd.DataFrame(index=np.arange(mask.sum()))
    for c in columns[:MAX_PROMPT_COLUMNS]:
        v = train[c].to_numpy()[mask]
        if pd.api.types.is_numeric_dtype(train[c]):
            x[c] = pd.to_numeric(pd.Series(v), errors="coerce").fillna(MISSING).to_numpy()
        else:
            x[c] = pd.Series(v).astype("category").cat.codes.to_numpy()
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(MIN_SLICE_ROWS, int(0.01 * len(x))),
                                 random_state=SEED).fit(x, err)
    t, overall = tree.tree_, err.mean()
    rules = []

    def walk(node: int, path: list[str]) -> None:
        if t.children_left[node] == -1:
            rules.append((t.value[node][0][0], int(t.n_node_samples[node]), path))
            return
        col, thr = x.columns[t.feature[node]], t.threshold[node]
        if thr < MISSING / 2:
            left, right = f"{col} is missing", f"{col} is present"
        elif pd.api.types.is_numeric_dtype(train[col]):
            left, right = f"{col} <= {thr:.4g}", f"{col} > {thr:.4g}"
        else:
            left, right = f"{col} in first {int(thr) + 1} category codes", f"{col} in later category codes"
        walk(t.children_left[node], path + [left])
        walk(t.children_right[node], path + [right])
    walk(0, [])
    unit = "mean squared error" if spec.kind == "regression" else "error rate"
    out = []
    for value, n, path in sorted(rules, reverse=True)[:5]:
        if value > overall * 1.2:
            out.append(f"{' and '.join(path)}: {unit} {value:.4g} (overall {overall:.4g}) on {n} rows")
    return out


def build_prompt(task_text: str, spec: TaskSpec, train: pd.DataFrame, features: list[str],
                 dictionary: pd.DataFrame | None, result, scheme: str, journal: list[Trial],
                 plan: str = "", params: dict | None = None) -> str:
    direction = "higher is better" if GREATER_IS_BETTER[spec.metric] else "lower is better"
    history = "\n".join(
        f"- [{t.status}] {t.action} {t.columns} {t.group_by or ''}{t.params or ''}: {t.hypothesis[:140]}"
        + (f" -> {t.score:.5g} vs {t.before:.5g}" if t.score is not None else f" ({t.note[:80]})")
        for t in journal) or "- nothing yet"
    return f"""Task description (from task.md):
{task_text[:MAX_TASK_CHARS]}

Research plan written by the analyst before modelling:
{plan or "- none"}

Target: {spec.target}; task type: {spec.kind}; metric: {spec.metric} ({direction}); validation: {scheme}.
Current CV score: {result.score:.5g}; per fold: {[round(s, 5) for s in result.fold_scores]}
LightGBM changes accepted so far: {params or DEFAULTS_NOTE}; trees in final model: {result.rounds}

Columns (name | dtype | unique | missing | description):
{describe_columns(train, features, dictionary)}

Where the model errs:
{error_slices(train, spec, result)}
Regions found by a shallow tree on OOF errors:
{chr(10).join(residual_rules(train, spec, result, features)) or "- none stand out"}

Journal of tried hypotheses:
{history}

{ACTION_HELP}

The target column `{spec.target}` is NOT available to features (no target encoding).
First write `observation`: what in the error slices or the journal motivates the idea and what effect you
expect (which slice should improve). Then propose the single most promising NEW hypothesis."""


DEFAULTS_NOTE = ("none (defaults: learning_rate 0.05, num_leaves 63, min_child_samples 20, feature_fraction 0.8, "
                 "bagging_fraction 0.8, lambda_l2 1)")


def check_code(code: str) -> list[str]:
    """Запрещённые имена в коде LLM. Смотрим токены, а не текст: имя колонки в кавычках
    («import_share») не должно отклонять гипотезу."""
    bad = set()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as error:
        return [f"синтаксис: {error}"]
    for tok in tokens:
        if tok.type != tokenize.NAME:
            continue
        name = tok.string
        if name in FORBIDDEN_NAMES or name.startswith("__") or name.startswith("read_") or name in FILE_METHODS:
            bad.add(name)
    return sorted(bad)


def _sandbox(code: str, frame: pd.DataFrame, conn) -> None:
    """Выполняется в дочернем процессе: зависание или порча файлов не задевают агента."""
    try:
        os.chdir(tempfile.mkdtemp(prefix="llm_feature_"))
        namespace = {"np": np, "pd": pd, "__builtins__": SAFE_BUILTINS}
        exec(compile(code, "<llm_feature>", "exec"), namespace)  # noqa: S102 — песочница без импортов и builtins
        fn = namespace.get("feature")
        if not callable(fn):
            raise ValueError("в коде нет функции feature(df)")
        out = fn(frame)
        # Series с чужим индексом (groupby без transform) не должна молча сдвинуться: выравниваем.
        values = out.reindex(frame.index).to_numpy() if isinstance(out, pd.Series) else np.asarray(out).ravel()
        conn.send(("ok", values))
    except BaseException as error:  # noqa: BLE001 — любая ошибка кода LLM уходит родителю текстом
        conn.send(("error", f"{type(error).__name__}: {str(error)[:300]}"))
    finally:
        conn.close()


def run_code(code: str, frame: pd.DataFrame) -> pd.Series:
    bad = check_code(code)
    if bad:
        raise ValueError(f"запрещённые конструкции в коде: {bad}")
    # Строки перемешиваем: в frame сначала train, потом test, и по позиции их можно различить.
    order = np.random.default_rng(SEED).permutation(len(frame))
    shuffled = frame.iloc[order].reset_index(drop=True)
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(method)
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_sandbox, args=(code, shuffled, child), daemon=True)
    proc.start()
    child.close()
    try:
        if not parent.poll(CODE_TIMEOUT_S):
            raise ValueError(f"код не уложился в {CODE_TIMEOUT_S} с")
        status, payload = parent.recv()
    except EOFError:
        raise ValueError("процесс с кодом упал без ответа (память?)") from None
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(5)
    if status != "ok":
        raise ValueError(payload)
    values = np.asarray(payload).ravel()
    if len(values) != len(frame):
        raise ValueError(f"длина признака {len(values)} вместо {len(frame)}")
    restored = np.empty(len(frame), dtype=values.dtype)
    restored[order] = values
    return pd.Series(restored)


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


def parse_params(text) -> dict:
    """Параметры LightGBM от LLM: только известные ключи, значения обрезаем по границам."""
    if isinstance(text, dict):
        raw = text
    else:
        try:
            raw = json.loads(text or "")
        except (json.JSONDecodeError, TypeError):
            return {}
    out = {}
    for key, value in (raw.items() if isinstance(raw, dict) else []):
        if key not in PARAM_RANGES:
            continue
        try:
            lo, hi = PARAM_RANGES[key]
            value = min(max(float(value), lo), hi)
        except (TypeError, ValueError):
            continue
        out[key] = int(round(value)) if key in INT_PARAMS else value
    if "extra_trees" in out:
        out["extra_trees"] = bool(out["extra_trees"] >= 0.5)
    return out


def validate(action: str, cols: list[str], group_by: str | None, code: str | None, columns: set[str],
             params: dict | None = None, protected: set | None = None) -> str | None:
    """Текст ошибки или None. Проверяем имена до того, как что-то считать."""
    if action in ("drop", "as_category") and protected and set(cols) & protected:
        return f"колонки {sorted(set(cols) & protected)} нужны валидации и истории сущности, их не трогаем"
    if action == "params":
        return None if params else "нет допустимых параметров или исчерпан лимит попыток: предложи признак"
    unknown = [c for c in cols + ([group_by] if group_by else []) if c not in columns]
    if action != "code" and unknown:
        return f"нет колонок {unknown}"
    need = {"ratio": 2, "diff": 2, "product": 2, "group_mean": 1, "group_std": 1, "freq": 1}
    if action in need and len(cols) < need[action]:
        return f"для {action} нужно {need[action]} колонки"
    if action.startswith("group_") and not group_by:
        return "нет group_by"
    if action in ("drop", "as_category") and not cols:
        return "не указаны колонки"
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


def critique(llm: LLM, trial: Trial, journal: list[Trial], target: str) -> tuple[bool, str]:
    """Критик: быстрый вызов без рассуждения. Не ответил — гипотезу запускаем."""
    history = "\n".join(f"- {t.action} {t.columns} {t.group_by or ''}{t.params or ''}: {t.hypothesis[:100]}"
                        for t in journal[-15:]) or "- empty"
    answer = llm.ask_json(CRITIC_SYSTEM, f"Target column: {target}\nJournal:\n{history}\n\nNew hypothesis: "
                                         f"{trial.action} columns={trial.columns} group_by={trial.group_by} "
                                         f"params={trial.params} code={trial.code}\n{trial.hypothesis}",
                          CRITIC_SCHEMA, "review", fast=True)
    if not answer:
        return True, ""
    return bool(answer.get("run", True)), str(answer.get("reason", ""))[:300]


def trial_key(trial: Trial) -> str:
    """Ключ для поиска повторов: одно действие над теми же колонками с теми же параметрами."""
    return json.dumps([trial.action, sorted(trial.columns), trial.group_by or "", trial.params or {},
                       " ".join((trial.code or "").split())], sort_keys=True, ensure_ascii=False)


def research_loop(llm: LLM, state, spec: TaskSpec, folds, task_text: str, dictionary: pd.DataFrame | None,
                  evaluate_stage, deadline: float, journal_path: Path, plan: str = "",
                  base_columns: list[str] | None = None, protected: set | None = None) -> list[Trial]:
    """evaluate_stage(name, train, test, params) -> (принят ли, скор кандидата): оценивает вариант
    признаков и параметров и, если он выиграл по правилу принятия, делает его текущим (и пишет сабмит).

    base_columns — исходные колонки задачи. Агрегаты по многим строкам (group_*, freq, свой код)
    строятся только из них и из признаков самого цикла: среднее ent_last по группе при валидации
    по времени захватывает строки следующего периода, а их ent_last равен таргету текущей строки,
    и гипотеза выигрывает CV за счёт утечки (reg_h5: OOF 49,5 → 48,0, holdout хуже). Построчные
    действия над производными колонками безопасны и разрешены."""
    journal: list[Trial] = []
    seen: set[str] = set()
    for it in range(MAX_ITERS):
        if not llm.enabled or time.monotonic() > deadline:
            log.info("research: стоп на итерации %d (llm %s, время %s)", it, llm.enabled,
                     time.monotonic() > deadline)
            break
        features = [c for c in state.train.columns if c not in (spec.id_col, spec.target)]
        derived = set() if base_columns is None else {c for c in features if c not in base_columns
                                                      and not c.startswith("llm_")}
        before = state.result.score
        prompt = build_prompt(task_text, spec, state.train, features, dictionary, state.result,
                              folds.scheme, journal, plan, state.params)
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
        params = parse_params(answer.get("params")) if action == "params" else None
        if action == "params" and sum(t.action == "params" for t in journal) >= MAX_PARAM_TRIALS:
            params = None  # validate отметит гипотезу как invalid, LLM увидит это в журнале
        trial = Trial(it, str(answer.get("hypothesis", ""))[:500], action, cols, group_by, code, "invalid", None,
                      before, params=params, observation=str(answer.get("observation", ""))[:500])
        error = "неизвестное действие" if action not in ACTIONS else validate(action, cols, group_by, code,
                                                                                  set(features), params, protected)
        if error is None and action in CROSS_ROW_ACTIONS and derived & set(cols + ([group_by] if group_by else [])):
            error = (f"агрегаты по строкам только из исходных колонок, {sorted(derived & set(cols + [group_by]))} "
                     "производные: построчно (ratio, diff, product) их можно")
        if error is None and trial_key(trial) in seen:
            error, trial.status = "повтор уже проверенной гипотезы", "duplicate"
        seen.add(trial_key(trial))
        if error is None:
            run, reason = critique(llm, trial, journal, spec.target)
            if not run:
                error, trial.status = f"критик: {reason}", "vetoed"
        if error is None:
            try:
                accepted, score, note = try_hypothesis(trial, state, spec, features, evaluate_stage, derived)
                # score None — оценка упала: LLM должна видеть ошибку, а не «проверено и отвергнуто».
                status = "accepted" if accepted else "rejected" if score is not None else "error"
                trial.status, trial.score, trial.note = status, score, note or ("" if score is not None
                                                                                else "оценка на фолдах упала")
            except Exception as exc:  # одна кривая гипотеза не должна останавливать цикл
                trial.status, trial.note = "error", f"{type(exc).__name__}: {str(exc)[:200]}"
        else:
            trial.note = error
        journal.append(trial)
        log.info("research %d: [%s] %s %s %s%s — %s", it, trial.status, action, cols, group_by or "",
                 params or "", trial.hypothesis[:150])
        try:
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(trial), ensure_ascii=False) + "\n")
        except OSError:
            pass
    accepted = sum(t.status == "accepted" for t in journal)
    log.info("research: %d гипотез, принято %d, вызовов LLM %d, токенов %d", len(journal), accepted,
             llm.calls, llm.tokens)
    return journal


def try_hypothesis(trial: Trial, state, spec: TaskSpec, features: list[str], evaluate_stage,
                   derived: set | None = None):
    train, test = state.train, state.test
    if trial.action == "params":
        params = {**(state.params or {}), **trial.params}
        ok, score = evaluate_stage(f"research params {trial.params}", train, test, params)
        return ok, score, ""
    if trial.action == "drop":
        keep = [c for c in train.columns if c not in trial.columns]
        ok, score = evaluate_stage(f"research drop {trial.columns}", train[keep],
                                   test[[c for c in keep if c in test]], state.params)
        return ok, score, ""
    if trial.action == "as_category":
        def as_cat(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            for c in trial.columns:
                out[c] = frame[c].astype("string")
            return out
        ok, score = evaluate_stage(f"research as_category {trial.columns}", as_cat(train), as_cat(test),
                                   state.params)
        return ok, score, ""

    # Свой код LLM может агрегировать по строкам, поэтому производных колонок истории ему не даём.
    usable = [c for c in features if not (trial.action == "code" and derived and c in derived)]
    frame = pd.concat([train[usable], test[usable]], ignore_index=True)
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
    ok, score = evaluate_stage(f"research {name}", new_train, new_test, state.params)
    return ok, score, ""
