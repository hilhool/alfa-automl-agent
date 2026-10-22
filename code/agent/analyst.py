"""Агент-аналитик: читает задачу и профиль данных до моделирования и пишет план.

LLM получает task.md, описания колонок из data_dictionary и сводку по данным
(типы, пропуски, примеры значений, сдвиг train/test, распределение таргета).
Возвращает понимание задачи, роли колонок, подозрения на утечку и идеи.
Каждое утверждение о роли колонки проверяется правилом: «это id» — почти все
значения уникальны, «это код категории» — целые числа с небольшим числом значений.
Что подтвердилось, агент пробует на фолдах; идеи идут в цикл гипотез.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agent.llm import LLM
from agent.spec import TaskSpec

log = logging.getLogger("solution")

MAX_PROFILE_COLUMNS = 80
MAX_TASK_CHARS = 4000
ID_UNIQUE_SHARE = 0.95
MAX_CODE_LEVELS = 200
ROLES = ("id", "time", "entity", "category_code", "numeric", "category", "text", "leak", "noise")

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "column_roles": {"type": "array", "items": {
            "type": "object",
            "properties": {"column": {"type": "string"}, "role": {"type": "string", "enum": list(ROLES)},
                           "reason": {"type": "string"}},
            "required": ["column", "role", "reason"]}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "ideas": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "column_roles", "risks", "ideas"],
}

SYSTEM = """You are the lead data scientist of an automated ML agent. Before any modelling you read the task
and a profile of the data and write a short research plan. You never see test labels. Be concrete: name columns.
Answer with JSON only."""

PROMPT_TAIL = """Write the plan:
- summary: 2-4 sentences: what is predicted, what drives the target, what makes the task hard
  (shift between train and test, imbalance, repeated entities, noisy columns).
- column_roles: only columns whose role is NOT obvious from dtype. Roles: id (row identifier, useless or leaky),
  time (period or date), entity (identifies a repeated object), category_code (a number that is really a category
  code), text (free text), leak (derived from the target or unavailable at prediction time), noise (random values
  with no relation to the target, e.g. sequence numbers or hashes).
- risks: what can make cross-validation lie or the model fail on test.
- ideas: up to 8 concrete hypotheses, most promising first, each naming exact columns. The agent's toolbox:
  new features computed with pandas from existing columns (ratios, differences, group statistics, frequencies,
  custom code), dropping columns, treating numeric codes as categories, LightGBM parameters. CatBoost, XGBoost,
  log-transform of the target, ensembling and class-threshold tuning run automatically later, so do not propose
  them. Other model families (linear, KNN, SVM, neural nets) are not available."""


@dataclass
class Plan:
    summary: str = ""
    roles: dict = field(default_factory=dict)       # колонка -> (роль, причина)
    risks: list = field(default_factory=list)
    ideas: list = field(default_factory=list)
    confirmed: dict = field(default_factory=dict)   # колонка -> роль, подтверждённая правилом
    rejected: dict = field(default_factory=dict)    # колонка -> почему правило не подтвердило

    def as_prompt(self) -> str:
        if not self.summary:
            return ""
        lines = [f"Summary: {self.summary}"]
        if self.confirmed:
            lines.append("Confirmed column roles: " + ", ".join(f"{c}={r}" for c, r in self.confirmed.items()))
        lines += [f"Risk: {r}" for r in self.risks[:5]]
        lines += [f"Idea: {i}" for i in self.ideas[:8]]
        return "\n".join(lines)


def profile(train: pd.DataFrame, test: pd.DataFrame, features: list[str], spec: TaskSpec,
            dictionary: pd.DataFrame | None, scheme: str, shifted: list | None) -> str:
    notes = {}
    if dictionary is not None and "feature" in dictionary.columns:
        text_col = next((c for c in dictionary.columns if "desc" in c.lower()), None)
        if text_col:
            notes = dict(zip(dictionary["feature"].astype(str), dictionary[text_col].astype(str)))
    y = train[spec.target]
    if spec.kind == "regression":
        q = y.quantile([0, 0.05, 0.5, 0.95, 1]).round(4).tolist()
        target = f"mean {y.mean():.4g}, std {y.std():.4g}, quantiles 0/5/50/95/100%: {q}, skew {y.skew():.2f}"
    else:
        target = "class shares: " + ", ".join(f"{k}={v:.3f}" for k, v in y.value_counts(normalize=True).items())
    rows = []
    for col in features[:MAX_PROFILE_COLUMNS]:
        s, t = train[col], test[col] if col in test else None
        if pd.api.types.is_numeric_dtype(s):
            values = f"range [{s.min():.4g}, {s.max():.4g}]"
        else:
            values = "e.g. " + ", ".join(map(str, s.dropna().astype(str).value_counts().index[:4]))[:80]
        unseen = ""
        if t is not None and not pd.api.types.is_numeric_dtype(s):
            unseen = f" | unseen in train {(~t.dropna().astype(str).isin(set(s.dropna().astype(str)))).mean():.0%}"
        rows.append(f"{col} | {s.dtype} | uniq {s.nunique()} | miss train {s.isna().mean():.0%} test "
                    f"{t.isna().mean() if t is not None else float('nan'):.0%} | {values}{unseen} | "
                    f"{notes.get(col, '')[:100]}")
    more = len(features) - MAX_PROFILE_COLUMNS
    shift = ", ".join(f"{c} ({v:.0%})" for c, v in (shifted or [])) or "unknown"
    return (f"train {len(train)} rows, test {len(test)} rows, {len(features)} feature columns.\n"
            f"Target `{spec.target}` ({spec.kind}, metric {spec.metric}): {target}\n"
            f"Validation chosen by rules: {scheme}\n"
            f"Columns that best separate test from train (adversarial gain share): {shift}\n\n"
            "Columns (name | dtype | unique | missing | values | description):\n" + "\n".join(rows)
            + (f"\n... and {more} more columns" if more > 0 else ""))


def analyze(llm: LLM, task_text: str, profile_text: str) -> Plan:
    answer = llm.ask_json(SYSTEM, f"Task description (task.md):\n{task_text[:MAX_TASK_CHARS]}\n\n"
                                  f"Data profile:\n{profile_text}\n\n{PROMPT_TAIL}", SCHEMA, "plan")
    if not answer:
        return Plan()
    roles = {}
    for item in answer.get("column_roles") or []:
        if isinstance(item, dict) and item.get("role") in ROLES and item.get("column"):
            roles[str(item["column"])] = (str(item["role"]), str(item.get("reason", ""))[:200])
    return Plan(str(answer.get("summary", ""))[:1500], roles,
                [str(r)[:300] for r in answer.get("risks") or []][:8],
                [str(i)[:300] for i in answer.get("ideas") or []][:8])


def verify(plan: Plan, train: pd.DataFrame, test: pd.DataFrame, features: list[str],
           period_col: str | None, entity_key: list | None) -> Plan:
    """Каждое утверждение LLM о роли колонки проверяем правилом по данным."""
    for col, (role, _) in plan.roles.items():
        if col not in features:
            plan.rejected[col] = "такой колонки нет среди признаков"
            continue
        s = train[col]
        if role == "id":
            share = s.nunique() / max(len(s), 1)
            ok = share >= ID_UNIQUE_SHARE
            why = f"уникальных {share:.0%}"
        elif role == "category_code":
            values = pd.to_numeric(s, errors="coerce").dropna()
            ok = (pd.api.types.is_numeric_dtype(s) and len(values) > 0
                  and bool(np.all(values == np.round(values))) and s.nunique() <= MAX_CODE_LEVELS)
            why = f"целые {pd.api.types.is_numeric_dtype(s)}, значений {s.nunique()}"
        elif role == "time":
            ok = col == period_col
            why = f"правила выбрали период {period_col}"
        elif role == "entity":
            ok = bool(entity_key) and col in entity_key
            why = f"ключ сущности по правилам {entity_key}"
        elif role in ("leak", "noise"):
            # Проверка — только на фолдах: агент пробует убрать колонку и смотрит на скор.
            ok, why = True, "проверим удалением на фолдах"
        else:
            continue
        if ok:
            plan.confirmed[col] = role
        else:
            plan.rejected[col] = f"{role}: {why}"
    log.info("analyst: подтверждено %s, не подтверждено %s", plan.confirmed, plan.rejected)
    return plan
