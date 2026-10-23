"""Отчёт агента: что он понял о задаче, что пробовал, что принял и почему.

Пишется в agent_report.md рядом с сабмитом после каждого крупного этапа, чтобы
при kill по таймауту на диске остался отчёт о том, что успели. Для ML-инженера
это второй результат работы после submission.csv: по нему видно, на чём держится
итоговая модель и что проверять дальше.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("solution")


@dataclass
class Step:
    name: str
    status: str
    score: float | None
    before: float | None
    seconds: float


@dataclass
class Report:
    path: Path
    started: float
    facts: dict = field(default_factory=dict)
    plan: object = None
    steps: list = field(default_factory=list)
    trials: list = field(default_factory=list)
    models: dict = field(default_factory=dict)
    final: str = ""

    def step(self, name: str, status: str, score: float | None, before: float | None) -> None:
        self.steps.append(Step(name, status, score, before, time.monotonic() - self.started))

    def write(self) -> None:
        try:
            self.path.write_text(self.render(), encoding="utf-8")
        except Exception:
            log.exception("Отчёт не записался")

    def render(self) -> str:
        out = ["# Отчёт агента", ""]
        out += [f"- {k}: {v}" for k, v in self.facts.items()]
        plan = self.plan
        if plan is not None and getattr(plan, "summary", ""):
            out += ["", "## Как агент понял задачу", "", plan.summary, ""]
            if plan.roles:
                out += ["| Колонка | Роль по LLM | Почему | Проверка правилом |", "|---|---|---|---|"]
                for col, (role, reason) in plan.roles.items():
                    check = ("подтверждено" if col in plan.confirmed else plan.rejected.get(col, "не проверялось"))
                    out.append(f"| {col} | {role} | {_cell(reason)} | {_cell(check)} |")
                out.append("")
            out += [f"- Риск: {r}" for r in plan.risks]
            out += [f"- Идея: {i}" for i in plan.ideas]
        if self.steps:
            out += ["", "## Шаги", "", "| Время | Шаг | Итог | Скор | Было |", "|---|---|---|---|---|"]
            for s in self.steps:
                out.append(f"| {s.seconds / 60:.1f} мин | {_cell(s.name)} | {s.status} | {_num(s.score)} | "
                           f"{_num(s.before)} |")
        if self.trials:
            accepted = sum(t.status == "accepted" for t in self.trials)
            out += ["", f"## Гипотезы LLM ({len(self.trials)}, принято {accepted})", "",
                    "| # | Наблюдение | Действие | Гипотеза | Итог | Скор | Было | Примечание |",
                    "|---|---|---|---|---|---|---|---|"]
            for t in self.trials:
                what = f"{t.action} {t.columns}" + (f" по {t.group_by}" if t.group_by else "") + (
                    f" {t.params}" if t.params else "")
                out.append(f"| {t.iteration} | {_cell(getattr(t, 'observation', '')[:300])} | {_cell(what)} | "
                           f"{_cell(t.hypothesis[:300])} | {t.status} | "
                           f"{_num(t.score)} | {_num(t.before)} | {_cell(t.note[:150])} |")
        if self.models:
            out += ["", "## Кандидаты зоопарка (OOF)", "", "| Модель | Скор | По фолдам |", "|---|---|---|"]
            for name, r in self.models.items():
                out.append(f"| {name} | {_num(r.score)} | {', '.join(_num(x) for x in r.fold_scores)} |")
        if self.final:
            out += ["", "## Итог", "", self.final]
        return "\n".join(out) + "\n"


def _num(x) -> str:
    return "—" if x is None else f"{x:.5g}"


def _cell(text) -> str:
    return str(text).replace("|", "/").replace("\n", " ")
