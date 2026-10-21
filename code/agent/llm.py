"""Клиент LLM организаторов: JSON-ответы по схеме, таймауты, ретраи.

Ключ и адрес подставляет платформа (OPENAI_API_KEY, OPENAI_BASE_URL), OpenAI()
читает их сам. Нет ключа или API лёг — клиент выключается, и агент работает
без LLM: каждый вызов возвращает None, вызывающий код обязан это переживать.

Рассуждением управляет параметр прокси `reasoning` (как у OpenRouter). Проба 21.10 на промпте
гипотезы в 9,6 тыс. символов: по умолчанию модель тратит все 6000 токенов на рассуждение
и возвращает пустой content; effort=low — около 2600 токенов рассуждения и 30–40 с;
enabled=false — 165 токенов и 5 с. enable_thinking=false и reasoning_effort не работают.
Первая попытка с коротким рассуждением, повтор без него.
"""

import json
import logging
import os
import re
import time

log = logging.getLogger("solution")

DEFAULT_MODEL = "qwen/qwen3.8-27b"
TIMEOUT_S = 120
MAX_TOKENS = 8000
REASONING_TRIES = ({"effort": "low"}, {"enabled": False})
RETRIES = 2
MAX_FAILURES = 4          # подряд упавших вызовов — выключаемся до конца прогона
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class LLM:
    def __init__(self) -> None:
        self.client = None
        self.model = os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        self.calls = self.failures = self.tokens = 0
        if not os.getenv("OPENAI_API_KEY"):
            log.info("llm: ключа нет, работаем без LLM")
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(timeout=TIMEOUT_S, max_retries=0)
        except Exception:
            log.exception("llm: клиент не создался, работаем без LLM")

    @property
    def enabled(self) -> bool:
        return self.client is not None and self.failures < MAX_FAILURES

    def ask_json(self, system: str, user: str, schema: dict, name: str = "answer") -> dict | None:
        """Ответ модели как dict по схеме или None. Схему передаём и в response_format,
        и проверяем сами: прокси может её проигнорировать."""
        if not self.enabled:
            return None
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt, reasoning in enumerate(REASONING_TRIES[:RETRIES]):
            started = time.monotonic()
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=MAX_TOKENS, temperature=0.7,
                    response_format={"type": "json_schema", "json_schema": {"name": name, "schema": schema}},
                    extra_body={"reasoning": reasoning})
                self.calls += 1
                if response.usage:
                    self.tokens += response.usage.total_tokens or 0
                answer = parse_json(response.choices[0].message.content or "")
                missing = [k for k in schema.get("required", []) if answer is None or k not in answer]
                if not missing:
                    self.failures = 0
                    log.info("llm: ответ за %.1f с, всего вызовов %d, токенов %d",
                             time.monotonic() - started, self.calls, self.tokens)
                    return answer
                log.info("llm: в ответе нет полей %s, попытка %d", missing, attempt + 1)
            except Exception as error:
                log.info("llm: ошибка %s: %s", type(error).__name__, str(error)[:200])
        self.failures += 1
        return None


def parse_json(text: str) -> dict | None:
    text = THINK_BLOCK.sub("", text).strip()
    fenced = JSON_FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
