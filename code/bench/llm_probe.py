"""Проверка API организаторов: что сервер умеет и как отвечает.

  python bench/llm_probe.py        (ключи из окружения или code/.env)

Печатает сырые ответы. Каждая проверка независима: упавшая не мешает остальным.
Параметры enable_thinking и guided_json — расширения vLLM; есть ли они у
организаторов, как раз и выясняем.
"""

import json
import time

from openai import OpenAI

from run import load_env

SCHEMA = {"type": "object",
          "properties": {"task_type": {"type": "string", "enum": ["regression", "binary", "multiclass"]},
                         "target": {"type": "string"}},
          "required": ["task_type", "target"]}
QUESTION = [{"role": "user", "content": "Таргет site_eui, RMSE. Верни JSON с task_type и target."}]
TOOL = [{"type": "function", "function": {
    "name": "run_python", "description": "Выполнить Python",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]

CHECKS = {
    "plain": {},
    "no_thinking": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    "json_object": {"response_format": {"type": "json_object"},
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    "json_schema": {"response_format": {"type": "json_schema",
                                        "json_schema": {"name": "answer", "schema": SCHEMA}},
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    "guided_json": {"extra_body": {"guided_json": SCHEMA, "chat_template_kwargs": {"enable_thinking": False}}},
    "effort_low": {"extra_body": {"reasoning_effort": "low"}},
    "tools": {"tools": TOOL, "messages": [{"role": "user", "content": "Посчитай 2+2 через run_python."}]},
}


def main() -> None:
    env = load_env()
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL"), timeout=120)
    model = env["OPENAI_MODEL"]
    try:
        print("models:", [m.id for m in client.models.list().data])
    except Exception as error:  # noqa: BLE001
        print("models: ошибка", error)
    for name, kwargs in CHECKS.items():
        kwargs = dict(kwargs)
        messages = kwargs.pop("messages", QUESTION)
        started = time.monotonic()
        try:
            response = client.chat.completions.create(model=model, messages=messages, max_tokens=800, **kwargs)
            raw = response.model_dump(exclude_none=True)
            print(f"\n=== {name}: {time.monotonic() - started:.1f} с, usage={raw.get('usage')}")
            print(json.dumps(raw["choices"][0], ensure_ascii=False, indent=1)[:3000])
        except Exception as error:  # noqa: BLE001
            print(f"\n=== {name}: ошибка {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
