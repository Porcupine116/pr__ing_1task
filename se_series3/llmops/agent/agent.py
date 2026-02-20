"""Демонстрационный ИИ-агент + observability.

Цель:
- показать логи + метрики + трейсы
- логировать user_id, trace_id, prompt versioning
- собирать latency и (упрощённо) tokens

Важно:
- Чтобы проект был воспроизводим без тяжёлой LLM, по умолчанию используется STUB-режим.
  При желании можно подключить реальную LLM через Ollama и добавить подсчёт токенов.

Запуск:
    py se_series3/llmops/agent/agent.py

Метрики Prometheus:
- http://127.0.0.1:8001/metrics
"""

from __future__ import annotations

# Позволяет запускать файл напрямую: `py se_series3/llmops/agent/agent.py`
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
import uuid
from dataclasses import dataclass
from typing import Optional

from prometheus_client import make_asgi_app
import uvicorn

from se_series3.llmops.observability.logging_config import get_logger, setup_logging
from se_series3.llmops.observability.metrics import (
    AGENT_REQUESTS_TOTAL,
    LLM_TOKENS_IN,
    LLM_TOKENS_OUT,
    observe_latency,
)
from se_series3.llmops.observability.tracing import get_tracer, setup_tracing


PROMPT_ID = "review_agent"
PROMPT_VERSION = "v1"
MODEL_NAME = "stub-llm"


@dataclass
class AgentInput:
    user_id: str
    text: str


def approx_token_count(s: str) -> int:
    # Упрощённая оценка токенов: длина/4 (приближение для EN; для RU грубо, но демонстрационно)
    return max(1, len(s) // 4)


def stub_llm(prompt: str) -> str:
    # Заглушка: имитируем задержку
    time.sleep(0.1)
    return (
        "1) Тональность: смешанная (скорее негатив)\n"
        "2) Топ-3 темы: стабильность приложения, скорость поддержки, баги\n"
        "3) Рекомендации: ускорить поддержку; приоритезировать crash-баги; добавить мониторинг ошибок"
    )


def run_agent(inp: AgentInput) -> str:
    logger = get_logger()
    tracer = get_tracer()

    trace_id = uuid.uuid4().hex

    with tracer.start_as_current_span("agent.request") as span:
        span.set_attribute("user.id", inp.user_id)
        span.set_attribute("trace.id", trace_id)
        span.set_attribute("prompt.id", PROMPT_ID)
        span.set_attribute("prompt.version", PROMPT_VERSION)
        span.set_attribute("llm.model", MODEL_NAME)

        prompt = (
            "Ты — аналитик отзывов. Сформируй структурированный ответ.\n"
            f"Отзыв: {inp.text}"
        )

        in_tokens = approx_token_count(prompt)
        LLM_TOKENS_IN.labels(model=MODEL_NAME).inc(in_tokens)

        t0 = time.perf_counter()
        with observe_latency(route="agent"):
            response = stub_llm(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000

        out_tokens = approx_token_count(response)
        LLM_TOKENS_OUT.labels(model=MODEL_NAME).inc(out_tokens)

        AGENT_REQUESTS_TOTAL.labels(route="agent", status="ok").inc()

        logger.info(
            "agent_completed",
            user_id=inp.user_id,
            trace_id=trace_id,
            prompt_id=PROMPT_ID,
            prompt_version=PROMPT_VERSION,
            latency_ms=round(latency_ms, 2),
            tokens_in=in_tokens,
            tokens_out=out_tokens,
        )

        return response


def main() -> None:
    setup_logging()
    setup_tracing()

    # Поднимаем отдельный сервер только для метрик (простая демонстрация)
    metrics_app = make_asgi_app()

    # Запускаем метрики в фоне
    config = uvicorn.Config(metrics_app, host="127.0.0.1", port=8001, log_level="warning")
    server = uvicorn.Server(config)

    # Старт сервера в другом потоке (простое решение для демо)
    import threading

    threading.Thread(target=server.run, daemon=True).start()

    # Пример запроса
    inp = AgentInput(
        user_id="student_001",
        text="В целом неплохо, но приложение часто вылетает и поддержка отвечает долго.",
    )

    print(run_agent(inp))
    print("\nMetrics: http://127.0.0.1:8001/metrics")

    # Не завершаем сразу, чтобы можно было открыть /metrics
    time.sleep(2)


if __name__ == "__main__":
    main()

