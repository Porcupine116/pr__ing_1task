from __future__ import annotations

import logging
import sys
from typing import Any, Dict

import structlog


def setup_logging() -> None:
    """Настройка структурных логов.

    Поля, которые мы хотим видеть в каждом событии:
    - trace_id
    - user_id
    - prompt_version
    - latency_ms
    """

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


def get_logger() -> structlog.BoundLogger:
    return structlog.get_logger("se_series3")

