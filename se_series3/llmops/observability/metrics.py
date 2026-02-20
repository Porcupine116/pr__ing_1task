from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Optional

from prometheus_client import Counter, Histogram


AGENT_REQUESTS_TOTAL = Counter(
    "agent_requests_total",
    "Total number of agent requests",
    labelnames=("route", "status"),
)

AGENT_REQUEST_LATENCY = Histogram(
    "agent_request_latency_seconds",
    "Agent request latency (seconds)",
    labelnames=("route",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

LLM_TOKENS_IN = Counter(
    "llm_tokens_in_total",
    "Total input tokens to LLM",
    labelnames=("model",),
)

LLM_TOKENS_OUT = Counter(
    "llm_tokens_out_total",
    "Total output tokens from LLM",
    labelnames=("model",),
)


@contextmanager
def observe_latency(route: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        AGENT_REQUEST_LATENCY.labels(route=route).observe(dt)

