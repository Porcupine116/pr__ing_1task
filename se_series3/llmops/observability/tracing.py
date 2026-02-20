from __future__ import annotations

from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


def setup_tracing(service_name: str = "se_series3_agent") -> None:
    """Минимальная настройка OpenTelemetry tracing.

    Для учебного проекта экспортируем spans в консоль.
    В реальном проде вместо ConsoleSpanExporter настраивают OTLP exporter в collector.
    """

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)


def get_tracer():
    return trace.get_tracer("se_series3")

