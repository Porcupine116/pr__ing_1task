# LLMOps: Observability (OpenTelemetry + Prometheus + Grafana + Langfuse)

Эта часть демонстрирует, как организовать наблюдаемость для ИИ-агента:
- структурные логи,
- метрики Prometheus,
- трейсы OpenTelemetry,
- LLM-специфичные поля: tokens, latency, prompt versioning.

## Структура
```
llmops/
  agent/agent.py
  observability/
    logging_config.py
    metrics.py
    tracing.py
  docker-compose.yml
  prometheus.yml
```

## Быстрый запуск (без Docker)
```bash
py se_series3/llmops/agent/agent.py
```
Скрипт поднимает минимальный HTTP сервер метрик на `http://127.0.0.1:8001/metrics`.

## Полный стек через Docker Compose
```bash
cd se_series3/llmops
docker compose up -d
```

Открой:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Что смотреть в Grafana
- RPS / request count
- p95 latency
- error rate
- tokens_in / tokens_out (если LLM включена)

## Langfuse (опционально)
В учебном проекте Langfuse упоминается как LLM-specific слой. Для простоты Langfuse не является обязательной зависимостью запуска.
В отчёте приведены аргументы выбора и место Langfuse в архитектуре.

