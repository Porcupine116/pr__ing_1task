# SE 25/26 — Серия заданий 3 (LLMOps + RecSys)

См. `se_series3/REPORT.md` — оформлено как исследовательская работа: обзор LLM observability решений + архитектура мониторинга + реализация RecSys + проблемы и метрики.
## Отчёт

---

Подробности — в `se_series3/llmops/README.md`.

- Grafana: http://localhost:3000 (логин/пароль по умолчанию: admin/admin)
- Prometheus: http://localhost:9090
Откроется:

```
docker compose up -d
cd se_series3/llmops
```bash

Нужно: Docker Desktop.
## Запуск полного observability-стека (Docker Compose)

---

По умолчанию агент работает в режиме `stub` (без реальной LLM), но всё логируется и метрики доступны.
```
py se_series3/llmops/agent/agent.py
```bash
### LLMOps демо (агент + метрики)

Скрипт сам скачает MovieLens 100k при первом запуске и посчитает метрики для нескольких стратегий.
```
py se_series3/recsys/evaluation.py
```bash
### RecSys демо (оффлайн)

```
py -m pip install -r se_series3/requirements.txt
.\.venv\Scripts\activate
py -m venv .venv
```bash
## Быстрый старт (без Docker)

---

```
  README.md
  requirements.txt
  REPORT.md
    exploration.ipynb
  notebooks/
    data_utils.py
    evaluation.py
    two_tower.py
    heuristics.py
    hybrid.py
    collaborative.py
    content_based.py
  recsys/
    README.md
    prometheus.yml
    docker-compose.yml
      tracing.py
      metrics.py
      logging_config.py
    observability/
      agent.py
    agent/
  llmops/
se_series3/
```
## Структура

---

Режим: **локальный запуск**, CPU-first, без fine-tuning LLM.

2) **RecSys** — рекомендательная система для e-commerce/контента на датасете MovieLens 100k.
1) **LLMOps (Observability)** — обзор решений и практическая интеграция наблюдаемости для ИИ-агента.
Проект содержит две большие части:

