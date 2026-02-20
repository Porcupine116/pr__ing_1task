# SE 25/26 — Серия заданий 3 (LLMOps + RecSys)

## Аннотация
В работе реализован учебный проект, объединяющий практики **LLMOps (observability)** и **рекомендательные системы**. 
Цели:
1) Сравнить решения для LLM observability (Langfuse, Langtrace, OpenLLMetry).
2) Реализовать мониторинг для ИИ-агента: логи, метрики, трейсы, LLM-метрики (tokens, latency, prompt versioning).
3) Реализовать рекомендательную систему на MovieLens 100k: content-based, collaborative filtering, hybrid, эвристики и Two-Tower модель.
4) Проанализировать прод-риски и метрики качества.

---

# ЧАСТЬ 1 — LLMOps (Observability)

## 1.1 Обзор LLM Observability решений

### 1) Langfuse
**Описание:** open-source платформа для наблюдаемости LLM приложений (prompts, traces, evaluations, datasets). 
**Сильные стороны:**
- Отличная поддержка LLM-специфичных сущностей: prompt versioning, tracing, cost/tokens.
- Подходит для self-hosted (Docker).
- Удобные дашборды для анализа качества и поведения.

**Ограничения:**
- Self-host обычно требует инфраструктуру (БД), повышает сложность.

### 2) Langtrace
**Описание:** решение для трассировки/наблюдаемости LLM pipeline’ов, часто ориентировано на простую интеграцию.
**Сильные стороны:**
- Упор на трассировку и анализ цепочек.
- Может быть проще как «thin layer» поверх приложений.

**Ограничения:**
- В сравнении с Langfuse обычно менее развит prompt/versioning и продуктовая аналитика.

### 3) OpenLLMetry
**Описание:** подход/набор инструментов, интегрирующий LLM события в OpenTelemetry. 
**Сильные стороны:**
- Vendor-neutral наблюдаемость через OpenTelemetry.
- Хорошо сочетается с существующим стеком SRE (Prometheus/Grafana/Tempo/Jaeger).

**Ограничения:**
- Требует большего «инженерного» усилия: нужно собирать дашборды и хранение трейсинга.

---

## 1.2 Сравнение (feature-matrix)

| Критерий | Langfuse | Langtrace | OpenLLMetry |
|---|---|---|---|
| Простота интеграции | высокая | высокая | средняя |
| Поддержка LangChain | хорошая | хорошая | зависит от instrumentation |
| LLM-метрики (tokens/cost) | сильная | средняя | зависит от реализации |
| Self-hosted | да | зависит | да (через OTel стек) |
| Prompt versioning | да | ограниченно | через теги/атрибуты |
| Лицензия | open-source | зависит | open-source |

---

## 1.3 Выбор стека и аргументация
Выбран гибридный стек:
- **OpenTelemetry** как нейтральный слой для traces/metrics/logs;
- **Prometheus + Grafana** как стандарт SRE метрик и визуализации;
- **Langfuse** как LLM-специфичная аналитика (prompts, версии, токены).

Причины:
- OTel даёт совместимость и масштабирование (в перспективе можно менять бэкенд трейсинга).
- Prometheus/Grafana — индустриальный стандарт мониторинга.
- Langfuse закрывает продуктовые потребности LLM-приложений.

---

## 1.4 Реализация observability
Реализация находится в `se_series3/llmops/observability/`:
- `logging_config.py` — структурные логи, JSON-поля (user_id, trace_id, latency, prompt_version).
- `metrics.py` — Prometheus метрики (request count, latency, tokens).
- `tracing.py` — настройка OpenTelemetry tracer + span атрибуты.

Добавлена демонстрация middleware (FastAPI-совместимая) и вызовы из агента.

### Снимаемые сигналы
- **Logs:** user_id, trace_id, prompt_id/version, response preview.
- **Metrics:**
  - `agent_requests_total`
  - `agent_request_latency_seconds`
  - `llm_tokens_in_total`, `llm_tokens_out_total`
- **Traces:** корневой span на запрос + дочерние span’ы для LLM и RecSys.

---

# ЧАСТЬ 2 — RecSys

## 2.1 Датасет
Использован **MovieLens 100k**:
- ~100k рейтингов
- 943 пользователя
- 1682 фильма

Релевантность для top-K оценивалась как `rating >= 4`.

---

## 2.2 Реализованные подходы

### (A) Content-Based
- TF-IDF по тексту (title + genres)
- cosine similarity

### (B) Collaborative Filtering
- user-based cosine
- item-based cosine

### (C) Hybrid
- линейная комбинация скоров content + CF

### (D) Эвристики
- popular items
- same genre

### (E) Two-Tower (PyTorch)
Двухбашенная архитектура:
- embedding пользователя
- embedding айтема
- dot product
- обучение на implicit feedback (BPR-like)

Масштабирование:
- разделение башен позволяет предвычислять item embeddings,
- для ANN поиска в проде применяют FAISS/ScaNN.

---

## 2.3 Метрики
### ML метрики
- Precision@K
- Recall@K
- MAP@K
- NDCG@K
- RMSE (для rating prediction — опционально)

### Бизнес метрики
- CTR
- Conversion rate
- Retention
- ARPU
- Session time

### Инфраструктурные
- latency
- QPS
- memory usage
- GPU utilization (если применимо)

---

# ЧАСТЬ 3 — Проблемы и решения

## 3.1 RecSys: типовые проблемы
1) **Cold Start** — нет истории для новых пользователей/товаров.
   - Решение: content-based, onboarding, популярные товары.
   - Минусы: ниже персонализация.

2) **Sparsity** — разреженность матрицы взаимодействий.
   - Решение: эмбеддинги, регуляризация, hybrid.
   - Минусы: сложнее отладка.

3) **Popularity bias**
   - Решение: диверсификация, re-ranking.
   - Минусы: возможное падение CTR.

4) **Echo chamber**
   - Решение: exploration, diversity constraints.
   - Минусы: сложнее измерение.

5) **Этика**
   - Решение: privacy-by-design, контроль fairness.

6) **Feedback loop**
   - Решение: A/B тесты, causal методы, logging policy.

---

## 3.2 Продакшен риски
- Latency → кэширование, ANN, батчинг
- Scalability → горизонтальное масштабирование, precompute
- Memory usage → sparse матрицы, компрессия эмбеддингов
- Feature drift → мониторинг распределений
- Data leakage → строгие сплиты и time-aware evaluation

---

## Выводы
Построен учебный проект, демонстрирующий:
- Observability для LLM-агента (логи/метрики/трейсы + LLM-специфичные измерения),
- несколько подходов RecSys и единый протокол оценки,
- анализ рисков и метрик, приближенный к production мышлению.

