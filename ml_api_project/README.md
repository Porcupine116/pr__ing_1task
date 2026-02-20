# ml_api_project (Семестр 2, СЗ_1)
См. `REPORT.md` и папки проекта.
## Структура

Агент находится в `agent/agent.py` и запускается как обычный python-скрипт (использует Ollama через HTTP).
## Агент (LangChain)

```
curl -X POST "http://127.0.0.1:8000/generate" -H "Content-Type: application/json" -d "{\"prompt\":\"Объясни, что такое API\",\"model\":\"mistral:7b\"}"
```bash
Затем:

```
ollama run mistral:7b
```bash
Запусти Ollama и скачай модель, например:
### 2) LLM generate (нужен Ollama)

```
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\":\"Мне очень понравилась лекция!\"}"
```bash
### 1) Sentiment
## Примеры запросов

- http://127.0.0.1:8000/docs
Открой Swagger UI:

```
uvicorn app.main:app --reload
```bash
## Запуск API

```
py -m pip install -r requirements.txt
.\.venv\Scripts\activate
py -m venv .venv
```bash
## Установка

- LangChain-агент с `PromptTemplate`, `LLMChain` и простой memory
- GitHub Actions workflow для автоматического запуска тестов
- PyTest тесты для `/predict` и `/generate` (для `/generate` используется мок, чтобы CI не зависел от Ollama)
- `POST /generate` — генерация текста через локальный Ollama (модели: `mistral:7b`, `phi3:mini`, `gemma:2b`)
- `POST /predict` — тональность русского текста на `blanchefort/rubert-base-cased-sentiment` через `transformers.pipeline`
## Что реализовано

- Только предобученные модели, **без fine-tuning**
- CPU-first (по умолчанию)
- Python 3.10+
## Требования

Проект для университетского задания: FastAPI для одной ML-модели + API для локальной LLM (Ollama) + тесты (PyTest) + CI (GitHub Actions) + простой LLM-агент (LangChain) + сравнение нескольких LLM.


