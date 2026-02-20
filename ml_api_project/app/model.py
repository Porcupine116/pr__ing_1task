"""ML модель (sentiment) как сервисный слой.

Здесь мы прячем работу с transformers.pipeline за простой функцией.
Важно: никакого обучения — только инференс предобученной модели.

Техническое замечание:
- Импорт `torch/transformers` выполняем ЛЕНИВО (внутри функции),
  чтобы:
  1) тесты могли мокать инференс,
  2) в окружениях, где torch не установлен/не загружается, модуль всё равно импортировался.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict


MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"


@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    # Ленивый импорт (см. docstring)
    from transformers import pipeline

    # device=-1 => CPU
    return pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        device=-1,
    )


def predict_sentiment(text: str) -> Dict[str, float | str]:
    nlp = get_sentiment_pipeline()
    result = nlp(text)
    item = result[0]
    return {
        "label": str(item.get("label")),
        "score": float(item.get("score", 0.0)),
    }
