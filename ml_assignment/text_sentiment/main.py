"""NLP: анализ тональности русского текста (inference-only).

Требования (из задания):
- Использовать предобученную модель blanchefort/rubert-base-cased-sentiment
- Использовать transformers pipeline
- Никакого обучения и fine-tuning
- CPU по умолчанию

Запуск:
    python main.py
    python main.py "Ваш текст"
"""

from __future__ import annotations

import sys
import time
from typing import List

from transformers import pipeline


def build_text_from_argv(argv: List[str]) -> str:
    """Берём текст из argv, иначе используем пример."""
    if len(argv) >= 2:
        return " ".join(argv[1:]).strip()
    return "Мне понравилась лекция: материал подан ясно и интересно."  # пример входных данных


def main() -> None:
    text = build_text_from_argv(sys.argv)

    # device=-1 => CPU
    sentiment = pipeline(
        task="sentiment-analysis",
        model="blanchefort/rubert-base-cased-sentiment",
        device=-1,
    )

    t0 = time.perf_counter()
    result = sentiment(text)
    dt = time.perf_counter() - t0

    # pipeline возвращает список словарей
    item = result[0]
    label = item.get("label")
    score = float(item.get("score", 0.0))

    print("=== NLP: Sentiment Analysis (RU) ===")
    print(f"Input text: {text}")
    print(f"Prediction: {label} (score={score:.4f})")
    print(f"Inference time: {dt:.3f}s (CPU)")


if __name__ == "__main__":
    main()

