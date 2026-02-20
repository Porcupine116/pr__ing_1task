"""Клиент к локальному Ollama и функции генерации.

Ollama по умолчанию слушает http://127.0.0.1:11434
Endpoint генерации: POST /api/generate
Документация: https://github.com/ollama/ollama/blob/main/docs/api.md

Важно: CI не должен зависеть от наличия Ollama — в тестах этот вызов мокается.
"""

from __future__ import annotations

import json
from typing import Optional

import requests


OLLAMA_URL = "http://127.0.0.1:11434"


class OllamaError(RuntimeError):
    pass


def ollama_generate(prompt: str, model: str, timeout_s: float = 60.0) -> str:
    url = f"{OLLAMA_URL}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
    except requests.RequestException as e:
        raise OllamaError(
            "Не удалось подключиться к Ollama. Убедитесь, что Ollama запущена и доступна на 127.0.0.1:11434"
        ) from e

    if r.status_code != 200:
        raise OllamaError(f"Ollama вернула статус {r.status_code}: {r.text}")

    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise OllamaError("Некорректный JSON от Ollama") from e

    response = data.get("response")
    if not isinstance(response, str):
        raise OllamaError("Ollama: поле 'response' отсутствует или имеет неверный тип")

    return response

