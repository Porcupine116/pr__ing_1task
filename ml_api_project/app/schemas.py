from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., description="Русский текст для анализа тональности", min_length=1)


class PredictResponse(BaseModel):
    label: str
    score: float


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Промпт для LLM", min_length=1)
    model: str = Field("mistral:7b", description="Имя модели в Ollama")


class GenerateResponse(BaseModel):
    response: str

