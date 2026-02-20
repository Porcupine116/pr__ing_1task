"""FastAPI приложение.

Endpoints:
- GET /health
- POST /predict  (sentiment)
- POST /generate (ollama)

Запуск:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException

from app import llm_api, model
from app.schemas import GenerateRequest, GenerateResponse, PredictRequest, PredictResponse


app = FastAPI(title="ML API Project (Semester 2)")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    # Доп. проверка (Pydantic уже проверяет min_length=1)
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Поле 'text' не должно быть пустым")

    t0 = time.perf_counter()
    try:
        out = model.predict_sentiment(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса модели: {e}")
    dt = time.perf_counter() - t0

    # можно логировать dt; в ответ не включаем, чтобы соответствовать ТЗ
    _ = dt

    return PredictResponse(label=out["label"], score=float(out["score"]))


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Поле 'prompt' не должно быть пустым")

    try:
        text = llm_api.ollama_generate(prompt=prompt, model=req.model)
    except llm_api.OllamaError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Неизвестная ошибка генерации: {e}")

    return GenerateResponse(response=text)
