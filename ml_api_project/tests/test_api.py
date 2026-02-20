from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_predict_ok(monkeypatch) -> None:
    # Мокаем инференс, чтобы тест был быстрым и не зависел от torch/скачивания модели.
    from app import model

    def fake_predict(text: str):
        assert text
        return {"label": "POSITIVE", "score": 0.99}

    monkeypatch.setattr(model, "predict_sentiment", fake_predict)

    r = client.post("/predict", json={"text": "Мне очень понравилось!"})
    assert r.status_code == 200
    data = r.json()
    assert data == {"label": "POSITIVE", "score": 0.99}


def test_predict_empty_text() -> None:
    r = client.post("/predict", json={"text": "   "})
    # Pydantic пропустит строку, но наш код должен отловить
    assert r.status_code == 400

