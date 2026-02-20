from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_generate_ok_mock(monkeypatch) -> None:
    # Мокаем ollama_generate, чтобы тесты проходили без установленного Ollama
    from app import llm_api

    def fake_generate(prompt: str, model: str, timeout_s: float = 60.0) -> str:
        assert prompt
        assert model
        return "FAKE_RESPONSE"

    monkeypatch.setattr(llm_api, "ollama_generate", fake_generate)

    r = client.post("/generate", json={"prompt": "Объясни, что такое API", "model": "mistral:7b"})
    assert r.status_code == 200
    data = r.json()
    assert data["response"] == "FAKE_RESPONSE"


def test_generate_empty_prompt() -> None:
    r = client.post("/generate", json={"prompt": "  "})
    assert r.status_code in (400, 422)

