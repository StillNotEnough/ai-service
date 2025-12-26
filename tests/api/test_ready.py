from fastapi.testclient import TestClient

from ai_service.main import app

client = TestClient(app)


def test_ready_ok_when_key_present(monkeypatch):
    # ключ есть → сервис готов
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    r = client.get("/ready")

    assert r.status_code == 200
    assert r.json() == {"status": "ready"}


def test_ready_503_when_key_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    r = client.get("/ready")

    assert r.status_code == 503
    assert r.json()["detail"] == "OPENROUTER_API_KEY missing"