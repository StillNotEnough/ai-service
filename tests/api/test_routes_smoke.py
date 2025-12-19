from fastapi.testclient import TestClient

from ai_service.main import app

client = TestClient(app)


def test_root_ok():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["message"] == "Study Helper AI Service"
    assert "version" in data
    assert "docs" in data


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "features" in data