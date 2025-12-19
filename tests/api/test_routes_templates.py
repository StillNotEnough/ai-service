from fastapi.testclient import TestClient

from ai_service.main import app

client = TestClient(app)


def test_get_templates_ok():
    r = client.get("/api/templates")
    assert r.status_code == 200
    data = r.json()
    assert "templates" in data


def test_get_template_unknown_404():
    r = client.get("/api/templates/definitely-not-a-real-template-id")
    assert r.status_code == 404