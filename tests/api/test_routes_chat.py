from fastapi.testclient import TestClient

from ai_service.main import app

client = TestClient(app)


def test_chat_send_ok(monkeypatch):
    async def fake_call_openrouter(messages, model):
        return "fake response"

    monkeypatch.setattr("ai_service.api.routes.call_openrouter", fake_call_openrouter)

    r = client.post(
        "/api/chat/send",
        json={"message": "hi", "subject": "general", "conversationHistory": []},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["message"] == "fake response"
    assert "timestamp" in data


def test_chat_template_unknown_template_id_404(monkeypatch):
    async def fake_call_openrouter(messages, model):
        return "should not be called"

    monkeypatch.setattr("ai_service.api.routes.call_openrouter", fake_call_openrouter)

    r = client.post(
        "/api/chat/template",
        json={
            "template_id": "no_such_template",
            "user_input": "x",
            "subject": "general",
            "conversationHistory": [],
        },
    )
    assert r.status_code == 404