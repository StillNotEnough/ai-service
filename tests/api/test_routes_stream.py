from fastapi.testclient import TestClient

from ai_service.main import app

client = TestClient(app)


class _FakeStreamResponse:
    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        # один кусок контента и завершение
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
        yield "data: [DONE]"


class _FakeStreamCtx:
    async def __aenter__(self):
        return _FakeStreamResponse()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, *args, **kwargs):
        return _FakeStreamCtx()


def test_chat_stream_returns_sse(monkeypatch):
    monkeypatch.setattr("ai_service.api.routes.httpx.AsyncClient", _FakeAsyncClient)

    r = client.post(
        "/api/chat/stream",
        json={"message": "hi", "subject": "general", "conversationHistory": []},
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")