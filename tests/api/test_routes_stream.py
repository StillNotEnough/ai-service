from fastapi.testclient import TestClient

from ai_service.main import app

client = TestClient(app)


class _FakeStreamResponse:
    status_code = 200
    closed = False

    async def aiter_lines(self):
        if self.closed:
            raise RuntimeError("stream has been closed")
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
        yield "data: [DONE]"


class _FakeStreamCtx:
    async def __aenter__(self):
        self.resp = _FakeStreamResponse()
        return self.resp

    async def __aexit__(self, exc_type, exc, tb):
        self.resp.closed = True
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


class _FakeErrorResponse:
    status_code = 403

    async def aread(self):
        return b'{"error": "Access denied by security policy."}'


class _FakeErrorCtx:
    async def __aenter__(self):
        return _FakeErrorResponse()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeErrorClient(_FakeAsyncClient):
    def stream(self, *args, **kwargs):
        return _FakeErrorCtx()


def test_chat_stream_returns_sse(monkeypatch):
    monkeypatch.setattr("ai_service.api.routes.httpx.AsyncClient", _FakeAsyncClient)

    r = client.post(
        "/api/chat/stream",
        json={"message": "hi", "subject": "general", "conversationHistory": []},
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    assert '"content": "hi"' in r.text


def test_chat_stream_returns_upstream_error_body(monkeypatch):
    monkeypatch.setattr("ai_service.api.routes.httpx.AsyncClient", _FakeErrorClient)

    r = client.post(
        "/api/chat/stream",
        json={"message": "hi", "subject": "general", "conversationHistory": []},
    )
    assert r.status_code == 200
    assert "OpenRouter 403" in r.text
    assert "Access denied by security policy" in r.text