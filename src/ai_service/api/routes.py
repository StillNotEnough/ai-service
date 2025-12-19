import json
from datetime import datetime
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ai_service.config.settings import OPENROUTER_API_KEY, OPENROUTER_URL
from ai_service.models.schemas import ChatRequest, ChatResponse, TemplateRequest
from ai_service.services.ai_client import call_openrouter
from ai_service.services.message_builder import prepare_messages
from ai_service.services.template_service import template_service

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint - приветствие API"""
    return {
        "message": "Study Helper AI Service",
        "version": "1.1.0",
        "docs": "/docs",
    }


@router.get("/health")
async def health():
    """Health check для Docker и мониторинга"""
    return {
        "status": "ok",
        "service": "Study Helper AI Service",
        "version": "1.1.0",
        "features": ["chat", "templates", "streaming"],
    }


@router.get("/api/templates")
async def get_templates(subject: Optional[str] = None):
    """
    Получить все шаблоны или шаблоны для конкретного предмета

    Query params:
    - subject (optional): math, programming, english, general
    """
    try:
        if subject:
            templates = template_service.get_templates_by_subject(subject)
            categories = template_service.templates_cache.get("categories", [])
            return {"templates": templates, "categories": categories}
        return template_service.get_all_templates()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/templates/{template_id}")
async def get_template(template_id: str):
    """Получить конкретный шаблон по ID"""
    try:
        template = template_service.get_template_by_id(template_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
        return template
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/chat/template")
async def chat_with_template(request: TemplateRequest):
    """
    Отправить сообщение с использованием шаблона

    Body:
    {
      "template_id": "explain_topic",
      "user_input": "What is Newton's Third Law?",
      "subject": "physics",
      "conversationHistory": [],
      "stream": false
    }
    """
    try:
        print(f"📥 [Template Chat] template_id={request.template_id}, subject={request.subject}")

        final_prompt = template_service.apply_template(request.template_id, request.user_input)
        print(f"📝 [Template] Applied prompt: {final_prompt[:100]}...")

        messages, model = prepare_messages(
            message=final_prompt,
            subject=request.subject,
            history=request.conversationHistory or [],
        )

        ai_response = await call_openrouter(messages, model)

        return ChatResponse(
            message=ai_response,
            timestamp=datetime.utcnow().isoformat(),
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ [Template Chat] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/chat/send")
async def send_message(request: ChatRequest):
    """Обычный чат без шаблонов"""
    try:
        print(f"📥 [Chat] subject={request.subject}")

        messages, model = prepare_messages(
            message=request.message,
            subject=request.subject,
            history=request.conversationHistory or [],
        )

        ai_response = await call_openrouter(messages, model)

        return ChatResponse(
            message=ai_response,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        print(f"❌ [Chat] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint - ответ приходит частями (Server-Sent Events)
    """

    async def generate():
        try:
            messages, model = prepare_messages(
                request.message,
                request.subject,
                request.conversationHistory,
            )

            print(f"🌊 [Stream] Starting stream for model: {model}")

            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://study-helper.app",
                "X-Title": "Study Helper",
            }

            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 3000,
                "stream": True,
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    OPENROUTER_URL,
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.strip():
                            if line.startswith("data: "):
                                data_str = line[6:]

                                if data_str == "[DONE]":
                                    print("✅ [Stream] Completed")
                                    yield "data: [DONE]\n\n"
                                    break

                                try:
                                    data_json = json.loads(data_str)
                                    if "choices" in data_json and len(data_json["choices"]) > 0:
                                        delta = data_json["choices"][0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            yield f"data: {json.dumps({'content': content})}\n\n"
                                except json.JSONDecodeError:
                                    continue

        except Exception as e:
            error_msg = f"Stream error: {str(e)}"
            print(f"❌ [Stream] {error_msg}")
            yield f"data: {json.dumps({'error': error_msg})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
