from typing import List

import httpx
from fastapi import HTTPException

from ai_service.config.settings import OPENROUTER_API_KEY, OPENROUTER_URL


async def call_openrouter(messages: List[dict], model: str) -> str:
    """Обычный вызов OpenRouter"""
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
    }

    print(f"🚀 [OpenRouter] Request to model: {model}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"]
            print("✅ [OpenRouter] Success")
            return result
        except httpx.HTTPError as e:
            print(f"❌ [OpenRouter] Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")
