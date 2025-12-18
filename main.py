from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator, Dict, Any
import httpx
import os
import json
from datetime import datetime
from pathlib import Path

app = FastAPI(title="Study Helper AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELS ====================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    subject: Optional[str] = "general"
    conversationHistory: Optional[List[ChatMessage]] = []
    stream: Optional[bool] = False

class TemplateRequest(BaseModel):
    template_id: str
    user_input: str
    subject: Optional[str] = "general"
    conversationHistory: Optional[List[ChatMessage]] = []
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    message: str
    conversationId: Optional[str] = None
    timestamp: str

# 🆕 НОВАЯ МОДЕЛЬ для анализа изображений
class AnalyzeImageRequest(BaseModel):
    image_url: Optional[str] = None  # URL изображения (Cloudinary, Imgur, etc)
    image_base64: Optional[str] = None  # Или base64 строка
    subject: Optional[str] = "general"  # Предмет для контекста

class AnalyzeImageResponse(BaseModel):
    description: str  # AI описание стиля решения
    timestamp: str
    success: bool

# ==================== CONFIGURATION ====================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-or-v1-YOUR-KEY-HERE")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "math": "meta-llama/llama-3.3-70b-instruct:free",
    "programming": "qwen/qwen3-coder:free",
    "english": "deepseek/deepseek-chat-v3.1:free",
    "general": "meta-llama/llama-3.3-70b-instruct:free",
}

SUBJECT_PROMPTS = {
    "math": """You are an expert math tutor. Be patient, encouraging, and adapt explanations to the student's level.""",
    "programming": """You are an experienced programming mentor. Provide clear, practical code examples and explanations.""",
    "english": """You are an experienced English language tutor. Provide constructive feedback and practical examples.""",
    "general": """You are a helpful study assistant. Provide clear, accurate explanations and help students learn effectively."""
}


# ==================== TEMPLATE SERVICE ====================

class TemplateService:
    def __init__(self):
        self.templates_file = Path("templates.json")
        self.templates_cache = None
        self.load_templates()
    
    def load_templates(self):
        """Загрузка шаблонов из JSON файла"""
        try:
            if self.templates_file.exists():
                with open(self.templates_file, 'r', encoding='utf-8') as f:
                    self.templates_cache = json.load(f)
                print(f"✅ Loaded {len(self.templates_cache.get('templates', []))} templates")
            else:
                print("⚠️ templates.json not found, using empty templates")
                self.templates_cache = {"templates": [], "categories": []}
        except Exception as e:
            print(f"❌ Error loading templates: {e}")
            self.templates_cache = {"templates": [], "categories": []}
    
    def get_all_templates(self) -> Dict[str, Any]:
        """Получить все шаблоны"""
        return self.templates_cache
    
    def get_template_by_id(self, template_id: str) -> Optional[Dict]:
        """Получить конкретный шаблон по ID"""
        templates = self.templates_cache.get("templates", [])
        for template in templates:
            if template["id"] == template_id:
                return template
        return None
    
    def get_templates_by_subject(self, subject: str) -> List[Dict]:
        """Получить шаблоны для конкретного предмета"""
        templates = self.templates_cache.get("templates", [])
        return [t for t in templates if subject in t.get("subjects", [])]
    
    def apply_template(self, template_id: str, user_input: str) -> str:
        """Применить шаблон с подстановкой user_input"""
        template = self.get_template_by_id(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")
        
        prompt = template["prompt"]
        
        # Подстановка всех плейсхолдеров
        for placeholder in template.get("placeholders", []):
            prompt = prompt.replace(f"{{{placeholder}}}", user_input)
        
        return prompt

# Создаём глобальный экземпляр сервиса
template_service = TemplateService()

# ==================== AI SERVICE ====================

async def call_openrouter(messages: List[dict], model: str) -> str:
    """Обычный вызов OpenRouter"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://study-helper.app",
        "X-Title": "Study Helper"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 3000
    }
    
    print(f"🚀 [OpenRouter] Request to model: {model}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"]
            print(f"✅ [OpenRouter] Success")
            return result
        except httpx.HTTPError as e:
            print(f"❌ [OpenRouter] Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

def prepare_messages(message: str, subject: str, history: List[ChatMessage]) -> tuple[List[dict], str]:
    """Подготовка сообщений для AI"""
    model = MODELS.get(subject, MODELS["general"])
    system_prompt = SUBJECT_PROMPTS.get(subject, SUBJECT_PROMPTS["general"])
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # История
    if history:
        for msg in history[-10:]:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
    
    # Текущее сообщение
    messages.append({
        "role": "user",
        "content": message
    })
    
    return messages, model

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint - приветствие API"""
    return {
        "message": "Study Helper AI Service",
        "version": "1.1.0",
        "docs": "/docs"  # Ссылка на Swagger документацию
    }

@app.get("/health")
async def health():
    """Health check для Docker и мониторинга"""
    return {
        "status": "ok",
        "service": "Study Helper AI Service",
        "version": "1.1.0",
        "features": ["chat", "templates", "streaming"]
    }

@app.get("/api/templates")
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
        else:
            return template_service.get_all_templates()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/templates/{template_id}")
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

@app.post("/api/chat/template")
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
        
        # Применяем шаблон
        final_prompt = template_service.apply_template(request.template_id, request.user_input)
        print(f"📝 [Template] Applied prompt: {final_prompt[:100]}...")
        
        # Подготавливаем сообщения
        messages, model = prepare_messages(
            message=final_prompt,
            subject=request.subject,
            history=request.conversationHistory or []
        )
        
        # Вызываем AI
        ai_response = await call_openrouter(messages, model)
        
        return ChatResponse(
            message=ai_response,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ [Template Chat] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/send")
async def send_message(request: ChatRequest):
    """Обычный чат без шаблонов"""
    try:
        print(f"📥 [Chat] subject={request.subject}")
        
        messages, model = prepare_messages(
            message=request.message,
            subject=request.subject,
            history=request.conversationHistory or []
        )
        
        ai_response = await call_openrouter(messages, model)
        
        return ChatResponse(
            message=ai_response,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        print(f"❌ [Chat] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint - ответ приходит частями (Server-Sent Events)
    """
    async def generate():
        try:
            # Подготовка сообщений и выбор модели
            messages, model = prepare_messages(
                request.message,
                request.subject,
                request.conversationHistory
            )
            
            print(f"🌊 [Stream] Starting stream for model: {model}")
            
            # Заголовки для OpenRouter
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://study-helper.app",
                "X-Title": "Study Helper"
            }
            
            # Payload с включенным стримингом
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 3000,
                "stream": True
            }
            
            # Делаем streaming запрос к OpenRouter
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    OPENROUTER_URL,
                    json=payload,
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    
                    # Читаем и пересылаем каждую строку
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
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)