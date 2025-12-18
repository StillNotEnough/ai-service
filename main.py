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
    # 🆕 НОВАЯ МОДЕЛЬ для Vision
    "vision": "google/gemini-2.0-flash-exp:free"
}

SUBJECT_PROMPTS = {
    "math": """You are an expert math tutor. Be patient, encouraging, and adapt explanations to the student's level.""",
    "programming": """You are an experienced programming mentor. Provide clear, practical code examples and explanations.""",
    "english": """You are an experienced English language tutor. Provide constructive feedback and practical examples.""",
    "general": """You are a helpful study assistant. Provide clear, accurate explanations and help students learn effectively."""
}

# 🆕 Промпты для анализа изображений решений
VISION_ANALYSIS_PROMPTS = {
    "math": """Analyze this image of a solved math problem. Your task is to extract the SOLUTION METHOD so that an AI can solve SIMILAR problems in the same way.

Please provide:

1. **Problem Type**: What type of math problem is this? (derivative, integral, series, limit, equation, etc.)

2. **Solution Method**: Step-by-step methodology used:
   - What formulas/theorems were applied?
   - What substitutions were made?
   - What order were operations performed?
   - What techniques were used? (integration by parts, substitution, expansion, etc.)

3. **Step Structure**: How the solution is organized:
   - How are steps numbered/marked?
   - Which intermediate calculations are shown?
   - How detailed are the explanations?

4. **Formatting Style**: 
   - How formulas are written (separate lines, boxed, etc.)
   - How the final answer is highlighted
   - Special notations or marks used

5. **Key Patterns**: What should an AI remember to solve similar problems in this style?

Write as a guide for solving SIMILAR problems, not just describing this one.""",
    
    "programming": """Analyze this image of code or a programming solution. Extract the CODING APPROACH so that an AI can write similar code.

Please provide:

1. **Problem Type**: What does this code do? (algorithm, data structure, API call, etc.)

2. **Solution Approach**:
   - What algorithm/pattern is used?
   - What data structures are chosen?
   - What is the logic flow?

3. **Code Style**:
   - Naming conventions (camelCase, snake_case, etc.)
   - Comment style and frequency
   - Code organization (functions, classes, modules)

4. **Best Practices Shown**:
   - Error handling approach
   - Edge cases covered
   - Optimization techniques

5. **Formatting Details**:
   - Indentation style
   - Line breaks and spacing
   - How explanations are provided

Write as a guide for writing SIMILAR code, not just describing this one.""",
    
    "english": """Analyze this image of an English text or essay. Extract the WRITING APPROACH so that an AI can write similar texts.

Please provide:

1. **Text Type**: What kind of text is this? (essay, article, letter, report, etc.)

2. **Writing Approach**:
   - What structure is used? (5-paragraph, PEEL, etc.)
   - How are arguments developed?
   - What rhetorical devices are used?

3. **Language Style**:
   - Formality level (formal, informal, academic)
   - Sentence complexity and variety
   - Vocabulary level and word choice

4. **Organization**:
   - How paragraphs are structured
   - How transitions are made
   - How evidence/examples are introduced

5. **Formatting**:
   - Citation style if any
   - Heading/subheading usage
   - Special emphasis techniques

Write as a guide for writing SIMILAR texts, not just describing this one.""",
    
    "general": """Analyze this image of a student's work or solution. Extract the APPROACH and METHOD so that an AI can tackle similar tasks.

Please provide:

1. **Task Type**: What kind of task is shown?

2. **Solution Method**:
   - What approach/strategy is used?
   - What steps are taken?
   - What techniques/tools are applied?

3. **Organization**:
   - How is information structured?
   - What order are steps performed?
   - How detailed are explanations?

4. **Presentation Style**:
   - Visual elements (diagrams, lists, highlights)
   - Formatting choices
   - How key points are emphasized

5. **Replication Guide**: What should an AI remember to solve SIMILAR tasks in this way?

Write as a practical guide, not just a description."""
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

# 🆕 НОВАЯ ФУНКЦИЯ для Vision API
async def call_openrouter_vision(image_url: str, prompt: str, model: str) -> str:
    """Вызов OpenRouter с изображением (Vision API)"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://study-helper.app",
        "X-Title": "Study Helper"
    }
    
    # Формат для Vision API - content это массив с text и image_url
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    print(f"🖼️ [Vision API] Analyzing image with model: {model}")
    print(f"🖼️ [Vision API] Image URL: {image_url[:100]}...")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"]
            print(f"✅ [Vision API] Analysis complete")
            return result
        except httpx.HTTPError as e:
            print(f"❌ [Vision API] Error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"❌ [Vision API] Error details: {error_detail}")
                except:
                    pass
            raise HTTPException(status_code=500, detail=f"Vision AI error: {str(e)}")

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
        "features": ["chat", "templates", "streaming", "vision_analysis"]
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