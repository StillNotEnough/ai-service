from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    subject: Optional[str] = "general"
    conversationHistory: List[ChatMessage] = Field(default_factory=list)
    stream: Optional[bool] = False


class TemplateRequest(BaseModel):
    template_id: str
    user_input: str
    subject: Optional[str] = "general"
    conversationHistory: List[ChatMessage] = Field(default_factory=list)
    stream: Optional[bool] = False


class ChatResponse(BaseModel):
    message: str
    conversationId: Optional[str] = None
    timestamp: str


class AnalyzeImageRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    subject: Optional[str] = "general"


class AnalyzeImageResponse(BaseModel):
    description: str
    timestamp: str
    success: bool
