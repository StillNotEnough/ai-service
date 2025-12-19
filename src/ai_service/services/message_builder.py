from typing import List, Tuple

from ai_service.config.settings import MODELS, SUBJECT_PROMPTS
from ai_service.models.schemas import ChatMessage


def prepare_messages(
    message: str,
    subject: str,
    history: List[ChatMessage],
) -> Tuple[List[dict], str]:
    """Подготовка сообщений для AI"""
    model = MODELS.get(subject, MODELS["general"])
    system_prompt = SUBJECT_PROMPTS.get(subject, SUBJECT_PROMPTS["general"])

    messages = [{"role": "system", "content": system_prompt}]

    if history:
        for msg in history[-10:]:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })

    messages.append({
        "role": "user",
        "content": message,
    })

    return messages, model
