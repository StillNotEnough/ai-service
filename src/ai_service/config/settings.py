import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-or-v1-YOUR-KEY-HERE")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "math": "meta-llama/llama-3.3-70b-instruct:free",
    "programming": "qwen/qwen3-coder:free",
    "english": "deepseek/deepseek-chat-v3.1:free",
    "general": "meta-llama/llama-3.3-70b-instruct:free",
}

SUBJECT_PROMPTS = {
    "math": (
        "You are an expert math tutor. Be patient, encouraging, and adapt explanations "
        "to the student's level."
    ),
    "programming": (
        "You are an experienced programming mentor. Provide clear, practical code examples "
        "and explanations."
    ),
    "english": (
        "You are an experienced English language tutor. Provide constructive feedback and "
        "practical examples."
    ),
    "general": (
        "You are a helpful study assistant. Provide clear, accurate explanations and help "
        "students learn effectively."
    ),
}
