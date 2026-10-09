import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-or-v1-YOUR-KEY-HERE")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "math": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "programming": "nvidia/nemotron-3-super-120b-a12b:free",
    "english": "google/gemma-4-31b-it:free",
    "general": "nvidia/nemotron-3-ultra-550b-a55b:free",
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
