from teachers.base import BaseTeacher
from teachers.gemini import GeminiTeacher
from teachers.nim_teacher import NimTeacher
from teachers.ollama_teacher import OllamaTeacher
from teachers.openai_teacher import OpenAITeacher

TEACHER_PROVIDERS = {
    "ollama": OllamaTeacher,
    "nim": NimTeacher,
    "gemini": GeminiTeacher,
    "openai": OpenAITeacher,
}


def get_teacher(provider: str = "ollama", model_name: str | None = None) -> BaseTeacher:
    provider_key = provider.lower().strip()
    if provider_key not in TEACHER_PROVIDERS:
        raise ValueError(
            f"Unsupported provider '{provider}'. Available providers: {list(TEACHER_PROVIDERS.keys())}"
        )

    teacher_cls = TEACHER_PROVIDERS[provider_key]
    if model_name:
        return teacher_cls(model_name=model_name)
    return teacher_cls()
