from __future__ import annotations

from src.core.base_llm import BaseLLM
from src.core.config import Settings
from src.core.local_llm import OllamaLLM
from src.core.openai_client import OpenAICompatibleLLM


def build_llm(settings: Settings) -> BaseLLM:
    provider = settings.llm_default_provider.lower()
    if provider == "local":
        return OllamaLLM(settings)
    if provider in {"openai", "openai_compatible", "api"}:
        return OpenAICompatibleLLM(settings)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_default_provider}")
