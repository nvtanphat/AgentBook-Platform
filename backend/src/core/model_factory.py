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


def build_extraction_llm(settings: Settings) -> BaseLLM:
    """Return the LLM configured for extraction tasks (entity/relation).

    Allows routing cheap extraction to a smaller/local model while using a
    stronger model for answer generation. Falls back to build_llm() when no
    extraction-specific provider is configured (extraction_provider: "").
    """
    provider = (settings.llm_extraction_provider or "").lower().strip()
    if not provider:
        return build_llm(settings)

    if provider == "local":
        # Override local model name if specified
        local_model = (settings.llm_extraction_local_model or "").strip()
        if local_model:
            import copy
            override = copy.copy(settings)
            object.__setattr__(override, "llm_local_model", local_model)
            return OllamaLLM(override)
        return OllamaLLM(settings)

    if provider in {"openai", "openai_compatible", "api"}:
        return OpenAICompatibleLLM(settings)

    raise ValueError(f"Unsupported extraction LLM provider: {provider}")
