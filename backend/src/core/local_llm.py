from __future__ import annotations

import httpx

from src.core.base_llm import BaseLLM
from src.core.config import Settings


class OllamaLLM(BaseLLM):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            timeout=self.settings.llm_timeout_seconds,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    async def generate(self, *, prompt: str) -> str:
        response = await self._client.post(
            f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": self.settings.llm_local_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.settings.llm_temperature,
                    "num_predict": self.settings.llm_max_output_tokens,
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("response", "")).strip()

    async def close(self) -> None:
        await self._client.aclose()
