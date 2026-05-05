from __future__ import annotations

import json
from typing import AsyncGenerator

import httpx

from src.core.base_llm import BaseLLM
from src.core.config import Settings


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI-compatible LLM provider")
        self._client = httpx.AsyncClient(
            timeout=self.settings.llm_timeout_seconds,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    async def generate(self, *, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        response = await self._client.post(
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": self.settings.openai_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.settings.llm_temperature,
                "max_tokens": self.settings.llm_max_output_tokens,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"]).strip()

    async def stream(self, *, prompt: str) -> AsyncGenerator[str, None]:
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        async with self._client.stream(
            "POST",
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": self.settings.openai_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.settings.llm_temperature,
                "max_tokens": self.settings.llm_max_output_tokens,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    token = chunk["choices"][0]["delta"].get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def close(self) -> None:
        await self._client.aclose()
