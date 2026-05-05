from __future__ import annotations

import json
import logging
import re
from typing import AsyncGenerator

import httpx

from src.core.base_llm import BaseLLM
from src.core.config import Settings

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaLLM(BaseLLM):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.llm_timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    def _prepare_prompt(self, prompt: str) -> str:
        """Prepend /no_think for qwen3 models to disable chain-of-thought output."""
        if "qwen3" in self.settings.llm_local_model.lower():
            return "/no_think\n" + prompt
        return prompt

    async def generate(self, *, prompt: str) -> str:
        max_retries = 2
        prompt = self._prepare_prompt(prompt)
        # Use /api/chat for better thinking-mode control (qwen3, deepseek-r1, etc.)
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        logger.info(f"Calling Ollama at {url} with model {self.settings.llm_local_model}, prompt length: {len(prompt)}")

        for attempt in range(max_retries):
            try:
                response = await self._client.post(
                    url,
                    json={
                        "model": self.settings.llm_local_model,
                        "think": False,
                        "stream": False,
                        "messages": [{"role": "user", "content": prompt}],
                        "options": {
                            "temperature": self.settings.llm_temperature,
                            "num_predict": self.settings.llm_max_output_tokens,
                        },
                    },
                )
                logger.info(f"Ollama response status: {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                raw = str(payload.get("message", {}).get("content", "")).strip()
                # Strip any residual <think>...</think> blocks (fallback for older Ollama)
                result = _THINK_TAG_RE.sub("", raw).strip()
                logger.info(f"Ollama generation successful, response length: {len(result)}")
                return result
            except httpx.HTTPStatusError as exc:
                logger.error(
                    f"Ollama HTTP error (attempt {attempt + 1}/{max_retries}): status={exc.response.status_code}, body={exc.response.text[:500]}"
                )
                if attempt == max_retries - 1:
                    raise
            except httpx.TimeoutException as exc:
                logger.error(f"Ollama timeout (attempt {attempt + 1}/{max_retries}): {exc}")
                if attempt == max_retries - 1:
                    raise
            except Exception as exc:
                logger.error(f"Ollama request failed (attempt {attempt + 1}/{max_retries}): {type(exc).__name__} - {exc}", exc_info=True)
                if attempt == max_retries - 1:
                    raise
        return ""

    async def stream(self, *, prompt: str) -> AsyncGenerator[str, None]:
        prompt = self._prepare_prompt(prompt)
        try:
            async with self._client.stream(
                "POST",
                f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
                json={
                    "model": self.settings.llm_local_model,
                    "think": False,
                    "stream": True,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {
                        "temperature": self.settings.llm_temperature,
                        "num_predict": self.settings.llm_max_output_tokens,
                    },
                },
            ) as response:
                response.raise_for_status()
                in_think = False
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            if "<think>" in token:
                                in_think = True
                            if in_think:
                                if "</think>" in token:
                                    in_think = False
                                    token = token.split("</think>", 1)[-1]
                                else:
                                    token = ""
                            if token:
                                yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except httpx.HTTPStatusError as exc:
            logger.error(f"Ollama stream HTTP error: {exc.response.status_code} - {exc.response.text[:200]}")
            raise
        except httpx.TimeoutException as exc:
            logger.error(f"Ollama stream timeout: {exc}")
            raise
        except Exception as exc:
            logger.error(f"Ollama stream failed: {type(exc).__name__} - {exc}")
            raise

    async def close(self) -> None:
        await self._client.aclose()
