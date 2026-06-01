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

    async def generate(self, *, prompt: str) -> str:
        max_retries = 1
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        logger.info(
            "Calling local LLM",
            extra={"url": url, "model": self.settings.llm_local_model, "prompt_length": len(prompt)},
        )

        for attempt in range(max_retries):
            try:
                response = await self._client.post(
                    url,
                    json={
                        "model": self.settings.llm_local_model,
                        "stream": False,
                        "messages": [{"role": "user", "content": prompt}],
                        "options": {
                            "temperature": self.settings.llm_temperature,
                            "num_predict": self.settings.llm_max_output_tokens,
                        },
                    },
                )
                logger.info("Local LLM response received", extra={"status_code": response.status_code})
                response.raise_for_status()
                payload = response.json()
                raw = str(payload.get("message", {}).get("content", "")).strip()
                result = _THINK_TAG_RE.sub("", raw).strip()
                logger.info("Local LLM generation succeeded", extra={"response_length": len(result)})
                return result
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Local LLM HTTP error",
                    extra={"attempt": attempt + 1, "max_retries": max_retries, "status_code": exc.response.status_code},
                )
                if attempt == max_retries - 1:
                    raise
            except httpx.TimeoutException:
                logger.error("Local LLM timeout", extra={"attempt": attempt + 1, "max_retries": max_retries})
                if attempt == max_retries - 1:
                    raise
            except Exception as exc:
                logger.error(
                    "Local LLM request failed",
                    exc_info=True,
                    extra={"attempt": attempt + 1, "max_retries": max_retries, "error_type": type(exc).__name__},
                )
                if attempt == max_retries - 1:
                    raise
        return ""

    async def stream(self, *, prompt: str) -> AsyncGenerator[str, None]:
        try:
            async with self._client.stream(
                "POST",
                f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
                json={
                    "model": self.settings.llm_local_model,
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
            logger.error("Local LLM stream HTTP error", extra={"status_code": exc.response.status_code})
            raise
        except httpx.TimeoutException:
            logger.error("Local LLM stream timeout")
            raise
        except Exception as exc:
            logger.error("Local LLM stream failed", extra={"error_type": type(exc).__name__})
            raise

    async def close(self) -> None:
        await self._client.aclose()
