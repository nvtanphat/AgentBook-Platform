from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, *, prompt: str) -> str:
        """Generate text from a grounded prompt."""

    async def stream(self, *, prompt: str) -> AsyncGenerator[str, None]:
        """Stream tokens one by one. Default: single chunk from generate()."""
        yield await self.generate(prompt=prompt)
