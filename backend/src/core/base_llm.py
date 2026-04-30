from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, *, prompt: str) -> str:
        """Generate text from a grounded prompt."""
