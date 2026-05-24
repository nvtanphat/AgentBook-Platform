"""Base contract for self-describing tools used by specialist agents."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Uniform wrapper around tool output.

    Tools never raise to the coordinator — they catch all exceptions and
    return `success=False` with the error reason. The coordinator then
    decides whether to retry, skip, or fall back to a deterministic path.
    """

    tool: str
    success: bool
    data: Any = None
    duration_ms: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool:
    """Each tool inherits this; subclasses set `name` + `description` and
    implement `_run`. Public callers go through `run()` which adds telemetry
    + exception safety."""

    name: str = "base_tool"
    description: str = "Abstract tool."

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}

    async def run(self, **kwargs) -> ToolResult:
        started = time.perf_counter()
        try:
            data = await self._run(**kwargs)
            return ToolResult(
                tool=self.name,
                success=True,
                data=data,
                duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
                metadata={"input_keys": sorted(kwargs.keys())},
            )
        except Exception as exc:
            logger.warning(
                "Tool execution failed",
                extra={"tool": self.name, "error": str(exc), "error_type": type(exc).__name__},
            )
            return ToolResult(
                tool=self.name,
                success=False,
                data=None,
                duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
                error=str(exc),
            )

    async def _run(self, **kwargs) -> Any:  # noqa: D401
        raise NotImplementedError
