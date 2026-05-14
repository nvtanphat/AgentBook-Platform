from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)

_TASKS: set[asyncio.Task] = set()


def spawn_background_task(coro: Coroutine[Any, Any, Any], *, name: str) -> asyncio.Task:
    """Create a tracked background task and log failures deterministically."""
    task = asyncio.create_task(coro, name=name)
    _TASKS.add(task)
    task.add_done_callback(_on_task_done)
    return task


def _on_task_done(task: asyncio.Task) -> None:
    _TASKS.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("Background task failed", extra={"task_name": task.get_name()})


async def shutdown_background_tasks(timeout_seconds: float = 5.0) -> None:
    if not _TASKS:
        return
    tasks = list(_TASKS)
    for task in tasks:
        task.cancel()
    done, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
    for task in done:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Background task failed during shutdown", extra={"task_name": task.get_name()})
    for task in pending:
        logger.warning("Background task did not stop before shutdown timeout", extra={"task_name": task.get_name()})
