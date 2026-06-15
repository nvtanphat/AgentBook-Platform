"""Per-request tracing for observability (đo được / debug được).

A `RequestTrace` is created once per query (keyed by a uuid4 `query_id`), threaded
through `inference_engine.answer()/answer_stream()`, and persisted into
`QueryLog.trace`. It records per-stage latency plus arbitrary structured fields
(route, modality, retrieved chunk ids, rerank scores, prompt file, validator and
quality-gate verdicts) so any latency regression or quality drop is visible without
re-running the query.

Pure measurement — adding a trace never changes pipeline behaviour. Stage timing works
inside async code because the timer only brackets the wrapped block.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator


class RequestTrace:
    def __init__(self, query_id: str | None = None) -> None:
        self.query_id: str = query_id or uuid.uuid4().hex
        self.latency_by_stage: dict[str, int] = {}
        self.fields: dict[str, Any] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a pipeline stage; records elapsed milliseconds under `name`.

        Re-entered names accumulate (e.g. multiple retrieval sub-calls) so the
        total reflects real wall-time spent in that stage.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self.latency_by_stage[name] = self.latency_by_stage.get(name, 0) + elapsed_ms

    def set(self, key: str, value: Any) -> None:
        """Attach a structured field (route, modality, retrieved_ids, …)."""
        if value is not None:
            self.fields[key] = value

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            self.set(key, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "latency_by_stage": dict(self.latency_by_stage),
            **self.fields,
        }
