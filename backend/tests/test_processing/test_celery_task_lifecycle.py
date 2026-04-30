from __future__ import annotations

import asyncio
from beanie import PydanticObjectId

from src.tasks import celery_tasks
from src.services.parse_index_pipeline import ParseIndexPipeline


def test_celery_task_initializes_database_runs_pipeline_and_closes(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_init_database(settings) -> None:
        calls.append("init")

    async def fake_close_database() -> None:
        calls.append("close")

    class FakePipeline:
        def __init__(self, *, settings) -> None:
            calls.append("pipeline_init")

        async def run(self, *, material_id: str, job_id: str) -> None:
            calls.append(f"run:{material_id}:{job_id}")

    monkeypatch.setattr(celery_tasks, "init_database", fake_init_database)
    monkeypatch.setattr(celery_tasks, "close_database", fake_close_database)
    monkeypatch.setattr(celery_tasks, "ParseIndexPipeline", FakePipeline)

    result = celery_tasks.parse_and_index_material_task.run("mat-1", "job-1")

    assert result == "job-1"
    assert calls == ["init", "pipeline_init", "run:mat-1:job-1", "close"]


def test_parse_pipeline_raises_if_material_was_deleted(monkeypatch) -> None:
    async def fake_get(material_id):
        return None

    monkeypatch.setattr("src.services.parse_index_pipeline.Material.get", fake_get)

    try:
        asyncio.run(ParseIndexPipeline._ensure_material_exists(str(PydanticObjectId())))
    except LookupError as exc:
        assert "deleted" in str(exc)
    else:
        raise AssertionError("deleted material should abort the pipeline")
