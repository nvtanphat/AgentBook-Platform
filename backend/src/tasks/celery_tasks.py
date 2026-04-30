from __future__ import annotations

import asyncio

from celery import Celery

from src.core.config import get_settings
from src.database import close_database, init_database
from src.services.parse_index_pipeline import ParseIndexPipeline

settings = get_settings()

celery_app = Celery(
    "prism",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_always_eager=settings.celery_task_always_eager,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
)


@celery_app.task(name="prism.parse_and_index_material")
def parse_and_index_material_task(material_id: str, job_id: str) -> str:
    async def _run() -> None:
        await init_database(settings)
        try:
            pipeline = ParseIndexPipeline(settings=settings)
            await pipeline.run(material_id=material_id, job_id=job_id)
        finally:
            await close_database()

    asyncio.run(_run())
    return job_id
