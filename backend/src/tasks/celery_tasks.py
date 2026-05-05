from __future__ import annotations

import asyncio
import logging

from celery import Celery

logger = logging.getLogger(__name__)

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
    task_default_queue="ingest",
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_max_retries=3,
)


@celery_app.task(
    name="prism.parse_and_index_material",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
)
def parse_and_index_material_task(self, material_id: str, job_id: str) -> str:
    async def _run() -> None:
        await init_database(settings)
        try:
            pipeline = ParseIndexPipeline(settings=settings)
            await pipeline.run(material_id=material_id, job_id=job_id)
        finally:
            await close_database()

    try:
        asyncio.run(_run())
    except Exception as exc:
        logger.warning(
            "parse_and_index_material_task failed, scheduling retry",
            extra={"material_id": material_id, "job_id": job_id, "error": str(exc), "retries": self.request.retries},
        )
        raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
    return job_id
