from __future__ import annotations

from datetime import datetime

from beanie import Document, PydanticObjectId
from pydantic import Field
from pymongo import IndexModel

from src.models.common import JobType, PipelineStatus, utc_now


class PipelineJob(Document):
    material_id: PydanticObjectId
    job_id: str
    job_type: str = JobType.UPLOAD.value
    status: str = PipelineStatus.UPLOADED.value
    stage: str = PipelineStatus.UPLOADED.value
    retry_count: int = 0
    failed_stage: str | None = None
    last_error: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None

    class Settings:
        name = "pipeline_jobs"
        indexes = [
            IndexModel([("material_id", 1), ("status", 1), ("started_at", -1)], name="pipeline_jobs_material_status_started"),
            IndexModel([("job_id", 1)], unique=True, name="pipeline_jobs_job_id_unique"),
        ]
