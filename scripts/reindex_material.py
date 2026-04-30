from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

from beanie import PydanticObjectId

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from src.core.config import get_settings
from src.database import close_database, init_database
from src.models.common import JobType, PipelineStatus
from src.models.pipeline_job import PipelineJob
from src.services.parse_index_pipeline import ParseIndexPipeline


async def _run(material_id: str) -> str:
    settings = get_settings()
    await init_database(settings)
    try:
        job = PipelineJob(
            material_id=PydanticObjectId(material_id),
            job_id=str(uuid4()),
            job_type=JobType.PARSE_INDEX.value,
            status=PipelineStatus.UPLOADED.value,
            stage=PipelineStatus.UPLOADED.value,
        )
        await job.insert()
        await ParseIndexPipeline(settings=settings).run(material_id=material_id, job_id=job.job_id)
        return job.job_id
    finally:
        await close_database()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("material_id")
    args = parser.parse_args()
    job_id = asyncio.run(_run(args.material_id))
    print(job_id)


if __name__ == "__main__":
    main()
