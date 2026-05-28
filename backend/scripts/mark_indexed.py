"""One-off: mark a material as INDEXED when its data is already complete.

Use when a material is stuck in 'uploaded'/'parsing' (e.g. a double-retry left
it mid-run) but the chunks/entities from a prior successful run are already in
Qdrant + Mongo. Breaks the startup re-enqueue loop (_reenqueue_uploaded_materials).

Usage:
    cd backend
    python scripts/mark_indexed.py <material_id>
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beanie import PydanticObjectId

from src.core.config import get_settings
from src.database import init_database
from src.models.chunk import Chunk
from src.models.common import PipelineStatus, utc_now
from src.models.material import Material
from src.models.pipeline_job import PipelineJob


async def main(material_id: str) -> None:
    await init_database(get_settings())
    mid = PydanticObjectId(material_id)
    material = await Material.get(mid)
    if material is None:
        print(f"ERROR: material {material_id} not found")
        sys.exit(1)

    chunk_count = await Chunk.find(Chunk.material_id == mid).count()
    print(f"Material: {material.original_name} | current status={material.status} | chunks in Mongo={chunk_count}")
    if chunk_count == 0:
        print("Refusing: 0 chunks — data is NOT complete, do a real retry instead.")
        sys.exit(1)

    material.status = PipelineStatus.INDEXED.value
    material.failed_stage = None
    material.error_message = None
    material.updated_at = utc_now()
    await material.save()

    # Mark any non-terminal jobs for this material as done so startup won't resume.
    jobs = await PipelineJob.find(PipelineJob.material_id == mid).to_list()
    fixed = 0
    for job in jobs:
        if job.status != PipelineStatus.INDEXED.value:
            job.status = PipelineStatus.INDEXED.value
            job.stage = PipelineStatus.INDEXED.value
            job.finished_at = job.finished_at or utc_now()
            await job.save()
            fixed += 1

    print(f"Set status=indexed. Reconciled {fixed} pipeline job(s). Data preserved ({chunk_count} chunks).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/mark_indexed.py <material_id>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
