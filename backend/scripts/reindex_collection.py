"""
Delete all chunks/vectors for a collection and re-trigger parse+index for every material.
Run from backend/: python scripts/reindex_collection.py
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OWNER_ID = "user_demo"
COLLECTION_ID = "69fc3c0949fae4625be50223"

async def main() -> None:
    import motor.motor_asyncio
    from beanie import init_beanie, PydanticObjectId
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    from src.core.config import get_settings
    from src.models.chunk import Chunk
    from src.models.collection import KnowledgeCollection
    from src.models.material import Material, MaterialPageDocument
    from src.models.pipeline_job import PipelineJob
    from src.models.chat_memory import ChatSummaryMemory
    from src.models.feedback import Feedback
    from src.models.knowledge_graph import Entity, Event, Relation
    from src.models.query_log import QueryLog
    from src.models.translation_cache import TranslationCache

    settings = get_settings()
    motor_client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongodb_uri)
    await init_beanie(
        database=motor_client[settings.mongodb_database],
        document_models=[
            Chunk, KnowledgeCollection, Material, MaterialPageDocument,
            PipelineJob, ChatSummaryMemory, Feedback,
            Entity, Event, Relation, QueryLog, TranslationCache,
        ],
    )

    col_oid = PydanticObjectId(COLLECTION_ID)

    # ── 1. Delete all chunks from MongoDB ────────────────────────────────
    print("Deleting chunks from MongoDB…")
    result = await Chunk.find(
        Chunk.owner_id == OWNER_ID,
        Chunk.collection_id == col_oid,
    ).delete()
    print(f"  Deleted MongoDB chunks: {result}")

    # ── 2. Delete all vectors from Qdrant ────────────────────────────────
    print("Deleting vectors from Qdrant…")
    qdrant = QdrantClient(url=settings.qdrant_url)
    del_result = qdrant.delete(
        collection_name=settings.qdrant_collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(key="owner_id", match=MatchValue(value=OWNER_ID)),
                FieldCondition(key="collection_id", match=MatchValue(value=COLLECTION_ID)),
            ]
        ),
    )
    print(f"  Qdrant delete result: {del_result}")

    # ── 3. Reset material status and re-queue ─────────────────────────────
    materials = await Material.find(
        Material.owner_id == OWNER_ID,
        Material.collection_id == col_oid,
    ).to_list()
    print(f"\nFound {len(materials)} materials to re-index")

    from src.services.material_service import MaterialService

    svc = MaterialService(settings=settings)

    for mat in materials:
        print(f"\n  [{mat.original_name}] resetting status…")
        mat.status = "uploaded"
        mat.failed_stage = None
        mat.error_message = None
        await mat.save()

        # enqueue via service (bypasses the non_retryable guard since we already reset)
        from uuid import uuid4
        from src.models.pipeline_job import PipelineJob
        from src.models.common import JobType, PipelineStatus, utc_now

        job = PipelineJob(
            material_id=mat.id,
            job_id=str(uuid4()),
            job_type=JobType.PARSE_INDEX.value,
            status=PipelineStatus.UPLOADED.value,
            stage=PipelineStatus.UPLOADED.value,
        )
        await job.insert()
        await svc._enqueue_parse_index(material_id=str(mat.id), job_id=job.job_id)
        print(f"  [{mat.original_name}] queued job_id={job.job_id}")

    print("\nAll materials queued. Watching status…")
    import time
    for _ in range(60):  # wait up to 10 min
        await asyncio.sleep(10)
        statuses = []
        for mat in materials:
            m = await Material.get(mat.id)
            statuses.append((m.original_name[:35], m.status))
        done = sum(1 for _, s in statuses if s in ("indexed", "failed"))
        print(f"  [{time.strftime('%H:%M:%S')}] {done}/{len(materials)} done")
        for name, s in statuses:
            print(f"    {s:10s} {name}")
        if done == len(materials):
            break

    print("\nDone.")

asyncio.run(main())
