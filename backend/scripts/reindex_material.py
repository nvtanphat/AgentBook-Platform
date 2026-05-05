"""
Re-index a material to regenerate entities and relations with new RelationExtractor.
Usage: python reindex_material.py <material_id>
"""
import asyncio
import sys
import uuid
from beanie import init_beanie, PydanticObjectId
from motor.motor_asyncio import AsyncIOMotorClient

from src.core.config import get_settings
from src.models.material import Material
from src.models.knowledge_graph import Entity, Relation
from src.models.pipeline_job import PipelineJob
from src.models.common import PipelineStatus
from src.services.parse_index_pipeline import ParseIndexPipeline


async def reindex_material(material_id: str):
    settings = get_settings()

    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_database]

    await init_beanie(
        database=db,
        document_models=[Material, PipelineJob, Entity, Relation]
    )

    # Find material
    material = await Material.get(PydanticObjectId(material_id))
    if not material:
        print(f"Material {material_id} not found")
        return

    print(f"Re-indexing material: {material.original_name}")
    print(f"  ID: {material.id}")
    print(f"  Owner: {material.owner_id}")
    print(f"  Collection: {material.collection_id}")

    # Delete old entities and relations for this material
    print("\nDeleting old entities and relations...")
    deleted_entities = await Entity.find(
        {"mention_refs.material_id": material.id}
    ).delete()
    deleted_relations = await Relation.find(
        {"evidence_refs.material_id": material.id}
    ).delete()
    print(f"  Deleted {deleted_entities.deleted_count} entities")
    print(f"  Deleted {deleted_relations.deleted_count} relations")

    # Re-run pipeline
    print("\nRunning parse & index pipeline...")
    pipeline = ParseIndexPipeline(settings=settings)

    # Create a new pipeline job
    job_id = str(uuid.uuid4())
    job = PipelineJob(
        material_id=material.id,
        job_id=job_id,
        status=PipelineStatus.UPLOADED,
    )
    await job.insert()

    try:
        await pipeline.run(material_id=str(material.id), job_id=job_id)
        print("\n[OK] Re-indexing completed successfully!")

        # Show new stats
        new_entities = await Entity.find(
            {"mention_refs.material_id": material.id}
        ).count()
        new_relations = await Relation.find(
            {"evidence_refs.material_id": material.id}
        ).count()

        print(f"\nNew stats:")
        print(f"  Entities: {new_entities}")
        print(f"  Relations: {new_relations}")

        # Show relation types
        pipeline_result = [
            {"$match": {"evidence_refs.material_id": material.id}},
            {"$group": {"_id": "$relation_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]

        print("\nRelation types:")
        async for result in Relation.get_motor_collection().aggregate(pipeline_result):
            rel_type = result["_id"]
            count = result["count"]
            print(f"  {rel_type}: {count}")

    except Exception as exc:
        print(f"\n[FAIL] Re-indexing failed: {exc}")
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reindex_material.py <material_id>")
        sys.exit(1)

    material_id = sys.argv[1]
    asyncio.run(reindex_material(material_id))
