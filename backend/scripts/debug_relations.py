"""
Debug script to check if relations are being created in the database.
Run this to verify the RelationExtractor is working.
"""
import asyncio
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from src.core.config import get_settings
from src.models.knowledge_graph import Entity, Relation
from src.models.material import Material
from src.models.pipeline_job import PipelineJob


async def check_relations():
    settings = get_settings()

    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_database]

    await init_beanie(
        database=db,
        document_models=[Material, PipelineJob, Entity, Relation]
    )

    print("=== RELATION EXTRACTION DEBUG ===\n")

    # Count total relations
    total_relations = await Relation.count()
    print(f"Total relations in DB: {total_relations}")

    if total_relations == 0:
        print("\n⚠️  NO RELATIONS FOUND!")
        print("This means:")
        print("  1. No documents have been indexed yet, OR")
        print("  2. Documents were indexed before RelationExtractor was added")
        print("\n✅ Solution: Upload a new document to test relation extraction")
        return

    # Sample relations by type
    print("\n--- Relations by type ---")
    pipeline = [
        {"$group": {"_id": "$relation_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]

    async for result in Relation.get_motor_collection().aggregate(pipeline):
        rel_type = result["_id"]
        count = result["count"]
        print(f"  {rel_type}: {count}")

    # Show sample semantic relations (new ones)
    semantic_types = ["is_a", "part_of", "causes", "uses", "prevents", "improves"]
    semantic_relations = await Relation.find(
        {"relation_type": {"$in": semantic_types}}
    ).limit(5).to_list()

    if semantic_relations:
        print(f"\n✅ Found {len(semantic_relations)} semantic relations (NEW!)")
        print("\n--- Sample semantic relations ---")
        for rel in semantic_relations[:3]:
            print(f"  {rel.source_id} --[{rel.relation_type}]--> {rel.target_id}")
            print(f"    Confidence: {rel.confidence:.2f}")
            print(f"    Evidence: {len(rel.evidence_refs)} refs")
    else:
        print("\n⚠️  No semantic relations found")
        print("Only structural relations (mentioned_in_block, section_contains)")
        print("\n✅ Solution: Upload a new document to create semantic relations")

    # Check entities
    total_entities = await Entity.count()
    print(f"\n--- Entities ---")
    print(f"Total entities: {total_entities}")

    if total_entities > 0:
        sample_entities = await Entity.find().limit(5).to_list()
        print("\nSample entities:")
        for entity in sample_entities[:3]:
            print(f"  - {entity.canonical_name} ({entity.entity_type}, conf={entity.confidence:.2f})")


if __name__ == "__main__":
    asyncio.run(check_relations())
