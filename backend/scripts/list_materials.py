"""List all materials in the database."""
import asyncio
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from src.core.config import get_settings
from src.models.material import Material
from src.models.knowledge_graph import Entity, Relation
from src.models.pipeline_job import PipelineJob


async def list_materials():
    settings = get_settings()

    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_database]

    await init_beanie(
        database=db,
        document_models=[Material, PipelineJob, Entity, Relation]
    )

    materials = await Material.find().sort("-created_at").to_list()

    print(f"=== MATERIALS ({len(materials)} total) ===\n")

    for mat in materials:
        print(f"ID: {mat.id}")
        print(f"  Name: {mat.original_name}")
        print(f"  Status: {mat.status}")
        print(f"  Owner: {mat.owner_id}")
        print(f"  Collection: {mat.collection_id}")
        print(f"  Created: {mat.created_at}")

        # Count entities and relations
        entity_count = await Entity.find(
            {"mention_refs.material_id": mat.id}
        ).count()
        relation_count = await Relation.find(
            {"evidence_refs.material_id": mat.id}
        ).count()

        print(f"  Entities: {entity_count}")
        print(f"  Relations: {relation_count}")
        print()


if __name__ == "__main__":
    asyncio.run(list_materials())
