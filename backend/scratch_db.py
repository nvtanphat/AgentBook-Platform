import asyncio
import motor.motor_asyncio
import sys
from pathlib import Path
from bson import ObjectId
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.core.config import get_settings

async def main():
    settings = get_settings()
    client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongodb_uri)
    db = client["agentbook"]
    
    col_chunks = db["chunks"]
    col_id = "6a00478c82721158f7a69517"
    count_mongo = await col_chunks.count_documents({"collection_id": ObjectId(col_id)})
    print("MongoDB Chunk Count for 6a00478c82721158f7a69517:", count_mongo)

    # Count Qdrant points
    qdrant = QdrantClient(url=settings.qdrant_url)
    try:
        res = qdrant.count(
            collection_name=settings.qdrant_collection_name,
            count_filter=Filter(
                must=[
                    FieldCondition(key="collection_id", match=MatchValue(value=col_id))
                ]
            ),
            exact=True
        )
        print("Qdrant Point Count for 6a00478c82721158f7a69517:", res.count)
    except Exception as exc:
        print("Qdrant count error:", exc)

if __name__ == "__main__":
    asyncio.run(main())
