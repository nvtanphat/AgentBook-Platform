"""
Reset stuck materials and call the backend retry API one at a time.
Run from backend/: python scripts/retry_materials.py
"""
import asyncio
import sys
import os
import time
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OWNER_ID = "user_demo"
COLLECTION_ID = "69fc3c0949fae4625be50223"
BASE_URL = "http://localhost:8000/api/v1"

# Materials that need re-indexing (those NOT "indexed")
MATERIAL_IDS = [
    ("69fc415e49fae4625be50228", "ML_Starter_Pack_Slides.pptx"),   # stuck at parsing
    ("69fc415f49fae4625be5022a", "ML_Study_Workbook.xlsx"),         # uploaded
    ("69fc415f49fae4625be5022c", "ML_Tai_lieu_hoc_20_trang.docx"),  # uploaded
    ("69fc416049fae4625be5022e", "rag_mau_hoc_tap.pdf"),            # uploaded
]

async def main() -> None:
    import motor.motor_asyncio
    from beanie import init_beanie, PydanticObjectId
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

    # Reset any "parsing" materials back to "uploaded" so retry endpoint accepts them
    col_oid = PydanticObjectId(COLLECTION_ID)
    materials = await Material.find(
        Material.owner_id == OWNER_ID,
        Material.collection_id == col_oid,
    ).to_list()

    for mat in materials:
        if mat.status in ("parsing", "indexing", "failed"):
            mat.status = "uploaded"
            mat.failed_stage = None
            mat.error_message = None
            await mat.save()
            print(f"Reset {mat.original_name}: {mat.status} -> uploaded")

    print("\nProcessing materials one at a time via backend API...")

    for mat_id, name in MATERIAL_IDS:
        print(f"\n[{name}] calling retry...")

        # Check current status first
        status_resp = requests.get(
            f"{BASE_URL}/materials/{mat_id}/status",
            params={"owner_id": OWNER_ID},
        )
        if status_resp.ok:
            cur_status = status_resp.json().get("data", {}).get("status", "?")
            print(f"  current status: {cur_status}")
            if cur_status == "indexed":
                print(f"  already indexed, skipping")
                continue

        resp = requests.post(
            f"{BASE_URL}/materials/{mat_id}/retry",
            params={"owner_id": OWNER_ID},
        )
        if not resp.ok:
            print(f"  retry failed: {resp.status_code} {resp.text[:200]}")
            continue
        print(f"  retry accepted, polling...")

        # Poll until done
        for attempt in range(120):  # up to 20 minutes (10s poll)
            time.sleep(10)
            status_resp = requests.get(
                f"{BASE_URL}/materials/{mat_id}/status",
                params={"owner_id": OWNER_ID},
            )
            if status_resp.ok:
                data = status_resp.json().get("data", {})
                status = data.get("status", "?")
                progress = data.get("progress", "?")
                print(f"  [{time.strftime('%H:%M:%S')}] status={status} progress={progress}")
                if status in ("indexed", "failed"):
                    print(f"  [{name}] DONE: {status}")
                    break
            else:
                print(f"  status poll failed: {status_resp.status_code}")

    print("\n=== Final status ===")
    all_mats = await Material.find(
        Material.owner_id == OWNER_ID,
        Material.collection_id == col_oid,
    ).to_list()
    for mat in all_mats:
        print(f"  {mat.status:10s}  {mat.original_name}")

asyncio.run(main())
