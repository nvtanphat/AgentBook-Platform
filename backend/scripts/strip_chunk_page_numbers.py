"""One-off: strip OCR'd leading page-number prefixes from existing chunks.

The chunking pipeline now strips these at build time (chunking.py), but chunks
indexed before that fix still carry prefixes like "172 ..." / "95 ...". This
updates Chunk.content in Mongo in place — no re-OCR, no re-embed (the prefix's
effect on a ~1600-char embedding is negligible; retrieval display reads Mongo).

Usage:
    cd backend
    python scripts/strip_chunk_page_numbers.py <collection_id>
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
from src.processing.chunking import _strip_leading_page_number


async def main(collection_id: str) -> None:
    await init_database(get_settings())
    cid = PydanticObjectId(collection_id)
    chunks = await Chunk.find(Chunk.collection_id == cid).to_list()
    print(f"Scanning {len(chunks)} chunks in collection {collection_id}")

    changed = 0
    for chunk in chunks:
        original = chunk.content or ""
        cleaned = _strip_leading_page_number(original)
        if cleaned != original:
            chunk.content = cleaned
            await chunk.save()
            changed += 1

    print(f"Stripped leading page numbers from {changed}/{len(chunks)} chunks.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/strip_chunk_page_numbers.py <collection_id>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
