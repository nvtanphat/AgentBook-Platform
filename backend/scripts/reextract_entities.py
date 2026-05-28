"""One-off: re-run entity extraction on an already-indexed material.

Bypasses OCR/parse/chunking — pulls existing chunks from Mongo, runs the
current EntityExtractor with the current prompt/config, deletes old Entity
records for that material, inserts the new ones.

Use after changing extraction_config.yaml or the prompt template, to verify
the change without re-processing scanned PDFs (which is expensive).

Usage:
    cd backend
    python scripts/reextract_entities.py <material_id>
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beanie import PydanticObjectId

from src.core.config import get_settings
from src.core.model_factory import build_llm
from src.database import init_database
from src.models.chunk import Chunk
from src.models.collection import KnowledgeCollection
from src.models.knowledge_graph import Entity
from src.models.material import Material
from src.processing.entity_extractor import EntityExtractor
from src.processing.entity_resolution import EntityResolver
from src.processing.types import EvidenceBlock, EvidenceMap


async def _build_evidence_map_from_chunks(material: Material) -> EvidenceMap:
    """Reconstruct an EvidenceMap directly from indexed chunks.

    Block-level granularity is lost (one pseudo-block per chunk), but entity
    extraction only needs `snippet_original` so this is sufficient.
    """
    chunks = await Chunk.find(Chunk.material_id == material.id).to_list()
    blocks: list[EvidenceBlock] = []
    for idx, chunk in enumerate(chunks):
        page = chunk.source_pages[0] if chunk.source_pages else 1
        block_id = chunk.source_block_ids[0] if chunk.source_block_ids else f"chunk-{idx}"
        blocks.append(
            EvidenceBlock(
                owner_id=chunk.owner_id,
                collection_id=str(chunk.collection_id),
                material_id=str(chunk.material_id),
                document_name=material.original_name,
                page=page,
                block_id=block_id,
                block_type="paragraph",
                snippet_original=chunk.content or "",
                source_language=chunk.language or material.language,
            )
        )
    print(f"Reconstructed {len(blocks)} evidence blocks from {len(chunks)} chunks")
    return EvidenceMap(
        owner_id=material.owner_id,
        collection_id=str(material.collection_id),
        material_id=str(material.id),
        document_name=material.original_name,
        blocks=blocks,
    )


async def main(material_id: str) -> None:
    settings = get_settings()
    await init_database(settings)

    material = await Material.get(PydanticObjectId(material_id))
    if material is None:
        print(f"ERROR: material {material_id} not found")
        sys.exit(1)
    print(f"Material: {material.original_name} ({material.page_count} pages)")

    # Look up collection.subject as domain hint
    collection = await KnowledgeCollection.get(material.collection_id)
    domain_hint = getattr(collection, settings.extraction_domain_hint_field, None) if collection else None
    print(f"Domain hint: {domain_hint!r}")

    evidence_map = await _build_evidence_map_from_chunks(material)
    if not evidence_map.blocks:
        print("No chunks found, nothing to extract")
        sys.exit(1)

    llm = build_llm(settings)
    extractor = EntityExtractor(
        llm=llm,
        default_entity_types=settings.extraction_default_entity_types,
        few_shots=settings.extraction_few_shots,
        mode=settings.extraction_mode,
    )
    resolver = EntityResolver()

    print("Running entity extraction (this calls the LLM for each batch)...")
    raw = await extractor.extract_async(evidence_map, domain_hint=domain_hint)
    resolved = resolver.resolve(raw)
    print(f"Extracted {len(raw)} raw entities, resolved to {len(resolved)} canonical")

    # Wipe old entities for this material and insert new
    deleted = await Entity.find({"mention_refs.material_id": material.id}).delete()
    print(f"Deleted {deleted.deleted_count} old Entity records for this material")

    if resolved:
        docs = [
            Entity(
                owner_id=e.mention_refs[0].material_id if not e.mention_refs else material.owner_id,
                collection_id=material.collection_id,
                canonical_name=e.canonical_name,
                entity_type=e.entity_type,
                confidence=e.confidence,
                aliases=e.aliases,
                mention_refs=[
                    {
                        "material_id": PydanticObjectId(ref.material_id),
                        "page": ref.page,
                        "block_id": ref.block_id,
                    }
                    for ref in e.mention_refs
                ],
                chunk_ids=[],
            )
            for e in resolved
        ]
        await Entity.insert_many(docs)
        print(f"Inserted {len(docs)} new Entity records")

    # Sample preview
    print("\nTop 15 entities by mention count:")
    for e in sorted(resolved, key=lambda x: (len(x.mention_refs), x.confidence), reverse=True)[:15]:
        print(f"  - {e.canonical_name!r:40s} type={e.entity_type:15s} mentions={len(e.mention_refs):3d} conf={e.confidence:.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/reextract_entities.py <material_id>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
