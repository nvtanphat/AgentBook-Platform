from __future__ import annotations

import logging
import asyncio
from datetime import UTC, datetime
from itertools import islice
from typing import Awaitable, Callable, Iterable, TypeVar
from uuid import NAMESPACE_URL, uuid5

from beanie import PydanticObjectId
from qdrant_client import QdrantClient, models

from pathlib import Path

from src.core.config import Settings
from src.models.chunk import Chunk
from src.models.knowledge_graph import Entity, Event, EvidenceRef, Relation
from src.processing.types import ExtractedEntity, ExtractedEvent, ExtractedRelation, TextChunk
from src.rag.embedder import BGEM3Embedder, EmbeddedText
from src.rag.embedding_provider import VisualEmbeddingProvider
from src.rag.types import FigureIndexItem

logger = logging.getLogger(__name__)
T = TypeVar("T")


class QdrantMongoIndexer:
    def __init__(
        self,
        *,
        settings: Settings,
        qdrant_client: QdrantClient,
        embedder: BGEM3Embedder | None = None,
    ) -> None:
        self.settings = settings
        self.qdrant_client = qdrant_client
        self.embedder = embedder or BGEM3Embedder(settings)
        self._embedding_semaphore = asyncio.Semaphore(1)
        self._qdrant_semaphore = asyncio.Semaphore(4)

    async def index(
        self,
        *,
        chunks: list[TextChunk],
        entities: list[ExtractedEntity],
        events: list[ExtractedEvent],
        relations: list[ExtractedRelation],
        should_continue: Callable[[], Awaitable[bool]] | None = None,
    ) -> list[Chunk]:
        material_ids = _collect_material_ids(chunks=chunks, entities=entities, events=events, relations=relations)
        if material_ids:
            await self._ensure_collection_async()
            await self._cleanup_existing_material_artifacts(material_ids)
        stored_chunks: list[Chunk] = []
        if chunks:
            logger.info(
                "Indexing chunks",
                extra={
                    "chunk_count": len(chunks),
                    "batch_size": self.settings.index_batch_size,
                    "collection": self.settings.qdrant_collection_name,
                },
            )
            await self._ensure_collection_async()
            for batch_number, chunk_batch in enumerate(
                batched(chunks, max(1, self.settings.index_batch_size)),
                start=1,
            ):
                if should_continue is not None and not await should_continue():
                    raise LookupError("Indexing aborted because the material no longer exists")
                embed_texts = [chunk.contextualized_content or chunk.content for chunk in chunk_batch]
                async with self._embedding_semaphore:
                    embeddings = await asyncio.to_thread(self.embedder.encode, embed_texts)
                if should_continue is not None and not await should_continue():
                    raise LookupError("Indexing aborted because the material no longer exists")
                batch_stored_chunks = await self._store_chunks(chunk_batch)
                if should_continue is not None and not await should_continue():
                    raise LookupError("Indexing aborted because the material no longer exists")
                await self._upsert_qdrant_async(batch_stored_chunks, chunk_batch, embeddings)
                stored_chunks.extend(batch_stored_chunks)
                logger.info(
                    "Indexed chunk batch",
                    extra={
                        "batch_number": batch_number,
                        "batch_chunk_count": len(chunk_batch),
                        "indexed_chunk_count": len(stored_chunks),
                        "total_chunk_count": len(chunks),
                    },
                )

        if should_continue is not None and not await should_continue():
            raise LookupError("Indexing aborted because the material no longer exists")
        await self._store_graph(entities=entities, events=events, relations=relations)
        return stored_chunks

    async def _ensure_collection_async(self) -> None:
        async with self._qdrant_semaphore:
            await asyncio.to_thread(self._ensure_collection)

    async def _upsert_qdrant_async(
        self,
        stored_chunks: list[Chunk],
        chunks: list[TextChunk],
        embeddings: list[EmbeddedText],
    ) -> None:
        async with self._qdrant_semaphore:
            await asyncio.to_thread(self._upsert_qdrant, stored_chunks, chunks, embeddings)

    async def _cleanup_existing_material_artifacts(self, material_ids: set[str]) -> None:
        material_oids = []
        for material_id in material_ids:
            try:
                material_oids.append(PydanticObjectId(material_id))
            except Exception:
                logger.warning("Skipping invalid material_id during cleanup", extra={"material_id": material_id})
                continue
        if material_oids:
            await Chunk.find({"material_id": {"$in": material_oids}}).delete()
            await Entity.find({"mention_refs.material_id": {"$in": material_oids}}).delete()
            await Event.find({"evidence_refs.material_id": {"$in": material_oids}}).delete()
            await Relation.find({"evidence_refs.material_id": {"$in": material_oids}}).delete()
        try:
            async with self._qdrant_semaphore:
                await asyncio.to_thread(
                    self.qdrant_client.delete,
                    collection_name=self.settings.qdrant_collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="material_id",
                                    match=models.MatchAny(any=list(material_ids)),
                                )
                            ]
                        )
                    ),
                    wait=True,
                )
        except Exception as exc:
            logger.debug(
                "Qdrant cleanup before re-index failed or collection is empty",
                extra={"material_ids": sorted(material_ids), "error": str(exc), "error_type": type(exc).__name__},
            )

        # Also clean up any existing visual points for these materials.
        try:
            async with self._qdrant_semaphore:
                await asyncio.to_thread(
                    self.qdrant_client.delete,
                    collection_name=self.settings.qdrant_visual_collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="material_id",
                                    match=models.MatchAny(any=list(material_ids)),
                                )
                            ]
                        )
                    ),
                    wait=True,
                )
        except Exception as exc:
            logger.debug(
                "Qdrant visual cleanup skipped (collection may not exist yet)",
                extra={"material_ids": sorted(material_ids), "error": str(exc), "error_type": type(exc).__name__},
            )

    def _ensure_collection(self) -> None:
        collection_name = self.settings.qdrant_collection_name
        if not self.qdrant_client.collection_exists(collection_name):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(size=self.settings.embedding_dense_size, distance=models.Distance.COSINE),
                },
                sparse_vectors_config={
                    "bge_m3_sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
                },
            )
        self._ensure_payload_indexes(collection_name)

    def _ensure_payload_indexes(self, collection_name: str) -> None:
        for field_name in ["owner_id", "collection_id", "material_id", "language", "modality"]:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as exc:
                logger.debug(
                    "Qdrant payload index already exists or could not be created",
                    extra={"collection": collection_name, "field": field_name, "error": str(exc)},
                )
        try:
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="content_text",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.MULTILINGUAL,
                    lowercase=True,
                ),
                wait=True,
            )
        except Exception as exc:
            logger.debug(
                "Qdrant text payload index already exists or could not be created",
                extra={"collection": collection_name, "field": "content_text", "error": str(exc)},
            )

    async def _store_chunks(self, chunks: list[TextChunk]) -> list[Chunk]:
        # Pre-assign IDs so they are available immediately after insert_many
        # (Beanie 1.x insert_many does not populate .id on the document objects)
        now = datetime.now(UTC)
        documents = [
            Chunk(
                id=PydanticObjectId(),
                owner_id=chunk.owner_id,
                material_id=PydanticObjectId(chunk.material_id),
                collection_id=PydanticObjectId(chunk.collection_id),
                content=chunk.content,
                language=chunk.language,
                modality=chunk.modality,
                source_block_ids=chunk.source_block_ids,
                source_pages=chunk.source_pages,
                token_count=chunk.token_count,
                embedding_model=chunk.embedding_model,
                embedding_version=chunk.embedding_version,
                chunk_strategy=chunk.chunk_strategy,
                chunker_version=chunk.chunker_version,
                parser_version=chunk.parser_version,
                indexed_at=now,
            )
            for chunk in chunks
        ]
        await Chunk.insert_many(documents)
        return documents

    def _upsert_qdrant(
        self,
        stored_chunks: list[Chunk],
        chunks: list[TextChunk],
        embeddings: list[EmbeddedText],
    ) -> None:
        points: list[models.PointStruct] = []
        for stored, chunk, embedding in zip(stored_chunks, chunks, embeddings, strict=True):
            point_id = str(uuid5(NAMESPACE_URL, f"prism:chunk:{stored.id}"))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": embedding.dense,
                        "bge_m3_sparse": models.SparseVector(
                            indices=embedding.sparse.indices,
                            values=embedding.sparse.values,
                        ),
                    },
                    payload={
                        "owner_id": chunk.owner_id,
                        "collection_id": chunk.collection_id,
                        "material_id": chunk.material_id,
                        "chunk_id": str(stored.id),
                        "content_text": chunk.content,
                        "language": chunk.language,
                        "modality": chunk.modality,
                        "subject": None,
                        "topic": None,
                        "page_numbers": chunk.source_pages,
                        "block_types": [evidence.block_type for evidence in chunk.evidence],
                        "block_kinds": _collect_evidence_field(chunk, "block_kind"),
                        "sheet_names": _collect_evidence_field(chunk, "sheet_name"),
                        "row_indices": _collect_evidence_field(chunk, "row_index"),
                        "source_block_ids": chunk.source_block_ids,
                        "token_count": chunk.token_count,
                        "parser_version": chunk.parser_version,
                        "chunker_version": chunk.chunker_version,
                        "embedding_model": chunk.embedding_model,
                        "embedding_version": chunk.embedding_version,
                        "index_version": chunk.index_version,
                    },
                )
            )
        self.qdrant_client.upsert(collection_name=self.settings.qdrant_collection_name, points=points, wait=True)

    # ── Visual embedding ───────────────────────────────────────────────────────

    async def index_visual(
        self,
        *,
        figure_items: list[FigureIndexItem],
        visual_provider: VisualEmbeddingProvider,
    ) -> None:
        """Embed figure images with SigLIP and upsert into the visual collection.

        Items without an image_path are skipped with a debug log; they remain
        retrievable via the text collection through their caption chunk.
        Graceful: any single-batch failure is logged and skipped, pipeline continues.
        """
        if not figure_items:
            return

        visual_dim = visual_provider.dense_dimension
        await asyncio.to_thread(self._ensure_visual_collection, visual_dim)

        upsert_batch_size = max(1, self.settings.visual_embedding_batch_size)
        upserted = 0

        for batch in batched(figure_items, upsert_batch_size):
            image_items = [item for item in batch if item.image_path]
            for item in batch:
                if not item.image_path:
                    logger.debug(
                        "Visual embedding skipped — no image path",
                        extra={"block_id": item.block_id, "material_id": item.material_id},
                    )

            if not image_items:
                continue

            image_paths = [Path(item.image_path) for item in image_items]
            try:
                async with self._embedding_semaphore:
                    vecs = await asyncio.to_thread(visual_provider.embed_images, image_paths)
            except Exception as exc:
                logger.warning(
                    "SigLIP embed_images failed for batch — skipping",
                    extra={"batch_size": len(image_items), "error": str(exc)},
                )
                continue

            points: list[models.PointStruct] = []
            for item, vec in zip(image_items, vecs):
                point_id = str(
                    uuid5(NAMESPACE_URL, f"prism:visual:{item.material_id}:{item.block_id}")
                )
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={"visual_dense": vec},
                        payload=_build_visual_payload(item),
                    )
                )

            try:
                async with self._qdrant_semaphore:
                    await asyncio.to_thread(
                        self.qdrant_client.upsert,
                        collection_name=self.settings.qdrant_visual_collection_name,
                        points=points,
                        wait=True,
                    )
                upserted += len(points)
            except Exception as exc:
                logger.warning(
                    "Qdrant visual upsert failed for batch — skipping",
                    extra={"batch_size": len(points), "error": str(exc)},
                )

        logger.info(
            "Visual indexing complete",
            extra={
                "total_figures": len(figure_items),
                "upserted": upserted,
                "collection": self.settings.qdrant_visual_collection_name,
            },
        )

    def _ensure_visual_collection(self, dense_dim: int) -> None:
        collection_name = self.settings.qdrant_visual_collection_name
        if not self.qdrant_client.collection_exists(collection_name):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "visual_dense": models.VectorParams(
                        size=dense_dim, distance=models.Distance.COSINE
                    )
                },
            )
        self._ensure_visual_payload_indexes(collection_name)

    def _ensure_visual_payload_indexes(self, collection_name: str) -> None:
        for field_name in ["owner_id", "collection_id", "material_id", "block_type"]:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as exc:
                logger.debug(
                    "Qdrant visual payload index already exists or could not be created",
                    extra={"collection": collection_name, "field": field_name, "error": str(exc)},
                )

    # ── Graph storage ──────────────────────────────────────────────────────────

    async def _store_graph(
        self,
        *,
        entities: list[ExtractedEntity],
        events: list[ExtractedEvent],
        relations: list[ExtractedRelation],
    ) -> None:
        entity_docs = [
            Entity(
                owner_id=entity.mention_refs[0].owner_id,
                collection_id=PydanticObjectId(entity.mention_refs[0].collection_id),
                canonical_name=entity.canonical_name,
                aliases=entity.aliases,
                entity_type=entity.entity_type,
                mention_refs=[self._to_ref(ref) for ref in entity.mention_refs],
                normalized_value=entity.normalized_value,
                confidence=entity.confidence,
            )
            for entity in entities
            if entity.mention_refs and entity.confidence >= self.settings.min_graph_confidence
        ]
        if entity_docs:
            await Entity.insert_many(entity_docs)

        event_docs = [
            Event(
                owner_id=event.evidence_refs[0].owner_id,
                collection_id=PydanticObjectId(event.evidence_refs[0].collection_id),
                event_name=event.event_name,
                event_time=None,
                participants=event.participants,
                evidence_refs=[self._to_ref(ref) for ref in event.evidence_refs],
                temporal_status=event.temporal_status,
            )
            for event in events
            if event.evidence_refs and event.confidence >= self.settings.min_graph_confidence
        ]
        if event_docs:
            await Event.insert_many(event_docs)

        relation_docs = [
            Relation(
                owner_id=relation.evidence_refs[0].owner_id,
                collection_id=PydanticObjectId(relation.evidence_refs[0].collection_id),
                source_id=relation.source_id,
                target_id=relation.target_id,
                relation_type=relation.relation_type,
                evidence_refs=[self._to_ref(ref) for ref in relation.evidence_refs],
                confidence=relation.confidence,
                is_conflicting=relation.is_conflicting,
            )
            for relation in relations
            if relation.evidence_refs and relation.confidence >= self.settings.min_graph_confidence
        ]
        if relation_docs:
            await Relation.insert_many(relation_docs)

        await self._wire_chunk_graph_links(entity_docs=entity_docs, relation_docs=relation_docs)

    async def _wire_chunk_graph_links(
        self,
        *,
        entity_docs: list[Entity],
        relation_docs: list[Relation],
    ) -> None:
        """Back-fill Chunk.entity_ids / Chunk.relation_ids and Entity/Relation chunk_ids.

        Matching rule: a Chunk references an entity when the chunk's source_block_ids
        overlaps with any block_id in the entity's mention_refs for the same material.
        """
        # ── collect (material_id_str, block_id) → [entity_id_str] ────────────
        block_to_entity_ids: dict[tuple[str, str], list[str]] = {}
        for entity_doc in entity_docs:
            if entity_doc.id is None:
                continue
            eid = str(entity_doc.id)
            for ref in entity_doc.mention_refs:
                if ref.block_id:
                    block_to_entity_ids.setdefault((str(ref.material_id), ref.block_id), []).append(eid)

        block_to_relation_ids: dict[tuple[str, str], list[str]] = {}
        for relation_doc in relation_docs:
            if relation_doc.id is None:
                continue
            rid = str(relation_doc.id)
            for ref in relation_doc.evidence_refs:
                if ref.block_id:
                    block_to_relation_ids.setdefault((str(ref.material_id), ref.block_id), []).append(rid)

        all_material_ids: set[str] = {mid for mid, _ in block_to_entity_ids} | {mid for mid, _ in block_to_relation_ids}
        if not all_material_ids:
            return

        try:
            material_oids = [PydanticObjectId(mid) for mid in all_material_ids]
            candidate_chunks = await Chunk.find({"material_id": {"$in": material_oids}}).to_list()
        except Exception as exc:
            logger.warning("Graph-chunk wire-up: chunk lookup failed", extra={"error": str(exc)})
            return

        # ── build reverse maps ────────────────────────────────────────────────
        # entity_id → list[chunk_id]
        entity_chunk_map: dict[str, list[str]] = {str(e.id): [] for e in entity_docs if e.id}
        # relation_id → list[chunk_id]
        relation_chunk_map: dict[str, list[str]] = {str(r.id): [] for r in relation_docs if r.id}

        chunks_to_update: list[tuple[Chunk, list[str], list[str]]] = []

        for chunk in candidate_chunks:
            mat_str = str(chunk.material_id)
            chunk_id = str(chunk.id)
            new_eids: list[str] = []
            new_rids: list[str] = []
            for blk_id in chunk.source_block_ids:
                for eid in block_to_entity_ids.get((mat_str, blk_id), []):
                    if eid not in new_eids:
                        new_eids.append(eid)
                    entity_chunk_map.setdefault(eid, [])
                    if chunk_id not in entity_chunk_map[eid]:
                        entity_chunk_map[eid].append(chunk_id)
                for rid in block_to_relation_ids.get((mat_str, blk_id), []):
                    if rid not in new_rids:
                        new_rids.append(rid)
                    relation_chunk_map.setdefault(rid, [])
                    if chunk_id not in relation_chunk_map[rid]:
                        relation_chunk_map[rid].append(chunk_id)
            if new_eids or new_rids:
                chunks_to_update.append((chunk, new_eids, new_rids))

        # ── persist: update Chunk docs ────────────────────────────────────────
        for chunk, new_eids, new_rids in chunks_to_update:
            try:
                update_doc: dict = {}
                if new_eids:
                    update_doc["$addToSet"] = {"entity_ids": {"$each": new_eids}}
                if new_rids:
                    update_doc.setdefault("$addToSet", {})["relation_ids"] = {"$each": new_rids}
                if update_doc:
                    await Chunk.find({"_id": chunk.id}).update_many(update_doc)
            except Exception as exc:
                logger.debug("Graph-chunk wire-up: chunk update failed", extra={"chunk_id": str(chunk.id), "error": str(exc)})

        # ── persist: update Entity chunk_ids ─────────────────────────────────
        for entity_doc in entity_docs:
            if entity_doc.id is None:
                continue
            new_chunk_ids = entity_chunk_map.get(str(entity_doc.id), [])
            if new_chunk_ids:
                try:
                    await Entity.find({"_id": entity_doc.id}).update_many(
                        {"$set": {"chunk_ids": new_chunk_ids}}
                    )
                    entity_doc.chunk_ids = new_chunk_ids
                except Exception as exc:
                    logger.debug("Graph-chunk wire-up: entity update failed", extra={"entity_id": str(entity_doc.id), "error": str(exc)})

        # ── persist: update Relation evidence_chunk_ids ───────────────────────
        for relation_doc in relation_docs:
            if relation_doc.id is None:
                continue
            new_chunk_ids = relation_chunk_map.get(str(relation_doc.id), [])
            if new_chunk_ids:
                try:
                    await Relation.find({"_id": relation_doc.id}).update_many(
                        {"$set": {"evidence_chunk_ids": new_chunk_ids}}
                    )
                    relation_doc.evidence_chunk_ids = new_chunk_ids
                except Exception as exc:
                    logger.debug("Graph-chunk wire-up: relation update failed", extra={"relation_id": str(relation_doc.id), "error": str(exc)})

    @staticmethod
    def _to_ref(ref) -> EvidenceRef:
        return EvidenceRef(material_id=PydanticObjectId(ref.material_id), page=ref.page, block_id=ref.block_id)


def batched(items: Iterable[T], size: int) -> Iterable[list[T]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def _collect_evidence_field(chunk: TextChunk, field: str) -> list:
    seen: list = []
    for evidence in chunk.evidence:
        value = evidence.metadata.get(field) if evidence.metadata else None
        if value is None or value in seen:
            continue
        seen.append(value)
    return seen


def _build_visual_payload(item: FigureIndexItem) -> dict:
    payload: dict = {
        "owner_id": item.owner_id,
        "collection_id": item.collection_id,
        "material_id": item.material_id,
        "document_name": item.document_name,
        "page": item.page,
        "block_id": item.block_id,
        "block_type": item.block_type,
        "caption": item.caption,
        "source_language": item.source_language,
        "image_path": item.image_path,
    }
    if item.bbox is not None:
        payload["bbox_x1"] = item.bbox.x1
        payload["bbox_y1"] = item.bbox.y1
        payload["bbox_x2"] = item.bbox.x2
        payload["bbox_y2"] = item.bbox.y2
    return payload


def _collect_material_ids(
    *,
    chunks: list[TextChunk],
    entities: list[ExtractedEntity],
    events: list[ExtractedEvent],
    relations: list[ExtractedRelation],
) -> set[str]:
    material_ids = {chunk.material_id for chunk in chunks}
    for entity in entities:
        material_ids.update(ref.material_id for ref in entity.mention_refs)
    for event in events:
        material_ids.update(ref.material_id for ref in event.evidence_refs)
    for relation in relations:
        material_ids.update(ref.material_id for ref in relation.evidence_refs)
    return {material_id for material_id in material_ids if material_id}
