from __future__ import annotations

import logging
import asyncio
from datetime import UTC, datetime
from itertools import islice
from typing import Awaitable, Callable, Iterable, TypeVar
from uuid import NAMESPACE_URL, uuid5

from beanie import PydanticObjectId
from qdrant_client import QdrantClient, models

from src.core.config import Settings
from src.models.chunk import Chunk
from src.models.knowledge_graph import Entity, Event, EvidenceRef, Relation
from src.processing.types import ExtractedEntity, ExtractedEvent, ExtractedRelation, TextChunk
from src.rag.embedder import BGEM3Embedder, EmbeddedText

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
            self._ensure_collection()
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
            self._ensure_collection()
            for batch_number, chunk_batch in enumerate(
                batched(chunks, max(1, self.settings.index_batch_size)),
                start=1,
            ):
                if should_continue is not None and not await should_continue():
                    raise LookupError("Indexing aborted because the material no longer exists")
                embed_texts = [chunk.contextualized_content or chunk.content for chunk in chunk_batch]
                embeddings = await asyncio.to_thread(self.embedder.encode, embed_texts)
                if should_continue is not None and not await should_continue():
                    raise LookupError("Indexing aborted because the material no longer exists")
                batch_stored_chunks = await self._store_chunks(chunk_batch)
                if should_continue is not None and not await should_continue():
                    raise LookupError("Indexing aborted because the material no longer exists")
                self._upsert_qdrant(batch_stored_chunks, chunk_batch, embeddings)
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

    async def _cleanup_existing_material_artifacts(self, material_ids: set[str]) -> None:
        material_oids = []
        for material_id in material_ids:
            try:
                material_oids.append(PydanticObjectId(material_id))
            except Exception:
                continue
        if material_oids:
            await Chunk.find({"material_id": {"$in": material_oids}}).delete()
            await Entity.find({"mention_refs.material_id": {"$in": material_oids}}).delete()
            await Event.find({"evidence_refs.material_id": {"$in": material_oids}}).delete()
            await Relation.find({"evidence_refs.material_id": {"$in": material_oids}}).delete()
        try:
            self.qdrant_client.delete(
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
