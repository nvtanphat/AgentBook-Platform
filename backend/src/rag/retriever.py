from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from threading import RLock

from beanie import PydanticObjectId
from qdrant_client import QdrantClient, models

from src.core.config import Settings
from src.models.chunk import Chunk
from src.models.material import Material, get_material_pages_by_material_ids
from src.processing.types import BBox, EvidenceBlock
from src.rag.embedder import BGEM3Embedder, EmbeddedText
from src.rag.types import RetrievalScope, RetrievedChunk

logger = logging.getLogger(__name__)

# ── Query embedding TTL cache (per-process, not shared across workers) ─────────

_EMBEDDING_CACHE_TTL = 300  # 5 minutes
_EMBEDDING_CACHE_MAX = 1024


@dataclass
class _EmbeddingCacheEntry:
    value: EmbeddedText
    expires_at: float


_embedding_cache: dict[str, _EmbeddingCacheEntry] = {}
_embedding_cache_lock = RLock()


def _get_cached_embedding(text: str) -> EmbeddedText | None:
    with _embedding_cache_lock:
        entry = _embedding_cache.get(text)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            _embedding_cache.pop(text, None)
            return None
        return entry.value


def _set_cached_embedding(text: str, embedding: EmbeddedText) -> None:
    with _embedding_cache_lock:
        if len(_embedding_cache) >= _EMBEDDING_CACHE_MAX:
            # Evict oldest entry
            oldest = min(_embedding_cache, key=lambda k: _embedding_cache[k].expires_at)
            _embedding_cache.pop(oldest, None)
        _embedding_cache[text] = _EmbeddingCacheEntry(
            value=embedding,
            expires_at=time.monotonic() + _EMBEDDING_CACHE_TTL,
        )


class HybridRetriever:
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

    async def retrieve(self, *, query: str, scope: RetrievalScope, limit: int | None = None) -> list[RetrievedChunk]:
        scope.ensure_scoped()
        embedding = _get_cached_embedding(query)
        if embedding is None:
            embeddings = self.embedder.encode([query])
            if not embeddings:
                return []
            embedding = embeddings[0]
            _set_cached_embedding(query, embedding)
        else:
            logger.debug("Embedding cache hit", extra={"query_preview": query[:60]})
        query_filter = self._scope_filter(scope)

        # Dense prefetch is always included.
        prefetches: list[models.Prefetch] = [
            models.Prefetch(
                query=embedding.dense,
                using="dense",
                limit=self.settings.dense_top_k,
                filter=query_filter,
            ),
        ]

        has_sparse_signal = bool(embedding.sparse.indices)
        # Sparse prefetch only when the query produces a non-empty sparse vector.
        if has_sparse_signal:
            prefetches.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=embedding.sparse.indices,
                        values=embedding.sparse.values,
                    ),
                    using="bge_m3_sparse",
                    limit=self.settings.sparse_top_k,
                    filter=query_filter,
                )
            )
        else:
            logger.debug(
                "No sparse signal for query — using dense-only retrieval",
                extra={"owner_id": scope.owner_id, "query_preview": query[:60]},
            )

        result = self.qdrant_client.query_points(
            collection_name=self.settings.qdrant_collection_name,
            prefetch=prefetches,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit or self.settings.rerank_input_k,
            with_payload=True,
        )

        semantic_points = list(result.points)
        lexical_points = self._lexical_fallback_points(query=query, scope_filter=query_filter, limit=limit or self.settings.rerank_input_k)
        if lexical_points:
            points = list(lexical_points)
            existing_ids = {str(point.id) for point in points}
            points.extend(point for point in semantic_points if str(point.id) not in existing_ids)
        else:
            points = semantic_points
        hydrated = await self._hydrate_points(points)
        return hydrated

    def _lexical_fallback_points(
        self,
        *,
        query: str,
        scope_filter: models.Filter,
        limit: int,
    ) -> list[models.ScoredPoint]:
        lexical_filter = models.Filter(
            must=[
                *(scope_filter.must or []),
                models.FieldCondition(key="content_text", match=models.MatchText(text=query)),
            ]
        )
        try:
            records, _ = self.qdrant_client.scroll(
                collection_name=self.settings.qdrant_collection_name,
                scroll_filter=lexical_filter,
                limit=max(limit * 3, limit),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.debug(
                "Qdrant lexical fallback failed",
                extra={"query_preview": query[:60], "error": str(exc), "error_type": type(exc).__name__},
            )
            return []
        scored = [
            models.ScoredPoint(
                id=record.id,
                version=0,
                score=_lexical_score(query, str((record.payload or {}).get("content_text") or "")),
                payload=record.payload or {},
            )
            for record in records
        ]
        scored.sort(key=lambda point: point.score, reverse=True)
        return scored[:limit]

    def _scope_filter(self, scope: RetrievalScope) -> models.Filter:
        must: list[models.FieldCondition] = [
            models.FieldCondition(key="owner_id", match=models.MatchValue(value=scope.owner_id))
        ]
        if scope.collection_id:
            must.append(models.FieldCondition(key="collection_id", match=models.MatchValue(value=scope.collection_id)))
        if scope.material_ids:
            must.append(models.FieldCondition(key="material_id", match=models.MatchAny(any=scope.material_ids)))
        return models.Filter(must=must)

    async def _hydrate_points(self, points: list[models.ScoredPoint]) -> list[RetrievedChunk]:
        """Batch-fetch all Chunks and Materials to avoid N+1 sequential queries."""
        if not points:
            return []

        # Collect chunk IDs from Qdrant payloads
        chunk_id_by_point: dict[str, str] = {}
        for point in points:
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id") or point.id)
            chunk_id_by_point[str(point.id)] = chunk_id

        unique_chunk_ids = list(set(chunk_id_by_point.values()))

        # Batch fetch Chunks
        chunk_oids = []
        for cid in unique_chunk_ids:
            try:
                chunk_oids.append(PydanticObjectId(cid))
            except Exception:
                pass

        chunks_list = await Chunk.find({"_id": {"$in": chunk_oids}}).to_list()
        chunks_by_id: dict[str, Chunk] = {str(c.id): c for c in chunks_list}

        # Batch fetch Materials
        material_ids_needed = list({str(c.material_id) for c in chunks_list})
        material_oids = []
        for mid in material_ids_needed:
            try:
                material_oids.append(PydanticObjectId(mid))
            except Exception:
                pass

        materials_list = await Material.find({"_id": {"$in": material_oids}}).to_list()
        materials_by_id: dict[str, Material] = {str(m.id): m for m in materials_list}
        pages_by_material_id = await get_material_pages_by_material_ids(materials_list)

        # Assemble RetrievedChunk objects
        hydrated: list[RetrievedChunk] = []
        for point in points:
            chunk_id = chunk_id_by_point[str(point.id)]
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            material = materials_by_id.get(str(chunk.material_id))
            if material is None:
                continue
            material_pages = pages_by_material_id.get(str(material.id), [])
            evidence = self._chunk_evidence(chunk=chunk, material=material, material_pages=material_pages)
            hydrated.append(
                RetrievedChunk(
                    chunk_id=str(chunk.id),
                    owner_id=chunk.owner_id,
                    collection_id=str(chunk.collection_id),
                    material_id=str(chunk.material_id),
                    document_name=material.original_name,
                    content=chunk.content,
                    language=chunk.language,
                    modality=chunk.modality,
                    source_block_ids=chunk.source_block_ids,
                    source_pages=chunk.source_pages,
                    bboxes=[block.bbox for block in evidence if block.bbox is not None],
                    evidence=evidence,
                    fused_score=float(point.score),
                )
            )
        return hydrated

    @staticmethod
    def _chunk_evidence(*, chunk: Chunk, material: Material, material_pages) -> list[EvidenceBlock]:
        block_lookup = {
            block.block_id: (page.page_number, block)
            for page in material_pages
            for block in page.blocks
            if block.block_id in chunk.source_block_ids
        }
        evidence: list[EvidenceBlock] = []
        for block_id in chunk.source_block_ids:
            if block_id not in block_lookup:
                logger.warning(
                    "Evidence block missing from material",
                    extra={"block_id": block_id, "material_id": str(material.id)},
                )
                continue
            page_number, block = block_lookup[block_id]
            evidence.append(
                EvidenceBlock(
                    owner_id=material.owner_id,
                    collection_id=str(material.collection_id),
                    material_id=str(material.id),
                    document_name=material.original_name,
                    page=page_number,
                    block_id=block.block_id,
                    block_type=block.block_type,
                    snippet_original=block.content,
                    source_language=block.language,
                    bbox=HybridRetriever._to_processing_bbox(block.bbox),
                    confidence=block.ocr_confidence,
                    metadata=block.extra,
                )
            )
        return evidence

    @staticmethod
    def _to_processing_bbox(bbox) -> BBox | None:
        if bbox is None:
            return None
        return BBox(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2)


def dedupe_retrieved_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    by_id: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        existing = by_id.get(chunk.chunk_id)
        if existing is None or (chunk.rerank_score or chunk.fused_score) > (existing.rerank_score or existing.fused_score):
            by_id[chunk.chunk_id] = chunk
    return list(by_id.values())


def _lexical_score(query: str, content: str) -> float:
    query_lower = query.lower()
    content_lower = content.lower()
    query_terms = {
        term
        for term in re.findall(r"[\w@%]+", query_lower, flags=re.UNICODE)
        if len(term) >= 3
    }
    content_terms = set(re.findall(r"[\w@%]+", content_lower, flags=re.UNICODE))
    overlap = len(query_terms & content_terms)
    score = 1.0 + overlap
    if "precision" in query_lower and "precision" in content_lower:
        score += 3.0
    if re.search(r"\b(cao nhất|highest|max|maximum|lớn nhất)\b", query_lower, re.IGNORECASE) and re.search(r"\b\d+(?:[.,]\d+)?\b", content_lower):
        score += 2.0
    if re.search(r"\b(bảng|table|hình|chart|biểu đồ)\b", query_lower, re.IGNORECASE) and re.search(r"\b(docx|pdf|pptx|xlsx|png|ocr)\b", content_lower, re.IGNORECASE):
        score += 2.0
    return score
