from __future__ import annotations

import logging
import asyncio
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
from src.rag.embedding_provider import VisualEmbeddingProvider
from src.rag.types import RetrievalScope, RetrievedChunk, RetrievedVisualChunk

logger = logging.getLogger(__name__)

# ── Query embedding cache (Redis shared or in-memory fallback) ─────────

_EMBEDDING_CACHE_TTL = 300  # 5 minutes
_EMBEDDING_CACHE_MAX = 1024


@dataclass
class _EmbeddingCacheEntry:
    value: EmbeddedText
    expires_at: float


_embedding_cache: dict[str, _EmbeddingCacheEntry] = {}
_embedding_cache_lock = RLock()
_redis_cache = None


def _get_redis_cache():
    """Lazy-load Redis cache (shared across workers)."""
    global _redis_cache
    if _redis_cache is None:
        try:
            from src.rag.embedding_cache import RedisEmbeddingCache
            from src.core.config import get_settings
            settings = get_settings()
            _redis_cache = RedisEmbeddingCache(
                redis_url=settings.redis_url,
                ttl=_EMBEDDING_CACHE_TTL,
            )
        except Exception as exc:
            logger.info("Redis cache unavailable, using in-memory cache", extra={"error": str(exc)})
            _redis_cache = False  # Mark as unavailable
    return _redis_cache if _redis_cache is not False else None


def _get_cached_embedding(text: str) -> EmbeddedText | None:
    # Try Redis first (shared cache)
    redis_cache = _get_redis_cache()
    if redis_cache:
        result = redis_cache.get(text)
        if result:
            return result

    # Fallback to in-memory cache
    with _embedding_cache_lock:
        entry = _embedding_cache.get(text)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            _embedding_cache.pop(text, None)
            return None
        return entry.value


def _set_cached_embedding(text: str, embedding: EmbeddedText) -> None:
    # Set in Redis (shared cache)
    redis_cache = _get_redis_cache()
    if redis_cache:
        redis_cache.set(text, embedding)

    # Also set in-memory for fast local access
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
        self._embedding_semaphore = asyncio.Semaphore(1)
        self._qdrant_semaphore = asyncio.Semaphore(4)

    async def retrieve(self, *, query: str, scope: RetrievalScope, limit: int | None = None) -> list[RetrievedChunk]:
        scope.ensure_scoped()
        embedding = _get_cached_embedding(query)
        if embedding is None:
            async with self._embedding_semaphore:
                embedding = _get_cached_embedding(query)
                if embedding is None:
                    embeddings = await asyncio.to_thread(self.embedder.encode, [query])
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

        semantic_points = await self._query_with_rrf_async(
            embedding=embedding,
            prefetches=prefetches,
            has_sparse=has_sparse_signal,
            query_filter=query_filter,
            limit=limit or self.settings.rerank_input_k,
        )
        lexical_points = (
            []
            if has_sparse_signal and semantic_points
            else await self._lexical_fallback_points_async(query=query, scope_filter=query_filter, limit=limit or self.settings.rerank_input_k)
        )
        if lexical_points:
            points = list(lexical_points)
            existing_ids = {str(point.id) for point in points}
            points.extend(point for point in semantic_points if str(point.id) not in existing_ids)
        else:
            points = semantic_points
        hydrated = await self._hydrate_points(points)
        # Keep short but meaningful chunks. Ingestion already handles most noise;
        # a fixed length threshold here can drop valid atomic facts.
        hydrated = [c for c in hydrated if _has_retrievable_content(c.content)]
        return self._diversify(hydrated, max_per_doc=self.settings.max_chunks_per_doc)

    async def retrieve_fast(
        self, *, query: str, scope: RetrievalScope, limit: int | None = None
    ) -> list[RetrievedChunk]:
        """Phase B fast path: single hybrid retrieval (RRF dense+sparse) on the
        original query, returning fused-score-ordered chunks. The caller uses
        `fast_path_eligible` to decide whether the resulting bundle is strong
        enough to also skip graph retrieval, multi-query expansion AND the
        cross-encoder reranker — those are the real latency hogs.

        We use the *fused* score (already RRF-normalised in [0,1]) rather than
        raw dense cosine because the stored dense vectors in this deployment
        carry sparse signal only (dense vectors are zero post-indexing — a known
        data issue that would require reindexing to fix).
        """
        return await self.retrieve(query=query, scope=scope, limit=limit)

    @staticmethod
    def fast_path_eligible(
        *, chunks: list[RetrievedChunk], settings: Settings
    ) -> bool:
        """Decide if the quick retrieve() result is strong enough to skip the
        downstream rerank + graph + multi-query stages.

        Signal: fused RRF score (always populated, range [0,1]). Two gates:
          - top-1 fused score ≥ adaptive_dense_skip_threshold
          - at least `strong_hits_required` chunks fused ≥ strong_hit_min_score
        """
        if not settings.adaptive_retrieval_enabled or not chunks:
            return False
        scores = [c.fused_score for c in chunks if c.fused_score is not None]
        if not scores:
            return False
        top = max(scores)
        strong_count = sum(1 for s in scores if s >= settings.adaptive_strong_hit_min_score)
        return (
            top >= settings.adaptive_dense_skip_threshold
            and strong_count >= settings.adaptive_strong_hits_required
        )

    async def _lexical_fallback_points_async(
        self,
        *,
        query: str,
        scope_filter: models.Filter,
        limit: int,
    ) -> list[models.ScoredPoint]:
        async with self._qdrant_semaphore:
            return await asyncio.to_thread(
                self._lexical_fallback_points,
                query=query,
                scope_filter=scope_filter,
                limit=limit,
            )

    async def _query_with_rrf_async(
        self,
        *,
        embedding,
        prefetches: list[models.Prefetch],
        has_sparse: bool,
        query_filter: models.Filter,
        limit: int,
    ) -> list[models.ScoredPoint]:
        async with self._qdrant_semaphore:
            return await asyncio.to_thread(
                self._query_with_rrf,
                embedding=embedding,
                prefetches=prefetches,
                has_sparse=has_sparse,
                query_filter=query_filter,
                limit=limit,
            )

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
                limit=max(limit * self.settings.lexical_fallback_multiplier, limit),
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

    def _query_with_rrf(
        self,
        *,
        embedding,
        prefetches: list[models.Prefetch],
        has_sparse: bool,
        query_filter: models.Filter,
        limit: int,
    ) -> list[models.ScoredPoint]:
        """Try server-side prefetch+RRF; fall back to Python RRF for embedded client."""
        try:
            result = self.qdrant_client.query_points(
                collection_name=self.settings.qdrant_collection_name,
                prefetch=prefetches,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
            return list(result.points)
        except Exception as exc:
            logger.debug(
                "query_points prefetch/RRF not supported (embedded client), using Python RRF fallback",
                extra={"error": str(exc)},
            )

        # Embedded client fallback: run dense + sparse searches separately, fuse in Python.
        dense_hits: list[models.ScoredPoint] = self.qdrant_client.search(
            collection_name=self.settings.qdrant_collection_name,
            query_vector=models.NamedVector(name="dense", vector=embedding.dense),
            query_filter=query_filter,
            limit=self.settings.dense_top_k,
            with_payload=True,
        )
        results_list: list[list[models.ScoredPoint]] = [dense_hits]

        if has_sparse:
            try:
                sparse_hits: list[models.ScoredPoint] = self.qdrant_client.search(
                    collection_name=self.settings.qdrant_collection_name,
                    query_vector=models.NamedSparseVector(
                        name="bge_m3_sparse",
                        vector=models.SparseVector(
                            indices=embedding.sparse.indices,
                            values=embedding.sparse.values,
                        ),
                    ),
                    query_filter=query_filter,
                    limit=self.settings.sparse_top_k,
                    with_payload=True,
                )
                results_list.append(sparse_hits)
            except Exception as exc2:
                logger.debug("Sparse search fallback failed", extra={"error": str(exc2)})

        return self._rrf_fuse(results_list, limit=limit, k=self.settings.rrf_k)

    def _rrf_fuse(
        self,
        results_list: list[list[models.ScoredPoint]],
        limit: int,
        k: int = 60,
    ) -> list[models.ScoredPoint]:
        """Reciprocal Rank Fusion across multiple ranked lists."""
        scores: dict[str | int, float] = {}
        by_id: dict[str | int, models.ScoredPoint] = {}
        for results in results_list:
            for rank, point in enumerate(results):
                scores[point.id] = scores.get(point.id, 0.0) + 1.0 / (k + rank + 1)
                by_id[point.id] = point
        ranked = sorted(by_id.values(), key=lambda p: scores[p.id], reverse=True)
        for point in ranked:
            point.score = scores[point.id]
        return ranked[:limit]

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


    @staticmethod
    def _diversify(chunks: list[RetrievedChunk], max_per_doc: int) -> list[RetrievedChunk]:
        """Cap chunks per document to avoid one broad chunk dominating results.

        Bypassed when results come from a single document — capping there only
        starves the reranker of candidates without diversification benefit, and
        was observed to drop genuinely-relevant chunks ranked below rank
        `max_per_doc` (e.g. WAPE rationale at rank #18 cut by a cap of 3).
        """
        if max_per_doc <= 0:
            return chunks
        unique_docs = {chunk.document_name for chunk in chunks}
        if len(unique_docs) <= 1:
            return chunks
        counts: dict[str, int] = {}
        result: list[RetrievedChunk] = []
        for chunk in chunks:
            doc = chunk.document_name
            if counts.get(doc, 0) < max_per_doc:
                result.append(chunk)
                counts[doc] = counts.get(doc, 0) + 1
        return result


    async def retrieve_visual(
        self,
        *,
        query: str,
        scope: RetrievalScope,
        visual_provider: VisualEmbeddingProvider,
        limit: int | None = None,
    ) -> list[RetrievedVisualChunk]:
        """Text query → SigLIP cross-modal → ranked figures from the visual collection.

        Evidence-trace fields (owner_id, collection_id, material_id, page, block_id)
        are read directly from the Qdrant payload; no MongoDB hydration is needed.
        """
        scope.ensure_scoped()
        query_vec = await asyncio.to_thread(visual_provider.embed_query, query)
        return await self.retrieve_visual_with_vector(
            vector=query_vec, scope=scope, limit=limit,
        )

    async def retrieve_visual_with_vector(
        self,
        *,
        vector: list[float],
        scope: RetrievalScope,
        limit: int | None = None,
    ) -> list[RetrievedVisualChunk]:
        """Image-as-query path: caller already has a SigLIP vector (e.g. from an upload)."""
        scope.ensure_scoped()
        query_filter = self._scope_filter(scope)
        effective_limit = limit or self.settings.final_top_k

        try:
            async with self._qdrant_semaphore:
                raw_points = await asyncio.to_thread(
                    self.qdrant_client.search,
                    collection_name=self.settings.qdrant_visual_collection_name,
                    query_vector=models.NamedVector(name="visual_dense", vector=vector),
                    query_filter=query_filter,
                    limit=effective_limit,
                    with_payload=True,
                )
        except Exception as exc:
            # Visual collection often doesn't exist until figures have been
            # indexed for at least one material. Demote 404s to debug so the
            # warning stream stays clean for the common cold-start case.
            level = logging.DEBUG if "Not Found" in str(exc) or "404" in str(exc) else logging.WARNING
            logger.log(
                level,
                "Visual retrieval skipped — returning empty result",
                extra={
                    "owner_id": scope.owner_id,
                    "collection_id": scope.collection_id,
                    "error": str(exc),
                },
            )
            return []

        chunks: list[RetrievedVisualChunk] = []
        for point in raw_points:
            payload = point.payload or {}
            bbox: BBox | None = None
            if all(k in payload for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")):
                bbox = BBox(
                    x1=payload["bbox_x1"],
                    y1=payload["bbox_y1"],
                    x2=payload["bbox_x2"],
                    y2=payload["bbox_y2"],
                )
            chunks.append(
                RetrievedVisualChunk(
                    point_id=str(point.id),
                    owner_id=payload.get("owner_id", ""),
                    collection_id=payload.get("collection_id", ""),
                    material_id=payload.get("material_id", ""),
                    document_name=payload.get("document_name", ""),
                    page=int(payload.get("page", 0)),
                    block_id=payload.get("block_id", ""),
                    block_type=payload.get("block_type", "figure"),
                    caption=payload.get("caption", ""),
                    source_language=payload.get("source_language", "unknown"),
                    bbox=bbox,
                    image_path=payload.get("image_path"),
                    score=float(point.score),
                )
            )
        return chunks


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
    return 1.0 + overlap


def _has_retrievable_content(content: str | None) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    return bool(re.search(r"[\wÀ-ɏḀ-ỿ]{2,}", text, flags=re.UNICODE))
