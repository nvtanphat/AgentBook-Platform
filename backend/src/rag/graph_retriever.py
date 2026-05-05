from __future__ import annotations

import re
import logging

from beanie import PydanticObjectId

from src.core.config import Settings
from src.models.knowledge_graph import Entity, Event, EvidenceRef, Relation
from src.models.material import Material, get_material_pages_by_material_ids
from src.processing.types import EvidenceBlock
from src.rag.types import GraphPath, RetrievalScope

logger = logging.getLogger(__name__)


class GraphRetriever:
    def __init__(self, settings: Settings, embedder=None) -> None:
        self.settings = settings
        self.embedder = embedder  # Optional: for semantic entity matching

    async def retrieve_paths(self, *, query: str, scope: RetrievalScope, max_hops: int | None = None) -> list[GraphPath]:
        scope.ensure_scoped()
        bounded_hops = min(max_hops or self.settings.graph_max_hops, 2)
        entities = await self._matching_entities(query=query, scope=scope)
        if not entities:
            return []
        seed_ids = {f"entity:{self._slug(entity.canonical_name)}" for entity in entities}
        relations = await self._relations_touching(seed_ids=seed_ids, scope=scope)

        paths: list[GraphPath] = []
        for relation in relations:
            if relation.source_id in seed_ids or relation.target_id in seed_ids:
                evidence = await self._hydrate_evidence_refs(relation.evidence_refs)
                paths.append(
                    GraphPath(
                        path=[relation.source_id, f"relation:{relation.relation_type}", relation.target_id],
                        confidence=relation.confidence,
                        evidence_refs=evidence,
                    )
                )
        if bounded_hops >= 2:
            paths.extend(await self._two_hop_paths(seed_ids=seed_ids, first_hop=relations, scope=scope))

        paths.sort(key=lambda path: path.confidence, reverse=True)
        return paths[: self.settings.graph_top_k]

    _CROSS_MODAL_TYPES = frozenset({"table", "figure", "equation"})
    _MODAL_KEYWORD_RE = re.compile(
        r"\b(?:table|figure|fig|equation|eq|bảng|hình|công\s+thức|biểu\s+đồ)\b",
        re.IGNORECASE,
    )

    async def _matching_entities(self, *, query: str, scope: RetrievalScope) -> list[Entity]:
        """Find entities matching the query using keyword + semantic matching."""
        # 1. Keyword matching (existing)
        keyword_entities = await self._keyword_matching_entities(query=query, scope=scope)

        # 2. Semantic matching (new - optional if embedder available)
        if self.embedder is not None:
            semantic_entities = await self._semantic_matching_entities(query=query, scope=scope, limit=20)
            # Merge and deduplicate
            seen_ids = {str(e.id) for e in keyword_entities}
            keyword_entities.extend(e for e in semantic_entities if str(e.id) not in seen_ids)

        return keyword_entities

    async def _keyword_matching_entities(self, *, query: str, scope: RetrievalScope) -> list[Entity]:
        """Original keyword-based entity matching."""
        terms = [term for term in re.findall(r"[\w\-]{3,}", query, flags=re.UNICODE)[:8]]
        if not terms:
            return []
        or_conditions = [
            {"canonical_name": {"$regex": re.escape(term), "$options": "i"}}
            for term in terms
        ] + [{"aliases": {"$regex": re.escape(term), "$options": "i"}} for term in terms]
        text_entities = await Entity.find(self._scope_query(scope, {"$or": or_conditions})).limit(50).to_list()

        # When the query explicitly asks about a table/figure/equation, also seed from cross-modal nodes
        if self._MODAL_KEYWORD_RE.search(query):
            cm_entities = await Entity.find(
                self._scope_query(scope, {"entity_type": {"$in": list(self._CROSS_MODAL_TYPES)}})
            ).limit(20).to_list()
            seen_ids = {str(e.id) for e in text_entities}
            text_entities.extend(e for e in cm_entities if str(e.id) not in seen_ids)

        return text_entities

    async def _semantic_matching_entities(self, *, query: str, scope: RetrievalScope, limit: int = 20) -> list[Entity]:
        """Semantic entity matching using embeddings (optional enhancement)."""
        try:
            # Embed query
            query_embedding = await self.embedder.embed_query(query)

            # Get all entities in scope (cached or limited)
            all_entities = await Entity.find(
                self._scope_query(scope, {"confidence": {"$gte": self.settings.min_graph_confidence}})
            ).limit(200).to_list()

            if not all_entities:
                return []

            # Compute similarity scores
            entity_scores: list[tuple[Entity, float]] = []
            for entity in all_entities:
                # Embed entity name
                entity_text = f"{entity.canonical_name} {' '.join(entity.aliases)}"
                entity_embedding = await self.embedder.embed_query(entity_text)

                # Cosine similarity
                similarity = self._cosine_similarity(query_embedding, entity_embedding)
                if similarity >= 0.5:  # Threshold for relevance
                    entity_scores.append((entity, similarity))

            # Sort by similarity and return top-k
            entity_scores.sort(key=lambda x: x[1], reverse=True)
            return [entity for entity, _ in entity_scores[:limit]]

        except Exception as exc:
            logger.warning("Semantic entity matching failed", extra={"error": str(exc)})
            return []

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    async def _relations_touching(self, *, seed_ids: set[str], scope: RetrievalScope) -> list[Relation]:
        if not seed_ids:
            return []
        return await Relation.find(
            self._scope_query(
                scope,
                {
                    "confidence": {"$gte": self.settings.min_graph_confidence},
                    "$or": [
                        {"source_id": {"$in": list(seed_ids)}},
                        {"target_id": {"$in": list(seed_ids)}},
                    ],
                },
            )
        ).sort("-confidence").limit(max(self.settings.graph_top_k * 4, 50)).to_list()

    async def _two_hop_paths(self, *, seed_ids: set[str], first_hop: list[Relation], scope: RetrievalScope) -> list[GraphPath]:
        frontier_ids = {
            relation.target_id if relation.source_id in seed_ids else relation.source_id
            for relation in first_hop
        }
        second_hop = await self._relations_touching(seed_ids=frontier_ids - seed_ids, scope=scope)
        adjacency: dict[str, list[Relation]] = {}
        for relation in second_hop:
            adjacency.setdefault(relation.source_id, []).append(relation)
        paths: list[GraphPath] = []
        for first in first_hop:
            if first.source_id not in seed_ids and first.target_id not in seed_ids:
                continue
            for second in adjacency.get(first.target_id, []):
                evidence = await self._hydrate_evidence_refs(first.evidence_refs + second.evidence_refs)
                paths.append(
                    GraphPath(
                        path=[
                            first.source_id,
                            f"relation:{first.relation_type}",
                            first.target_id,
                            f"relation:{second.relation_type}",
                            second.target_id,
                        ],
                        confidence=min(first.confidence, second.confidence),
                        evidence_refs=evidence,
                    )
                )
        return paths

    @staticmethod
    def _scope_query(scope: RetrievalScope, extra: dict) -> dict:
        query: dict = {"owner_id": scope.owner_id, **extra}
        if scope.collection_id:
            query["collection_id"] = PydanticObjectId(scope.collection_id)
        if scope.material_ids:
            material_ids = [PydanticObjectId(material_id) for material_id in scope.material_ids]
            query["evidence_refs.material_id"] = {"$in": material_ids}
        return query

    async def _hydrate_evidence_refs(self, refs: list[EvidenceRef]) -> list[EvidenceBlock]:
        if not refs:
            return []
        unique_material_ids = list({ref.material_id for ref in refs})
        materials_list = await Material.find({"_id": {"$in": unique_material_ids}}).to_list()
        materials_by_id = {m.id: m for m in materials_list}
        pages_by_material_id = await get_material_pages_by_material_ids(materials_list)
        block_lookup: dict[tuple[PydanticObjectId, int | None, str | None], tuple[Material, object, object]] = {}
        for material in materials_list:
            for page in pages_by_material_id.get(str(material.id), []):
                for block in page.blocks:
                    block_lookup[(material.id, page.page_number, block.block_id)] = (material, page, block)

        evidence: list[EvidenceBlock] = []
        for ref in refs:
            found = block_lookup.get((ref.material_id, ref.page, ref.block_id))
            if found is None:
                continue
            material, page, block = found
            evidence.append(
                EvidenceBlock(
                    owner_id=material.owner_id,
                    collection_id=str(material.collection_id),
                    material_id=str(material.id),
                    document_name=material.original_name,
                    page=page.page_number,
                    block_id=block.block_id,
                    block_type=block.block_type,
                    snippet_original=block.content,
                    source_language=block.language,
                    bbox=block.bbox,
                    confidence=block.ocr_confidence,
                    metadata=block.extra,
                )
            )
        return evidence

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"
