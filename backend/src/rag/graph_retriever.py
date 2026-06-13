from __future__ import annotations

import asyncio
import re
import logging

from beanie import PydanticObjectId

from src.core.config import Settings
from src.models.knowledge_graph import Entity, EvidenceRef, Relation
from src.models.material import Material, get_material_pages_by_material_ids
from src.processing.types import BBox, EvidenceBlock
from src.rag.types import GraphPath, RetrievalScope

logger = logging.getLogger(__name__)


class GraphRetriever:
    def __init__(self, settings: Settings, embedder=None) -> None:
        self.settings = settings
        self.embedder = embedder  # Optional: for semantic entity matching

    async def retrieve_subgraph(self, query: str, scope: RetrievalScope, top_k: int = 5) -> list[GraphPath]:
        """Text-index entity matching → 1-hop relation expansion → GraphPath with chunk refs.

        Uses MongoDB $text search on canonical_name + aliases (requires the
        entities_text_search index to exist). Falls back to regex matching when
        $text is unavailable or returns no results.
        All queries are scoped to owner_id + collection_id.
        """
        scope.ensure_scoped()
        collection_oid = PydanticObjectId(scope.collection_id) if scope.collection_id else None

        # ── 1. Entity lookup via $text (primary) + regex fallback ─────────────
        scope_filter: dict = {"owner_id": scope.owner_id}
        if collection_oid is not None:
            scope_filter["collection_id"] = collection_oid

        entities: list[Entity] = []
        try:
            text_filter = {**scope_filter, "$text": {"$search": query}}
            entities = await Entity.find(text_filter).limit(top_k * 4).to_list()
        except Exception:
            pass  # text index may not exist yet on this deployment

        if not entities:
            entities = await self._keyword_matching_entities(query=query, scope=scope)

        if not entities:
            return []

        entity_ids_str = {f"entity:{self._slug(e.canonical_name)}" for e in entities}
        entity_id_map: dict[str, Entity] = {f"entity:{self._slug(e.canonical_name)}": e for e in entities}

        # ── 2. 1-hop relation expansion ───────────────────────────────────────
        relations = await Relation.find(
            {
                **scope_filter,
                "confidence": {"$gte": self.settings.min_graph_confidence},
                "$or": [
                    {"source_id": {"$in": list(entity_ids_str)}},
                    {"target_id": {"$in": list(entity_ids_str)}},
                ],
            }
        ).sort("-confidence").limit(top_k * 6).to_list()

        # ── 3. Collect chunk_ids + build paths ────────────────────────────────
        paths: list[GraphPath] = []
        for relation in relations:
            chunk_ids: list[str] = []
            # From matched entities at either endpoint
            for eid_str in (relation.source_id, relation.target_id):
                entity = entity_id_map.get(eid_str)
                if entity and entity.chunk_ids:
                    chunk_ids.extend(cid for cid in entity.chunk_ids if cid not in chunk_ids)
            # From the relation itself
            for cid in relation.evidence_chunk_ids:
                if cid not in chunk_ids:
                    chunk_ids.append(cid)

            evidence = await self._hydrate_evidence_refs(relation.evidence_refs)
            paths.append(
                GraphPath(
                    path=[relation.source_id, f"relation:{relation.relation_type}", relation.target_id],
                    confidence=relation.confidence,
                    evidence_refs=evidence,
                    source_chunk_ids=chunk_ids,
                )
            )

        paths.sort(key=lambda p: p.confidence, reverse=True)
        return paths[:top_k]

    async def retrieve_paths(self, *, query: str, scope: RetrievalScope, max_hops: int | None = None) -> list[GraphPath]:
        scope.ensure_scoped()
        bounded_hops = min(max_hops or self.settings.graph_max_hops, 2)
        entities = await self._matching_entities(query=query, scope=scope)
        if not entities:
            return []
        seed_ids = {f"entity:{self._slug(entity.canonical_name)}" for entity in entities}
        relations = await self._relations_touching(seed_ids=seed_ids, scope=scope)

        # Batch-fetch all materials/pages/blocks ONCE — avoids N×Mongo roundtrips
        # in the per-relation hydration loop below (saved 100s+ on warm collections).
        all_material_ids: set[PydanticObjectId] = set()
        for relation in relations:
            for ref in relation.evidence_refs:
                all_material_ids.add(ref.material_id)
        lookup_cache = await self._build_block_lookup(all_material_ids)

        paths: list[GraphPath] = []
        for relation in relations:
            if relation.source_id in seed_ids or relation.target_id in seed_ids:
                evidence = await self._hydrate_evidence_refs_with_cache(
                    relation.evidence_refs, lookup_cache=lookup_cache
                )
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
        """Find entities matching the query using keyword + semantic matching.

        Semantic matching re-encodes every candidate entity in a Python loop, which is
        prohibitively slow on CPU (~100s for 200 entities). Skip it whenever keyword
        matching already returned candidates — semantic only helps as a fallback when
        keyword matching finds nothing.
        """
        keyword_entities = await self._keyword_matching_entities(query=query, scope=scope)

        # Fallback to semantic only when keyword matching returned nothing.
        # Guarded by a 10s timeout — per-entity BGE-M3 encoding is O(N×embed_time)
        # and can block for 400s+ on collections with 200 entities.
        if not keyword_entities and self.embedder is not None:
            try:
                semantic_entities = await asyncio.wait_for(
                    self._semantic_matching_entities(query=query, scope=scope, limit=20),
                    timeout=10.0,
                )
                return semantic_entities
            except asyncio.TimeoutError:
                logger.warning(
                    "Semantic entity matching timed out — falling back to empty",
                    extra={"owner_id": scope.owner_id, "collection_id": scope.collection_id},
                )
                return []

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
        return await self._hydrate_evidence_refs_with_cache(refs, lookup_cache=None)

    async def _build_block_lookup(
        self, material_ids: set[PydanticObjectId]
    ) -> dict[tuple[PydanticObjectId, int | None, str | None], tuple[Material, object, object]]:
        """Fetch materials + pages + blocks once for the given material ids."""
        if not material_ids:
            return {}
        materials_list = await Material.find({"_id": {"$in": list(material_ids)}}).to_list()
        pages_by_material_id = await get_material_pages_by_material_ids(materials_list)
        lookup: dict[tuple[PydanticObjectId, int | None, str | None], tuple[Material, object, object]] = {}
        for material in materials_list:
            for page in pages_by_material_id.get(str(material.id), []):
                for block in page.blocks:
                    lookup[(material.id, page.page_number, block.block_id)] = (material, page, block)
        return lookup

    async def _hydrate_evidence_refs_with_cache(
        self,
        refs: list[EvidenceRef],
        *,
        lookup_cache: dict[tuple[PydanticObjectId, int | None, str | None], tuple[Material, object, object]] | None,
    ) -> list[EvidenceBlock]:
        if not refs:
            return []
        if lookup_cache is None:
            unique_material_ids = {ref.material_id for ref in refs}
            block_lookup = await self._build_block_lookup(unique_material_ids)
        else:
            block_lookup = lookup_cache

        evidence: list[EvidenceBlock] = []
        for ref in refs:
            found = block_lookup.get((ref.material_id, ref.page, ref.block_id))
            if found is None:
                continue
            material, page, block = found
            bbox = BBox.model_validate(block.bbox.model_dump()) if block.bbox is not None else None
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
                    bbox=bbox,
                    confidence=block.ocr_confidence,
                    metadata=block.extra,
                )
            )
        return evidence

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"

    # ── GraphRAG navigation surface (G1 + G2) ───────────────────────────────

    async def resolve_entities_by_slugs(
        self, *, entity_slugs: list[str], scope: RetrievalScope,
    ) -> list[Entity]:
        """Resolve `entity:slug-form` ids (used by the frontend graph UI) back to
        Mongo Entity documents within scope. Slugs map back via canonical_name."""
        if not entity_slugs:
            return []
        wanted: set[str] = set()
        for raw in entity_slugs:
            slug = raw.removeprefix("entity:") if raw.startswith("entity:") else raw
            if slug:
                wanted.add(slug)
        if not wanted:
            return []

        candidates = await Entity.find(self._scope_query(scope, {})).limit(2000).to_list()
        matched = [e for e in candidates if self._slug(e.canonical_name) in wanted]
        return matched

    async def subgraph_around_entities(
        self,
        *,
        entity_slugs: list[str],
        scope: RetrievalScope,
        hops: int = 2,
    ) -> tuple[list[Entity], list[Relation]]:
        """Return entities + relations within K hops of the given seed slugs.

        Used by G1 (subgraph endpoint) and G2 (graph-anchored query) to materialise
        a neighbourhood the user can drill into.
        """
        seeds = await self.resolve_entities_by_slugs(entity_slugs=entity_slugs, scope=scope)
        if not seeds:
            return [], []
        seed_ids = {f"entity:{self._slug(e.canonical_name)}" for e in seeds}

        first_hop = await self._relations_touching(seed_ids=seed_ids, scope=scope)
        frontier_ids: set[str] = set()
        for r in first_hop:
            frontier_ids.add(r.source_id)
            frontier_ids.add(r.target_id)

        all_relation_ids: set[str] = {*(r.id and str(r.id) for r in first_hop)}
        all_relations: list[Relation] = list(first_hop)
        if hops >= 2:
            second_hop = await self._relations_touching(
                seed_ids=frontier_ids - seed_ids, scope=scope,
            )
            for r in second_hop:
                rid = str(r.id)
                if rid not in all_relation_ids:
                    all_relations.append(r)
                    all_relation_ids.add(rid)
                frontier_ids.add(r.source_id)
                frontier_ids.add(r.target_id)

        # Pull all entities referenced in the relation set (the seed + neighbours).
        if frontier_ids:
            neighbours = await Entity.find(
                self._scope_query(scope, {})
            ).limit(3000).to_list()
            neighbours_in_frontier = [
                e for e in neighbours
                if f"entity:{self._slug(e.canonical_name)}" in (frontier_ids | seed_ids)
            ]
        else:
            neighbours_in_frontier = list(seeds)

        return neighbours_in_frontier, all_relations

    async def retrieve_around_entities(
        self,
        *,
        entity_slugs: list[str],
        scope: RetrievalScope,
        hops: int = 2,
    ) -> tuple[list[str], list[Entity], list[Relation]]:
        """Collect Chunk ids reachable from seed entities (chunks where seeds or
        their 1-2 hop neighbours are mentioned). Returns chunk_ids + the entity
        and relation sets that contributed — so the caller can track provenance.
        """
        entities, relations = await self.subgraph_around_entities(
            entity_slugs=entity_slugs, scope=scope, hops=hops,
        )
        if not entities:
            return [], [], []
        chunk_ids: list[str] = []
        seen: set[str] = set()
        for e in entities:
            for cid in e.chunk_ids or []:
                if cid not in seen:
                    seen.add(cid)
                    chunk_ids.append(cid)
        for r in relations:
            for cid in r.evidence_chunk_ids or []:
                if cid not in seen:
                    seen.add(cid)
                    chunk_ids.append(cid)
        return chunk_ids, entities, relations
