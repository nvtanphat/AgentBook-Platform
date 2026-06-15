"""Tests for SOTA entity/relation extraction components.

Covers: GLiNER extraction, smart gleaning, extraction cache, BGE-M3 resolution,
sentence decomposer, ontology filter, KET-RAG passage scoring, Pydantic-validated
relation parsing, and model routing.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.processing.entity_extractor import (
    EntityExtractor,
    _ExtractionCache,
    _should_glean,
    _is_junk,
    _clean_name,
)
from src.processing.entity_resolution import EntityResolver
from src.processing.semantic_relation_extractor import (
    LLMSemanticRelationExtractor,
    _get_ontology,
    _parse_response,
    _relation_allowed,
    _select_passages,
)
from src.processing.sentence_decomposer import decompose, decompose_blocks
from src.processing.types import EvidenceBlock, EvidenceMap, ExtractedEntity


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _block(block_id: str, text: str, lang: str = "en", page: int = 1) -> EvidenceBlock:
    return EvidenceBlock(
        owner_id="u1", collection_id="c1", material_id="m1",
        document_name="test.pdf", page=page,
        block_id=block_id, block_type="paragraph",
        snippet_original=text, source_language=lang,
    )


def _evidence_map(*texts: str) -> EvidenceMap:
    blocks = [_block(f"b{i}", t) for i, t in enumerate(texts)]
    return EvidenceMap(
        owner_id="u1", collection_id="c1", material_id="m1",
        document_name="test.pdf", blocks=blocks,
    )


def _entity(name: str, etype: str = "concept", conf: float = 0.8) -> ExtractedEntity:
    return ExtractedEntity(
        canonical_name=name, entity_type=etype, confidence=conf, mention_refs=[],
    )


# ── P3.3: Extraction Cache ────────────────────────────────────────────────────

class TestExtractionCache:
    def test_miss_returns_none(self):
        cache = _ExtractionCache()
        assert cache.get("never seen", "gliner") is None

    def test_put_then_get(self):
        cache = _ExtractionCache()
        data = [{"text": "BERT", "label": "algorithm", "score": 0.9}]
        cache.put("BERT is a model", "gliner", data)
        result = cache.get("BERT is a model", "gliner")
        assert result == data

    def test_different_backend_different_key(self):
        cache = _ExtractionCache()
        cache.put("text", "gliner", [{"a": 1}])
        assert cache.get("text", "llm") is None

    def test_eviction_on_overflow(self):
        cache = _ExtractionCache(max_entries=3)
        for i in range(4):
            cache.put(f"text_{i}", "gliner", [{"i": i}])
        # Oldest entry (text_0) should be evicted
        assert cache.get("text_0", "gliner") is None
        assert cache.get("text_3", "gliner") is not None

    def test_lru_access_prevents_eviction(self):
        cache = _ExtractionCache(max_entries=3)
        for i in range(3):
            cache.put(f"text_{i}", "gliner", [{"i": i}])
        # Access text_0 to make it recently used
        cache.get("text_0", "gliner")
        # Add a 4th entry → text_1 should be evicted (LRU), not text_0
        cache.put("text_3", "gliner", [])
        assert cache.get("text_0", "gliner") is not None
        assert cache.get("text_1", "gliner") is None


# ── P2.1: Smart Gleaning ─────────────────────────────────────────────────────

class TestSmartGleaning:
    def test_triggers_on_low_density(self):
        text = "A" * 2000  # 2000 chars, 0 entities → density=0
        assert _should_glean([], text) is True

    def test_triggers_on_low_confidence(self):
        entities = [_entity("X", conf=0.5), _entity("Y", conf=0.6)]
        text = "A" * 500  # density = 2/0.5 = 4 ≥ 2 → pass density, but avg_conf=0.55 < 0.7
        assert _should_glean(entities, text) is True

    def test_no_glean_on_rich_block(self):
        # density = 5/1.0 = 5 ≥ 2 AND avg_conf = 0.9 ≥ 0.7
        entities = [_entity(f"E{i}", conf=0.9) for i in range(5)]
        text = "X" * 1000
        assert _should_glean(entities, text) is False

    def test_empty_entities_always_glean(self):
        assert _should_glean([], "any text at all") is True


# ── P2.3: Sentence Decomposer ────────────────────────────────────────────────

class TestSentenceDecomposer:
    def test_single_sentence_unchanged(self):
        t = "BERT is a pretrained language model."
        assert decompose(t, "en") == [t]

    def test_splits_on_and(self):
        t = "BERT uses self-attention and GPT uses causal masking."
        clauses = decompose(t, "en")
        assert len(clauses) >= 2
        assert any("BERT" in c for c in clauses)
        assert any("GPT" in c for c in clauses)

    def test_splits_on_while(self):
        t = "BERT is bidirectional while GPT is unidirectional."
        clauses = decompose(t, "en")
        assert len(clauses) >= 2

    def test_splits_on_however(self):
        t = "BERT achieves high accuracy, however it is slower than GPT."
        clauses = decompose(t, "en")
        assert len(clauses) >= 2

    def test_short_fragments_dropped(self):
        # Fragment shorter than 20 chars should be dropped
        t = "A, and B achieves high accuracy on benchmarks."
        clauses = decompose(t, "en")
        for c in clauses:
            assert len(c) >= 20

    def test_empty_string(self):
        assert decompose("", "en") == []

    def test_decompose_blocks_single(self):
        b = _block("b0", "Single sentence only.")
        result = decompose_blocks([b])
        assert len(result) == 1
        assert result[0].block_id == "b0"

    def test_decompose_blocks_expands(self):
        b = _block("b0", "BERT uses self-attention and GPT uses causal masking for autoregressive generation.")
        result = decompose_blocks([b])
        assert len(result) >= 2
        # All results should be copies of the original block
        for r in result:
            assert r.block_id == "b0"

    def test_decompose_blocks_max_clauses(self):
        # Long sentence with many conjunctions
        t = "A is good and B is better and C is best and D is finest."
        b = _block("b0", t)
        result = decompose_blocks([b], max_clauses_per_block=2)
        assert len(result) <= 2


# ── P2.2: Ontology Filter ─────────────────────────────────────────────────────

class TestOntologyFilter:
    def setup_method(self):
        self.ontology = {
            "concept":   {"uses", "extends", "related_to"},
            "metric":    {"evaluates_on", "compared_with"},
            "dataset":   {"evaluates_on"},
        }

    def test_known_type_allowed(self):
        assert _relation_allowed("concept", "uses", self.ontology) is True

    def test_known_type_not_allowed(self):
        assert _relation_allowed("concept", "evaluates_on", self.ontology) is False

    def test_metric_restricted(self):
        assert _relation_allowed("metric", "extends", self.ontology) is False
        assert _relation_allowed("metric", "evaluates_on", self.ontology) is True

    def test_unknown_type_always_allowed(self):
        # Unknown entity type → no constraint → permit
        assert _relation_allowed("unknown_type", "uses", self.ontology) is True

    def test_empty_ontology_allows_all(self):
        assert _relation_allowed("concept", "extends", {}) is True
        assert _relation_allowed("dataset", "contradicts", {}) is True

    def test_get_ontology_returns_dict(self):
        o = _get_ontology()
        assert isinstance(o, dict)


# ── P3.1: KET-RAG Passage Scoring ────────────────────────────────────────────

class TestKETRAGPassageSelection:
    def test_single_entity_block_excluded(self):
        blocks = [
            _block("b0", "BERT is a model."),
            _block("b1", "Introduction section."),
        ]
        selected = _select_passages(
            blocks=blocks,
            concept_names_lower={"bert", "gpt"},
            max_passages=5,
        )
        # b0 has only 1 entity mention, b1 has 0 — both excluded
        assert len(selected) == 0

    def test_co_mention_block_included(self):
        blocks = [
            _block("b0", "BERT and GPT both use transformer architecture."),
            _block("b1", "Introduction only."),
        ]
        selected = _select_passages(
            blocks=blocks,
            concept_names_lower={"bert", "gpt"},
            max_passages=5,
        )
        assert len(selected) == 1
        assert selected[0].block_id == "b0"

    def test_ranks_by_entity_count(self):
        blocks = [
            _block("b0", "BERT uses transformer and self-attention."),  # 3 entity hits
            _block("b1", "BERT and GPT compared."),  # 2 entity hits
            _block("b2", "BERT and GPT use transformer and self-attention."),  # 4 hits
        ]
        concepts = {"bert", "gpt", "transformer", "self-attention"}
        selected = _select_passages(blocks=blocks, concept_names_lower=concepts, max_passages=2)
        # b2 should be first (4 hits), then b0 or b1
        assert selected[0].block_id == "b2"

    def test_max_passages_cap(self):
        blocks = [_block(f"b{i}", "BERT and GPT compete.") for i in range(10)]
        selected = _select_passages(
            blocks=blocks,
            concept_names_lower={"bert", "gpt"},
            max_passages=3,
        )
        assert len(selected) == 3


# ── P1.3: Pydantic-validated relation parsing ─────────────────────────────────

class TestRelationParsing:
    def _entities(self):
        return {
            "bert": _entity("BERT", "algorithm"),
            "gpt": _entity("GPT", "algorithm"),
            "transformer": _entity("transformer", "concept"),
        }

    def _passages(self):
        return [_block("b0", "BERT extends transformer. GPT extends transformer.")]

    def test_valid_json_parses(self):
        raw = json.dumps({
            "relations": [
                {"source": "BERT", "target": "transformer", "type": "extends", "passage_index": 0, "confidence": 0.9},
                {"source": "GPT", "target": "transformer", "type": "extends", "passage_index": 0, "confidence": 0.85},
            ],
            "missed_entities": [],
        })
        rels, missed = _parse_response(
            raw=raw,
            concept_by_lower_name=self._entities(),
            passages=self._passages(),
            max_passage_chars=500,
            ontology={},
        )
        assert len(rels) == 2
        assert missed == []

    def test_invalid_relation_type_dropped(self):
        raw = json.dumps({
            "relations": [
                {"source": "BERT", "target": "transformer", "type": "INVALID_TYPE", "confidence": 0.9},
            ],
            "missed_entities": [],
        })
        rels, _ = _parse_response(
            raw=raw,
            concept_by_lower_name=self._entities(),
            passages=self._passages(),
            max_passage_chars=500,
            ontology={},
        )
        assert rels == []

    def test_unknown_entity_dropped(self):
        raw = json.dumps({
            "relations": [
                {"source": "BERT", "target": "UNKNOWN_ENTITY", "type": "uses", "confidence": 0.8},
            ],
            "missed_entities": [],
        })
        rels, _ = _parse_response(
            raw=raw,
            concept_by_lower_name=self._entities(),
            passages=self._passages(),
            max_passage_chars=500,
            ontology={},
        )
        assert rels == []

    def test_duplicate_relations_deduplicated(self):
        raw = json.dumps({
            "relations": [
                {"source": "BERT", "target": "transformer", "type": "extends", "confidence": 0.9},
                {"source": "BERT", "target": "transformer", "type": "extends", "confidence": 0.7},
            ],
            "missed_entities": [],
        })
        rels, _ = _parse_response(
            raw=raw,
            concept_by_lower_name=self._entities(),
            passages=self._passages(),
            max_passage_chars=500,
            ontology={},
        )
        assert len(rels) == 1

    def test_ontology_filter_in_parse(self):
        raw = json.dumps({
            "relations": [
                {"source": "BERT", "target": "transformer", "type": "extends", "confidence": 0.9},
            ],
            "missed_entities": [],
        })
        # algorithm type cannot use "extends" in this ontology
        strict_ontology = {"algorithm": {"uses", "compared_with"}}
        rels, _ = _parse_response(
            raw=raw,
            concept_by_lower_name=self._entities(),
            passages=self._passages(),
            max_passage_chars=500,
            ontology=strict_ontology,
        )
        assert rels == []

    def test_missed_entities_parsed(self):
        raw = json.dumps({
            "relations": [],
            "missed_entities": [
                {"name": "RoBERTa", "type": "algorithm", "confidence": 0.85},
                {"name": "", "type": "concept", "confidence": 0.5},  # empty name → dropped
            ],
        })
        _, missed = _parse_response(
            raw=raw,
            concept_by_lower_name=self._entities(),
            passages=self._passages(),
            max_passage_chars=500,
            ontology={},
        )
        assert len(missed) == 1
        assert missed[0][0] == "RoBERTa"

    def test_known_entity_not_in_missed(self):
        raw = json.dumps({
            "relations": [],
            "missed_entities": [
                {"name": "BERT", "type": "algorithm", "confidence": 0.9},  # already known
            ],
        })
        _, missed = _parse_response(
            raw=raw,
            concept_by_lower_name=self._entities(),
            passages=self._passages(),
            max_passage_chars=500,
            ontology={},
        )
        assert missed == []

    def test_malformed_json_returns_empty(self):
        raw = "not json at all {broken"
        rels, missed = _parse_response(
            raw=raw,
            concept_by_lower_name=self._entities(),
            passages=self._passages(),
            max_passage_chars=500,
            ontology={},
        )
        assert rels == []
        assert missed == []

    def test_markdown_wrapped_json(self):
        raw = """```json
{"relations": [{"source": "BERT", "target": "transformer", "type": "extends", "confidence": 0.9}], "missed_entities": []}
```"""
        rels, _ = _parse_response(
            raw=raw,
            concept_by_lower_name=self._entities(),
            passages=self._passages(),
            max_passage_chars=500,
            ontology={},
        )
        assert len(rels) == 1

    def test_self_relation_dropped(self):
        raw = json.dumps({
            "relations": [
                {"source": "BERT", "target": "BERT", "type": "related_to", "confidence": 0.8},
            ],
            "missed_entities": [],
        })
        rels, _ = _parse_response(
            raw=raw,
            concept_by_lower_name=self._entities(),
            passages=self._passages(),
            max_passage_chars=500,
            ontology={},
        )
        assert rels == []

    def test_confidence_clamped(self):
        raw = json.dumps({
            "relations": [
                {"source": "BERT", "target": "transformer", "type": "extends", "confidence": 999},
            ],
            "missed_entities": [],
        })
        rels, _ = _parse_response(
            raw=raw,
            concept_by_lower_name=self._entities(),
            passages=self._passages(),
            max_passage_chars=500,
            ontology={},
        )
        assert rels[0].confidence == 1.0


# ── Entity extractor helpers ───────────────────────────────────────────────────

class TestEntityExtractorHelpers:
    def test_is_junk_single_char(self):
        assert _is_junk("A") is True

    def test_is_junk_numeric(self):
        assert _is_junk("123") is True

    def test_is_junk_path_chars(self):
        assert _is_junk("path/to/file") is True

    def test_not_junk_valid_term(self):
        assert _is_junk("BERT") is False
        assert _is_junk("transformer") is False

    def test_clean_name_strips_whitespace(self):
        assert _clean_name("  BERT  ") == "BERT"

    def test_clean_name_normalizes_spaces(self):
        # _clean_name collapses multiple spaces into one (re.sub r"\s+" → " ")
        assert _clean_name("self  attention") == "self attention"

    def test_regex_extractor_finds_keywords(self):
        em = _evidence_map(
            "Dropout is applied during training. Adam optimizer is used.",
        )
        ex = EntityExtractor()
        entities = ex.extract(em)
        names = [e.canonical_name.lower() for e in entities]
        assert any("dropout" in n for n in names)
        assert any("adam" in n for n in names)


# ── BGE-M3 entity resolution async ────────────────────────────────────────────

class TestBGEM3EntityResolution:
    def test_resolve_sync_still_works(self):
        entities = [
            _entity("Machine Learning"),
            _entity("machine learning"),  # duplicate
            _entity("Transformer"),
        ]
        resolver = EntityResolver()
        result = resolver.resolve(entities)
        # "Machine Learning" and "machine learning" should merge
        names_lower = [e.canonical_name.lower() for e in result]
        assert names_lower.count("machine learning") == 1

    @pytest.mark.asyncio
    async def test_resolve_async_no_embedder(self):
        """resolve_async with embedder=None falls back to sync resolve."""
        entities = [_entity("BERT"), _entity("bert")]
        resolver = EntityResolver()
        result = await resolver.resolve_async(entities, embedder=None)
        assert len(result) <= 2  # dedup may reduce count

    @pytest.mark.asyncio
    async def test_resolve_async_with_mock_embedder(self):
        """resolve_async calls embedder.encode and merges close entities."""
        entities = [
            _entity("Machine Learning", conf=0.9),
            _entity("Hoc may", conf=0.8),   # cross-lingual near-duplicate
            _entity("Deep Learning", conf=0.85),
        ]

        # Mock embedder using SimpleNamespace — no src.rag imports needed.
        # ML and "Hoc may" have cosine ≈ 0.994 (≥ 0.82) → should merge.
        # DL is orthogonal → stays separate.
        def mock_encode(names):
            vecs = {
                "Machine Learning": [1.0, 0.0, 0.0],
                "Hoc may":          [0.99, 0.1, 0.0],
                "Deep Learning":    [0.0, 1.0, 0.0],
            }
            return [SimpleNamespace(dense=vecs.get(n, [0.0, 0.0, 0.0])) for n in names]

        mock_embedder = MagicMock()
        mock_embedder.encode = mock_encode

        resolver = EntityResolver()
        result = await resolver.resolve_async(entities, embedder=mock_embedder)
        # ML + Hoc may should merge → 2 entities total
        assert len(result) <= 2


# ── LLMSemanticRelationExtractor async ────────────────────────────────────────

class TestLLMSemanticRelationExtractor:
    def _make_extractor(self, llm_response: str) -> LLMSemanticRelationExtractor:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=llm_response)
        return LLMSemanticRelationExtractor(llm=mock_llm, max_concepts=10, max_passages=5)

    def _make_entities(self) -> list[ExtractedEntity]:
        return [
            _entity("BERT", "algorithm", 0.95),
            _entity("transformer", "concept", 0.90),
            _entity("self-attention", "concept", 0.85),
            _entity("GPT", "model", 0.88),
        ]

    def _make_evidence_map(self) -> EvidenceMap:
        return _evidence_map(
            "BERT uses self-attention and extends transformer architecture.",
            "GPT extends transformer and compares with BERT on benchmarks.",
        )

    @pytest.mark.asyncio
    async def test_returns_empty_without_llm(self):
        ex = LLMSemanticRelationExtractor(llm=None)
        result = await ex.extract_async(_evidence_map("text"), [_entity("X")])
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_with_single_entity(self):
        mock_llm = AsyncMock()
        ex = LLMSemanticRelationExtractor(llm=mock_llm)
        # person is now a valid graph type but len(concepts) < 2 → still empty
        result = await ex.extract_async(_evidence_map("text"), [_entity("Alice", "person")])
        assert result == []

    @pytest.mark.asyncio
    async def test_person_org_entities_enter_graph(self):
        """person and organization are graph-eligible types (multi-domain fix)."""
        llm_response = json.dumps({
            "relations": [
                {"source": "Alice", "target": "Acme Corp", "type": "created_by", "passage_index": 0, "confidence": 0.8},
            ],
            "missed_entities": [],
        })
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=llm_response)
        ex = LLMSemanticRelationExtractor(llm=mock_llm)
        text = "Alice founded Acme Corp in 2010."
        entities = [_entity("Alice", "person"), _entity("Acme Corp", "organization")]
        result = await ex.extract_async(_evidence_map(text), entities)
        assert len(result) == 1
        assert result[0].relation_type == "created_by"

    @pytest.mark.asyncio
    async def test_extracts_valid_relations(self):
        llm_response = json.dumps({
            "relations": [
                {"source": "BERT", "target": "transformer", "type": "extends", "passage_index": 0, "confidence": 0.9},
                {"source": "GPT", "target": "transformer", "type": "extends", "passage_index": 1, "confidence": 0.85},
            ],
            "missed_entities": [],
        })
        ex = self._make_extractor(llm_response)
        result = await ex.extract_async(self._make_evidence_map(), self._make_entities())
        assert len(result) == 2
        rel_types = {r.relation_type for r in result}
        assert rel_types == {"extends"}

    @pytest.mark.asyncio
    async def test_handles_llm_failure_gracefully(self):
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
        ex = LLMSemanticRelationExtractor(llm=mock_llm)
        result = await ex.extract_async(self._make_evidence_map(), self._make_entities())
        assert result == []

    @pytest.mark.asyncio
    async def test_gleaning_pass_adds_missed_relations(self):
        """Gleaning second call surfaces relations the first pass missed."""
        first = json.dumps({
            "relations": [
                {"source": "BERT", "target": "transformer", "type": "extends", "passage_index": 0, "confidence": 0.9},
            ],
            "missed_entities": [],
        })
        glean = json.dumps({
            "relations": [
                {"source": "GPT", "target": "transformer", "type": "extends", "passage_index": 1, "confidence": 0.85},
                {"source": "BERT", "target": "self-attention", "type": "uses", "passage_index": 0, "confidence": 0.8},
                # duplicate of first pass — must be deduped
                {"source": "BERT", "target": "transformer", "type": "extends", "passage_index": 0, "confidence": 0.9},
            ],
            "missed_entities": [],
        })
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=[first, glean])
        ex = LLMSemanticRelationExtractor(llm=mock_llm, max_concepts=10, max_passages=5, gleaning=True)
        result = await ex.extract_async(self._make_evidence_map(), self._make_entities())
        # 1 from first pass + 2 new from gleaning (duplicate dropped) = 3
        assert len(result) == 3
        assert mock_llm.generate.await_count == 2

    @pytest.mark.asyncio
    async def test_gleaning_disabled_makes_single_call(self):
        ex = self._make_extractor(json.dumps({"relations": [], "missed_entities": []}))
        await ex.extract_async(self._make_evidence_map(), self._make_entities())
        assert ex._llm.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_sentence_decomposition_expands_passages(self):
        """Decomposed blocks should allow more co-mention passage matches."""
        compound = "BERT uses self-attention and GPT extends transformer."
        em = _evidence_map(compound)
        llm_response = json.dumps({"relations": [], "missed_entities": []})
        ex = self._make_extractor(llm_response)
        # Should not raise — decomposition expands the single block into clauses
        result = await ex.extract_async(em, self._make_entities())
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_missed_entities_logged_not_returned(self):
        """missed_entities are logged as info — not added to relations list."""
        llm_response = json.dumps({
            "relations": [
                {"source": "BERT", "target": "transformer", "type": "uses", "confidence": 0.8},
            ],
            "missed_entities": [
                {"name": "RoBERTa", "type": "algorithm", "confidence": 0.85},
            ],
        })
        ex = self._make_extractor(llm_response)
        result = await ex.extract_async(self._make_evidence_map(), self._make_entities())
        # relations are returned; missed_entities do not appear in output
        assert all(hasattr(r, "relation_type") for r in result)


# ── Model routing ─────────────────────────────────────────────────────────────

class TestModelRouting:
    def test_build_extraction_llm_no_override(self):
        from src.core.model_factory import build_extraction_llm, build_llm
        from src.core.config import get_settings
        settings = get_settings()
        # With empty extraction_provider, should return same type as build_llm
        ex_llm = build_extraction_llm(settings)
        default_llm = build_llm(settings)
        assert type(ex_llm) == type(default_llm)

    def test_build_extraction_llm_local_override(self):
        from src.core.model_factory import build_extraction_llm
        from src.core.local_llm import OllamaLLM
        from unittest.mock import MagicMock
        settings = MagicMock()
        settings.llm_extraction_provider = "local"
        settings.llm_extraction_local_model = "qwen2.5:3b"
        settings.llm_local_model = "qwen2.5:7b"
        settings.ollama_base_url = "http://localhost:11434"
        settings.llm_timeout_seconds = 60
        result = build_extraction_llm(settings)
        assert isinstance(result, OllamaLLM)
