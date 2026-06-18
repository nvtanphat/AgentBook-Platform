from __future__ import annotations

import asyncio

from PIL import Image

from src.core.config import Settings
from src.core.base_llm import BaseLLM
from src.guardrails.citation_aligner import CitationAligner
from src.guardrails.evidence_validator import EvidenceValidator
from src.inference.inference_engine import InferenceEngine
from src.processing.types import BBox, EvidenceBlock
from src.rag.evidence import CitationBuilder, EvidenceAssembler, EvidenceBundle, EvidenceFusionRanker, EvidenceKind
from src.rag.types import RetrievedChunk, RetrievedVisualChunk, RetrievalScope


def _text_chunk(modality: str = "text") -> RetrievedChunk:
    ev = EvidenceBlock(
        owner_id="u",
        collection_id="c",
        material_id="m",
        document_name="doc.pdf",
        page=2,
        block_id="b-text",
        block_type=modality,
        snippet_original="The chart shows accuracy increasing.",
        source_language="en",
        confidence=0.8,
        metadata={"start_seconds": 12.0, "end_seconds": 18.0} if modality == "audio" else {},
    )
    return RetrievedChunk(
        chunk_id="chunk-1",
        owner_id="u",
        collection_id="c",
        material_id="m",
        document_name="doc.pdf",
        content=ev.snippet_original,
        language="en",
        modality=modality,
        source_block_ids=[ev.block_id],
        source_pages=[2],
        evidence=[ev],
        metadata={"sheet_names": ["Sheet1"]} if modality == "table" else {},
        fused_score=0.7,
    )


def _visual_hit(image_path: str | None = None) -> RetrievedVisualChunk:
    return RetrievedVisualChunk(
        point_id="visual-1",
        owner_id="u",
        collection_id="c",
        material_id="m",
        document_name="slides.pdf",
        page=3,
        block_id="fig-1",
        block_type="figure",
        caption="Accuracy chart",
        source_language="en",
        bbox=BBox(x1=1, y1=2, x2=100, y2=80),
        image_path=image_path,
        score=0.91,
    )


def test_evidence_bundle_preserves_modality_order_and_legacy_adapter(tmp_path):
    img = tmp_path / "fig.png"
    Image.new("RGB", (16, 16)).save(img)

    bundle = EvidenceAssembler.assemble(
        text_chunks=[_text_chunk("table"), _text_chunk("audio")],
        visual_hits=[_visual_hit(str(img))],
        visual_first=True,
    )

    assert [item.kind for item in bundle.items] == [
        EvidenceKind.VISUAL.value,
        EvidenceKind.TABLE.value,
        EvidenceKind.AUDIO.value,
    ]
    legacy = bundle.to_legacy_chunks()
    assert legacy[0].modality == "figure"
    assert legacy[1].modality == "table"
    assert legacy[2].modality == "audio"


def test_citation_builder_maps_block_crop_cell_and_audio(tmp_path):
    img = tmp_path / "fig.png"
    Image.new("RGB", (16, 16)).save(img)
    bundle = EvidenceAssembler.assemble(
        text_chunks=[_text_chunk("table"), _text_chunk("audio")],
        visual_hits=[_visual_hit(str(img))],
        visual_first=True,
    )

    citations = CitationBuilder.from_evidence_bundle(bundle, owner_id="u")

    assert citations[0].kind == "visual"
    assert citations[0].block_id == "fig-1"
    assert citations[0].figure_image_url is not None
    assert citations[1].kind == "table"
    assert citations[1].sheet_name == "Sheet1"
    assert citations[2].kind == "audio"
    assert citations[2].audio_start_seconds == 12.0
    assert citations[2].audio_end_seconds == 18.0


def test_citation_aligner_accepts_evidence_bundle():
    bundle = EvidenceBundle.from_visual_hits([_visual_hit()])
    result = CitationAligner().align(
        answer="The figure shows an accuracy chart [1].",
        evidence_bundle=bundle,
        preferred_modality="figure",
    )
    assert result.stage == "PASS"
    assert result.invalid_citation_count == 0


def test_evidence_validator_checks_bundle_modality():
    bundle = EvidenceBundle.from_visual_hits([_visual_hit()])
    result = EvidenceValidator().validate(
        query="What does the chart show?",
        evidence_bundle=bundle,
        preferred_modality="figure",
    )
    assert result.sufficient
    assert result.selected_evidence_ids == ["visual-1"]

    missing = EvidenceValidator().validate(
        query="What does the table show?",
        evidence_bundle=bundle,
        preferred_modality="table",
    )
    assert not missing.sufficient
    assert "table_evidence" in missing.missing


def test_fusion_ranker_prioritizes_route_modality():
    ranker = EvidenceFusionRanker()
    bundle = ranker.fuse(
        query="What does the chart show?",
        text_chunks=[_text_chunk("text")],
        visual_hits=[_visual_hit()],
        preferred_modality="figure",
        final_limit=2,
    )

    assert [item.kind for item in bundle.items] == ["visual", "text"]
    trace = ranker.trace_metadata(bundle)
    assert trace["modality_policy"] == "visual_first"
    assert trace["selected_evidence_ids"] == ["visual-1", "chunk-1"]


class FakeTextLLM(BaseLLM):
    async def generate(self, *, prompt: str) -> str:
        return "Fallback answer [1]."


class FakeVisionLLM:
    def __init__(self) -> None:
        self.calls = []

    async def generate_with_images(self, *, prompt: str, image_paths=None, image_bytes=None):
        self.calls.append({"prompt": prompt, "image_paths": list(image_paths or []), "image_bytes": list(image_bytes or [])})
        return "The retrieved figure is an accuracy chart [1]."

    async def verify_with_images_structured(self, *, answer: str, prompt_context: str, image_paths):
        class Verdict:
            supported = True

            def model_dump(self, mode=None):
                return {"supported": True, "unsupported_claims": [], "unreadable_regions": []}

        return Verdict()


class DummyRetriever:
    async def retrieve(self, **kwargs):
        return []

    async def retrieve_visual(self, **kwargs):
        return []


def test_visual_answer_path_calls_vlm_with_retrieved_image(tmp_path):
    img = tmp_path / "fig.png"
    Image.new("RGB", (16, 16)).save(img)
    vision = FakeVisionLLM()
    settings = Settings(
        testing=True,
        vlm_query_enabled=True,
        vlm_query_verify_enabled=False,
        visual_embedding_enabled=False,
    )
    engine = InferenceEngine(
        settings=settings,
        retriever=DummyRetriever(),
        llm=FakeTextLLM(),
        vision_llm=vision,
    )

    response = asyncio.run(
        engine.answer_with_visual_evidence(
            query="What does the chart show?",
            scope=RetrievalScope(owner_id="u", collection_id="c"),
            visual_hits=[_visual_hit(str(img))],
            answer_language="en",
        )
    )

    assert not response.was_refused
    assert response.citations[0].kind == "visual"
    assert response.citations[0].block_id == "fig-1"
    assert vision.calls
    assert vision.calls[0]["image_paths"] == [img]


class StreamRetriever(DummyRetriever):
    def __init__(self, hit):
        self.hit = hit

    async def retrieve_visual(self, **kwargs):
        return [self.hit]


def test_visual_stream_route_uses_vlm_without_legacy_visual_chunk(tmp_path):
    img = tmp_path / "fig.png"
    Image.new("RGB", (16, 16)).save(img)
    vision = FakeVisionLLM()
    settings = Settings(
        testing=True,
        llm_router_enabled=False,
        reranker_enabled=False,
        adaptive_retrieval_enabled=False,
        vlm_query_enabled=True,
        vlm_query_verify_enabled=False,
        visual_embedding_enabled=True,
    )
    engine = InferenceEngine(
        settings=settings,
        retriever=StreamRetriever(_visual_hit(str(img))),
        llm=FakeTextLLM(),
        vision_llm=vision,
        visual_provider=object(),
    )

    def _boom(_):
        raise AssertionError("legacy visual conversion should not be used")

    engine._visual_to_text_chunk = _boom  # type: ignore[method-assign]

    async def _collect():
        return [
            event async for event in engine.answer_stream(
                query="What does the chart show?",
                scope=RetrievalScope(owner_id="u", collection_id="c"),
                answer_language="en",
            )
        ]

    events = asyncio.run(_collect())

    assert any("event: agent_step" in event for event in events)
    assert not any("event: token" in event for event in events)
    assert any("event: done" in event for event in events)
    assert vision.calls
