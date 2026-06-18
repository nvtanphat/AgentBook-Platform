"""Visual retrieval utilities — extracted from InferenceEngine for maintainability.

Owns:
  - Figure-number filtering of SigLIP visual hits
  - VLM verifier refuse decision
  - Visual hit → text chunk conversion
  - Inline image injection into markdown answers
  - Visual citation construction
  - Visual retrieval (hits + chunk forms)
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    from src.core.config import Settings
    from src.rag.embedding_provider import VisualEmbeddingProvider
    from src.rag.retriever import HybridRetriever
    from src.rag.types import RetrievalScope, RetrievedChunk, RetrievedVisualChunk

logger = logging.getLogger(__name__)


# ── Pure static helpers ───────────────────────────────────────────────────────


def visual_verifier_should_refuse(
    *, visual_verdict, image_paths: list | None, threshold: float
) -> bool:
    if not image_paths:
        return False
    if getattr(visual_verdict, "supported", True):
        return False
    confidence = float(getattr(visual_verdict, "confidence", 0.0) or 0.0)
    return confidence >= threshold


def requested_figure_number(query: str) -> int | None:
    from src.processing.slug import ascii_fold

    folded = ascii_fold(query or "").lower()
    match = re.search(r"\b(?:figure|fig|hinh|hinh ve|so do|diagram)\s*\.?\s*(\d{1,3})\b", folded)
    return int(match.group(1)) if match else None


def visual_hit_figure_number(hit: "RetrievedVisualChunk") -> int | None:
    from src.processing.slug import ascii_fold

    haystack = " ".join([hit.block_id or "", hit.caption or "", hit.document_name or ""])
    folded = ascii_fold(haystack).lower()
    for pattern in (
        r"\b(?:figure|fig|hinh|hinh ve|diagram)\s*\.?\s*(\d{1,3})\b",
        r"\b(?:fig|figure|picture|image|pic)[-_/# ]+(\d{1,3})\b",
        r"\b(\d{1,3})\b",
    ):
        match = re.search(pattern, folded)
        if match:
            return int(match.group(1))
    return None


def visual_hit_label(hit: "RetrievedVisualChunk") -> str:
    figure_no = visual_hit_figure_number(hit)
    label = f"Figure {figure_no}" if figure_no is not None else "Figure"
    if hit.page:
        label = f"{label}, trang {hit.page}"
    return label


def filter_visual_hits_for_query(
    query: str, hits: "list[RetrievedVisualChunk]"
) -> "list[RetrievedVisualChunk]":
    """Filter visual hits by the figure number in the query; fall back to unfiltered on miss."""
    requested = requested_figure_number(query)
    if requested is None:
        return hits
    filtered = [hit for hit in hits if visual_hit_figure_number(hit) == requested]
    return filtered if filtered else hits


def strip_inline_image_markdown(answer: str) -> str:
    if not answer or "![" not in answer:
        return answer
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", answer)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def visual_to_text_chunk(v: "RetrievedVisualChunk") -> "RetrievedChunk":
    """Convert a visual chunk to RetrievedChunk so it flows through the text pipeline."""
    from src.processing.types import EvidenceBlock
    from src.rag.types import RetrievedChunk

    ev = EvidenceBlock(
        owner_id=v.owner_id,
        collection_id=v.collection_id,
        material_id=v.material_id,
        document_name=v.document_name,
        page=v.page,
        block_id=v.block_id,
        block_type=v.block_type,
        snippet_original=v.caption,
        source_language=v.source_language,
        bbox=v.bbox,
        confidence=v.score,
        metadata={},
    )
    return RetrievedChunk(
        chunk_id=v.point_id,
        owner_id=v.owner_id,
        collection_id=v.collection_id,
        material_id=v.material_id,
        document_name=v.document_name,
        content=f"[Figure] {v.caption}",
        language=v.source_language,
        modality="figure",
        source_block_ids=[v.block_id],
        source_pages=[v.page],
        bboxes=[v.bbox] if v.bbox else [],
        evidence=[ev],
        fused_score=v.score,
    )


def inject_inline_images(
    answer: str,
    visual_hits: "list[RetrievedVisualChunk]",
    owner_id: str,
) -> str:
    """Embed top-N visual hits as markdown ![]() blocks inside the answer.

    Placement: directly after the first paragraph break.
    URLs are relative paths under /api/v1/materials/...; frontend resolves them.
    """
    if not visual_hits:
        return answer
    encoded_owner = quote(owner_id, safe="")
    image_blocks: list[str] = []
    for hit in visual_hits:
        alt = visual_hit_label(hit).replace("]", " ").replace("[", " ")
        url = f"/api/v1/materials/{hit.material_id}/raw?owner_id={encoded_owner}"
        image_blocks.append(f"![{alt}]({url})")
    joined = "\n\n".join(image_blocks)

    stripped = answer.rstrip()
    if not stripped:
        return f"{joined}\n"
    split_idx = stripped.find("\n\n")
    if split_idx == -1:
        return f"{stripped}\n\n{joined}\n"
    head = stripped[:split_idx].rstrip()
    tail = stripped[split_idx + 2:].lstrip()
    if tail:
        return f"{head}\n\n{joined}\n\n{tail}"
    return f"{head}\n\n{joined}\n"


# ── Stateful handler (needs runtime dependencies) ─────────────────────────────


class VisualHandler:
    """Handles visual retrieval and citation construction for InferenceEngine.

    Constructed once in InferenceEngine.__init__; only carries the I/O dependencies
    it actually uses. Prompt-building helpers (build_prompt, language_lock, etc.)
    remain on InferenceEngine to avoid circular coupling.
    """

    def __init__(
        self,
        *,
        retriever: "HybridRetriever",
        visual_provider: "VisualEmbeddingProvider | None",
        settings: "Settings",
    ) -> None:
        self.retriever = retriever
        self.visual_provider = visual_provider
        self.settings = settings

    async def retrieve_hits(
        self, *, query: str, scope: "RetrievalScope"
    ) -> "list[RetrievedVisualChunk]":
        """Run visual retrieval and keep modality-native visual hits."""
        if self.visual_provider is None:
            return []
        try:
            hits = await self.retriever.retrieve_visual(
                query=query,
                scope=scope,
                visual_provider=self.visual_provider,
                limit=self.settings.visual_retrieval_top_k,
            )
            hits = filter_visual_hits_for_query(query, hits)
            if hits:
                logger.info(
                    "Visual retrieval returned %d figure(s)",
                    len(hits),
                    extra={"owner_id": scope.owner_id, "collection_id": scope.collection_id},
                )
            return hits
        except Exception as exc:
            logger.warning(
                "Visual retrieval failed - skipping",
                extra={"owner_id": scope.owner_id, "error": str(exc)},
            )
            return []

    async def retrieve_chunks(
        self, *, query: str, scope: "RetrievalScope"
    ) -> "list[RetrievedChunk]":
        """Run visual retrieval and convert results to RetrievedChunk; [] on error."""
        if self.visual_provider is None:
            return []
        try:
            raw = await self.retriever.retrieve_visual(
                query=query,
                scope=scope,
                visual_provider=self.visual_provider,
                limit=self.settings.visual_retrieval_top_k,
            )
            chunks = [visual_to_text_chunk(v) for v in raw]
            if chunks:
                logger.info(
                    "Visual retrieval returned %d figure(s)",
                    len(chunks),
                    extra={"owner_id": scope.owner_id, "collection_id": scope.collection_id},
                )
            return chunks
        except Exception as exc:
            logger.warning(
                "Visual retrieval failed — skipping",
                extra={"owner_id": scope.owner_id, "error": str(exc)},
            )
            return []

    def build_citation(
        self, hit: "RetrievedVisualChunk"
    ):
        """Convert a visual hit to a CitationSchema for the response."""
        from src.rag.evidence import CitationBuilder, EvidenceBundle

        citation = CitationBuilder.from_evidence_bundle(
            EvidenceBundle.from_visual_hits([hit]),
            owner_id=hit.owner_id,
            api_v1_prefix=self.settings.api_v1_prefix,
        )[0]
        return citation.model_copy(
            update={
                "role": "visual_match",
                "confidence": float(min(max(hit.score or citation.confidence, 0.0), 1.0)),
            }
        )
