from __future__ import annotations

import html
import math
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, Field

from src.processing.types import BBox, EvidenceBlock
from src.rag.types import RetrievedChunk, RetrievedVisualChunk
from src.schemas.evidence import BoundingBoxSchema, CitationSchema, EvidenceBlockSchema


class EvidenceKind(StrEnum):
    TEXT = "text"
    VISUAL = "visual"
    TABLE = "table"
    AUDIO = "audio"


def _confidence_from_chunk(chunk: RetrievedChunk) -> float:
    if chunk.rerank_score is not None:
        score = chunk.rerank_score
        if score >= 0:
            value = 1.0 / (1.0 + math.exp(-score))
        else:
            exp_score = math.exp(score)
            value = exp_score / (1.0 + exp_score)
    elif chunk.graph_score is not None:
        value = chunk.graph_score
    else:
        value = chunk.fused_score if chunk.fused_score else 0.5
    return min(1.0, max(0.0, float(value)))


def _bbox_schema(bbox: BBox | None) -> BoundingBoxSchema | None:
    return BoundingBoxSchema.model_validate(bbox.model_dump()) if bbox is not None else None


def _primary_block_from_chunk(chunk: RetrievedChunk) -> EvidenceBlock | None:
    return chunk.evidence[0] if chunk.evidence else None


def _audio_bounds(evidence: list[EvidenceBlock], metadata: dict[str, Any]) -> tuple[float | None, float | None, str | None]:
    starts: list[float] = []
    ends: list[float] = []
    audio_file = metadata.get("audio_file")
    for ev in evidence:
        meta = ev.metadata or {}
        if audio_file is None:
            audio_file = meta.get("audio_file")
        if meta.get("start_seconds") is not None:
            starts.append(float(meta["start_seconds"]))
        if meta.get("end_seconds") is not None:
            ends.append(float(meta["end_seconds"]))
    if metadata.get("audio_start_seconds") is not None:
        starts.append(float(metadata["audio_start_seconds"]))
    if metadata.get("audio_end_seconds") is not None:
        ends.append(float(metadata["audio_end_seconds"]))
    return (min(starts) if starts else None, max(ends) if ends else None, audio_file)


class BaseEvidence(BaseModel):
    kind: str
    evidence_id: str
    owner_id: str
    collection_id: str
    material_id: str
    document_name: str
    language: str = "unknown"
    score: float = 0.0
    page: int | None = None
    pages: list[int] = Field(default_factory=list)
    block_id: str | None = None
    block_type: str | None = None
    snippet: str = ""
    bbox: BBox | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence_blocks: list[EvidenceBlock] = Field(default_factory=list)

    def prompt_text(self) -> str:
        return self.snippet

    def to_legacy_chunk(self, *, fallback_index: int = 0) -> RetrievedChunk:
        block_id = self.block_id or f"evidence-{fallback_index}"
        ev_blocks = list(self.evidence_blocks)
        if not ev_blocks:
            ev_blocks = [
                EvidenceBlock(
                    owner_id=self.owner_id,
                    collection_id=self.collection_id,
                    material_id=self.material_id,
                    document_name=self.document_name,
                    page=self.page or (self.pages[0] if self.pages else 0),
                    block_id=block_id,
                    block_type=self.block_type or self.kind,
                    snippet_original=self.snippet,
                    source_language=self.language,
                    bbox=self.bbox,
                    confidence=self.score,
                    metadata=dict(self.metadata),
                )
            ]
        chunk_modality = self.kind if self.kind in {"table", "audio"} else "text"
        if self.kind == EvidenceKind.VISUAL.value:
            chunk_modality = "figure"
        return RetrievedChunk(
            chunk_id=self.evidence_id,
            owner_id=self.owner_id,
            collection_id=self.collection_id,
            material_id=self.material_id,
            document_name=self.document_name,
            content=self.prompt_text(),
            language=self.language,
            modality=chunk_modality,
            source_block_ids=[block_id] if block_id else [],
            source_pages=self.pages or ([self.page] if self.page else []),
            bboxes=[self.bbox] if self.bbox else [],
            evidence=ev_blocks,
            metadata=dict(self.metadata),
            fused_score=self.score,
            rerank_score=self.score,
        )


class TextEvidence(BaseEvidence):
    kind: Literal["text"] = EvidenceKind.TEXT.value
    chunk_id: str | None = None
    source_block_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk, *, evidence_id: str | None = None) -> "TextEvidence":
        primary = _primary_block_from_chunk(chunk)
        pages = sorted({e.page for e in chunk.evidence}) if chunk.evidence else list(chunk.source_pages)
        return cls(
            evidence_id=evidence_id or chunk.chunk_id,
            chunk_id=chunk.chunk_id,
            owner_id=chunk.owner_id,
            collection_id=chunk.collection_id,
            material_id=chunk.material_id,
            document_name=chunk.document_name,
            language=chunk.language,
            score=_confidence_from_chunk(chunk),
            page=(primary.page if primary else (pages[0] if pages else None)),
            pages=pages,
            block_id=primary.block_id if primary else (chunk.source_block_ids[0] if chunk.source_block_ids else None),
            block_type=primary.block_type if primary else "text",
            snippet=chunk.content,
            bbox=primary.bbox if primary else (chunk.bboxes[0] if chunk.bboxes else None),
            metadata=dict(chunk.metadata or {}),
            evidence_blocks=list(chunk.evidence or []),
            source_block_ids=list(chunk.source_block_ids or []),
        )


class VisualEvidence(BaseEvidence):
    kind: Literal["visual"] = EvidenceKind.VISUAL.value
    point_id: str | None = None
    caption: str = ""
    image_path: str | None = None

    @classmethod
    def from_visual_chunk(cls, hit: RetrievedVisualChunk, *, evidence_id: str | None = None) -> "VisualEvidence":
        caption = (hit.caption or hit.document_name or "Visual evidence").strip()
        metadata = {"image_path": hit.image_path} if hit.image_path else {}
        ev = EvidenceBlock(
            owner_id=hit.owner_id,
            collection_id=hit.collection_id,
            material_id=hit.material_id,
            document_name=hit.document_name,
            page=hit.page,
            block_id=hit.block_id,
            block_type=hit.block_type or "figure",
            snippet_original=caption,
            source_language=hit.source_language or "unknown",
            bbox=hit.bbox,
            confidence=hit.score,
            metadata={"figure_image_path": hit.image_path} if hit.image_path else {},
        )
        return cls(
            evidence_id=evidence_id or hit.point_id,
            point_id=hit.point_id,
            owner_id=hit.owner_id,
            collection_id=hit.collection_id,
            material_id=hit.material_id,
            document_name=hit.document_name,
            language=hit.source_language or "unknown",
            score=float(hit.score or 0.0),
            page=hit.page,
            pages=[hit.page] if hit.page else [],
            block_id=hit.block_id,
            block_type=hit.block_type or "figure",
            snippet=caption,
            bbox=hit.bbox,
            metadata=metadata,
            evidence_blocks=[ev],
            caption=caption,
            image_path=hit.image_path,
        )

    def prompt_text(self) -> str:
        marker = "attached image available" if self.image_path else "caption-only"
        return f"[Figure: {marker}] {self.caption or self.snippet}".strip()

    def image_path_obj(self) -> Path | None:
        if not self.image_path:
            return None
        path = Path(self.image_path)
        return path if path.exists() else None


class TableEvidence(BaseEvidence):
    kind: Literal["table"] = EvidenceKind.TABLE.value
    chunk_id: str | None = None
    sheet_name: str | None = None
    cell_ref: str | None = None
    source_block_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk, *, evidence_id: str | None = None) -> "TableEvidence":
        base = TextEvidence.from_chunk(chunk, evidence_id=evidence_id or chunk.chunk_id)
        metadata = dict(chunk.metadata or {})
        sheets = metadata.get("sheet_names") or []
        return cls(
            evidence_id=base.evidence_id,
            chunk_id=chunk.chunk_id,
            owner_id=base.owner_id,
            collection_id=base.collection_id,
            material_id=base.material_id,
            document_name=base.document_name,
            language=base.language,
            score=base.score,
            page=base.page,
            pages=base.pages,
            block_id=base.block_id,
            block_type=base.block_type or "table",
            snippet=base.snippet,
            bbox=base.bbox,
            metadata=metadata,
            evidence_blocks=base.evidence_blocks,
            sheet_name=sheets[0] if sheets else metadata.get("sheet_name"),
            cell_ref=metadata.get("cell_ref"),
            source_block_ids=list(chunk.source_block_ids or []),
        )

    @classmethod
    def from_aggregation(cls, *, result: Any, chunk: RetrievedChunk, answer: str, evidence_id: str = "table-aggregation") -> "TableEvidence":
        table = cls.from_chunk(chunk, evidence_id=evidence_id)
        metadata = dict(table.metadata)
        metadata.update(
            {
                "operation": getattr(result, "operation", None),
                "column": getattr(result, "column", None),
                "value": getattr(result, "value", None),
                "n_rows": getattr(result, "n_rows", None),
                "arg_label": getattr(result, "arg_label", None),
                "label_column": getattr(result, "label_column", None),
            }
        )
        table.metadata = metadata
        table.sheet_name = getattr(result, "sheet_name", None) or table.sheet_name
        table.source_block_ids = list(getattr(result, "source_block_ids", None) or table.source_block_ids)
        table.snippet = answer
        return table

    @classmethod
    def from_lookup(cls, *, result: Any, chunk: RetrievedChunk, answer: str, evidence_id: str = "table-lookup") -> "TableEvidence":
        table = cls.from_chunk(chunk, evidence_id=evidence_id)
        metadata = dict(table.metadata)
        metadata.update(
            {
                "operation": "lookup",
                "column": getattr(result, "column", None),
                "row_label": getattr(result, "row_label", None),
                "value": getattr(result, "value", None),
                "row_index": getattr(result, "row_index", None),
                "column_index": getattr(result, "column_index", None),
            }
        )
        table.metadata = metadata
        table.sheet_name = getattr(result, "sheet_name", None) or table.sheet_name
        table.cell_ref = getattr(result, "cell_ref", None) or table.cell_ref
        table.source_block_ids = list(getattr(result, "source_block_ids", None) or table.source_block_ids)
        table.snippet = answer
        return table


class AudioEvidence(BaseEvidence):
    kind: Literal["audio"] = EvidenceKind.AUDIO.value
    chunk_id: str | None = None
    transcript: str = ""
    audio_start_seconds: float | None = None
    audio_end_seconds: float | None = None
    audio_file: str | None = None

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk, *, evidence_id: str | None = None) -> "AudioEvidence":
        base = TextEvidence.from_chunk(chunk, evidence_id=evidence_id or chunk.chunk_id)
        start, end, audio_file = _audio_bounds(list(chunk.evidence or []), dict(chunk.metadata or {}))
        return cls(
            evidence_id=base.evidence_id,
            chunk_id=chunk.chunk_id,
            owner_id=base.owner_id,
            collection_id=base.collection_id,
            material_id=base.material_id,
            document_name=base.document_name,
            language=base.language,
            score=base.score,
            page=base.page,
            pages=base.pages,
            block_id=base.block_id,
            block_type=base.block_type or "audio",
            snippet=base.snippet,
            bbox=base.bbox,
            metadata=dict(chunk.metadata or {}),
            evidence_blocks=base.evidence_blocks,
            transcript=base.snippet,
            audio_start_seconds=start,
            audio_end_seconds=end,
            audio_file=audio_file,
        )


EvidenceItem = TextEvidence | VisualEvidence | TableEvidence | AudioEvidence


class EvidenceBundle(BaseModel):
    items: list[EvidenceItem] = Field(default_factory=list)

    @classmethod
    def from_chunks(cls, chunks: list[RetrievedChunk]) -> "EvidenceBundle":
        return cls(items=[evidence_from_chunk(chunk) for chunk in chunks])

    @classmethod
    def from_visual_hits(cls, hits: list[RetrievedVisualChunk]) -> "EvidenceBundle":
        return cls(items=[VisualEvidence.from_visual_chunk(hit) for hit in hits])

    def with_items(self, items: list[EvidenceItem]) -> "EvidenceBundle":
        return EvidenceBundle(items=[*self.items, *items])

    def select_indices(self, indices: list[int]) -> "EvidenceBundle":
        selected: list[EvidenceItem] = []
        for idx in indices:
            if 0 <= idx < len(self.items):
                selected.append(self.items[idx])
        return EvidenceBundle(items=selected)

    def to_legacy_chunks(self) -> list[RetrievedChunk]:
        return [item.to_legacy_chunk(fallback_index=i) for i, item in enumerate(self.items, start=1)]

    def visual_items(self) -> list[VisualEvidence]:
        return [item for item in self.items if isinstance(item, VisualEvidence)]

    def image_paths(self, *, limit: int | None = None) -> list[Path]:
        paths: list[Path] = []
        for item in self.visual_items():
            path = item.image_path_obj()
            if path is None:
                continue
            paths.append(path)
            if limit is not None and len(paths) >= limit:
                break
        return paths

    def kind_counts(self) -> dict[str, int]:
        counts = {kind.value: 0 for kind in EvidenceKind}
        for item in self.items:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        return counts

    def format_for_prompt(self) -> str:
        lines: list[str] = []
        for index, item in enumerate(self.items, start=1):
            page_str = "N/A"
            pages = item.pages or ([item.page] if item.page else [])
            if pages:
                page_str = f"trang {pages[0]}" if len(pages) == 1 else f"trang {pages[0]}-{pages[-1]}"
            meta_bits = []
            if isinstance(item, TableEvidence) and item.sheet_name:
                meta_bits.append(f"sheet={item.sheet_name}")
            if isinstance(item, AudioEvidence) and item.audio_start_seconds is not None:
                end = item.audio_end_seconds if item.audio_end_seconds is not None else item.audio_start_seconds
                meta_bits.append(f"time={item.audio_start_seconds:.1f}-{end:.1f}s")
            if isinstance(item, VisualEvidence):
                meta_bits.append("image_attached=true" if item.image_path else "image_attached=false")
            meta = f' metadata="{html.escape("; ".join(meta_bits), quote=True)}"' if meta_bits else ""
            lines.append(
                f'<EVIDENCE id="{index}" citation="[{index}]" kind="{html.escape(item.kind, quote=True)}" '
                f'source="{html.escape(item.document_name or "", quote=True)}" pages="{html.escape(page_str, quote=True)}"{meta}>\n'
                f"{html.escape(item.prompt_text() or '')}\n"
                f"</EVIDENCE>"
            )
        return "\n\n".join(lines)


def evidence_from_chunk(chunk: RetrievedChunk, *, evidence_id: str | None = None) -> EvidenceItem:
    modality = (chunk.modality or "").lower()
    if modality == "table" or (chunk.metadata or {}).get("sheet_names"):
        return TableEvidence.from_chunk(chunk, evidence_id=evidence_id)
    if modality == "audio" or (chunk.metadata or {}).get("audio_start_seconds"):
        return AudioEvidence.from_chunk(chunk, evidence_id=evidence_id)
    return TextEvidence.from_chunk(chunk, evidence_id=evidence_id)


class CitationBuilder:
    @classmethod
    def from_evidence_bundle(
        cls,
        bundle: EvidenceBundle,
        *,
        owner_id: str | None = None,
        api_v1_prefix: str = "/api/v1",
    ) -> list[CitationSchema]:
        citations: list[CitationSchema] = []
        for index, item in enumerate(bundle.items, start=1):
            citations.append(
                CitationSchema(
                    doc_id=item.material_id,
                    doc_name=item.document_name,
                    page=item.page,
                    pages=item.pages or ([item.page] if item.page else []),
                    block_id=item.block_id,
                    block_type=item.block_type or item.kind,
                    snippet_original=item.snippet or item.prompt_text(),
                    snippet_translated=None,
                    bbox=_bbox_schema(item.bbox),
                    role="primary" if index == 1 else "supporting",
                    source_language=item.language,
                    confidence=float(min(max(item.score or 0.0, 0.0), 1.0)),
                    evidence_blocks=cls._evidence_blocks(item, owner_id=owner_id, api_v1_prefix=api_v1_prefix),
                    evidence_id=item.evidence_id,
                    kind=item.kind,
                    figure_image_url=cls._figure_url(item, owner_id=owner_id, api_v1_prefix=api_v1_prefix),
                    sheet_name=getattr(item, "sheet_name", None),
                    cell_ref=getattr(item, "cell_ref", None),
                    audio_start_seconds=getattr(item, "audio_start_seconds", None),
                    audio_end_seconds=getattr(item, "audio_end_seconds", None),
                )
            )
        return citations

    @classmethod
    def _evidence_blocks(
        cls,
        item: EvidenceItem,
        *,
        owner_id: str | None,
        api_v1_prefix: str,
    ) -> list[EvidenceBlockSchema]:
        blocks = list(item.evidence_blocks or [])
        if not blocks:
            blocks = [
                EvidenceBlock(
                    owner_id=item.owner_id,
                    collection_id=item.collection_id,
                    material_id=item.material_id,
                    document_name=item.document_name,
                    page=item.page or 0,
                    block_id=item.block_id or item.evidence_id,
                    block_type=item.block_type or item.kind,
                    snippet_original=item.snippet,
                    source_language=item.language,
                    bbox=item.bbox,
                    confidence=item.score,
                    metadata=dict(item.metadata),
                )
            ]
        result: list[EvidenceBlockSchema] = []
        for block in blocks:
            meta = block.metadata or {}
            fig_url = None
            if block.block_type == "figure" and owner_id and (meta.get("figure_image_path") or getattr(item, "image_path", None)):
                fig_url = cls._figure_url(item, owner_id=owner_id, api_v1_prefix=api_v1_prefix)
            result.append(
                EvidenceBlockSchema(
                    block_id=block.block_id,
                    block_type=block.block_type,
                    page=block.page,
                    snippet_original=block.snippet_original,
                    source_language=block.source_language,
                    bbox=_bbox_schema(block.bbox),
                    confidence=block.confidence,
                    material_id=block.material_id,
                    doc_name=block.document_name,
                    audio_start_seconds=meta.get("start_seconds"),
                    audio_end_seconds=meta.get("end_seconds"),
                    audio_file=meta.get("audio_file"),
                    figure_image_url=fig_url,
                )
            )
        return result

    @staticmethod
    def _figure_url(
        item: EvidenceItem,
        *,
        owner_id: str | None,
        api_v1_prefix: str,
    ) -> str | None:
        has_image = bool(getattr(item, "image_path", None) or (item.metadata or {}).get("image_path") or (item.metadata or {}).get("figure_image_path"))
        if not owner_id or item.kind != EvidenceKind.VISUAL.value or not item.block_id or not has_image:
            return None
        return (
            f"{api_v1_prefix}/evidence/figure/{item.material_id}"
            f"?block_id={quote(item.block_id, safe='')}&owner_id={quote(owner_id, safe='')}"
        )


class EvidenceAssembler:
    """Fuse modality-native retrieval outputs into a citation-ordered bundle."""

    @staticmethod
    def assemble(
        *,
        text_chunks: list[RetrievedChunk] | None = None,
        visual_hits: list[RetrievedVisualChunk] | None = None,
        visual_first: bool = False,
    ) -> EvidenceBundle:
        text_items = [evidence_from_chunk(chunk) for chunk in (text_chunks or [])]
        visual_items = [VisualEvidence.from_visual_chunk(hit) for hit in (visual_hits or [])]
        items: list[EvidenceItem] = [*visual_items, *text_items] if visual_first else [*text_items, *visual_items]
        return EvidenceBundle(items=items)


_VISUAL_SIGNAL_RE = re.compile(
    r"\b(image|figure|picture|photo|chart|diagram|graph|plot|visual|hinh|anh|bieu do|so do)\b",
    re.IGNORECASE,
)


class EvidenceFusionRanker:
    """Modality-aware fusion for citation-ordered evidence bundles.

    The low-level retrievers still return their native objects. This class is the
    single place where route policy decides which evidence becomes first-class
    context and in what citation order.
    """

    _POLICY_BOOSTS: dict[str, dict[str, float]] = {
        "visual_first": {EvidenceKind.VISUAL.value: 1.0, EvidenceKind.TEXT.value: 0.12},
        "table_first": {EvidenceKind.TABLE.value: 1.0, EvidenceKind.TEXT.value: 0.10},
        "audio_first": {EvidenceKind.AUDIO.value: 1.0, EvidenceKind.TEXT.value: 0.10},
        "text": {EvidenceKind.TEXT.value: 0.25, EvidenceKind.TABLE.value: 0.08, EvidenceKind.AUDIO.value: 0.08},
    }

    def __init__(self, *, settings: Any | None = None) -> None:
        self.settings = settings
        self.last_scores: dict[str, float] = {}
        self.last_policy: str = "text"

    @staticmethod
    def policy_for_modality(preferred_modality: str | None) -> str:
        if preferred_modality == "figure":
            return "visual_first"
        if preferred_modality == "table":
            return "table_first"
        if preferred_modality == "audio":
            return "audio_first"
        return "text"

    def fuse(
        self,
        *,
        query: str,
        text_chunks: list[RetrievedChunk] | None = None,
        visual_hits: list[RetrievedVisualChunk] | None = None,
        preferred_modality: str | None = None,
        route_type: str | None = None,
        final_limit: int | None = None,
        include_visual: bool | None = None,
    ) -> EvidenceBundle:
        policy = self.policy_for_modality(preferred_modality)
        self.last_policy = policy
        text_items = [evidence_from_chunk(chunk) for chunk in (text_chunks or [])]
        visual_items = [VisualEvidence.from_visual_chunk(hit) for hit in (visual_hits or [])]

        if include_visual is None:
            include_visual = policy == "visual_first" or self._query_has_visual_signal(query)
        if policy == "text" and visual_items and include_visual:
            linked_materials = {item.material_id for item in text_items}
            visual_items = [
                item for item in visual_items
                if self._query_has_visual_signal(query) or item.material_id in linked_materials
            ]
        elif not include_visual:
            visual_items = []

        candidates: list[tuple[EvidenceItem, int, float]] = []
        for order, item in enumerate([*text_items, *visual_items]):
            candidates.append((item, order, self._score_item(item=item, query=query, policy=policy)))

        deduped: dict[tuple[str, str], tuple[EvidenceItem, int, float]] = {}
        for item, order, score in candidates:
            key = (item.kind, item.evidence_id)
            existing = deduped.get(key)
            if existing is None or score > existing[2]:
                deduped[key] = (item, order, score)

        ranked = sorted(
            deduped.values(),
            key=lambda entry: (
                entry[2],
                -entry[1],
            ),
            reverse=True,
        )
        if final_limit is not None and final_limit > 0:
            ranked = ranked[:final_limit]

        items: list[EvidenceItem] = []
        for item, _, score in ranked:
            raw_score = float(item.score or 0.0)
            item.metadata = {
                **dict(item.metadata or {}),
                "retrieval_score_raw": raw_score,
                "fusion_score": round(float(score), 4),
                "fusion_policy": policy,
            }
            item.score = self._confidence_from_scores(item=item, raw_score=raw_score, policy=policy)
            items.append(item)
        self.last_scores = {item.evidence_id: round(score, 4) for item, _, score in ranked}
        return EvidenceBundle(items=items)

    def trace_metadata(self, bundle: EvidenceBundle) -> dict[str, Any]:
        return {
            "fusion_scores": dict(self.last_scores),
            "selected_evidence_ids": [item.evidence_id for item in bundle.items],
            "modality_policy": self.last_policy,
        }

    @staticmethod
    def _query_has_visual_signal(query: str) -> bool:
        from src.processing.slug import ascii_fold

        return bool(_VISUAL_SIGNAL_RE.search(ascii_fold(query or "").lower()))

    def _score_item(self, *, item: EvidenceItem, query: str, policy: str) -> float:
        boosts = self._POLICY_BOOSTS.get(policy, self._POLICY_BOOSTS["text"])
        score = float(item.score or 0.0) + boosts.get(item.kind, 0.0)
        if isinstance(item, TableEvidence):
            score += self._lexical_overlap_bonus(query, " ".join([
                item.snippet or "",
                item.sheet_name or "",
                item.cell_ref or "",
                str(item.metadata.get("column") or ""),
                str(item.metadata.get("row_label") or ""),
            ]))
        elif isinstance(item, AudioEvidence):
            score += self._lexical_overlap_bonus(query, item.transcript or item.snippet)
            if item.audio_start_seconds is not None:
                score += 0.05
        elif isinstance(item, VisualEvidence):
            score += self._lexical_overlap_bonus(query, item.caption or item.snippet)
            if item.image_path:
                score += 0.12
        else:
            score += self._lexical_overlap_bonus(query, item.snippet)
        return score

    @staticmethod
    def _confidence_from_scores(*, item: EvidenceItem, raw_score: float, policy: str) -> float:
        raw_conf = max(0.0, min(1.0, raw_score))
        if isinstance(item, VisualEvidence):
            # SigLIP/Qdrant cosine scores for good image matches are often small
            # decimals, so raw score alone can collapse confidence to ~0. Keep
            # ranking boosts separate, and expose a calibrated confidence floor
            # only when actual image evidence exists.
            floor = 0.55 if item.image_path else 0.30
            return round(max(raw_conf, min(0.85, floor + raw_conf * 0.8)), 4)
        if isinstance(item, TableEvidence) and policy == "table_first":
            return round(max(raw_conf, 0.65), 4)
        if isinstance(item, AudioEvidence) and policy == "audio_first":
            floor = 0.60 if item.audio_start_seconds is not None else 0.45
            return round(max(raw_conf, floor), 4)
        return round(raw_conf, 4)

    @staticmethod
    def _lexical_overlap_bonus(query: str, text: str) -> float:
        from src.processing.slug import ascii_fold

        q_tokens = {t for t in re.findall(r"[a-z0-9]{2,}", ascii_fold(query or "").lower())}
        if not q_tokens:
            return 0.0
        t_tokens = {t for t in re.findall(r"[a-z0-9]{2,}", ascii_fold(text or "").lower())}
        if not t_tokens:
            return 0.0
        overlap = len(q_tokens & t_tokens)
        return min(0.18, overlap / max(1, len(q_tokens)) * 0.18)
