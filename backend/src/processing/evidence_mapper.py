from __future__ import annotations

import re
import unicodedata

from src.processing.types import EvidenceBlock, EvidenceMap, ParsedDocument


def _content_signature(text: str) -> str:
    """Stable hash for dedup: lowercase + collapse whitespace + strip punctuation.
    Catches PDF double-extraction where same paragraph appears with minor whitespace diffs.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"[^\w\sÀ-ỹ]+", "", normalized)
    return normalized


class EvidenceMapper:
    def build(
        self,
        *,
        parsed: ParsedDocument,
        owner_id: str,
        collection_id: str,
        material_id: str,
        document_name: str,
    ) -> EvidenceMap:
        evidence_blocks: list[EvidenceBlock] = []
        # Track signatures to drop duplicate blocks (PDF text-layer + redraw artifacts).
        # Key by (page, signature) — same content on different pages is kept.
        seen_signatures: set[tuple[int, str]] = set()
        dropped_count = 0
        for block in parsed.blocks:
            sig = _content_signature(block.content)
            # Only dedup substantive blocks (>= 20 chars normalized) to avoid false merges
            if sig and len(sig) >= 20:
                key = (block.page_number, sig)
                if key in seen_signatures:
                    dropped_count += 1
                    continue
                seen_signatures.add(key)
            evidence_blocks.append(
                EvidenceBlock(
                    owner_id=owner_id,
                    collection_id=collection_id,
                    material_id=material_id,
                    document_name=document_name,
                    page=block.page_number,
                    block_id=block.block_id,
                    block_type=block.block_type,
                    snippet_original=block.content,
                    source_language=block.language,
                    bbox=block.bbox,
                    confidence=block.ocr_confidence,
                    metadata={
                        "reading_order": block.reading_order,
                        "source": block.source,
                        **block.extra,
                    },
                )
            )
        if dropped_count > 0:
            import logging
            logging.getLogger(__name__).info(
                "Evidence dedup dropped %d duplicate blocks",
                dropped_count,
                extra={"material_id": material_id, "document_name": document_name},
            )
        return EvidenceMap(
            owner_id=owner_id,
            collection_id=collection_id,
            material_id=material_id,
            document_name=document_name,
            blocks=evidence_blocks,
        )
