from __future__ import annotations

from src.processing.types import EvidenceBlock, EvidenceMap, ParsedDocument


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
        for block in parsed.blocks:
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
        return EvidenceMap(
            owner_id=owner_id,
            collection_id=collection_id,
            material_id=material_id,
            document_name=document_name,
            blocks=evidence_blocks,
        )
