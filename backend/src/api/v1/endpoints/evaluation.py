from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from beanie import PydanticObjectId

from src.dependencies import get_settings, require_admin_access, verify_owner_access
from src.evaluation.ragas_evaluator import RAGASEvaluator
from src.models.chunk import Chunk
from src.models.material import Material
from src.rag.embedder import BGEM3Embedder

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

_evaluator = RAGASEvaluator()
_embed_semaphore = asyncio.Semaphore(1)


class EvalSample(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=12000)
    was_refused: bool = False
    chunk_scores: list[float] = Field(default_factory=list, max_length=100, description="Reranker/fused scores of retrieved chunks")


class EvalBatchRequest(BaseModel):
    samples: list[EvalSample] = Field(min_length=1, max_length=200)


class EvalResponse(BaseModel):
    faithfulness: float
    citation_coverage: float
    context_precision: float
    refusal_rate: float
    sample_count: int
    note: str = ""


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


@router.post("/embed", response_model=EmbedResponse)
async def embed_texts(
    body: EmbedRequest,
    settings=Depends(get_settings),
    _: None = Depends(require_admin_access),
) -> EmbedResponse:
    """Embed texts using BGE-M3 dense vectors. Used for semantic eval metrics."""
    embedder = BGEM3Embedder(settings)
    async with _embed_semaphore:
        results = await asyncio.to_thread(embedder.encode, body.texts)
    return EmbedResponse(embeddings=[r.dense for r in results])


@router.post("/ragas", response_model=EvalResponse)
async def run_ragas_evaluation(
    body: EvalBatchRequest,
    settings=Depends(get_settings),
    _: None = Depends(require_admin_access),
) -> EvalResponse:
    """
    Run lightweight RAGAS-style evaluation on a batch of QA samples.

    Metrics computed (no LLM calls — score-based):
    - faithfulness: fraction of answer sentences that contain a citation [N]
    - citation_coverage: fraction of paragraphs with at least one citation
    - context_precision: fraction of chunks with score >= 0.4
    - refusal_rate: fraction of samples that were refused
    """
    ev = RAGASEvaluator()

    for sample in body.samples:
        faith = ev.evaluate_faithfulness(answer=sample.answer)
        cov = ev.evaluate_citation_coverage(answer=sample.answer)

        # Build dummy chunk proxies from scores
        class _FakeChunk:
            def __init__(self, score: float):
                self.reranker_score = score
                self.fused_score = score

        fake_chunks = [_FakeChunk(s) for s in sample.chunk_scores]
        ctx_prec = ev.evaluate_context_precision(chunks=fake_chunks)  # type: ignore[arg-type]

        ev.record(
            query=sample.query,
            answer=sample.answer,
            chunks=fake_chunks,  # type: ignore[arg-type]
            was_refused=sample.was_refused,
            faithfulness=faith,
            answer_relevancy=-1.0,
            context_precision=ctx_prec,
            citation_coverage=cov,
        )

    metrics = ev.aggregate()
    return EvalResponse(
        faithfulness=round(metrics.faithfulness, 4),
        citation_coverage=round(metrics.citation_coverage, 4),
        context_precision=round(metrics.context_precision, 4),
        refusal_rate=round(metrics.refusal_rate, 4),
        sample_count=metrics.sample_count,
        note="answer_relevancy requires LLM — use /evaluation/ragas-llm endpoint for full metrics",
    )


class ChunkMeta(BaseModel):
    chunk_id: str
    material_id: str
    document_name: str
    content_preview: str
    token_count: int
    modality: str
    page: int | None
    block_id: str
    source_language: str


@router.get("/chunks", response_model=list[ChunkMeta])
async def list_chunks_for_eval(
    request: Request,
    owner_id: str = Query(..., min_length=1),
    collection_id: str = Query(..., min_length=1),
    limit: int = Query(default=500, ge=1, le=2000),
    min_content_len: int = Query(default=150, ge=0),
) -> list[ChunkMeta]:
    """Return chunk metadata for eval dataset generation.

    Used by evaluation/cli/generate_dataset.py when --api-url is set,
    so the script doesn't need a direct MongoDB Atlas connection.
    """
    verify_owner_access(request, owner_id)
    col_oid = PydanticObjectId(collection_id)

    # Fetch materials first for document name lookup
    mats = await Material.find(
        Material.owner_id == owner_id,
        Material.collection_id == col_oid,
    ).to_list()
    mat_names: dict[str, str] = {str(m.id): m.original_name for m in mats}

    chunks = await Chunk.find(
        Chunk.owner_id == owner_id,
        Chunk.collection_id == col_oid,
    ).limit(limit).to_list()

    result: list[ChunkMeta] = []
    for c in chunks:
        content = (c.content or "").strip()
        if len(content) < min_content_len:
            continue
        result.append(ChunkMeta(
            chunk_id=str(c.id),
            material_id=str(c.material_id),
            document_name=mat_names.get(str(c.material_id), str(c.material_id)),
            content_preview=content[:600],
            token_count=c.token_count or 0,
            modality=c.modality or "text",
            page=(c.source_pages or [None])[0],
            block_id=(c.source_block_ids or [""])[0],
            source_language=c.language or "vi",
        ))
    return result
