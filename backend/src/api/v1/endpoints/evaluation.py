from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.dependencies import get_settings
from src.evaluation.ragas_evaluator import RAGASEvaluator, RAGASMetrics
from src.rag.embedder import BGEM3Embedder

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

_evaluator = RAGASEvaluator()


class EvalSample(BaseModel):
    query: str
    answer: str
    was_refused: bool = False
    chunk_scores: list[float] = Field(default_factory=list, description="Reranker/fused scores of retrieved chunks")


class EvalBatchRequest(BaseModel):
    samples: list[EvalSample]


class EvalResponse(BaseModel):
    faithfulness: float
    citation_coverage: float
    context_precision: float
    refusal_rate: float
    sample_count: int
    note: str = ""


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


@router.post("/embed", response_model=EmbedResponse)
async def embed_texts(body: EmbedRequest, settings=Depends(get_settings)) -> EmbedResponse:
    """Embed texts using BGE-M3 dense vectors. Used for semantic eval metrics."""
    embedder = BGEM3Embedder(settings)
    results = embedder.encode(body.texts)
    return EmbedResponse(embeddings=[r.dense for r in results])


@router.post("/ragas", response_model=EvalResponse)
async def run_ragas_evaluation(
    body: EvalBatchRequest,
    settings=Depends(get_settings),
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
