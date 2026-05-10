"""
Auto-generate golden QA dataset from uploaded documents.

Usage:
    cd backend
    python scripts/generate_eval_dataset.py \
        --owner-id user_demo \
        --collection-id 69fc3c0949fae4625be50223 \
        --output eval_results/eval_dataset.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import httpx
from beanie import PydanticObjectId

# ── bootstrap ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import get_settings
from src.database import init_database
from src.models.chunk import Chunk

_QUESTION_GEN_PROMPT = """\
Read the following passage from a learning document and write {n} questions a student might ask.
One question per line. No numbering. No extra text. Questions only.

Passage:
{chunk}

Questions:\
"""

# Queries that should be REFUSED (off-topic / false premise)
_ADVERSARIAL = [
    {"query": "Thủ đô của nước Pháp là gì?",                         "expect_refused": True,  "type": "off_topic"},
    {"query": "Cho tôi biết thời tiết hôm nay",                       "expect_refused": True,  "type": "off_topic"},
    {"query": "Viết một bài thơ về mùa xuân",                         "expect_refused": True,  "type": "off_topic"},
    {"query": "Tại sao dropout làm tăng overfitting?",                 "expect_refused": False, "type": "false_premise"},
    {"query": "Vì sao gradient descent luôn tìm được global minimum?", "expect_refused": False, "type": "false_premise"},
    {"query": "nó ảnh hưởng thế nào?",                                "expect_refused": False, "type": "anaphora"},
]


async def fetch_chunks(*, collection_id: str, owner_id: str, limit: int = 30) -> list[Chunk]:
    """Sample diverse chunks from collection via MongoDB."""
    col_oid = PydanticObjectId(collection_id)
    chunks = await Chunk.find(
        Chunk.collection_id == col_oid,
        Chunk.owner_id == owner_id,
    ).limit(limit).to_list()
    # Filter: at least 150 chars of content
    return [c for c in chunks if len((c.content or "").strip()) >= 150]


async def generate_questions(*, chunk_content: str, n: int, ollama_url: str, model: str) -> list[str]:
    """Call Ollama to generate questions from a chunk — one per line."""
    prompt = _QUESTION_GEN_PROMPT.format(chunk=chunk_content[:800], n=n)
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.3, "num_predict": 300}},
            )
            if resp.status_code != 200:
                return []
            raw = resp.json().get("response", "")
            # Parse line by line — robust to broken JSON
            questions = []
            for line in raw.splitlines():
                line = line.strip().lstrip("-•*123456789. ")
                if len(line) >= 10 and "?" in line:
                    questions.append(line)
                if len(questions) >= n:
                    break
            return questions
    except Exception as exc:
        print(f"    [warn] question gen failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    return []


async def run_query(*, api_url: str, owner_id: str, collection_id: str, query: str) -> dict:
    """Send query to the system and return response dict."""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{api_url}/api/v1/query/ask",
                json={
                    "owner_id": owner_id,
                    "collection_id": collection_id,
                    "query": query,
                    "conversation_id": "eval_run",
                    "answer_language": "vi",
                },
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                return resp.json().get("data", {})
            print(f"    [warn] HTTP {resp.status_code}: {resp.text[:100]}", file=sys.stderr)
    except Exception as exc:
        print(f"    [error] {type(exc).__name__}: {exc}", file=sys.stderr)
    return {}


def compute_baseline(samples: list[dict]) -> dict:
    total = len(samples)
    if not total:
        return {}
    refused = sum(1 for s in samples if s.get("was_refused"))
    non_refused = [s for s in samples if not s.get("was_refused")]
    has_cit = sum(1 for s in non_refused if s.get("citations"))
    avg_conf = sum(s.get("confidence", 0) for s in samples) / total
    cite_re = re.compile(r"\[\d+\]")
    sent_re = re.compile(r"[^.!?\n]+[.!?]?")
    faith_scores = []
    for s in non_refused:
        sents = [t.strip() for t in sent_re.findall(s.get("answer", "")) if len(t.strip()) >= 10]
        if sents:
            faith_scores.append(sum(1 for t in sents if cite_re.search(t)) / len(sents))
    faithfulness = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0
    latencies = [s["latency_s"] for s in samples if s.get("latency_s")]
    return {
        "total": total,
        "refusal_rate": round(refused / total, 3),
        "citation_rate": round(has_cit / len(non_refused), 3) if non_refused else 0.0,
        "avg_confidence": round(avg_conf, 3),
        "faithfulness_proxy": round(faithfulness, 3),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "p95_latency_s": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) >= 5 else 0.0,
    }


async def main(args: argparse.Namespace) -> None:
    settings = get_settings()

    print(f"\n{'='*60}")
    print(f"  Noelys Eval Dataset Generator")
    print(f"  Owner:      {args.owner_id}")
    print(f"  Collection: {args.collection_id}")
    print(f"  API:        {args.api_url}")
    print(f"{'='*60}\n")

    # ── 1. Connect DB & fetch chunks ─────────────────────────────────────────
    print("Step 1/4  Connecting to database and sampling chunks...")
    await init_database(settings)
    chunks = await fetch_chunks(
        collection_id=args.collection_id,
        owner_id=args.owner_id,
        limit=args.max_chunks,
    )
    if not chunks:
        print("[ERROR] No indexed chunks found for this collection/owner.", file=sys.stderr)
        sys.exit(1)
    print(f"  Sampled {len(chunks)} chunks\n")

    # ── 2. Generate questions via Ollama ─────────────────────────────────────
    print("Step 2/4  Generating questions from chunk content...")
    all_queries: list[dict] = []
    for i, chunk in enumerate(chunks):
        mat_id = str(chunk.material_id)
        print(f"  [{i+1}/{len(chunks)}] chunk {str(chunk.id)[:12]}... ({chunk.token_count or '?'} tokens)")
        questions = await generate_questions(
            chunk_content=chunk.content or "",
            n=args.questions_per_chunk,
            ollama_url=settings.ollama_base_url,
            model=settings.llm_local_model,
        )
        print(f"    -> {len(questions)} questions generated")
        for q in questions:
            all_queries.append({
                "query": q,
                "query_type": "generated",
                "source_chunk_id": str(chunk.id),
                "source_material_id": mat_id,
                "expect_refused": False,
            })

    # Add adversarial queries
    for adv in _ADVERSARIAL:
        all_queries.append({**adv, "source_chunk_id": None, "source_material_id": None})

    print(f"\n  Total queries: {len(all_queries)}\n")

    # ── 3. Run each query through the system ─────────────────────────────────
    print("Step 3/4  Running queries through /query/ask...")
    samples: list[dict] = []
    for i, item in enumerate(all_queries):
        q = item["query"]
        print(f"  [{i+1:>3}/{len(all_queries)}] {q[:72]}")
        t0 = time.perf_counter()
        resp = await run_query(
            api_url=args.api_url,
            owner_id=args.owner_id,
            collection_id=args.collection_id,
            query=q,
        )
        latency = round(time.perf_counter() - t0, 2)
        refused = resp.get("was_refused", True)
        conf = resp.get("confidence", 0.0)
        n_cit = len(resp.get("citations", []))
        print(f"        conf={conf:.2f}  citations={n_cit}  refused={refused}  {latency}s")

        samples.append({
            "id": f"q{i+1:04d}",
            "query": q,
            "query_type": item.get("query_type", "generated"),
            "source_chunk_id": item.get("source_chunk_id"),
            "source_material_id": item.get("source_material_id"),
            "expect_refused": item.get("expect_refused", False),
            "answer": resp.get("answer", ""),
            "answer_language": resp.get("answer_language", ""),
            "confidence": conf,
            "was_refused": refused,
            "refusal_reason": resp.get("refusal_reason"),
            "citations": resp.get("citations", []),
            "latency_s": latency,
            # Fill these manually after review:
            "human_verdict": None,   # "correct" | "partial" | "wrong" | "refused_correctly" | "should_not_refuse"
            "human_notes": "",
        })
        await asyncio.sleep(1.0)

    # ── 4. Save & print baseline ──────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    metrics = compute_baseline(samples)
    print(f"\nStep 4/4  Saved {len(samples)} samples → {out.resolve()}\n")
    print(f"{'='*50}")
    print("  BASELINE METRICS (before human review)")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")
    print(f"{'='*50}")
    print(f"\n  Next: open {out.name}, set 'human_verdict' per sample")
    print(f"  Then: python scripts/score_eval_dataset.py --input {out}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--questions-per-chunk", type=int, default=4)
    parser.add_argument("--max-chunks", type=int, default=15)
    parser.add_argument("--output", default="eval_results/eval_dataset.jsonl")
    args = parser.parse_args()
    asyncio.run(main(args))
