"""
Run retrieval gold dataset through the API and save retrieved chunks for scoring.

Usage:
    cd backend
    python scripts/run_retrieval_eval.py \
        --gold ../evaluation/datasets/agentbook_retrieval_gold.jsonl \
        --owner-id nguyenvtp69_gmail_com \
        --collection-id 6a2ce7456f898beeba9f44db \
        --output eval_results/retrieval_gold_run.jsonl \
        --api-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ask(*, api_url: str, owner_id: str, collection_id: str, query: str, timeout: int = 120) -> dict:
    resp = requests.post(
        f"{api_url}/api/v1/query/ask",
        json={"query": query, "owner_id": owner_id, "collection_id": collection_id, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def main(args: argparse.Namespace) -> None:
    gold_path = Path(args.gold)
    cases = []
    with gold_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    print(f"\n{'='*60}", flush=True)
    print(f"  Retrieval Eval Runner — {len(cases)} queries", flush=True)
    print(f"  API: {args.api_url}", flush=True)
    print(f"{'='*60}\n", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, case in enumerate(cases):
        query = case["query"]
        print(f"[{i+1:>2}/{len(cases)}] {query[:60]}", flush=True)
        t0 = time.time()
        try:
            data = ask(
                api_url=args.api_url,
                owner_id=args.owner_id or case.get("owner_id", ""),
                collection_id=args.collection_id or case.get("collection_id", ""),
                query=query,
                timeout=args.timeout,
            )
            elapsed = time.time() - t0
            payload = data.get("data") or data
            citations = payload.get("citations") or []

            retrieved_chunks = []
            for c in citations:
                chunk = {
                    "doc_name": c.get("doc_name") or c.get("document_name") or "",
                    "document_name": c.get("doc_name") or c.get("document_name") or "",
                    "page": c.get("page"),
                    "score": c.get("confidence") or 0.0,
                    "content_preview": (c.get("snippet_original") or c.get("snippet") or "")[:300],
                }
                for blk in (c.get("evidence_blocks") or [])[:1]:
                    chunk["block_id"] = blk.get("block_id") or blk.get("id") or ""
                    break
                retrieved_chunks.append(chunk)

            retrieved_docs = list(dict.fromkeys(c["document_name"] for c in retrieved_chunks if c["document_name"]))
            top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0.0

            print(f"        {elapsed:.1f}s  chunks={len(retrieved_chunks)}  top_doc={retrieved_docs[0][:40] if retrieved_docs else 'none'}", flush=True)

            results.append({
                **case,
                "retrieved_chunks": retrieved_chunks,
                "retrieved_docs": retrieved_docs,
                "top_reranker_score": top_score,
                "elapsed_s": round(elapsed, 2),
                "error": None,
            })
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"        ERROR {elapsed:.1f}s: {exc}", flush=True)
            results.append({**case, "retrieved_chunks": [], "retrieved_docs": [], "error": str(exc)})

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} retrieval results → {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--owner-id", default="nguyenvtp69_gmail_com")
    parser.add_argument("--collection-id", default="6a2ce7456f898beeba9f44db")
    parser.add_argument("--output", default="eval_results/retrieval_gold_run.jsonl")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    main(args)
