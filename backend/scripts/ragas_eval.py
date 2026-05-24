"""RAGAS evaluation over saved e2e eval predictions.

Replaces the homegrown semantic_faithfulness/answer_relevance cosine metrics
with the proper RAGAS LLM-as-judge metrics. Cosine-based metrics suffer when
the LLM paraphrases (our v22 sem_faith = 0.62 is misleadingly low — answers
ARE grounded, just worded differently from source).

Usage:
    python backend/scripts/ragas_eval.py \\
        --input eval_results/e2e_eval_v22_phaseB_only.jsonl \\
        --output eval_results/ragas_v22.json \\
        --judge-model qwen2.5:3b \\
        [--limit 5]   # subset for quick smoke

Notes:
- Uses Ollama locally as the judge LLM (no API key required).
- BGE-M3 from local Ollama for embeddings.
- Skips refused / off-topic queries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_contexts(citations: list[dict], max_blocks: int = 6) -> list[str]:
    """Flatten each citation into a single context string, including all
    evidence_blocks (the snippet_original alone is too narrow for RAGAS).
    """
    contexts: list[str] = []
    for c in citations[:5]:
        parts: list[str] = []
        primary = c.get("snippet_original") or c.get("snippet") or ""
        if primary:
            parts.append(primary)
        for blk in (c.get("evidence_blocks") or [])[:max_blocks]:
            snippet = blk.get("snippet_original") or ""
            if snippet and snippet not in parts:
                parts.append(snippet)
        merged = " ".join(parts).strip()
        if merged:
            contexts.append(merged[:2000])
    return contexts or ["[no context]"]


def to_ragas_dataset(rows: list[dict]):
    from datasets import Dataset
    questions, answers, contexts = [], [], []
    for r in rows:
        if r.get("refused") or r.get("query_type") == "off_topic_should_refuse":
            continue
        if r.get("error", "").startswith("timeout"):
            continue
        questions.append(r["query"])
        answers.append(r["answer"])
        contexts.append(build_contexts(r.get("citations") or []))
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }), len(questions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judge-model", default="qwen2.5:3b", help="Ollama model used by RAGAS as judge")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--limit", type=int, default=None, help="Eval at most N items")
    args = parser.parse_args()

    print(f"\n{'='*65}", flush=True)
    print(f"  RAGAS Evaluation — input: {args.input.name}", flush=True)
    print(f"  Judge LLM: ollama:{args.judge_model}", flush=True)
    print(f"  Embeddings: {args.embedding_model}", flush=True)
    print(f"{'='*65}\n", flush=True)

    rows = load_predictions(args.input)
    print(f"Loaded {len(rows)} prediction rows from {args.input}", flush=True)

    if args.limit:
        rows = rows[: args.limit]
        print(f"--limit {args.limit} → evaluating subset", flush=True)

    dataset, n = to_ragas_dataset(rows)
    if n == 0:
        print("No usable answered/non-refused rows found.")
        sys.exit(2)
    print(f"RAGAS dataset built: {n} examples (refused/off-topic/timeout excluded)\n", flush=True)

    # ── Wire RAGAS to local Ollama LLM ──────────────────────────────────────
    # answer_relevancy needs embeddings — skipped here because langchain-huggingface
    # has a TensorFlow protobuf conflict on this Windows env. faithfulness +
    # context_precision are LLM-only and cover the two metrics that mattered most
    # in v22 (cosine sem_faith=0.62 was misleading; RAGAS LLM-judge clarifies).
    print("Bootstrapping RAGAS LLM (Ollama, embeddings-free metrics only)...", flush=True)
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    # Long timeout: each RAGAS judge call can run ~150-200s on CPU Qwen3-3b.
    # Default httpx 60s would trigger TimeoutError on every prompt.
    judge_llm = LangchainLLMWrapper(
        ChatOllama(
            model=args.judge_model,
            base_url=args.ollama_url,
            temperature=0,
            num_predict=512,
            num_ctx=4096,
            client_kwargs={"timeout": 600},
            async_client_kwargs={"timeout": 600},
        )
    )

    from ragas import evaluate
    from ragas.metrics import faithfulness, LLMContextPrecisionWithoutReference

    metrics = [faithfulness, LLMContextPrecisionWithoutReference()]
    print(f"Running RAGAS over {n} examples...", flush=True)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        raise_exceptions=False,
        show_progress=True,
    )

    print()
    print(f"\n{'='*65}", flush=True)
    print(f"  RAGAS RESULTS", flush=True)
    print(f"{'='*65}", flush=True)
    summary = {}
    for metric_name in ("faithfulness", "llm_context_precision_without_reference"):
        try:
            val = float(result[metric_name])
        except Exception:
            val = None
        summary[metric_name] = val
        print(f"  {metric_name:25s}: {val if val is not None else 'N/A'}")
    print(f"{'='*65}\n", flush=True)

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = result.to_pandas()
    per_query = df.to_dict(orient="records")
    out = {
        "input": str(args.input),
        "n_examples": n,
        "judge_model": args.judge_model,
        "embedding_model": args.embedding_model,
        "aggregate": summary,
        "per_query": per_query,
    }
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
