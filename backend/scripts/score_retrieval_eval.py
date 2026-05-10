"""
Score retrieval eval dataset.
Supports both manual annotation and auto-annotation via keyword matching.

Usage:
    cd backend
    # Auto-annotate + score:
    python scripts/score_retrieval_eval.py --input eval_results/retrieval_eval.jsonl --auto

    # Score after manual annotation:
    python scripts/score_retrieval_eval.py --input eval_results/retrieval_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Keywords expected in top chunk content, by query type
_KEYWORDS: dict[str, list[str]] = {
    "factual": [],        # generic — check query terms in chunk
    "summarization": [],
    "comparison": [],
    "graph_relation": [],
    "claim_check": [],
    "cross_lingual": [],
    "anaphora": [],
    "off_topic_should_refuse": [],   # should NOT find relevant content
    "false_premise": [],
}

# Off-topic queries should retrieve irrelevant chunks (low semantic match)
_OFF_TOPIC_TYPES = {"off_topic_should_refuse"}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def auto_verdict(sample: dict) -> str:
    """
    Auto-annotate based on keyword overlap between query and retrieved content.
    Returns: "correct" | "partial" | "wrong" | "off_topic_ok" | "off_topic_bad"
    """
    qtype = sample.get("query_type", "")
    query_tokens = _tokenize(sample.get("query", ""))
    chunks = sample.get("retrieved_chunks", [])

    if qtype in _OFF_TOPIC_TYPES:
        # For off-topic: if top chunk score < 0.3, retrieval appropriately uncertain
        top_score = chunks[0]["score"] if chunks else 0.0
        # Since IntentClassifier handles refusal at API level, retrieval always returns something.
        # We check: are the retrieved chunks semantically unrelated to the query?
        if not chunks:
            return "off_topic_ok"
        top_content_tokens = _tokenize(chunks[0].get("content_preview", ""))
        overlap = len(query_tokens & top_content_tokens) / max(len(query_tokens), 1)
        return "off_topic_ok" if overlap < 0.15 else "off_topic_bad"

    if not chunks:
        return "wrong"

    # Check overlap of query keywords in top-3 chunks
    all_content = " ".join(c.get("content_preview", "") for c in chunks[:3])
    content_tokens = _tokenize(all_content)

    # Remove stopwords
    stops = {"là", "gì", "và", "của", "trong", "như", "nào", "thế", "có",
              "không", "tại", "vì", "sao", "thế", "nào", "the", "is", "what",
              "how", "does", "why", "a", "an", "of", "in", "to", "do"}
    query_keywords = query_tokens - stops

    if not query_keywords:
        return "partial"

    overlap_ratio = len(query_keywords & content_tokens) / len(query_keywords)

    if overlap_ratio >= 0.5:
        return "correct"
    elif overlap_ratio >= 0.2:
        return "partial"
    else:
        return "wrong"


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save(samples: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def report(samples: list[dict]) -> None:
    reviewed = [s for s in samples if s.get("retrieval_ok") is not None]
    unreviewed = len(samples) - len(reviewed)
    if not reviewed:
        print("No annotations found. Run with --auto or annotate manually.")
        return

    verdicts = Counter(s["retrieval_ok"] for s in reviewed)
    total = len(reviewed)

    correct = verdicts.get("correct", 0)
    partial = verdicts.get("partial", 0)
    wrong = verdicts.get("wrong", 0)
    off_ok = verdicts.get("off_topic_ok", 0)
    off_bad = verdicts.get("off_topic_bad", 0)

    # Retrieval metrics (excluding off-topic)
    rag_samples = [s for s in reviewed if s.get("query_type") not in _OFF_TOPIC_TYPES]
    rag_total = len(rag_samples)

    # Score distribution
    scores = [s.get("top_reranker_score", 0) or (s["retrieved_chunks"][0]["score"] if s.get("retrieved_chunks") else 0)
              for s in rag_samples]
    avg_score = sum(scores) / len(scores) if scores else 0

    # Off-topic handling
    off_topic = [s for s in reviewed if s.get("query_type") in _OFF_TOPIC_TYPES]

    # By query type
    by_type: dict[str, Counter] = {}
    for s in reviewed:
        qt = s.get("query_type", "unknown")
        if qt not in by_type:
            by_type[qt] = Counter()
        by_type[qt][s["retrieval_ok"]] += 1

    w = 52
    print(f"\n{'='*w}")
    print(f"  NOELYS RETRIEVAL EVAL REPORT")
    print(f"{'='*w}")
    print(f"  Samples:     {total}  ({unreviewed} unreviewed)")
    print(f"{'─'*w}")
    print(f"  RAG QUERIES  ({rag_total} samples, excl. off-topic)")
    print(f"    Correct:   {correct:>4}  ({correct/rag_total:.1%})" if rag_total else "")
    print(f"    Partial:   {partial:>4}  ({partial/rag_total:.1%})" if rag_total else "")
    print(f"    Wrong:     {wrong:>4}  ({wrong/rag_total:.1%})" if rag_total else "")
    acc = (correct + 0.5 * partial) / rag_total if rag_total else 0
    print(f"    Accuracy:  {acc:.3f}  (correct + 0.5*partial)")
    print(f"    Avg score: {avg_score:.3f}")
    print(f"{'─'*w}")
    print(f"  OFF-TOPIC HANDLING  ({len(off_topic)} samples)")
    print(f"    Correctly irrelevant: {off_ok}/{len(off_topic)}")
    print(f"    Incorrectly relevant: {off_bad}/{len(off_topic)}")
    print(f"{'─'*w}")
    print(f"  BY QUERY TYPE")
    for qt, counts in sorted(by_type.items()):
        total_t = sum(counts.values())
        ok = counts.get("correct", 0) + counts.get("off_topic_ok", 0)
        print(f"    {qt:<28} {ok}/{total_t}  {dict(counts)}")
    print(f"{'─'*w}")

    # Wrong cases
    wrong_samples = [s for s in reviewed if s.get("retrieval_ok") == "wrong"]
    if wrong_samples:
        print(f"  WRONG RETRIEVALS (fix chunking/embedding for these):")
        for s in wrong_samples[:5]:
            docs = ", ".join(s.get("retrieved_docs", []))
            print(f"    [{s['query_type']}] {s['query'][:55]}")
            print(f"      -> retrieved: {docs[:60]}")
    print(f"{'='*w}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--auto", action="store_true", help="Auto-annotate using keyword matching")
    args = parser.parse_args()

    path = Path(args.input)
    samples = load(path)
    print(f"Loaded {len(samples)} samples from {path.name}")

    if args.auto:
        print("Auto-annotating via keyword matching...")
        for s in samples:
            if s.get("retrieval_ok") is None:
                s["retrieval_ok"] = auto_verdict(s)
        save(samples, path)
        print(f"Saved annotated results -> {path}")

    report(samples)
