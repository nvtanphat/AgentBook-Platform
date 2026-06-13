"""
Score retrieval eval dataset.
Supports both manual annotation and auto-annotation via keyword matching.
When --gold is provided, computes proper IR metrics: Recall@K, MRR, nDCG@K.

Usage:
    cd backend
    # Auto-annotate + score:
    python scripts/score_retrieval_eval.py --input eval_results/retrieval_eval.jsonl --auto

    # Score with gold labels + IR metrics + Markdown report:
    python scripts/score_retrieval_eval.py \\
        --input eval_results/retrieval_gold_run.jsonl \\
        --gold ../evaluation/datasets/agentbook_retrieval_gold.jsonl \\
        --metrics recall@1,recall@3,recall@5,mrr,ndcg@5 \\
        --report eval_results/retrieval_gold_report.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Keywords expected in top chunk content, by query type
_KEYWORDS: dict[str, list[str]] = {
    "factual": [],
    "summarization": [],
    "comparison": [],
    "graph_relation": [],
    "claim_check": [],
    "cross_lingual": [],
    "anaphora": [],
    "off_topic_should_refuse": [],
    "false_premise": [],
}

_OFF_TOPIC_TYPES = {"off_topic_should_refuse"}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


# ── Auto-annotation ────────────────────────────────────────────────────────────

def auto_verdict(sample: dict) -> str:
    """
    Auto-annotate based on keyword overlap between query and retrieved content.
    Returns: "correct" | "partial" | "wrong" | "off_topic_ok" | "off_topic_bad"
    """
    qtype = sample.get("query_type", "")
    query_tokens = _tokenize(sample.get("query", ""))
    chunks = sample.get("retrieved_chunks", [])

    if qtype in _OFF_TOPIC_TYPES:
        if not chunks:
            return "off_topic_ok"
        top_content_tokens = _tokenize(chunks[0].get("content_preview", ""))
        overlap = len(query_tokens & top_content_tokens) / max(len(query_tokens), 1)
        return "off_topic_ok" if overlap < 0.15 else "off_topic_bad"

    if not chunks:
        return "wrong"

    all_content = " ".join(c.get("content_preview", "") for c in chunks[:3])
    content_tokens = _tokenize(all_content)

    stops = {"là", "gì", "và", "của", "trong", "như", "nào", "thế", "có",
              "không", "tại", "vì", "sao", "the", "is", "what",
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


# ── IR Metrics ─────────────────────────────────────────────────────────────────

def _is_relevant(chunk: dict, expected_docs: list[dict]) -> bool:
    """True when a retrieved chunk matches any gold expected_doc entry."""
    chunk_doc = (chunk.get("doc_name") or chunk.get("document_name") or "").strip().lower()
    chunk_page = chunk.get("page") or chunk.get("page_number")
    for ed in expected_docs:
        ed_doc = (ed.get("document_name") or "").strip().lower()
        if not ed_doc or not chunk_doc:
            continue
        # Substring match: handles path prefix differences
        if ed_doc in chunk_doc or chunk_doc in ed_doc:
            ed_page = ed.get("page")
            # If gold specifies a page, require page match; otherwise doc match suffices
            if ed_page is None or chunk_page is None or int(ed_page) == int(chunk_page):
                return True
    return False


def _dcg(relevances: list[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )


def compute_ir_metrics(
    samples: list[dict],
    gold_map: dict[str, list[dict]],
    ks: list[int],
) -> dict[str, float]:
    """
    Compute Recall@K, MRR@K, nDCG@K for each k in ks.

    gold_map: query -> list[expected_doc dicts]
    samples: list of retrieval eval rows with "retrieved_chunks"
    """
    results: dict[str, list[float]] = {
        f"recall@{k}": [] for k in ks
    }
    results.update({f"mrr@{k}": [] for k in ks})
    results.update({f"ndcg@{k}": [] for k in ks})

    for sample in samples:
        query = sample.get("query", "").strip()
        expected_docs = gold_map.get(query, [])
        if not expected_docs:
            continue

        chunks = sample.get("retrieved_chunks", [])
        relevances = [1 if _is_relevant(c, expected_docs) else 0 for c in chunks]

        for k in ks:
            top_k = relevances[:k]

            # Recall@K: any relevant in top-k
            results[f"recall@{k}"].append(1.0 if any(top_k) else 0.0)

            # MRR@K: reciprocal rank of first relevant
            rr = 0.0
            for rank, rel in enumerate(top_k, 1):
                if rel:
                    rr = 1.0 / rank
                    break
            results[f"mrr@{k}"].append(rr)

            # nDCG@K
            ideal = sorted(relevances, reverse=True)
            dcg = _dcg(top_k, k)
            idcg = _dcg(ideal[:k], k)
            results[f"ndcg@{k}"].append(dcg / idcg if idcg > 0 else 0.0)

    return {
        metric: round(sum(vals) / len(vals), 4) if vals else 0.0
        for metric, vals in results.items()
    }


# ── Markdown report ────────────────────────────────────────────────────────────

def write_md_report(
    samples: list[dict],
    ir_metrics: dict[str, float],
    path: str,
) -> None:
    reviewed = [s for s in samples if s.get("retrieval_ok") is not None]
    rag = [s for s in reviewed if s.get("query_type") not in _OFF_TOPIC_TYPES]
    correct = sum(1 for s in rag if s.get("retrieval_ok") == "correct")
    partial = sum(1 for s in rag if s.get("retrieval_ok") == "partial")
    wrong = sum(1 for s in rag if s.get("retrieval_ok") == "wrong")
    acc = (correct + 0.5 * partial) / len(rag) if rag else 0.0

    lines: list[str] = [
        "# AgentBook Retrieval Evaluation Report",
        "",
        f"**Date:** {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Samples:** {len(samples)}  **Reviewed:** {len(reviewed)}",
        "",
    ]

    if ir_metrics:
        lines += [
            "## IR Metrics (Gold-based)",
            "",
            "| Metric | Score | Dev threshold |",
            "|--------|-------|---------------|",
        ]
        thresholds = {"recall@5": 0.85, "mrr": 0.65, "ndcg@5": 0.70}
        for metric, score in sorted(ir_metrics.items()):
            thr = next((v for k, v in thresholds.items() if k in metric), None)
            thr_str = f"≥{thr:.2f}" if thr else "—"
            badge = ("✅" if score >= thr else "❌") if thr else "—"
            lines.append(f"| {metric} | {score:.4f} | {thr_str} {badge} |")
        lines.append("")

    lines += [
        "## Keyword Annotation Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Correct | {correct}/{len(rag)} ({correct/len(rag):.1%} )" if rag else "| Correct | 0/0 |",
        f"| Partial | {partial}/{len(rag)} ({partial/len(rag):.1%} )" if rag else "| Partial | 0/0 |",
        f"| Wrong | {wrong}/{len(rag)} ({wrong/len(rag):.1%} )" if rag else "| Wrong | 0/0 |",
        f"| Accuracy (correct+0.5×partial) | {acc:.3f} |",
        "",
    ]

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {p.resolve()}")


# ── Load / save ────────────────────────────────────────────────────────────────

def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save(samples: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


# ── Console report ─────────────────────────────────────────────────────────────

def report(samples: list[dict], ir_metrics: dict[str, float] | None = None) -> None:
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

    rag_samples = [s for s in reviewed if s.get("query_type") not in _OFF_TOPIC_TYPES]
    rag_total = len(rag_samples)

    scores = [
        s.get("top_reranker_score", 0) or (s["retrieved_chunks"][0]["score"] if s.get("retrieved_chunks") else 0)
        for s in rag_samples
    ]
    avg_score = sum(scores) / len(scores) if scores else 0

    off_topic = [s for s in reviewed if s.get("query_type") in _OFF_TOPIC_TYPES]

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

    if ir_metrics:
        print(f"  IR METRICS (gold-based)")
        for metric, score in sorted(ir_metrics.items()):
            print(f"    {metric:<16} {score:.4f}")
        print(f"{'─'*w}")

    print(f"  RAG QUERIES  ({rag_total} samples, excl. off-topic)")
    if rag_total:
        print(f"    Correct:   {correct:>4}  ({correct/rag_total:.1%})")
        print(f"    Partial:   {partial:>4}  ({partial/rag_total:.1%})")
        print(f"    Wrong:     {wrong:>4}  ({wrong/rag_total:.1%})")
        acc = (correct + 0.5 * partial) / rag_total
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

    wrong_samples = [s for s in reviewed if s.get("retrieval_ok") == "wrong"]
    if wrong_samples:
        print(f"  WRONG RETRIEVALS (fix chunking/embedding for these):")
        for s in wrong_samples[:5]:
            docs = ", ".join(s.get("retrieved_docs", []))
            print(f"    [{s['query_type']}] {s['query'][:55]}")
            print(f"      -> retrieved: {docs[:60]}")
    print(f"{'='*w}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Retrieval eval JSONL")
    parser.add_argument("--auto", action="store_true", help="Auto-annotate using keyword matching")
    parser.add_argument("--gold", default=None,
                        help="Gold JSONL with {query, expected_docs} for IR metrics")
    parser.add_argument("--metrics", default="recall@1,recall@3,recall@5,mrr,ndcg@5",
                        help="Comma-separated IR metrics to compute (requires --gold)")
    parser.add_argument("--report", default=None, help="Save Markdown report to this path")
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

    # IR metrics from gold labels
    ir_metrics: dict[str, float] = {}
    ks: list[int] = []
    if args.gold:
        gold_path = Path(args.gold)
        if not gold_path.exists():
            print(f"[WARN] Gold file not found: {gold_path}", file=sys.stderr)
        else:
            gold_rows = load(gold_path)
            gold_map: dict[str, list[dict]] = {}
            for row in gold_rows:
                q = row.get("query", "").strip()
                if q:
                    gold_map[q] = row.get("expected_docs", [])

            # Parse requested metrics to extract K values
            metric_names = [m.strip().lower() for m in args.metrics.split(",")]
            for m in metric_names:
                for part in m.split("@")[1:]:
                    try:
                        ks.append(int(part))
                    except ValueError:
                        pass
            ks = sorted(set(ks)) or [1, 3, 5]

            print(f"Computing IR metrics {metric_names} at k={ks} against {len(gold_map)} gold queries...")
            ir_metrics = compute_ir_metrics(samples, gold_map, ks)

    report(samples, ir_metrics or None)

    if args.report:
        write_md_report(samples, ir_metrics, args.report)
