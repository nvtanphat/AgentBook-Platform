"""
Score an eval dataset after human annotation.

Usage:
    cd backend
    python scripts/score_eval_dataset.py --input eval_results/eval_dataset.jsonl

Expects each line in the JSONL to have a "human_verdict" field:
    "correct"           — answer is accurate and well-grounded
    "partial"           — answer is partially correct or incomplete
    "wrong"             — answer is incorrect or hallucinated
    "refused_correctly" — system correctly refused an off-topic/false-premise query
    "should_not_refuse" — system refused but should have answered
    null                — not yet reviewed (skipped in scoring)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

_CITATION_RE = re.compile(r"\[\d+\]")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")


def load_samples(path: Path) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def score(samples: list[dict]) -> None:
    reviewed = [s for s in samples if s.get("human_verdict") is not None]
    unreviewed = len(samples) - len(reviewed)

    if not reviewed:
        print("No reviewed samples found. Fill in 'human_verdict' for each entry first.")
        return

    verdicts = Counter(s["human_verdict"] for s in reviewed)
    total = len(reviewed)

    # Accuracy metrics
    correct = verdicts.get("correct", 0)
    partial = verdicts.get("partial", 0)
    wrong = verdicts.get("wrong", 0)
    refused_ok = verdicts.get("refused_correctly", 0)
    should_not_refuse = verdicts.get("should_not_refuse", 0)

    # Retrieval proxy: avg confidence on correct answers
    correct_samples = [s for s in reviewed if s.get("human_verdict") == "correct"]
    avg_conf_correct = (
        sum(s.get("confidence", 0) for s in correct_samples) / len(correct_samples)
        if correct_samples else 0.0
    )

    # Faithfulness proxy: citation coverage in correct + partial answers
    grounded = [s for s in reviewed if s.get("human_verdict") in ("correct", "partial")]
    faithfulness_scores = []
    for s in grounded:
        sentences = [t.strip() for t in _SENTENCE_RE.findall(s.get("answer", "")) if len(t.strip()) >= 10]
        if sentences:
            supported = sum(1 for t in sentences if _CITATION_RE.search(t))
            faithfulness_scores.append(supported / len(sentences))
    faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0

    # Latency stats
    latencies = [s.get("latency_s", 0) for s in reviewed if s.get("latency_s")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else avg_latency

    # Per query type breakdown
    by_type: dict[str, Counter] = defaultdict(Counter)
    for s in reviewed:
        qtype = s.get("query_type", "unknown")
        verdict = s.get("human_verdict", "unknown")
        by_type[qtype][verdict] += 1

    # Refusal behavior
    refusal_precision = (
        refused_ok / (refused_ok + should_not_refuse)
        if (refused_ok + should_not_refuse) > 0 else 1.0
    )

    # Print report
    w = 42
    print(f"\n{'='*w}")
    print(f"  NOELYS EVALUATION REPORT")
    print(f"{'='*w}")
    print(f"  Samples reviewed:   {total:>6}  ({unreviewed} pending)")
    print(f"{'─'*w}")
    print(f"  ANSWER QUALITY")
    print(f"    Correct:          {correct:>6}  ({correct/total:.1%})")
    print(f"    Partial:          {partial:>6}  ({partial/total:.1%})")
    print(f"    Wrong:            {wrong:>6}  ({wrong/total:.1%})")
    print(f"    Accuracy (exact): {correct/total:.3f}")
    print(f"    Accuracy (+part): {(correct+partial)/total:.3f}")
    print(f"{'─'*w}")
    print(f"  REFUSAL BEHAVIOR")
    print(f"    Refused correctly: {refused_ok:>5}")
    print(f"    Should not refuse: {should_not_refuse:>5}")
    print(f"    Refusal precision: {refusal_precision:.3f}")
    print(f"{'─'*w}")
    print(f"  QUALITY PROXIES")
    print(f"    Faithfulness:     {faithfulness:.3f}  (citation coverage in sentences)")
    print(f"    Avg confidence:   {avg_conf_correct:.3f}  (on correct answers only)")
    print(f"{'─'*w}")
    print(f"  LATENCY")
    print(f"    Avg:              {avg_latency:.2f}s")
    print(f"    P95:              {p95_latency:.2f}s")
    print(f"{'─'*w}")
    print(f"  BY QUERY TYPE")
    for qtype, counts in sorted(by_type.items()):
        total_type = sum(counts.values())
        c = counts.get("correct", 0) + counts.get("refused_correctly", 0)
        print(f"    {qtype:<22} {c}/{total_type} ok  {dict(counts)}")
    print(f"{'='*w}")

    # Flag worst samples for review
    worst = [
        s for s in reviewed
        if s.get("human_verdict") == "wrong"
    ]
    if worst:
        print(f"\n  TOP WRONG ANSWERS (fix these first):")
        for s in worst[:5]:
            print(f"    [{s['id']}] {s['query'][:60]}")
            print(f"           conf={s.get('confidence', 0):.2f}  doc={s.get('source_document', '?')[:30]}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score annotated eval dataset")
    parser.add_argument("--input", required=True, help="Path to annotated JSONL file")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}")
        raise SystemExit(1)

    samples = load_samples(path)
    print(f"Loaded {len(samples)} samples from {path}")
    score(samples)
