"""
Ablation study: so sánh từng kỹ thuật RAG nâng cao.

Chạy 4 config trên 8 câu đại diện, so sánh kết quả.
Không cần restart server — dùng rag_flags per-request.

Usage:
    cd backend
    python scripts/ablation_eval.py \
        --owner-id user_demo \
        --collection-id 69fc3c0949fae4625be50223
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 8 câu đại diện — bao phủ đủ loại query
EVAL_QUERIES = [
    ("factual",               "Regularization là gì và tại sao cần dùng?"),
    ("factual",               "L1 và L2 regularization khác nhau như thế nào?"),
    ("summarization",         "Tóm tắt các kỹ thuật regularization phổ biến"),
    ("comparison",            "So sánh L1 và L2 regularization"),
    ("claim_check",           "Dropout có giúp giảm overfitting không?"),
    ("cross_lingual",         "What is the difference between L1 and L2 regularization?"),
    ("off_topic_should_refuse", "Thủ đô của nước Pháp là gì?"),
    ("false_premise",         "Tại sao L2 regularization tạo ra sparse weights?"),
]

_OFF_TOPIC_TYPES = {"off_topic_should_refuse"}

_CITATION_RE = re.compile(r"\[[a-zA-Z]?(\d+)\]")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")

# Configs đúng — agentic_rag_enabled phải được tắt tường minh để test inference_engine
CONFIGS = [
    {
        "name": "agentic",
        "label": "Agentic RAG (default config, multi-step)",
        "rag_flags": {},  # dùng server default (agentic=true)
    },
    {
        "name": "baseline",
        "label": "Baseline (inference engine, reranker ON, no rewriter)",
        "rag_flags": {"agentic_rag_enabled": False, "query_rewriter_enabled": False},
    },
    {
        "name": "multi_query",
        "label": "Multi-Query Rewriter (inference engine, rewriter ON)",
        "rag_flags": {"agentic_rag_enabled": False, "query_rewriter_enabled": True},
    },
    {
        "name": "no_reranker",
        "label": "No Reranker (inference engine, fusion only)",
        "rag_flags": {"agentic_rag_enabled": False, "reranker_enabled": False, "query_rewriter_enabled": False},
    },
]


def faithfulness_citation(answer: str) -> float:
    """RAGAS faithfulness: % câu có citation marker [N]."""
    sentences = [s.strip() for s in _SENTENCE_RE.findall(answer) if len(s.strip()) >= 10]
    if not sentences:
        return 1.0
    return sum(1 for s in sentences if _CITATION_RE.search(s)) / len(sentences)


def context_precision(citations: list[dict], threshold: float = 0.4) -> float:
    """% citations có confidence >= threshold."""
    if not citations:
        return 0.0
    scores = [c.get("confidence") or 0.0 for c in citations]
    return sum(1 for s in scores if s >= threshold) / len(scores)


def refused(answer: str) -> bool:
    if not answer:
        return True
    lower = answer.lower()
    hard = ["nằm ngoài phạm vi", "không thuộc phạm vi", "outside the scope", "không thể trả lời câu hỏi này"]
    if any(p in lower for p in hard):
        return True
    soft = ["không tìm thấy đủ bằng chứng", "không có thông tin", "không đề cập", "not found", "không liên quan"]
    return len(answer.strip()) < 200 and any(p in lower for p in soft)


def ask(*, api_url: str, owner_id: str, collection_id: str, query: str,
        rag_flags: dict, timeout: int) -> dict:
    payload = {
        "query": query,
        "owner_id": owner_id,
        "collection_id": collection_id,
        "stream": False,
        "rag_flags": rag_flags,
    }
    resp = requests.post(f"{api_url}/api/v1/query/ask", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def run_config(*, cfg: dict, api_url: str, owner_id: str, collection_id: str,
               timeout: int) -> dict:
    name = cfg["name"]
    label = cfg["label"]
    rag_flags = cfg["rag_flags"]

    print(f"\n{'='*65}", flush=True)
    print(f"  CONFIG: {label}", flush=True)
    print(f"{'='*65}", flush=True)

    results = []
    for i, (qtype, query) in enumerate(EVAL_QUERIES):
        print(f"  [{i+1}/{len(EVAL_QUERIES)}] [{qtype}] {query[:50]}", flush=True)
        t0 = time.time()
        try:
            data = ask(
                api_url=api_url, owner_id=owner_id, collection_id=collection_id,
                query=query, rag_flags=rag_flags, timeout=timeout,
            )
            elapsed = time.time() - t0
            payload = data.get("data") or data
            answer = payload.get("answer") or ""
            citations = payload.get("citations") or []
            top_score = payload.get("confidence") or 0.0
            if citations:
                top_score = citations[0].get("confidence") or top_score

            faith = faithfulness_citation(answer)
            cov = context_precision(citations)
            is_refused = payload.get("was_refused", False) or refused(answer)
            status = "refused" if is_refused else "answered"

            off_topic_ok = None
            if qtype in _OFF_TOPIC_TYPES:
                off_topic_ok = is_refused

            print(f"        [{status}] {elapsed:.0f}s  score={top_score:.3f}  faith={faith:.2f}  ctx_prec={cov:.2f}", flush=True)
            results.append({
                "query_type": qtype,
                "query": query,
                "elapsed_s": round(elapsed, 1),
                "top_score": top_score,
                "faithfulness": round(faith, 3),
                "context_precision": round(cov, 3),
                "refused": is_refused,
                "off_topic_ok": off_topic_ok,
            })
        except requests.Timeout:
            print(f"        TIMEOUT {timeout}s", flush=True)
            results.append({"query_type": qtype, "query": query, "elapsed_s": timeout,
                            "top_score": 0.0, "citation_coverage": 0.0,
                            "faithfulness_proxy": 0.0, "refused": False, "off_topic_ok": None,
                            "error": "timeout"})
        except Exception as exc:
            print(f"        ERROR: {type(exc).__name__}: {exc}", flush=True)
            results.append({"query_type": qtype, "query": query, "elapsed_s": 0.0,
                            "top_score": 0.0, "citation_coverage": 0.0,
                            "faithfulness_proxy": 0.0, "refused": False, "off_topic_ok": None,
                            "error": str(exc)})

    # Tổng hợp
    rag_rows = [r for r in results if r.get("query_type") not in _OFF_TOPIC_TYPES]
    off_rows = [r for r in results if r.get("query_type") in _OFF_TOPIC_TYPES]
    answered = [r for r in rag_rows if not r.get("error") and not r.get("refused")]

    avg_score = sum(r["top_score"] for r in rag_rows) / max(len(rag_rows), 1)
    avg_faith = sum(r["faithfulness"] for r in rag_rows) / max(len(rag_rows), 1)
    avg_prec = sum(r["context_precision"] for r in rag_rows) / max(len(rag_rows), 1)
    avg_elapsed = sum(r["elapsed_s"] for r in results) / max(len(results), 1)
    correct_refuse = sum(1 for r in off_rows if r.get("off_topic_ok"))

    return {
        "config_name": name,
        "config_label": label,
        "rag_flags": rag_flags,
        "avg_top_score": round(avg_score, 3),
        "avg_faithfulness": round(avg_faith, 3),
        "avg_context_precision": round(avg_prec, 3),
        "avg_elapsed_s": round(avg_elapsed, 1),
        "answered_count": len(answered),
        "off_topic_correct": correct_refuse,
        "off_topic_total": len(off_rows),
        "results": results,
    }


def compare_report(summaries: list[dict]) -> None:
    w = 65
    print(f"\n{'='*w}", flush=True)
    print("  ABLATION STUDY — TỔNG HỢP SO SÁNH", flush=True)
    print(f"{'='*w}", flush=True)
    print(f"  {'Config':<30} {'Score':>6} {'Faith':>6} {'CtxPr':>6} {'Time':>7} {'Ans':>5} {'OffOK':>6}", flush=True)
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*5} {'-'*6}", flush=True)
    for s in summaries:
        print(
            f"  {s['config_name']:<30} "
            f"{s['avg_top_score']:>6.3f} "
            f"{s['avg_faithfulness']:>6.3f} "
            f"{s['avg_context_precision']:>6.3f} "
            f"{s['avg_elapsed_s']:>6.1f}s "
            f"{s['answered_count']:>3}/{len(EVAL_QUERIES)-s['off_topic_total']} "
            f"{s['off_topic_correct']}/{s['off_topic_total']}",
            flush=True,
        )
    print(f"{'='*w}", flush=True)

    # Nhận xét tự động
    baseline = next((s for s in summaries if s["config_name"] == "baseline"), None)
    if baseline:
        print("\n  NHẬN XÉT:", flush=True)
        for s in summaries:
            if s["config_name"] == "baseline":
                continue
            delta_score = s["avg_top_score"] - baseline["avg_top_score"]
            delta_faith = s["avg_faithfulness"] - baseline["avg_faithfulness"]
            verdict = "↑ CẢI THIỆN" if delta_score > 0.01 else ("↓ GIẢM" if delta_score < -0.01 else "≈ TƯƠNG ĐƯƠNG")
            print(
                f"  [{s['config_name']}] {verdict}  "
                f"score Δ={delta_score:+.3f}  faith Δ={delta_faith:+.3f}",
                flush=True,
            )
    print(f"{'='*w}\n", flush=True)


def main(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    configs_to_run = CONFIGS
    if args.configs:
        names = set(args.configs.split(","))
        configs_to_run = [c for c in CONFIGS if c["name"] in names]

    summaries = []
    for cfg in configs_to_run:
        summary = run_config(
            cfg=cfg,
            api_url=args.api_url,
            owner_id=args.owner_id,
            collection_id=args.collection_id,
            timeout=args.timeout,
        )
        summaries.append(summary)

        # Lưu intermediate
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)

    compare_report(summaries)
    print(f"  Saved: {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--output", default="eval_results/ablation_eval.json")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--configs", default=None,
                        help="Comma-separated config names to run (default: all). "
                             "Options: baseline,no_reranker,multi_query,agentic")
    args = parser.parse_args()
    main(args)
