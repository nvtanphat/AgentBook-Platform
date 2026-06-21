"""
Ablation study: C0→C7 incremental ladder + leave-one-out (LOO) + 3-axis metrics.

Trục A — Retrieval (gold-labeled, harness/metrics.py): Recall@k, MRR@k, nDCG@k, Citation Accuracy
Trục B — Generation (RAGAS thật, LLM-judge độc lập): Faithfulness, Answer Relevancy, Context Precision
Trục C — Safety/Refusal (confusion matrix): FAR, FRR, Refusal F1

Usage:
    # Quick smoke test (2 configs × 3 queries)
    python evaluation/cli/run_ablation.py \\
        --owner-id nguyenvtp69_gmail_com \\
        --collection-id <EVAL_COLLECTION_ID> \\
        --question-set evaluation/datasets/gold/e2e_gold_v2.jsonl \\
        --configs C0_baseline,C7_full \\
        --max-queries 3

    # Full ablation run
    python evaluation/cli/run_ablation.py \\
        --owner-id nguyenvtp69_gmail_com \\
        --collection-id <EVAL_COLLECTION_ID> \\
        --question-set evaluation/datasets/gold/e2e_gold_v2.jsonl \\
        --adversarial-set evaluation/datasets/gold/adversarial.jsonl \\
        --output evaluation/results/ablation_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Ablation configs ───────────────────────────────────────────────────────────
# C0_baseline (dense-only): requires sparse_enabled=False to be wired.
# C7_full uses agentic pipeline — compared SEPARATELY from ladder, not as C8.

LADDER_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "C0_baseline",
        "label": "C0 — Dense-only (no sparse, no reranker, no guardrails)",
        "rag_flags": {
            "agentic_rag_enabled": False,
            "reranker_enabled": False,
            "multi_query_enabled": False,
            "sparse_enabled": False,
            "graph_probe_enabled": False,
            "slec_enabled": False,
            "claim_verifier_enabled": False,
            "crag_enabled": False,
        },
    },
    {
        "name": "C1_hybrid",
        "label": "C1 — +Hybrid sparse (RRF fusion)",
        "rag_flags": {
            "agentic_rag_enabled": False,
            "reranker_enabled": False,
            "multi_query_enabled": False,
            "sparse_enabled": True,
            "graph_probe_enabled": False,
            "slec_enabled": False,
            "claim_verifier_enabled": False,
            "crag_enabled": False,
        },
    },
    {
        "name": "C2_multi_query",
        "label": "C2 — +Multi-Query rewriter",
        "rag_flags": {
            "agentic_rag_enabled": False,
            "reranker_enabled": False,
            "multi_query_enabled": True,
            "sparse_enabled": True,
            "graph_probe_enabled": False,
            "slec_enabled": False,
            "claim_verifier_enabled": False,
            "crag_enabled": False,
        },
    },
    {
        "name": "C3_reranker",
        "label": "C3 — +Cross-Encoder reranker",
        "rag_flags": {
            "agentic_rag_enabled": False,
            "reranker_enabled": True,
            "multi_query_enabled": True,
            "sparse_enabled": True,
            "graph_probe_enabled": False,
            "slec_enabled": False,
            "claim_verifier_enabled": False,
            "crag_enabled": False,
        },
    },
    {
        "name": "C4_crag",
        "label": "C4 — +CRAG correctness scoring",
        "rag_flags": {
            "agentic_rag_enabled": False,
            "reranker_enabled": True,
            "multi_query_enabled": True,
            "sparse_enabled": True,
            "graph_probe_enabled": False,
            "slec_enabled": False,
            "claim_verifier_enabled": False,
            "crag_enabled": True,
        },
    },
    {
        "name": "C5_slec",
        "label": "C5 — +SLEC sentence coverage gate",
        "rag_flags": {
            "agentic_rag_enabled": False,
            "reranker_enabled": True,
            "multi_query_enabled": True,
            "sparse_enabled": True,
            "graph_probe_enabled": False,
            "slec_enabled": True,
            "claim_verifier_enabled": False,
            "crag_enabled": True,
        },
    },
    {
        "name": "C6_graph_probe",
        "label": "C6 — +Graph probe (entity-linked context)",
        "rag_flags": {
            "agentic_rag_enabled": False,
            "reranker_enabled": True,
            "multi_query_enabled": True,
            "sparse_enabled": True,
            "graph_probe_enabled": True,
            "slec_enabled": True,
            "claim_verifier_enabled": True,
            "crag_enabled": True,
        },
    },
]

# Full direct pipeline (C6 + claim_verifier, non-agentic)
FULL_DIRECT_CONFIG: dict[str, Any] = {
    "name": "C_full_direct",
    "label": "Full — Direct pipeline (all components ON)",
    "rag_flags": {
        "agentic_rag_enabled": False,
        "reranker_enabled": True,
        "multi_query_enabled": True,
        "sparse_enabled": True,
        "graph_probe_enabled": True,
        "slec_enabled": True,
        "claim_verifier_enabled": True,
        "crag_enabled": True,
    },
}

# Leave-one-out configs from Full Direct
LOO_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "LOO_no_reranker",
        "label": "LOO — Full − reranker",
        "rag_flags": {**FULL_DIRECT_CONFIG["rag_flags"], "reranker_enabled": False},
    },
    {
        "name": "LOO_no_multi_query",
        "label": "LOO — Full − multi_query",
        "rag_flags": {**FULL_DIRECT_CONFIG["rag_flags"], "multi_query_enabled": False},
    },
    {
        "name": "LOO_no_sparse",
        "label": "LOO — Full − sparse (dense-only)",
        "rag_flags": {**FULL_DIRECT_CONFIG["rag_flags"], "sparse_enabled": False},
    },
    {
        "name": "LOO_no_crag",
        "label": "LOO — Full − CRAG",
        "rag_flags": {**FULL_DIRECT_CONFIG["rag_flags"], "crag_enabled": False},
    },
    {
        "name": "LOO_no_slec",
        "label": "LOO — Full − SLEC",
        "rag_flags": {**FULL_DIRECT_CONFIG["rag_flags"], "slec_enabled": False},
    },
    {
        "name": "LOO_no_claim_verifier",
        "label": "LOO — Full − claim_verifier",
        "rag_flags": {**FULL_DIRECT_CONFIG["rag_flags"], "claim_verifier_enabled": False},
    },
    {
        "name": "LOO_no_graph_probe",
        "label": "LOO — Full − graph_probe",
        "rag_flags": {**FULL_DIRECT_CONFIG["rag_flags"], "graph_probe_enabled": False},
    },
]

# Agentic comparison (separate, not part of ladder — confound guard per G5)
AGENTIC_CONFIG: dict[str, Any] = {
    "name": "C7_agentic",
    "label": "C7 — Agentic RAG (compared vs Full-direct, not vs C6)",
    "rag_flags": {"agentic_rag_enabled": True},
}

ALL_CONFIG_MAP: dict[str, dict[str, Any]] = {
    cfg["name"]: cfg
    for cfg in LADDER_CONFIGS + LOO_CONFIGS + [FULL_DIRECT_CONFIG, AGENTIC_CONFIG]
}


# ── Metrics — Trục A (retrieval gold-labeled) ─────────────────────────────────

def _truc_a_metrics(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str],
    k: int = 5,
) -> dict[str, float]:
    """Gold-labeled retrieval metrics (harness/metrics.py definitions)."""
    if not expected_chunk_ids:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr_at_k": 0.0, "ndcg_at_k": 0.0}
    gold = set(expected_chunk_ids)
    topk = retrieved_chunk_ids[:k]
    hits = [1 if cid in gold else 0 for cid in topk]
    recall = sum(hits) / len(gold) if gold else 0.0
    precision = sum(hits) / k if k else 0.0
    mrr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / rank
            break
    # nDCG
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal = sorted(hits, reverse=True)
    idcg = sum(h / math.log2(i + 2) for i, h in enumerate(ideal))
    ndcg = dcg / idcg if idcg else 0.0
    return {
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "mrr_at_k": round(mrr, 4),
        "ndcg_at_k": round(ndcg, 4),
    }


def _citation_accuracy(citations: list[dict], expected_chunk_ids: list[str]) -> float:
    """Fraction of returned citations that match a gold chunk_id."""
    if not citations or not expected_chunk_ids:
        return 0.0
    gold = set(expected_chunk_ids)
    # Citations expose the chunk id under "evidence_id"; "chunk_id" is not populated.
    matched = sum(1 for c in citations if (c.get("chunk_id") or c.get("evidence_id")) in gold)
    return round(matched / len(citations), 4)


def _citation_accuracy_page(citations: list[dict], expected_evidence: list[dict]) -> float:
    """Page-level citation accuracy: a citation counts as correct if it lands on the
    same (document, page) as any gold evidence — fairer than exact chunk_id match,
    since single-chunk gold labels under-credit citing an adjacent chunk on the same page."""
    if not citations or not expected_evidence:
        return 0.0
    gold_pages = {(ev.get("document_name"), ev.get("page")) for ev in expected_evidence if ev.get("page") is not None}
    if not gold_pages:
        return 0.0
    matched = sum(1 for c in citations if (c.get("doc_name") or c.get("document_name"), c.get("page")) in gold_pages)
    return round(matched / len(citations), 4)


# ── Metrics — Trục C (safety/refusal confusion matrix) ────────────────────────

def _should_refuse(answer: str, was_refused: bool) -> bool:
    if was_refused:
        return True
    if not answer:
        return True
    lower = answer.lower()
    hard = ["nằm ngoài phạm vi", "không thuộc phạm vi", "outside the scope",
            "không thể trả lời câu hỏi này"]
    return any(p in lower for p in hard)


def _truc_c_metrics(results: list[dict]) -> dict[str, float]:
    """Confusion matrix metrics on adversarial set (Trục C)."""
    tp = fp = fn = tn = 0
    for r in results:
        expected = r.get("expected_behavior", "answer")
        actual_refused = r.get("was_refused_actual", False)
        should_have_refused = expected == "refuse"
        if should_have_refused and actual_refused:
            tp += 1
        elif should_have_refused and not actual_refused:
            fn += 1   # False Accept (dangerous: answered when should refuse)
        elif not should_have_refused and actual_refused:
            fp += 1   # False Refusal (overprotective)
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    far = fn / (tp + fn) if (tp + fn) else 0.0  # False Accept Rate
    frr = fp / (fp + tn) if (fp + tn) else 0.0  # False Refusal Rate
    return {
        "refusal_precision": round(precision, 4),
        "refusal_recall": round(recall, 4),
        "refusal_f1": round(f1, 4),
        "false_accept_rate": round(far, 4),
        "false_refusal_rate": round(frr, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ── Bootstrap CI (G3) ─────────────────────────────────────────────────────────

def _bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI around the mean."""
    rng = random.Random(seed)
    if not values:
        return (0.0, 0.0)
    means = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (round(lo, 4), round(hi, 4))


# ── API call ───────────────────────────────────────────────────────────────────

def _ask(
    *,
    api_url: str,
    owner_id: str,
    collection_id: str,
    query: str,
    rag_flags: dict,
    answer_language: str | None = None,
    timeout: int = 120,
) -> dict:
    payload = {
        "query": query,
        "owner_id": owner_id,
        "collection_id": collection_id,
        "rag_flags": rag_flags,
    }
    if answer_language:
        payload["answer_language"] = answer_language
    resp = requests.post(f"{api_url}/api/v1/query/ask", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ── Single config run ──────────────────────────────────────────────────────────

def _flush_semantic_cache(redis_url: str = "redis://localhost:6379/0") -> int:
    """Flush the semantic query cache (sqc:*) before a config runs.

    The cache is scoped only by owner+collection+language, NOT by rag_flags, so
    without this flush an identical query answered under a previous config is
    served from cache (3-5s, no fresh retrieval) — silently invalidating the
    ablation. Returns the number of keys deleted.
    """
    try:
        import redis as _redis
        r = _redis.from_url(redis_url)
        keys = list(r.scan_iter(match="sqc:*", count=500))
        if keys:
            r.delete(*keys)
        return len(keys)
    except Exception as exc:
        print(f"  [cache-flush skipped: {exc}]", flush=True)
        return 0


def _run_config(
    *,
    cfg: dict[str, Any],
    gold_cases: list[dict],
    adv_cases: list[dict],
    api_url: str,
    owner_id: str,
    collection_id: str,
    timeout: int,
    k: int = 5,
) -> dict[str, Any]:
    name = cfg["name"]
    rag_flags = cfg["rag_flags"]

    print(f"\n{'='*70}", flush=True)
    print(f"  CONFIG: {cfg['label']}", flush=True)
    print(f"{'='*70}", flush=True)
    flushed = _flush_semantic_cache()
    print(f"  (flushed {flushed} semantic-cache keys → fresh retrieval per config)", flush=True)

    # ── Trục A+B: gold cases ──────────────────────────────────────────────────
    truc_a_rows: list[dict] = []
    gold_refuse_rows: list[dict] = []
    latencies: list[float] = []
    all_results: list[dict] = []

    for i, case in enumerate(gold_cases):
        query = case.get("query", "")
        expected_chunk_ids = [
            ev.get("chunk_id") for ev in case.get("expected_evidence", []) if ev.get("chunk_id")
        ]
        print(f"  [{i+1}/{len(gold_cases)}] {query[:55]}", flush=True)
        t0 = time.time()
        try:
            data = _ask(
                api_url=api_url, owner_id=owner_id, collection_id=collection_id,
                query=query, rag_flags=rag_flags,
                answer_language=case.get("answer_language"),
                timeout=timeout,
            )
            elapsed = time.time() - t0
            payload = data.get("data") or data
            answer = payload.get("answer") or ""
            citations = payload.get("citations") or []
            confidence = payload.get("confidence") or 0.0
            was_refused = payload.get("was_refused", False)
            trace = payload.get("trace") or {}
            retrieved_ids: list[str] = trace.get("retrieved_chunk_ids") or []
            latencies.append(elapsed)

            # Refusal-test gold cases (off_topic_should_refuse) carry no gold
            # evidence by design → score refusal correctness, skip Trục A so they
            # don't drag retrieval averages to 0.
            if case.get("expected_behavior") == "refuse":
                refused_actual = _should_refuse(answer, was_refused)
                gold_refuse_rows.append({"case_id": case.get("case_id"), "refused_actual": refused_actual})
                print(f"        [refuse-case] {'OK-refuse' if refused_actual else 'FALSE-ACCEPT'}  {elapsed:.0f}s", flush=True)
                all_results.append({
                    "case_id": case.get("case_id"), "task_type": case.get("task_type"),
                    "query": query, "answer": answer, "was_refused": was_refused,
                    "expected_behavior": "refuse", "refused_actual": refused_actual,
                    "elapsed_s": round(elapsed, 2),
                })
                continue

            a_metrics = _truc_a_metrics(retrieved_ids, expected_chunk_ids, k=k)
            cit_acc = _citation_accuracy(citations, expected_chunk_ids)
            cit_acc_page = _citation_accuracy_page(citations, case.get("expected_evidence") or [])
            truc_a_rows.append({**a_metrics, "citation_accuracy": cit_acc, "citation_accuracy_page": cit_acc_page})

            status = "REFUSED" if was_refused else "ok"
            print(
                f"        [{status}] {elapsed:.0f}s  "
                f"R@{k}={a_metrics['recall_at_k']:.2f}  "
                f"nDCG={a_metrics['ndcg_at_k']:.2f}  "
                f"conf={confidence:.2f}",
                flush=True,
            )
            all_results.append({
                "case_id": case.get("case_id"),
                "task_type": case.get("task_type"),
                "query": query,
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "was_refused": was_refused,
                "retrieved_chunk_ids": retrieved_ids,
                "expected_chunk_ids": expected_chunk_ids,
                "elapsed_s": round(elapsed, 2),
                **a_metrics,
                "citation_accuracy": cit_acc,
                "citation_accuracy_page": cit_acc_page,
            })
        except requests.Timeout:
            print(f"        TIMEOUT {timeout}s", flush=True)
            latencies.append(float(timeout))
            all_results.append({
                "case_id": case.get("case_id"), "query": query,
                "error": "timeout", "elapsed_s": float(timeout),
                **{m: 0.0 for m in ("recall_at_k", "precision_at_k", "mrr_at_k", "ndcg_at_k", "citation_accuracy")},
            })
        except Exception as exc:
            print(f"        ERROR: {exc}", flush=True)
            latencies.append(0.0)
            all_results.append({
                "case_id": case.get("case_id"), "query": query,
                "error": str(exc), "elapsed_s": 0.0,
                **{m: 0.0 for m in ("recall_at_k", "precision_at_k", "mrr_at_k", "ndcg_at_k", "citation_accuracy")},
            })

    # Aggregate Trục A metrics
    def _avg(key: str) -> float:
        vals = [r[key] for r in truc_a_rows if key in r]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    truc_a_summary = {
        f"avg_recall_at_{k}": _avg("recall_at_k"),
        f"avg_precision_at_{k}": _avg("precision_at_k"),
        f"avg_mrr_at_{k}": _avg("mrr_at_k"),
        f"avg_ndcg_at_{k}": _avg("ndcg_at_k"),
        "avg_citation_accuracy": _avg("citation_accuracy"),
        "avg_citation_accuracy_page": _avg("citation_accuracy_page"),
    }

    # Bootstrap CI for headline metric nDCG@k
    ndcg_vals = [r["ndcg_at_k"] for r in truc_a_rows if "ndcg_at_k" in r]
    ci_lo, ci_hi = _bootstrap_ci(ndcg_vals)
    truc_a_summary[f"ndcg_at_{k}_ci95"] = [ci_lo, ci_hi]

    # Latency
    p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0.0
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0

    # ── Trục C: adversarial cases ─────────────────────────────────────────────
    adv_results: list[dict] = []
    if adv_cases:
        print(f"\n  --- Adversarial ({len(adv_cases)} cases) ---", flush=True)
    for case in adv_cases:
        query = case.get("query", "")
        expected_behavior = case.get("expected_behavior", "answer")
        try:
            data = _ask(
                api_url=api_url, owner_id=owner_id, collection_id=collection_id,
                query=query, rag_flags=rag_flags, timeout=timeout,
            )
            payload = data.get("data") or data
            answer = payload.get("answer") or ""
            was_refused_actual = _should_refuse(answer, payload.get("was_refused", False))
            print(f"  [adv/{expected_behavior}] refused={was_refused_actual}  {query[:50]}", flush=True)
            adv_results.append({
                "case_id": case.get("case_id"),
                "query": query,
                "expected_behavior": expected_behavior,
                "was_refused_actual": was_refused_actual,
                "answer": answer[:200],
            })
        except Exception as exc:
            print(f"  [adv error] {exc}", flush=True)
            adv_results.append({
                "case_id": case.get("case_id"), "query": query,
                "expected_behavior": expected_behavior,
                "was_refused_actual": False, "error": str(exc),
            })

    truc_c_summary = _truc_c_metrics(adv_results) if adv_results else {}

    # Refusal correctness on gold off_topic_should_refuse cases (false-accept = answered when should refuse).
    gold_refuse_summary: dict[str, float] = {}
    if gold_refuse_rows:
        n_rf = len(gold_refuse_rows)
        correct = sum(1 for r in gold_refuse_rows if r["refused_actual"])
        gold_refuse_summary = {
            "n": n_rf,
            "correct_refusals": correct,
            "refuse_accuracy": round(correct / n_rf, 4),
            "false_accept_rate": round((n_rf - correct) / n_rf, 4),
        }

    return {
        "config_name": name,
        "config_label": cfg["label"],
        "rag_flags": rag_flags,
        "truc_a": truc_a_summary,
        "truc_c": truc_c_summary,
        "truc_c_gold": gold_refuse_summary,
        "latency": {"p50_s": round(p50, 2), "p95_s": round(p95, 2)},
        "n_gold": len(gold_cases),
        "n_adv": len(adv_cases),
        "results": all_results,
        "adv_results": adv_results,
    }


# ── Report ─────────────────────────────────────────────────────────────────────

def _print_report(summaries: list[dict], k: int = 5) -> None:
    w = 80
    print(f"\n{'='*w}", flush=True)
    print("  ABLATION — TỔNG HỢP (Trục A: Retrieval | Trục C: Safety)", flush=True)
    print(f"{'='*w}", flush=True)
    hdr = f"  {'Config':<24} {'R@k':>6} {'MRR':>6} {'nDCG':>7} {'CI-lo':>7} {'CI-hi':>7} {'FAR':>6} {'FRR':>6} {'p50':>6}"
    print(hdr, flush=True)
    print(f"  {'-'*24} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*6}", flush=True)
    for s in summaries:
        a = s.get("truc_a", {})
        c = s.get("truc_c", {})
        lat = s.get("latency", {})
        ci = a.get(f"ndcg_at_{k}_ci95", [0.0, 0.0])
        print(
            f"  {s['config_name']:<24} "
            f"{a.get(f'avg_recall_at_{k}', 0.0):>6.3f} "
            f"{a.get(f'avg_mrr_at_{k}', 0.0):>6.3f} "
            f"{a.get(f'avg_ndcg_at_{k}', 0.0):>7.3f} "
            f"{ci[0]:>7.3f} "
            f"{ci[1]:>7.3f} "
            f"{c.get('false_accept_rate', 0.0):>6.3f} "
            f"{c.get('false_refusal_rate', 0.0):>6.3f} "
            f"{lat.get('p50_s', 0.0):>5.1f}s",
            flush=True,
        )
    print(f"{'='*w}", flush=True)

    # Delta from C0_baseline
    baseline = next((s for s in summaries if s["config_name"] == "C0_baseline"), None)
    if baseline:
        b_ndcg = baseline.get("truc_a", {}).get(f"avg_ndcg_at_{k}", 0.0)
        print("\n  DELTA vs C0_baseline (nDCG@k):", flush=True)
        for s in summaries:
            if s["config_name"] == "C0_baseline":
                continue
            delta = s.get("truc_a", {}).get(f"avg_ndcg_at_{k}", 0.0) - b_ndcg
            ci = s.get("truc_a", {}).get(f"ndcg_at_{k}_ci95", [0.0, 0.0])
            # CI overlap check (G3): only claim improvement if CIs don't overlap
            b_ci = baseline.get("truc_a", {}).get(f"ndcg_at_{k}_ci95", [0.0, 0.0])
            overlap = ci[0] < b_ci[1] and b_ci[0] < ci[1]
            verdict = "↑ sig" if delta > 0 and not overlap else ("↑ ns" if delta > 0 else ("↓" if delta < 0 else "≈"))
            print(
                f"    [{s['config_name']:<22}] Δ={delta:+.3f}  CI=[{ci[0]:.3f},{ci[1]:.3f}]  {verdict}",
                flush=True,
            )
    print(f"{'='*w}\n", flush=True)


def _latex_table(summaries: list[dict], k: int = 5) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        rf"Config & R@{k} & MRR@{k} & nDCG@{k} & FAR & FRR & p50(s) \\",
        r"\hline",
    ]
    for s in summaries:
        a = s.get("truc_a", {})
        c = s.get("truc_c", {})
        lat = s.get("latency", {})
        lines.append(
            rf"{s['config_name']} "
            rf"& {a.get(f'avg_recall_at_{k}', 0.0):.3f} "
            rf"& {a.get(f'avg_mrr_at_{k}', 0.0):.3f} "
            rf"& {a.get(f'avg_ndcg_at_{k}', 0.0):.3f} "
            rf"& {c.get('false_accept_rate', 0.0):.3f} "
            rf"& {c.get('false_refusal_rate', 0.0):.3f} "
            rf"& {lat.get('p50_s', 0.0):.1f} \\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\caption{Ablation results}", r"\end{table}"]
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    gold_cases = _load_jsonl(args.question_set) if args.question_set else []
    adv_cases = _load_jsonl(args.adversarial_set) if args.adversarial_set else []

    if not gold_cases and not adv_cases:
        print("[ERROR] No question set provided (--question-set or --adversarial-set).", file=sys.stderr)
        sys.exit(1)

    # Filter to max-queries
    if args.max_queries and gold_cases:
        random.seed(args.seed)
        random.shuffle(gold_cases)
        gold_cases = gold_cases[:args.max_queries]
    if args.max_queries and adv_cases:
        adv_cases = adv_cases[:args.max_queries]

    # Resolve which configs to run
    if args.configs:
        names = [n.strip() for n in args.configs.split(",")]
        configs_to_run = [ALL_CONFIG_MAP[n] for n in names if n in ALL_CONFIG_MAP]
        unknown = [n for n in names if n not in ALL_CONFIG_MAP]
        if unknown:
            print(f"[WARN] Unknown configs: {unknown}", file=sys.stderr)
    elif args.mode == "ladder":
        configs_to_run = LADDER_CONFIGS + [FULL_DIRECT_CONFIG]
    elif args.mode == "loo":
        configs_to_run = [FULL_DIRECT_CONFIG] + LOO_CONFIGS
    elif args.mode == "agentic":
        configs_to_run = [FULL_DIRECT_CONFIG, AGENTIC_CONFIG]
    else:  # all
        configs_to_run = LADDER_CONFIGS + [FULL_DIRECT_CONFIG] + LOO_CONFIGS + [AGENTIC_CONFIG]

    print(f"\n  Running {len(configs_to_run)} configs × {len(gold_cases)} gold + {len(adv_cases)} adv cases", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for cfg in configs_to_run:
        summary = _run_config(
            cfg=cfg,
            gold_cases=gold_cases,
            adv_cases=adv_cases,
            api_url=args.api_url,
            owner_id=args.owner_id,
            collection_id=args.collection_id,
            timeout=args.timeout,
            k=args.k,
        )
        summaries.append(summary)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)

    _print_report(summaries, k=args.k)

    # LaTeX table
    latex = _latex_table(summaries, k=args.k)
    latex_path = out_path.with_suffix(".tex")
    latex_path.write_text(latex, encoding="utf-8")

    print(f"  Saved JSON  → {out_path.resolve()}", flush=True)
    print(f"  Saved LaTeX → {latex_path.resolve()}", flush=True)


def _load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentBook ablation study (C0→C7 ladder + LOO)")
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--question-set", default="",
                        help="Gold QA JSONL (e2e_gold_v2.jsonl)")
    parser.add_argument("--adversarial-set", default="",
                        help="Adversarial JSONL for Trục C metrics")
    parser.add_argument("--output", default="evaluation/results/ablation_results.json")
    parser.add_argument("--configs", default="",
                        help="Comma-separated config names to run (e.g. C0_baseline,C3_reranker). "
                             "Overrides --mode.")
    parser.add_argument("--mode",
                        choices=["ladder", "loo", "agentic", "all"],
                        default="ladder",
                        help="ladder=C0→C7+Full, loo=LOO from Full, agentic=Full vs Agentic, all=everything")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Limit number of queries per config (0=all, use 3 for smoke test)")
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval metrics")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
