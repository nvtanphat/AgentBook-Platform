"""
End-to-end evaluation: gọi API thật, đánh giá answer quality với RAGAS-style metrics.

Metrics:
  faithfulness         — % câu trong answer có citation marker [N] (grounded)
  citation_coverage    — % đoạn văn có ít nhất 1 citation marker
  answer_relevance     — cosine_sim(embed(query), embed(answer))  [semantic]
  semantic_faithfulness— mean max cosine_sim(embed(answer_sentence), embed(citation_snippet))
  context_precision    — % retrieved chunks có score > 0.4 (reranker confidence)

Usage:
    cd backend

    # Built-in ML question set
    python scripts/e2e_eval.py \\
        --owner-id user_demo \\
        --collection-id 69fc3c0949fae4625be50223 \\
        --api-url http://localhost:8000

    # External gold question set
    python scripts/e2e_eval.py \\
        --owner-id nguyenvtp69_gmail_com \\
        --collection-id 6a16f8d1a0d535db39664088 \\
        --question-set ../evaluation/datasets/agentbook_e2e_gold.jsonl \\
        --report eval_results/e2e_report.md \\
        --ci-mode --ci-threshold 0.70
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ML_QUESTIONS = [
    # factual — metrics cheatsheet + docx content
    ("factual",               "Accuracy, Precision, Recall và F1-score là gì?"),
    ("factual",               "Confusion matrix dùng để làm gì?"),
    ("factual",               "Overfitting là gì và làm sao để tránh?"),
    ("factual",               "What is supervised learning?"),
    ("factual",               "Machine learning là gì?"),
    # summarization
    ("summarization",         "Tóm tắt các bước cơ bản để xây dựng một mô hình machine learning"),
    ("summarization",         "Các metric đánh giá mô hình phân loại phổ biến là gì?"),
    # comparison
    ("comparison",            "Precision và Recall khác nhau như thế nào?"),
    ("comparison",            "Supervised learning và unsupervised learning khác nhau thế nào?"),
    # graph_relation
    ("graph_relation",        "F1-score liên quan đến Precision và Recall như thế nào?"),
    ("graph_relation",        "Dữ liệu training ảnh hưởng như thế nào đến hiệu suất mô hình?"),
    # claim_check
    ("claim_check",           "F1-score là trung bình cộng của Precision và Recall đúng không?"),
    ("claim_check",           "Accuracy luôn là metric tốt nhất để đánh giá mô hình không?"),
    # off_topic — should refuse
    ("off_topic_should_refuse", "Thủ đô của nước Pháp là gì?"),
    ("off_topic_should_refuse", "Hôm nay thời tiết thế nào?"),
    # false_premise
    ("false_premise",         "Tại sao F1-score là tổng của Precision và Recall?"),
    ("false_premise",         "Vì sao mô hình có accuracy cao luôn tốt hơn mô hình có F1 cao?"),
    # cross_lingual — EN query over VI docs
    ("cross_lingual",         "What metrics are used to evaluate a classification model?"),
    ("cross_lingual",         "What is the difference between precision and recall?"),
    # anaphora
    ("anaphora",              "Nó khác với accuracy như thế nào?"),
]

_OFF_TOPIC_TYPES = {"off_topic_should_refuse"}
_FALSE_PREMISE_TYPES = {"false_premise"}
_CITATION_RE = re.compile(r"\[[a-zA-Z]?(\d+)\]")  # matches [1], [s1], [L1], etc.
_CITATION_CLEAN_RE = re.compile(r"\[[a-zA-Z]?\d+\]")  # for stripping before embedding
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_CORRECTION_KEYWORDS = [
    # Explicit refutation markers
    "không chính xác", "tiền đề", "sai", "incorrect", "not correct",
    "premise", "does not", "không phải", "thực ra", "ngược lại",
    "in fact", "actually", "however", "trái lại",
    # Implicit corrections — LLM rewrites the false premise into the correct fact
    # without flagging it. v14 eval showed Qwen3 4B often does this on VN claims.
    "trung bình điều hòa", "trung bình hòa", "harmonic mean",
    "không nhất thiết", "not necessarily",
    "phụ thuộc vào", "depends on",
    "không phải lúc nào", "not always",
]

_TASK_TYPE_MAP: dict[str, str] = {
    "factual": "factual",
    "compare": "comparison",
    "comparison": "comparison",
    "summarize": "summarization",
    "summarization": "summarization",
    "study_guide": "summarization",
    "graph_relation": "graph_relation",
    "table": "factual",
    "ocr": "factual",
    "audio": "factual",
    "refusal": "off_topic_should_refuse",
    "off_topic": "off_topic_should_refuse",
    "off_topic_should_refuse": "off_topic_should_refuse",
    "cross_lingual": "cross_lingual",
    "false_premise": "false_premise",
    "anaphora": "anaphora",
    "claim_check": "claim_check",
    "prompt_injection": "off_topic_should_refuse",
    "no_evidence": "off_topic_should_refuse",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _avg(rows: list[dict], key: str) -> float:
    vals = [r.get(key, 0) for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def _load_question_set(path: str) -> list[tuple[str, str]]:
    """Load questions from JSONL file.

    Accepts both legacy format (query_type, query) and gold format
    (task_type, query, expected_behavior).
    """
    questions: list[tuple[str, str]] = []
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Question set not found: {path}", file=sys.stderr)
        return questions
    with p.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [WARN] {p.name}:{lineno} bad JSON: {exc}", file=sys.stderr)
                continue
            query = item.get("query", "").strip()
            if not query:
                continue
            # Determine query type
            if item.get("expected_behavior") == "refuse":
                qtype = "off_topic_should_refuse"
            else:
                raw_type = item.get("query_type") or item.get("task_type") or "factual"
                qtype = _TASK_TYPE_MAP.get(raw_type, raw_type)
            questions.append((qtype, query))
    return questions


def _write_md_report(
    results: list[dict],
    path: str,
    args: argparse.Namespace,
    ci_threshold: float,
) -> None:
    """Write a Markdown evaluation report."""
    answered = [r for r in results if r.get("answer") and not r.get("error")]
    rag_answered = [r for r in answered
                    if r.get("query_type") not in _OFF_TOPIC_TYPES and not r.get("refused")]
    off_topic = [r for r in results if r.get("query_type") in _OFF_TOPIC_TYPES]
    correct_refuse = [r for r in off_topic if r.get("off_topic_verdict") == "correct_refuse"]
    errors = sum(1 for r in results if r.get("error"))
    false_refusals = [r for r in results if r.get("false_refusal")]
    false_premise_rows = [
        r for r in results
        if r.get("query_type") in _FALSE_PREMISE_TYPES and r.get("false_premise_corrected") is not None
    ]
    fp_corrected_count = sum(1 for r in false_premise_rows if r.get("false_premise_corrected"))

    faith_avg = _avg(rag_answered, "faithfulness")
    refusal_precision = len(correct_refuse) / len(off_topic) if off_topic else 1.0
    false_refusal_rate = len(false_refusals) / len(results) if results else 0.0

    q_set_label = Path(args.question_set).name if getattr(args, "question_set", None) else "built-in"

    lines: list[str] = [
        "# E2E Evaluation Report",
        "",
        f"**Date:** {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Owner:** `{args.owner_id}`  **Collection:** `{args.collection_id}`  ",
        f"**Question set:** `{q_set_label}`  **API:** `{args.api_url}`  **Timeout:** {args.timeout}s/query",
        "",
        "## Summary",
        "",
        "| Metric | Value | Threshold | Status |",
        "|--------|-------|-----------|--------|",
        f"| Total queries | {len(results)} | — | — |",
        f"| Answered | {len(answered)}/{len(results)} | — | — |",
        f"| Errors / Timeout | {errors} | — | — |",
        f"| Faithfulness (citation) | {faith_avg:.3f} | ≥{ci_threshold:.2f} | {'✅' if faith_avg >= ci_threshold else '❌'} |",
        f"| Refusal precision | {refusal_precision:.3f} | ≥0.85 | {'✅' if refusal_precision >= 0.85 else '❌'} |",
        f"| False refusal rate | {false_refusal_rate:.3f} | ≤0.10 | {'✅' if false_refusal_rate <= 0.10 else '❌'} |",
        f"| Avg latency | {_avg(answered, 'elapsed_s'):.1f}s | — | — |",
        "",
        "## By Query Type",
        "",
        "| Type | Answered | Faith | Rel | Sem. Faith | Grounded |",
        "|------|----------|-------|-----|------------|---------|",
    ]

    by_type: dict[str, list] = {}
    for r in results:
        by_type.setdefault(r.get("query_type", "unknown"), []).append(r)
    for qt, rows in sorted(by_type.items()):
        n_ans = sum(1 for r in rows
                    if r.get("answer") and not r.get("refused") and not r.get("error"))
        lines.append(
            f"| {qt} | {n_ans}/{len(rows)} "
            f"| {_avg(rows, 'faithfulness'):.2f} "
            f"| {_avg(rows, 'answer_relevance'):.2f} "
            f"| {_avg(rows, 'semantic_faithfulness'):.2f} "
            f"| {_avg(rows, 'grounded_sentence_ratio'):.2f} |"
        )

    # ── Trace aggregate section ───────────────────────────────────────────────
    traced = [r for r in answered if r.get("trace_latency_total_ms") is not None]
    if traced:
        avg_lat = _avg(traced, "trace_latency_total_ms")
        avg_cit_err = _avg(traced, "trace_citation_error_count")
        route_dist: dict[str, int] = {}
        for r in traced:
            rt = r.get("trace_route") or "unknown"
            route_dist[rt] = route_dist.get(rt, 0) + 1
        gate_pass = sum(1 for r in traced
                        if all(v.get("verdict") in {"PASS", "CAUTION"}
                               for v in (r.get("trace_quality_verdicts") or {}).values()))
        lines += [
            "",
            "## Trace Metrics",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Queries with trace | {len(traced)} |",
            f"| Avg latency (total ms) | {avg_lat:.0f} |",
            f"| Avg latency (generate ms) | {_avg(traced, 'trace_latency_generate_ms'):.0f} |",
            f"| Avg latency (retrieve ms) | {_avg(traced, 'trace_latency_retrieve_ms'):.0f} |",
            f"| Avg latency (rerank ms) | {_avg(traced, 'trace_latency_rerank_ms'):.0f} |",
            f"| Avg citation errors | {avg_cit_err:.2f} |",
            f"| Quality gate pass/caution | {gate_pass}/{len(traced)} |",
            f"| Route distribution | {'; '.join(f'{k}={v}' for k, v in sorted(route_dist.items()))} |",
        ]

    lines += [
        "",
        "## Per-Query Results",
        "",
        "| ID | Type | Latency | Faith | Rel | Route | CitErr | Refused | Error |",
        "|----|------|---------|-------|-----|-------|--------|---------|-------|",
    ]
    for r in results:
        err = (r.get("error") or "")[:40] or "—"
        lines.append(
            f"| {r['id']} | {r.get('query_type','?')} | {r.get('elapsed_s',0):.1f}s "
            f"| {r.get('faithfulness',0):.2f} | {r.get('answer_relevance',0):.2f} "
            f"| {r.get('trace_route') or '—'} "
            f"| {r.get('trace_citation_error_count') if r.get('trace_citation_error_count') is not None else '—'} "
            f"| {'yes' if r.get('refused') else 'no'} | {err} |"
        )

    if false_premise_rows:
        lines += [
            "",
            f"## False Premise Correction: {fp_corrected_count}/{len(false_premise_rows)}",
        ]

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved: {p.resolve()}", flush=True)


# ── Embedding helpers ──────────────────────────────────────────────────────────

def _embed(texts: list[str], api_url: str, timeout: int = 60) -> list[list[float]]:
    """Get BGE-M3 dense embeddings via /evaluation/embed endpoint."""
    resp = requests.post(
        f"{api_url}/api/v1/evaluation/embed",
        json={"texts": texts},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


# ── RAGAS-style metrics ────────────────────────────────────────────────────────

def faithfulness_citation(answer: str) -> float:
    """RAGAS faithfulness proxy: % câu trong answer có citation marker [N]."""
    sentences = [s.strip() for s in _SENTENCE_RE.findall(answer) if len(s.strip()) >= 10]
    if not sentences:
        return 1.0
    supported = sum(1 for s in sentences if _CITATION_RE.search(s))
    return supported / len(sentences)


def citation_coverage(answer: str) -> float:
    """% đoạn văn (paragraph) có ít nhất 1 citation [N]."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", answer) if p.strip()]
    if not paragraphs:
        return 1.0
    covered = sum(1 for p in paragraphs if _CITATION_RE.search(p))
    return covered / len(paragraphs)


def answer_relevance(query_emb: list[float], answer_emb: list[float]) -> float:
    """cosine_sim(embed(query), embed(answer)) — RAGAS answer relevance."""
    return _cosine(query_emb, answer_emb)


def semantic_faithfulness(
    answer: str,
    answer_embs: list[list[float]],
    citation_embs: list[list[float]],
) -> float:
    """
    RAGAS-style semantic faithfulness:
    For each answer sentence, find max cosine_sim with any citation snippet.
    Score = mean of those max similarities.
    """
    sentences = [s.strip() for s in _SENTENCE_RE.findall(answer) if len(s.strip()) >= 10]
    if not sentences or not citation_embs:
        return 0.0
    if not answer_embs:
        return 0.0

    total = 0.0
    for sent_emb in answer_embs:
        max_sim = max(_cosine(sent_emb, c_emb) for c_emb in citation_embs)
        total += max_sim
    return total / len(answer_embs)


def context_precision(citations: list[dict], threshold: float = 0.4) -> float:
    """% citations có confidence score >= threshold."""
    if not citations:
        return 0.0
    scores = [c.get("confidence") or c.get("score") or 0.0 for c in citations]
    return sum(1 for s in scores if s >= threshold) / len(scores)


def citation_validity(answer: str, num_citations: int) -> float:
    """% citation markers [N] in answer where N is within valid range (1..num_citations)."""
    if num_citations == 0:
        return 1.0
    matches = _CITATION_RE.findall(answer)
    if not matches:
        return 1.0
    valid = sum(1 for m in matches if 1 <= int(m) <= num_citations)
    return valid / len(matches)


def false_premise_corrected(answer: str) -> bool:
    """True if the answer contains correction language (for false_premise queries)."""
    lower = answer.lower()
    return any(kw in lower for kw in _CORRECTION_KEYWORDS)


def grounded_sentence_ratio(
    answer_embs: list[list[float]],
    citation_embs: list[list[float]],
    threshold: float = 0.4,
) -> float:
    """% answer sentences with max cosine_sim to any citation snippet >= threshold."""
    if not answer_embs or not citation_embs:
        return 0.0
    grounded = sum(
        1 for s_emb in answer_embs
        if max(_cosine(s_emb, c_emb) for c_emb in citation_embs) >= threshold
    )
    return grounded / len(answer_embs)


def refused_check(answer: str) -> bool:
    """True chỉ khi answer là refusal thực sự — ngắn và chứa refusal pattern.
    Tránh false positive khi LLM có partial content + disclaimer."""
    if not answer:
        return True
    lower = answer.lower()
    hard_patterns = [
        "nằm ngoài phạm vi", "không thuộc phạm vi",
        "i don't have information", "outside the scope",
        "không thể trả lời câu hỏi này",
    ]
    if any(p in lower for p in hard_patterns):
        return True
    # Soft patterns chỉ từ chối nếu answer ngắn (< 200 chars = không có real content)
    soft_patterns = [
        "không tìm thấy đủ bằng chứng",
        "không có thông tin",
        "không đề cập",
        "không có trong tài liệu",
        "not found",
        "không liên quan",
    ]
    if len(answer.strip()) < 200 and any(p in lower for p in soft_patterns):
        return True
    return False


# ── API calls ──────────────────────────────────────────────────────────────────

def ask(*, api_url: str, owner_id: str, collection_id: str, query: str, timeout: int) -> dict:
    payload = {
        "query": query,
        "owner_id": owner_id,
        "collection_id": collection_id,
        "stream": False,
    }
    resp = requests.post(f"{api_url}/api/v1/query/ask", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}", flush=True)
    print(f"  E2E Eval (RAGAS-style metrics) — {args.owner_id}", flush=True)
    print(f"  API: {args.api_url}  timeout={args.timeout}s/query", flush=True)
    print(f"{'='*65}\n", flush=True)

    # Load question set
    if args.question_set:
        print(f"Loading question set: {args.question_set}", flush=True)
        questions = _load_question_set(args.question_set)
        if not questions:
            print("[ERROR] No valid questions loaded.", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(questions)} questions\n", flush=True)
    else:
        questions = list(ML_QUESTIONS)

    if args.types:
        questions = [(qt, q) for qt, q in questions if qt in args.types]

    # Resume: load existing results if output file exists
    results = []
    errors = 0
    start_idx = 0
    if args.start_from > 0 or (out_path.exists() and args.resume):
        if out_path.exists():
            with out_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
            start_idx = len(results)
            errors = sum(1 for r in results if r.get("error"))
            print(f"Resuming from Q{start_idx + 1} ({len(results)} results loaded)", flush=True)
        if args.start_from > 0:
            start_idx = args.start_from

    for i, (qtype, query) in enumerate(questions):
        if i < start_idx:
            continue
        print(f"[{i+1:>2}/{len(questions)}] [{qtype}] {query[:55]}", flush=True)
        t0 = time.time()
        try:
            data = ask(
                api_url=args.api_url,
                owner_id=args.owner_id,
                collection_id=args.collection_id,
                query=query,
                timeout=args.timeout,
            )
            elapsed = time.time() - t0
            payload = data.get("data") or data
            answer = payload.get("answer") or ""
            citations = payload.get("citations") or []
            top_conf = payload.get("confidence") or 0.0
            is_refused = payload.get("was_refused", False) or refused_check(answer)

            # ── Citation-based metrics (fast, no embedding) ──────────────────
            faith_cit = faithfulness_citation(answer)
            cit_cov = citation_coverage(answer)
            ctx_prec = context_precision(citations)
            cit_valid = citation_validity(answer, len(citations))
            sentences = [s.strip() for s in _SENTENCE_RE.findall(answer) if len(s.strip()) >= 10]
            n_sentences = len(sentences)
            fp_corrected = false_premise_corrected(answer) if qtype in _FALSE_PREMISE_TYPES else None

            # ── Embedding-based metrics (semantic) ───────────────────────────
            ans_rel = 0.0
            sem_faith = 0.0
            grounded_ratio = 0.0
            try:
                citation_snippets = []
                for c in citations[:5]:
                    parts: list[str] = []
                    primary = c.get("snippet_original") or c.get("snippet") or c.get("content") or ""
                    if primary:
                        parts.append(primary)
                    for blk in (c.get("evidence_blocks") or [])[:4]:
                        snippet = blk.get("snippet_original") or blk.get("snippet") or ""
                        if snippet and snippet not in parts:
                            parts.append(snippet)
                    merged = " ".join(parts)[:1200]
                    citation_snippets.append(merged)
                answer_sentences = [
                    _CITATION_CLEAN_RE.sub("", s).strip()
                    for s in _SENTENCE_RE.findall(answer)
                    if len(_CITATION_CLEAN_RE.sub("", s).strip()) >= 20
                ][:8]

                texts_to_embed: list[str] = []
                q_idx = len(texts_to_embed); texts_to_embed.append(query)
                a_full_idx = len(texts_to_embed); texts_to_embed.append(answer[:1000])
                sent_start = len(texts_to_embed); texts_to_embed.extend(answer_sentences)
                cit_start = len(texts_to_embed); texts_to_embed.extend(citation_snippets)

                if texts_to_embed:
                    all_embs = _embed(texts_to_embed, args.api_url)
                    q_emb = all_embs[q_idx]
                    a_emb = all_embs[a_full_idx]
                    sent_embs = all_embs[sent_start:sent_start + len(answer_sentences)]
                    cit_embs = all_embs[cit_start:cit_start + len(citation_snippets)]

                    ans_rel = answer_relevance(q_emb, a_emb)
                    sem_faith = semantic_faithfulness(
                        answer=answer,
                        answer_embs=sent_embs,
                        citation_embs=cit_embs,
                    )
                    grounded_ratio = grounded_sentence_ratio(sent_embs, cit_embs, threshold=0.4)
            except Exception as emb_exc:
                print(f"        [embed error] {type(emb_exc).__name__}: {emb_exc}", flush=True)

            # ── Trace fields (Phase 6) ───────────────────────────────────────
            trace_raw: dict = payload.get("trace") or {}
            trace_route = trace_raw.get("route")
            trace_modality = trace_raw.get("modality")
            trace_difficulty = trace_raw.get("difficulty")
            trace_latency: dict = trace_raw.get("latency_by_stage") or {}
            trace_latency_total = trace_latency.get("total")
            trace_cit_errors: int | None = trace_raw.get("citation_error_count")
            trace_claim_count: int | None = trace_raw.get("claim_count")
            trace_verdicts: dict = trace_raw.get("quality_stage_verdicts") or {}

            # ── Off-topic verdict ────────────────────────────────────────────
            status = "refused" if is_refused else "answered"
            if qtype in _OFF_TOPIC_TYPES:
                verdict = "correct_refuse" if is_refused else "wrong_answered"
            else:
                verdict = None

            fp_str = f"  fp_corrected={fp_corrected}" if fp_corrected is not None else ""
            route_str = f"  route={trace_route}" if trace_route else ""
            lat_str = f"  lat={trace_latency_total}ms" if trace_latency_total is not None else ""
            print(
                f"        [{status}] {elapsed:.1f}s  conf={top_conf:.3f}  "
                f"faith={faith_cit:.2f}  cit_cov={cit_cov:.2f}  cit_valid={cit_valid:.2f}{fp_str}",
                flush=True,
            )
            print(
                f"        ans_rel={ans_rel:.3f}  sem_faith={sem_faith:.3f}  "
                f"grounded={grounded_ratio:.2f}  ctx_prec={ctx_prec:.2f}  sents={n_sentences}",
                flush=True,
            )
            if trace_raw:
                print(
                    f"        trace:{route_str}  modality={trace_modality}  diff={trace_difficulty}"
                    f"{lat_str}  cit_errors={trace_cit_errors}  claims={trace_claim_count}",
                    flush=True,
                )
                if trace_verdicts:
                    vdict_str = "  ".join(
                        f"{s}={v.get('verdict','-')}({v.get('score',0):.2f})"
                        for s, v in trace_verdicts.items()
                    )
                    print(f"        quality: {vdict_str}", flush=True)
            if answer:
                print(f"        answer[:80]: {answer[:80]}", flush=True)

            results.append({
                "id": f"q{i+1:03d}",
                "query_type": qtype,
                "query": query,
                "answer": answer,
                "citations": citations,
                "confidence": top_conf,
                "elapsed_s": round(elapsed, 2),
                # RAGAS-style metrics
                "faithfulness": round(faith_cit, 3),
                "citation_coverage": round(cit_cov, 3),
                "citation_validity": round(cit_valid, 3),
                "answer_relevance": round(ans_rel, 3),
                "semantic_faithfulness": round(sem_faith, 3),
                "grounded_sentence_ratio": round(grounded_ratio, 3),
                "context_precision": round(ctx_prec, 3),
                "answer_sentences": n_sentences,
                "false_premise_corrected": fp_corrected,
                "false_refusal": (qtype not in _OFF_TOPIC_TYPES and is_refused),
                "refused": is_refused,
                "off_topic_verdict": verdict,
                "human_verdict": None,
                # Trace fields (Phase 6)
                "trace_route": trace_route,
                "trace_modality": trace_modality,
                "trace_difficulty": trace_difficulty,
                "trace_latency_total_ms": trace_latency_total,
                "trace_latency_generate_ms": trace_latency.get("generate"),
                "trace_latency_retrieve_ms": trace_latency.get("retrieve"),
                "trace_latency_rerank_ms": trace_latency.get("rerank"),
                "trace_citation_error_count": trace_cit_errors,
                "trace_claim_count": trace_claim_count,
                "trace_quality_verdicts": trace_verdicts or None,
            })

        except requests.Timeout:
            print(f"        TIMEOUT {args.timeout}s", flush=True)
            errors += 1
            results.append({"id": f"q{i+1:03d}", "query_type": qtype, "query": query,
                            "answer": None, "error": "timeout", "human_verdict": None})
        except Exception as exc:
            print(f"        ERROR: {type(exc).__name__}: {exc}", flush=True)
            errors += 1
            results.append({"id": f"q{i+1:03d}", "query_type": qtype, "query": query,
                            "answer": None, "error": str(exc), "human_verdict": None})
        print(flush=True)

    # Save results (incremental write)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregate
    answered = [r for r in results if r.get("answer") and not r.get("error")]
    rag_answered = [r for r in answered
                    if r.get("query_type") not in _OFF_TOPIC_TYPES and not r.get("refused")]
    off_topic = [r for r in results if r.get("query_type") in _OFF_TOPIC_TYPES]
    correct_refuse = [r for r in off_topic if r.get("off_topic_verdict") == "correct_refuse"]

    w = 65
    print(f"\n{'='*w}", flush=True)
    print(f"  E2E EVAL REPORT (RAGAS-style)", flush=True)
    print(f"{'='*w}", flush=True)
    print(f"  Tổng queries:           {len(results)}", flush=True)
    print(f"  Answered:               {len(answered)}/{len(results)}", flush=True)
    print(f"  Errors/Timeout:         {errors}", flush=True)
    print(f"  Avg elapsed:            {_avg(answered, 'elapsed_s'):.1f}s/query", flush=True)
    print(f"{'─'*w}", flush=True)
    false_premise_rows = [r for r in results
                          if r.get("query_type") in _FALSE_PREMISE_TYPES
                          and r.get("false_premise_corrected") is not None]
    fp_corrected_count = sum(1 for r in false_premise_rows if r.get("false_premise_corrected"))
    false_refusals = [r for r in results if r.get("false_refusal")]

    print(f"  RAG QUERIES ({len(rag_answered)} answered non-refused)", flush=True)
    print(f"    Faithfulness (cit.):  {_avg(rag_answered, 'faithfulness'):.3f}  (% câu có [N] marker)", flush=True)
    print(f"    Citation coverage:    {_avg(rag_answered, 'citation_coverage'):.3f}  (% đoạn có citation)", flush=True)
    print(f"    Citation validity:    {_avg(rag_answered, 'citation_validity'):.3f}  (% [N] trong range citations)", flush=True)
    print(f"    Answer relevance:     {_avg(rag_answered, 'answer_relevance'):.3f}  (cosine query↔answer)", flush=True)
    print(f"    Semantic faithfulness:{_avg(rag_answered, 'semantic_faithfulness'):.3f}  (cosine answer↔citation)", flush=True)
    print(f"    Grounded ratio:       {_avg(rag_answered, 'grounded_sentence_ratio'):.3f}  (% câu sem≥0.4 vs citation)", flush=True)
    print(f"    Context precision:    {_avg(rag_answered, 'context_precision'):.3f}  (% chunks score≥0.4)", flush=True)
    print(f"    Avg confidence:       {_avg(rag_answered, 'confidence'):.3f}", flush=True)
    print(f"    Avg sentences/answer: {_avg(rag_answered, 'answer_sentences'):.1f}", flush=True)
    print(f"{'─'*w}", flush=True)
    print(f"  OFF-TOPIC ({len(off_topic)} queries)", flush=True)
    print(f"    Correctly refused:    {len(correct_refuse)}/{len(off_topic)}", flush=True)
    print(f"  FALSE PREMISE ({len(false_premise_rows)} queries)", flush=True)
    print(f"    Correction detected:  {fp_corrected_count}/{len(false_premise_rows)}", flush=True)
    print(f"  FALSE REFUSALS (in-scope refused): {len(false_refusals)}", flush=True)
    print(f"{'─'*w}", flush=True)
    # ── Trace aggregate (Phase 6) ─────────────────────────────────────────────
    traced = [r for r in answered if r.get("trace_latency_total_ms") is not None]
    if traced:
        print(f"  TRACE FIELDS ({len(traced)} queries with trace)", flush=True)
        avg_lat = _avg(traced, "trace_latency_total_ms")
        avg_gen = _avg(traced, "trace_latency_generate_ms")
        avg_ret = _avg(traced, "trace_latency_retrieve_ms")
        avg_rnk = _avg(traced, "trace_latency_rerank_ms")
        avg_cit_err = _avg(traced, "trace_citation_error_count")
        print(f"    Avg latency total:    {avg_lat:.0f}ms", flush=True)
        print(f"    Avg latency generate: {avg_gen:.0f}ms", flush=True)
        print(f"    Avg latency retrieve: {avg_ret:.0f}ms", flush=True)
        print(f"    Avg latency rerank:   {avg_rnk:.0f}ms", flush=True)
        print(f"    Avg citation errors:  {avg_cit_err:.2f}", flush=True)
        # Route distribution
        route_dist: dict[str, int] = {}
        for r in traced:
            rt = r.get("trace_route") or "unknown"
            route_dist[rt] = route_dist.get(rt, 0) + 1
        routes_str = "  ".join(f"{k}={v}" for k, v in sorted(route_dist.items()))
        print(f"    Route distribution:   {routes_str}", flush=True)
        # Quality gate verdicts
        gate_pass = sum(1 for r in traced
                        if all(v.get("verdict") in {"PASS", "CAUTION"}
                               for v in (r.get("trace_quality_verdicts") or {}).values()))
        gate_fail = len(traced) - gate_pass
        print(f"    Quality gate: {gate_pass} pass/caution  {gate_fail} with FAIL stage", flush=True)
        print(f"{'─'*w}", flush=True)
    print(f"  BY QUERY TYPE", flush=True)
    by_type: dict[str, list] = {}
    for r in results:
        by_type.setdefault(r.get("query_type", "unknown"), []).append(r)
    for qt, rows in sorted(by_type.items()):
        n_ans = sum(1 for r in rows
                    if r.get("answer") and not r.get("refused") and not r.get("error"))
        avg_faith = _avg(rows, "faithfulness")
        avg_rel = _avg(rows, "answer_relevance")
        print(f"    {qt:<30} answered={n_ans}/{len(rows)}  faith={avg_faith:.2f}  rel={avg_rel:.2f}", flush=True)
    print(f"{'─'*w}", flush=True)
    print(f"  Saved: {out_path.resolve()}", flush=True)
    print(f"{'='*w}\n", flush=True)

    # ── Optional: Markdown report ─────────────────────────────────────────────
    if args.report:
        _write_md_report(results, args.report, args, ci_threshold=args.ci_threshold)

    # ── Optional: CI mode exit code ───────────────────────────────────────────
    if args.ci_mode:
        faith_avg = _avg(rag_answered, "faithfulness")
        if faith_avg < args.ci_threshold:
            print(
                f"\n[CI] FAIL: faithfulness={faith_avg:.3f} < threshold={args.ci_threshold:.2f}",
                file=sys.stderr, flush=True,
            )
            sys.exit(1)
        else:
            print(f"\n[CI] PASS: faithfulness={faith_avg:.3f} >= {args.ci_threshold:.2f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--output", default="eval_results/e2e_eval.jsonl")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--types", nargs="+", default=None,
                        help="Filter by query type(s), e.g. --types factual comparison")

    # Tier A additions
    parser.add_argument("--question-set", default=None,
                        help="Path to JSONL with {query, query_type|task_type} records "
                             "(e.g. agentbook_e2e_gold.jsonl). Overrides built-in ML_QUESTIONS.")
    parser.add_argument("--report", default=None,
                        help="Save Markdown evaluation report to this path")
    parser.add_argument("--ci-mode", action="store_true",
                        help="Exit 1 when faithfulness < --ci-threshold (use in CI pipelines)")
    parser.add_argument("--ci-threshold", type=float, default=0.70,
                        help="Faithfulness threshold for --ci-mode (default 0.70)")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip first N questions (0-based). Use to resume after crash.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output JSONL (auto-detect start index).")
    args = parser.parse_args()
    main(args)
