"""
LLM-as-judge evaluation for AgentBook E2E results.

Scores each answer on 5 axes using an OpenAI-compatible model:
  groundedness        — claims supported by cited evidence
  answer_relevance    — answer addresses the question
  citation_correctness— [N] markers map to correct evidence
  refusal_correctness — refuse when expected, answer when expected
  vietnamese_quality  — fluency, tone marks, no unnecessary EN mixing

Usage:
    cd backend
    python scripts/judge_eval_with_gpt4o.py \
        --input  eval_results/e2e_eval.jsonl \
        --gold   ../evaluation/datasets/agentbook_e2e_gold.jsonl \
        --rubric ../evaluation/datasets/agentbook_judge_rubric.yaml \
        --model  gpt-5.4-mini \
        --api-base https://luongchidung.online/v1 \
        --api-key  sk-... \
        --output eval_results/e2e_judged.jsonl \
        --report eval_results/e2e_judge_report.md
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

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── LLM call ──────────────────────────────────────────────────────────────────

async def _call_llm(
    *,
    system: str,
    user: str,
    model: str,
    api_base: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    retries: int = 3,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{api_base}/chat/completions", headers=headers, json=body)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_exc}")


def _extract_json(text: str) -> dict:
    """Extract JSON object from model output (handles markdown fences)."""
    # Try direct parse first
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    # Try extracting from markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding any JSON object
    m = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


# ── Judge prompt ───────────────────────────────────────────────────────────────

def _build_judge_prompt(
    *,
    query: str,
    answer: str,
    citations: list[dict],
    gold: dict | None,
    rubric: dict,
) -> str:
    required_facts = gold.get("required_facts", []) if gold else []
    forbidden_claims = gold.get("forbidden_claims", []) if gold else []
    expected_behavior = gold.get("expected_behavior", "answer") if gold else "answer"
    answer_language = gold.get("answer_language", "vi") if gold else "vi"

    citations_text = ""
    for i, c in enumerate(citations[:6], 1):
        snippet = c.get("snippet_original") or c.get("snippet") or c.get("content") or ""
        doc = c.get("document_name") or c.get("doc_name") or "?"
        page = c.get("page") or c.get("page_number") or "?"
        citations_text += f"[{i}] {doc} p.{page}: {snippet[:300]}\n"

    facts_text = "\n".join(f"- {f}" for f in required_facts) if required_facts else "(none specified)"
    forbidden_text = "\n".join(f"- {f}" for f in forbidden_claims) if forbidden_claims else "(none specified)"

    return f"""Evaluate this system answer for a Vietnamese document Q&A system.

QUERY:
{query}

EXPECTED BEHAVIOR: {expected_behavior}
ANSWER LANGUAGE REQUIRED: {answer_language}

REQUIRED FACTS (must appear if expected_behavior=answer):
{facts_text}

FORBIDDEN CLAIMS (must NOT appear):
{forbidden_text}

SYSTEM ANSWER:
{answer[:2000] if answer else "(no answer — system refused)"}

CITED EVIDENCE:
{citations_text or "(no citations)"}

Score the answer on these 5 axes using the rubric criteria.
Each score: 0.0 to 1.0 (see rubric for guidance: 1.0=best, 0.0=worst).

Also:
- required_facts_covered: fraction of required facts present in answer (0.0–1.0)
- forbidden_claims_violated: true if any forbidden claim appears

Output ONLY this JSON object, no other text:
{{
  "groundedness": <float>,
  "answer_relevance": <float>,
  "citation_correctness": <float>,
  "refusal_correctness": <float>,
  "vietnamese_quality": <float>,
  "required_facts_covered": <float>,
  "forbidden_claims_violated": <bool>,
  "rationale": "<1-2 sentences>"
}}"""


# ── Load / save helpers ────────────────────────────────────────────────────────

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


def _save_jsonl(rows: list[dict], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Markdown report ────────────────────────────────────────────────────────────

def _write_report(judged: list[dict], path: str, thresholds: dict) -> None:
    import datetime as dt

    axes = ["groundedness", "answer_relevance", "citation_correctness",
            "refusal_correctness", "vietnamese_quality"]

    def _avg(rows: list[dict], key: str) -> float:
        vals = [r["scores"][key] for r in rows if isinstance(r.get("scores", {}).get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    smoke = thresholds.get("smoke", {})

    lines: list[str] = []
    lines.append("# AgentBook E2E Judge Report")
    lines.append("")
    lines.append(f"**Date:** {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Samples judged:** {len(judged)}")
    lines.append("")
    lines.append("## Aggregate Scores")
    lines.append("")
    lines.append("| Axis | Score | Smoke threshold | Status |")
    lines.append("|------|-------|-----------------|--------|")
    for ax in axes:
        score = _avg(judged, ax)
        thr = smoke.get(ax, 0.0)
        badge = "✅" if score >= thr else "❌"
        lines.append(f"| {ax} | {score:.3f} | ≥{thr:.2f} | {badge} |")

    # required_facts_covered and forbidden_claims
    rf_scores = [r["scores"].get("required_facts_covered", 0) for r in judged if r.get("scores")]
    rf_avg = sum(rf_scores) / len(rf_scores) if rf_scores else 0.0
    fc_violated = sum(1 for r in judged if r.get("scores", {}).get("forbidden_claims_violated") is True)
    lines.append(f"| required_facts_covered | {rf_avg:.3f} | — | — |")
    lines.append(f"| forbidden_claims_violated | {fc_violated}/{len(judged)} | 0 | {'✅' if fc_violated==0 else '❌'} |")
    lines.append("")
    lines.append("## Per-Query Results")
    lines.append("")
    lines.append("| ID | Query (50) | Ground. | Rel. | Citation | Refusal | VI qual. | Note |")
    lines.append("|----|-----------|---------|------|----------|---------|----------|------|")
    for r in judged:
        sc = r.get("scores") or {}
        q = (r.get("query") or "")[:50]
        lines.append(
            f"| {r.get('id','?')} | {q} "
            f"| {sc.get('groundedness',0):.2f} "
            f"| {sc.get('answer_relevance',0):.2f} "
            f"| {sc.get('citation_correctness',0):.2f} "
            f"| {sc.get('refusal_correctness',0):.2f} "
            f"| {sc.get('vietnamese_quality',0):.2f} "
            f"| {(sc.get('rationale',''))[:60]} |"
        )
    lines.append("")

    # Worst performers
    scored = [r for r in judged if r.get("scores")]
    if scored:
        def _total(r: dict) -> float:
            sc = r.get("scores", {})
            return sum(sc.get(ax, 0) for ax in axes) / len(axes)
        worst = sorted(scored, key=_total)[:5]
        lines.append("## Worst Performers (by avg score)")
        lines.append("")
        for r in worst:
            lines.append(f"- **{r.get('id')}** `{(r.get('query',''))[:60]}` — avg={_total(r):.2f}")
            lines.append(f"  {r.get('scores',{}).get('rationale','')}")
        lines.append("")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved: {p.resolve()}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    # Load rubric
    rubric: dict = {}
    if args.rubric and Path(args.rubric).exists():
        try:
            import yaml  # type: ignore[import]
            with open(args.rubric, encoding="utf-8") as f:
                rubric = yaml.safe_load(f)
        except ImportError:
            # Fallback: read YAML as text (only system_prompt is critical)
            pass

    system_prompt = (rubric.get("system_prompt") or
        "You are an expert evaluator for a Vietnamese document Q&A system. "
        "Output only valid JSON.")
    thresholds = rubric.get("thresholds", {})

    # Load eval results
    eval_rows = _load_jsonl(args.input)
    if not eval_rows:
        print(f"[ERROR] No results in {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load gold cases
    gold_map: dict[str, dict] = {}
    if args.gold:
        for row in _load_jsonl(args.gold):
            q = row.get("query", "").strip()
            if q:
                gold_map[q] = row

    print(f"\n{'='*60}")
    print(f"  AgentBook LLM Judge")
    print(f"  Model:   {args.model}")
    print(f"  API:     {args.api_base}")
    print(f"  Inputs:  {len(eval_rows)} results, {len(gold_map)} gold cases")
    print(f"{'='*60}\n")

    judged: list[dict] = []

    for i, row in enumerate(eval_rows):
        query = row.get("query", "")
        answer = row.get("answer") or ""
        citations = row.get("citations") or []
        gold = gold_map.get(query.strip())

        print(f"[{i+1:>3}/{len(eval_rows)}] {query[:60]}", flush=True)

        scores: dict = {}
        error: str | None = None

        if row.get("error"):
            # No answer to judge
            scores = {
                "groundedness": 0.0,
                "answer_relevance": 0.0,
                "citation_correctness": 0.0,
                "refusal_correctness": 0.0,
                "vietnamese_quality": 0.0,
                "required_facts_covered": 0.0,
                "forbidden_claims_violated": False,
                "rationale": f"Eval error: {row.get('error')}",
            }
        else:
            try:
                t0 = time.perf_counter()
                raw = await _call_llm(
                    system=system_prompt,
                    user=_build_judge_prompt(
                        query=query,
                        answer=answer,
                        citations=citations,
                        gold=gold,
                        rubric=rubric,
                    ),
                    model=args.model,
                    api_base=args.api_base,
                    api_key=args.api_key,
                    max_tokens=512,
                )
                elapsed = time.perf_counter() - t0
                scores = _extract_json(raw)
                if not scores:
                    error = f"Could not parse JSON from judge: {raw[:100]}"
                    print(f"  [WARN] {error}", flush=True)
                else:
                    gr = scores.get("groundedness", 0)
                    ar = scores.get("answer_relevance", 0)
                    print(f"  ground={gr:.2f}  rel={ar:.2f}  {elapsed:.1f}s", flush=True)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"  [ERROR] {error}", flush=True)

        judged.append({
            "id": row.get("id", f"q{i+1:03d}"),
            "query": query,
            "query_type": row.get("query_type", ""),
            "answer": answer[:400] if answer else "",
            "refused": row.get("refused", False),
            "scores": scores,
            "judge_error": error,
        })

        # Rate limit pause
        await asyncio.sleep(0.5)

    # Save output
    _save_jsonl(judged, args.output)
    print(f"\nSaved {len(judged)} judged records → {args.output}", flush=True)

    # Aggregate
    scored = [r for r in judged if r.get("scores") and not r.get("judge_error")]
    axes = ["groundedness", "answer_relevance", "citation_correctness",
            "refusal_correctness", "vietnamese_quality"]

    def _avg(rows: list[dict], key: str) -> float:
        vals = [r["scores"][key] for r in rows if isinstance(r.get("scores", {}).get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    w = 55
    print(f"\n{'='*w}")
    print(f"  JUDGE AGGREGATE ({len(scored)} scored)")
    print(f"{'─'*w}")
    smoke = thresholds.get("smoke", {})
    for ax in axes:
        score = _avg(scored, ax)
        thr = smoke.get(ax, 0.0)
        badge = "✅" if score >= thr else "❌"
        print(f"  {ax:<28} {score:.3f}  (smoke≥{thr:.2f}) {badge}")
    print(f"{'='*w}\n")

    if args.report:
        _write_report(judged, args.report, thresholds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation for AgentBook")
    parser.add_argument("--input", required=True, help="e2e_eval.jsonl output")
    parser.add_argument("--gold", help="agentbook_e2e_gold.jsonl for required_facts etc.")
    parser.add_argument("--rubric", help="agentbook_judge_rubric.yaml")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--api-base", default="https://luongchidung.online/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--output", default="eval_results/e2e_judged.jsonl")
    parser.add_argument("--report", help="Path to save Markdown report")
    args = parser.parse_args()

    if not args.api_key:
        import os
        args.api_key = os.getenv("OPENAI_API_KEY", "")
    if not args.api_key:
        print("[ERROR] --api-key or OPENAI_API_KEY required", file=sys.stderr)
        sys.exit(1)

    asyncio.run(main(args))
