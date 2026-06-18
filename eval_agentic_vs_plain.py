"""
Evaluation: Plain RAG vs Agentic RAG
Chạy 5 câu hỏi multi-hop khó, dùng LLM-as-judge chấm điểm.

Usage:
    python eval_agentic_vs_plain.py

Output:
    eval_results.json  — raw scores
    eval_report.md     — bảng tổng kết
"""

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import httpx

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE   = "http://localhost:8000/api/v1"
OWNER_ID   = "nguyenvtp69_gmail_com"
COLLECTION = "6a16f8d1a0d535db39664088"
TOP_K      = 6

# LLM judge trực tiếp qua gateway (không cần khởi InferenceEngine)
LLM_BASE   = "https://luongchidung.online/v1"
LLM_KEY    = "sk-Jf9O8lgvf6qEstD2yTmKQE1m5Cw5KiYCiV41t8sZhHPmmVOF"
LLM_MODEL  = "gpt-5.4-mini"

# ── Test questions ─────────────────────────────────────────────────────────────
# Được chọn để khai thác điểm mạnh của agentic RAG:
#   Q1 – multi-hop: điều kiện kết hôn → hậu quả vi phạm (2 section khác nhau)
#   Q2 – multi-hop: vợ/chồng chết → thừa kế tài sản + quyền nuôi con (2 chủ đề)
#   Q3 – comparison: quyền nuôi con dưới 36 tháng vs trên 36 tháng
#   Q4 – chained: hôn nhân vô hiệu → hậu quả pháp lý tài sản + con cái
#   Q5 – complex factual: nghĩa vụ cấp dưỡng sau ly hôn, mức tính + thay đổi
QUESTIONS = [
    {
        "id": "Q1",
        "type": "multi_hop_condition_consequence",
        "text": "Điều kiện để kết hôn hợp lệ theo Luật Hôn nhân và Gia đình là gì, và nếu vi phạm các điều kiện đó thì hậu quả pháp lý sẽ như thế nào?",
        "expected_hops": ["điều kiện kết hôn", "hậu quả vi phạm / hôn nhân vô hiệu"],
    },
    {
        "id": "Q2",
        "type": "multi_hop_death",
        "text": "Khi một bên vợ hoặc chồng chết thì người còn lại có quyền và nghĩa vụ gì đối với tài sản chung và việc nuôi dưỡng con cái?",
        "expected_hops": ["quyền tài sản khi vợ/chồng chết", "nghĩa vụ nuôi con sau khi bên kia mất"],
    },
    {
        "id": "Q3",
        "type": "comparison",
        "text": "Quyền trực tiếp nuôi con sau ly hôn khác nhau như thế nào giữa trường hợp con dưới 36 tháng tuổi và con từ 36 tháng tuổi trở lên?",
        "expected_hops": ["nuôi con dưới 36 tháng", "nuôi con từ 36 tháng trở lên"],
    },
    {
        "id": "Q4",
        "type": "chained_consequence",
        "text": "Hôn nhân vô hiệu được xác định dựa trên những căn cứ nào, và khi hôn nhân bị tuyên vô hiệu thì tài sản và quyền nuôi con được giải quyết ra sao?",
        "expected_hops": ["căn cứ hôn nhân vô hiệu", "hậu quả pháp lý tài sản + con cái"],
    },
    {
        "id": "Q5",
        "type": "complex_factual",
        "text": "Mức cấp dưỡng cho con sau khi ly hôn được xác định như thế nào, và người có nghĩa vụ cấp dưỡng có thể yêu cầu thay đổi mức cấp dưỡng trong trường hợp nào?",
        "expected_hops": ["cách tính mức cấp dưỡng", "điều kiện thay đổi mức cấp dưỡng"],
    },
]

# ── Judge prompt ──────────────────────────────────────────────────────────────
JUDGE_PROMPT = """Bạn là chuyên gia đánh giá chất lượng câu trả lời RAG cho tài liệu pháp luật.

Câu hỏi: {question}
Các hop cần trả lời: {hops}

=== Câu trả lời A ===
{answer_a}

=== Câu trả lời B ===
{answer_b}

Chấm điểm MỖI câu trả lời trên 3 tiêu chí (mỗi tiêu chí 0-10):
1. **completeness**: Trả lời đầy đủ tất cả các hop không? Có bỏ sót thông tin quan trọng không?
2. **grounding**: Có citation cụ thể không? Số điều luật được trích dẫn không? Không bịa thông tin không?
3. **legal_precision**: Dùng đúng thuật ngữ pháp lý không? Không nhầm lẫn khái niệm không?

Output JSON duy nhất (không có text khác):
{{
  "A": {{"completeness": <0-10>, "grounding": <0-10>, "legal_precision": <0-10>, "comment": "<ngắn gọn>"}},
  "B": {{"completeness": <0-10>, "grounding": <0-10>, "legal_precision": <0-10>, "comment": "<ngắn gọn>"}},
  "winner": "A" | "B" | "tie",
  "reason": "<1-2 câu giải thích>"
}}"""


@dataclass
class PipelineResult:
    pipeline: str          # "plain" | "agentic"
    question_id: str
    latency_s: float
    answer: str
    citations: int
    confidence: float
    slec_coverage: float
    guardrail_verdict: str
    guardrail_confidence: float
    evidence_retrieved: int
    evidence_final: int
    was_refused: bool
    raw: dict = field(default_factory=dict)


@dataclass
class JudgeResult:
    question_id: str
    winner: str
    plain_scores: dict
    agentic_scores: dict
    reason: str


# ── HTTP helpers ──────────────────────────────────────────────────────────────
async def ask(client: httpx.AsyncClient, question: str, agentic: bool) -> dict[str, Any]:
    body = {
        "owner_id": OWNER_ID,
        "collection_id": COLLECTION,
        "query": question,
        "top_k": TOP_K,
        "rag_flags": {"agentic_rag_enabled": agentic},
    }
    r = await client.post(f"{API_BASE}/query/ask", json=body, timeout=300)
    r.raise_for_status()
    return r.json()


async def llm_judge(client: httpx.AsyncClient, question: str, hops: list[str],
                    answer_plain: str, answer_agentic: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        hops=", ".join(hops),
        answer_a=answer_plain,
        answer_b=answer_agentic,
    )
    r = await client.post(
        f"{LLM_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_KEY}"},
        json={"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.0},
        timeout=60,
    )
    r.raise_for_status()
    raw_text = r.json()["choices"][0]["message"]["content"].strip()
    # Strip markdown if wrapped
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    return json.loads(raw_text.strip())


# ── Extraction helpers ────────────────────────────────────────────────────────
def extract_result(resp: dict, pipeline: str, qid: str, latency: float) -> PipelineResult:
    d = resp.get("data", {})
    trace = d.get("agent_trace") or {}
    ver = trace.get("verification") or {}
    slec = d.get("sentence_coverage") or {}
    steps = trace.get("steps") or []

    ev_retrieved = 0
    ev_final = 0
    for s in steps:
        if s.get("name") == "retrieve_evidence":
            ev_retrieved = s.get("evidence_count") or 0
        if s.get("name") == "rerank_evidence":
            ev_final = s.get("evidence_count") or 0
    # Plain RAG: use reasoning_path
    if not ev_retrieved:
        rp = d.get("reasoning_path") or []
        for step in rp:
            if step.get("step_type") == "retrieve":
                desc = step.get("description", "")
                import re
                m = re.search(r"Retrieved (\d+)", desc)
                if m:
                    ev_retrieved = int(m.group(1))
            if step.get("step_type") == "synthesize":
                desc = step.get("description", "")
                m = re.search(r"top (\d+)", desc)
                if m:
                    ev_final = int(m.group(1))

    return PipelineResult(
        pipeline=pipeline,
        question_id=qid,
        latency_s=round(latency, 1),
        answer=d.get("answer", ""),
        citations=len(d.get("citations") or []),
        confidence=round(float(d.get("confidence") or 0), 3),
        slec_coverage=round(float((slec.get("coverage_ratio") or 0)), 3),
        guardrail_verdict=ver.get("verdict") or "—",
        guardrail_confidence=round(float(ver.get("confidence") or 0), 3),
        evidence_retrieved=ev_retrieved,
        evidence_final=ev_final,
        was_refused=bool(d.get("was_refused")),
        raw=d,
    )


# ── Main eval loop ────────────────────────────────────────────────────────────
async def run_eval():
    plain_results: list[PipelineResult] = []
    agentic_results: list[PipelineResult] = []
    judge_results: list[JudgeResult] = []

    async with httpx.AsyncClient() as client:
        for q in QUESTIONS:
            print(f"\n{'='*60}")
            print(f"[{q['id']}] {q['text'][:80]}...")

            # Run plain RAG
            print(f"  → Plain RAG ...", end="", flush=True)
            t0 = time.time()
            try:
                resp_plain = await ask(client, q["text"], agentic=False)
                t_plain = time.time() - t0
                pr = extract_result(resp_plain, "plain", q["id"], t_plain)
                print(f" {t_plain:.0f}s | citations={pr.citations} | slec={pr.slec_coverage}")
            except Exception as e:
                print(f" ERROR: {e}")
                continue

            # Run agentic RAG
            print(f"  → Agentic RAG ...", end="", flush=True)
            t0 = time.time()
            try:
                resp_agentic = await ask(client, q["text"], agentic=True)
                t_agentic = time.time() - t0
                ar = extract_result(resp_agentic, "agentic", q["id"], t_agentic)
                print(f" {t_agentic:.0f}s | citations={ar.citations} | guardrail={ar.guardrail_verdict}")
            except Exception as e:
                print(f" ERROR: {e}")
                continue

            plain_results.append(pr)
            agentic_results.append(ar)

            # LLM judge
            print(f"  → LLM Judge ...", end="", flush=True)
            try:
                verdict = await llm_judge(
                    client, q["text"], q["expected_hops"],
                    pr.answer, ar.answer,
                )
                jr = JudgeResult(
                    question_id=q["id"],
                    winner=verdict.get("winner", "tie"),
                    plain_scores=verdict.get("A", {}),
                    agentic_scores=verdict.get("B", {}),
                    reason=verdict.get("reason", ""),
                )
                judge_results.append(jr)
                print(f" winner={jr.winner} | {jr.reason[:60]}")
            except Exception as e:
                print(f" Judge ERROR: {e}")

    # ── Save raw ───────────────────────────────────────────────────────────────
    raw_output = {
        "plain": [asdict(r) for r in plain_results],
        "agentic": [asdict(r) for r in agentic_results],
        "judge": [asdict(j) for j in judge_results],
    }
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(raw_output, f, ensure_ascii=False, indent=2)

    # ── Build report ───────────────────────────────────────────────────────────
    report = build_report(plain_results, agentic_results, judge_results)
    with open("eval_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "="*60)
    print(report)
    print("\nSaved: eval_results.json, eval_report.md")


def avg(vals):
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def build_report(plains, agentcs, judges) -> str:
    lines = ["# Evaluation: Plain RAG vs Agentic RAG\n"]
    lines.append(f"Collection: Pháp Luật | Questions: {len(plains)} | Top-K: {TOP_K}\n")

    # Per-question table
    lines.append("## Per-question results\n")
    lines.append("| ID | Type | Plain latency | Agentic latency | Plain SLEC | Agentic guardrail | Winner | Reason |")
    lines.append("|---|---|---|---|---|---|---|---|")
    q_map = {q["id"]: q for q in QUESTIONS}
    for pr in plains:
        ar = next((a for a in agentcs if a.question_id == pr.question_id), None)
        jr = next((j for j in judges if j.question_id == pr.question_id), None)
        if not ar:
            continue
        qtype = q_map.get(pr.question_id, {}).get("type", "")
        winner = jr.winner if jr else "—"
        reason = (jr.reason[:50] + "…") if jr and len(jr.reason) > 50 else (jr.reason if jr else "—")
        lines.append(
            f"| {pr.question_id} | {qtype} "
            f"| {pr.latency_s}s | {ar.latency_s}s "
            f"| {pr.slec_coverage:.0%} | {ar.guardrail_verdict}({ar.guardrail_confidence:.2f}) "
            f"| **{winner}** | {reason} |"
        )

    # Judge scores
    if judges:
        lines.append("\n## LLM Judge scores (0-10)\n")
        lines.append("| ID | Plain complete | Plain grounding | Plain legal | Agentic complete | Agentic grounding | Agentic legal | Winner |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for jr in judges:
            ps = jr.plain_scores
            as_ = jr.agentic_scores
            lines.append(
                f"| {jr.question_id} "
                f"| {ps.get('completeness','—')} | {ps.get('grounding','—')} | {ps.get('legal_precision','—')} "
                f"| {as_.get('completeness','—')} | {as_.get('grounding','—')} | {as_.get('legal_precision','—')} "
                f"| **{jr.winner}** |"
            )

        # Aggregate
        p_comp  = [j.plain_scores.get("completeness", 0) for j in judges]
        p_grd   = [j.plain_scores.get("grounding", 0) for j in judges]
        p_leg   = [j.plain_scores.get("legal_precision", 0) for j in judges]
        a_comp  = [j.agentic_scores.get("completeness", 0) for j in judges]
        a_grd   = [j.agentic_scores.get("grounding", 0) for j in judges]
        a_leg   = [j.agentic_scores.get("legal_precision", 0) for j in judges]
        wins    = {"A": 0, "B": 0, "tie": 0}
        for j in judges:
            wins[j.winner] = wins.get(j.winner, 0) + 1

        lines.append("\n## Aggregate\n")
        lines.append("| Metric | Plain RAG | Agentic RAG | Delta |")
        lines.append("|---|---|---|---|")
        for name, pv, av in [
            ("Completeness (avg)", avg(p_comp), avg(a_comp)),
            ("Grounding (avg)",    avg(p_grd),  avg(a_grd)),
            ("Legal precision (avg)", avg(p_leg), avg(a_leg)),
        ]:
            delta = round(av - pv, 2)
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
            lines.append(f"| {name} | {pv} | {av} | {arrow}{abs(delta)} |")

        p_total = avg(p_comp) + avg(p_grd) + avg(p_leg)
        a_total = avg(a_comp) + avg(a_grd) + avg(a_leg)
        lines.append(f"| **Total score** | **{round(p_total,2)}** | **{round(a_total,2)}** | **{round(a_total-p_total,2)}** |")
        lines.append(f"\n**Win/Tie/Loss (Agentic vs Plain):** {wins.get('B',0)} / {wins.get('tie',0)} / {wins.get('A',0)}")

    # Latency
    if plains and agentcs:
        p_lat = avg([r.latency_s for r in plains])
        a_lat = avg([r.latency_s for r in agentcs])
        lines.append(f"\n**Avg latency:** Plain {p_lat}s | Agentic {a_lat}s | Overhead {round(a_lat-p_lat,1)}s")

    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(run_eval())
