from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_e2e import (  # noqa: E402
    _CITATION_CLEAN_RE,
    _SENTENCE_RE,
    _embed,
    answer_relevance,
    citation_coverage,
    citation_validity,
    context_precision,
    faithfulness_citation,
    grounded_sentence_ratio,
    semantic_faithfulness,
)
from judge import (  # noqa: E402
    _build_judge_prompt,
    _call_llm,
    _extract_json,
)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ask(api_url: str, case: dict, timeout: int) -> dict:
    payload = {
        "owner_id": case["owner_id"],
        "collection_id": case["collection_id"],
        "query": case["query"],
        "stream": False,
    }
    resp = requests.post(f"{api_url}/api/v1/query/ask", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("data") or resp.json()


def _ragas_proxy(api_url: str, query: str, answer: str, citations: list[dict]) -> dict:
    faith = faithfulness_citation(answer)
    coverage = citation_coverage(answer)
    validity = citation_validity(answer, len(citations))
    ctx_precision = context_precision(citations)

    answer_sentences = [
        _CITATION_CLEAN_RE.sub("", s).strip()
        for s in _SENTENCE_RE.findall(answer)
        if len(_CITATION_CLEAN_RE.sub("", s).strip()) >= 20
    ][:8]
    citation_snippets: list[str] = []
    for c in citations[:5]:
        parts: list[str] = []
        primary = c.get("snippet_original") or c.get("snippet") or c.get("content") or ""
        if primary:
            parts.append(primary)
        for blk in (c.get("evidence_blocks") or [])[:4]:
            snippet = blk.get("snippet_original") or blk.get("snippet") or ""
            if snippet and snippet not in parts:
                parts.append(snippet)
        citation_snippets.append(" ".join(parts)[:1200])

    ans_rel = sem_faith = grounded = 0.0
    try:
        texts: list[str] = []
        q_idx = len(texts); texts.append(query)
        a_idx = len(texts); texts.append(answer[:1000])
        s_start = len(texts); texts.extend(answer_sentences)
        c_start = len(texts); texts.extend(citation_snippets)
        embs = _embed(texts, api_url)
        q_emb = embs[q_idx]
        a_emb = embs[a_idx]
        sent_embs = embs[s_start:s_start + len(answer_sentences)]
        cit_embs = embs[c_start:c_start + len(citation_snippets)]
        ans_rel = answer_relevance(q_emb, a_emb)
        sem_faith = semantic_faithfulness(answer=answer, answer_embs=sent_embs, citation_embs=cit_embs)
        grounded = grounded_sentence_ratio(sent_embs, cit_embs, threshold=0.4)
    except Exception as exc:
        print(f"LIVE_WARN embed_error={type(exc).__name__}: {exc}", flush=True)

    return {
        "faithfulness": round(faith, 3),
        "citation_coverage": round(coverage, 3),
        "citation_validity": round(validity, 3),
        "answer_relevance": round(ans_rel, 3),
        "semantic_faithfulness": round(sem_faith, 3),
        "grounded_ratio": round(grounded, 3),
        "context_precision": round(ctx_precision, 3),
        "answer_sentences": len(answer_sentences),
    }


async def _judge(case: dict, answer: str, citations: list[dict], rubric: dict, args: argparse.Namespace) -> dict:
    raw = await _call_llm(
        system=rubric.get("system_prompt")
        or "You are an expert evaluator for a Vietnamese document Q&A system. Output only valid JSON.",
        user=_build_judge_prompt(
            query=case["query"],
            answer=answer,
            citations=citations,
            gold=case,
            rubric=rubric,
        ),
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=512,
    )
    return _extract_json(raw)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="evaluation/datasets/gold/e2e_gold.jsonl")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--rubric", default="evaluation/config/judge_rubric.yaml")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--output", default="evaluation/results/live_eval_results.jsonl")
    args = parser.parse_args()

    _load_dotenv(Path(".env"))
    args.api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    args.api_base = args.api_base or os.getenv("AGENTBOOK_OPENAI_BASE_URL", "https://api.openai.com/v1")
    args.model = args.model or os.getenv("AGENTBOOK_OPENAI_MODEL", "gpt-4o-mini")
    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY missing")

    try:
        import yaml
        rubric = yaml.safe_load(Path(args.rubric).read_text(encoding="utf-8")) or {}
    except Exception:
        rubric = {}

    cases = _load_jsonl(Path(args.gold))[: args.limit]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    print(f"LIVE_START cases={len(cases)} model={args.model} api={args.api_base}", flush=True)
    for idx, case in enumerate(cases, start=1):
        qid = f"q{idx:03d}"
        t0 = time.time()
        error = None
        payload: dict = {}
        ragas: dict = {}
        judge: dict = {}
        try:
            payload = _ask(args.api_url, case, args.timeout)
            answer = payload.get("answer") or ""
            citations = payload.get("citations") or []
            ragas = _ragas_proxy(args.api_url, case["query"], answer, citations)
            judge = await _judge(case, answer, citations, rubric, args)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        trace = payload.get("trace") or {}
        result = {
            "id": qid,
            "case_id": case.get("case_id"),
            "query": case.get("query"),
            "elapsed_s": round(time.time() - t0, 1),
            "route": trace.get("route"),
            "modality": trace.get("modality"),
            "citation_error_count": trace.get("citation_error_count"),
            "has_image_markdown": "![" in (payload.get("answer") or ""),
            "was_refused": payload.get("was_refused"),
            "citations": len(payload.get("citations") or []),
            "ragas": ragas,
            "judge": judge,
            "error": error,
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print("LIVE_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
        await asyncio.sleep(0.2)
    print(f"LIVE_DONE output={out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
