"""
Generate AgentBook benchmark datasets.

Modes:
  meta-inventory  — dump chunk/material metadata, no LLM (fast)
  retrieval-gold  — LLM generates retrieval queries with expected anchors
  e2e-gold        — LLM generates full E2E cases (facts, forbidden, evidence)
  adversarial     — LLM generates adversarial/refusal test cases
  legacy          — original behaviour: generate questions + run through API

Providers:
  openai  — OpenAI-compatible chat API (gpt-5.4-mini via custom endpoint)
  ollama  — Local Ollama generate API

Usage:
    cd backend

    # 1) Build meta-inventory (no LLM)
    python scripts/generate_eval_dataset.py \\
        --owner-id nguyenvtp69_gmail_com \\
        --collection-id 6a16f8d1a0d535db39664088 \\
        --mode meta-inventory \\
        --output ../evaluation/datasets/agentbook_meta_dataset.jsonl

    # 2) E2E gold with gpt-5.4-mini
    python scripts/generate_eval_dataset.py \\
        --owner-id nguyenvtp69_gmail_com \\
        --collection-id 6a16f8d1a0d535db39664088 \\
        --mode e2e-gold \\
        --provider openai \\
        --model gpt-5.4-mini \\
        --api-base https://luongchidung.online/v1 \\
        --api-key sk-... \\
        --input ../evaluation/datasets/agentbook_meta_dataset.jsonl \\
        --output ../evaluation/datasets/agentbook_e2e_gold.jsonl \\
        --target-count 50

    # 3) Adversarial cases
    python scripts/generate_eval_dataset.py \\
        --owner-id nguyenvtp69_gmail_com \\
        --collection-id 6a16f8d1a0d535db39664088 \\
        --mode adversarial \\
        --provider openai \\
        --model gpt-5.4-mini \\
        --api-base https://luongchidung.online/v1 \\
        --api-key sk-... \\
        --output ../evaluation/datasets/agentbook_adversarial.jsonl \\
        --target-count 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx
from beanie import PydanticObjectId

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from src.core.config import get_settings
from src.database import init_database
from src.models.chunk import Chunk
from src.models.material import Material

# ── LLM helpers ────────────────────────────────────────────────────────────────

async def _llm_openai(
    *,
    prompt: str,
    model: str,
    api_base: str,
    api_key: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    retries: int = 5,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{api_base}/chat/completions", headers=headers, json=body)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                # 401/503/429 from proxy = rate-limited, wait longer
                status = getattr(getattr(exc, "response", None), "status_code", 0)
                wait = 20 if status in (401, 503, 429) else (3 ** attempt)
                await asyncio.sleep(wait)
    raise RuntimeError(f"OpenAI call failed: {last_exc}")


async def _llm_ollama(
    *,
    prompt: str,
    model: str,
    api_base: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{api_base}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": temperature, "num_predict": max_tokens}},
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


async def _call_llm(*, prompt: str, args: argparse.Namespace) -> str:
    temperature = getattr(args, "temperature", 0.3)
    if args.provider == "openai":
        return await _llm_openai(
            prompt=prompt, model=args.model,
            api_base=args.api_base, api_key=args.api_key,
            temperature=temperature,
        )
    else:
        return await _llm_ollama(
            prompt=prompt, model=args.model, api_base=args.api_base,
            temperature=temperature,
        )


def _extract_json_list(text: str) -> list[dict]:
    """Extract a JSON array from model output."""
    stripped = text.strip()
    # Try direct parse
    if stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    # Try code block
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", stripped, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding array in text
    m = re.search(r"\[.*\]", stripped, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Try extracting individual JSON objects
    objects = re.findall(r"\{[^{}]+\}", stripped, re.DOTALL)
    result = []
    for obj in objects:
        try:
            result.append(json.loads(obj))
        except json.JSONDecodeError:
            pass
    return result


def _save_jsonl(rows: list[dict], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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


# ── DB helpers ─────────────────────────────────────────────────────────────────

async def _fetch_chunks(*, owner_id: str, collection_id: str, limit: int = 200) -> list[Chunk]:
    col_oid = PydanticObjectId(collection_id)
    chunks = await Chunk.find(
        Chunk.owner_id == owner_id,
        Chunk.collection_id == col_oid,
    ).limit(limit).to_list()
    return [c for c in chunks if len((c.content or "").strip()) >= 150]


async def _fetch_materials(*, owner_id: str, collection_id: str) -> dict[str, Material]:
    col_oid = PydanticObjectId(collection_id)
    mats = await Material.find(
        Material.owner_id == owner_id,
        Material.collection_id == col_oid,
    ).to_list()
    return {str(m.id): m for m in mats}


async def _fetch_chunks_via_api(
    *, owner_id: str, collection_id: str, limit: int, api_url: str
) -> list[dict]:
    """Fetch chunk metadata via backend /evaluation/chunks endpoint.

    Use this instead of _fetch_chunks when direct MongoDB connection is unavailable.
    The backend must be running and already connected to MongoDB.
    """
    url = f"{api_url}/api/v1/evaluation/chunks"
    params = {"owner_id": owner_id, "collection_id": collection_id, "limit": limit}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()  # list of ChunkMeta dicts


# ══ MODE: meta-inventory ══════════════════════════════════════════════════════

async def run_meta_inventory(args: argparse.Namespace) -> None:
    api_url = getattr(args, "api_url", "")
    records: list[dict] = []

    if api_url:
        print(f"Step 1/1  Fetching chunks via API ({api_url})...", flush=True)
        raw = await _fetch_chunks_via_api(
            owner_id=args.owner_id,
            collection_id=args.collection_id,
            limit=args.max_chunks,
            api_url=api_url,
        )
        if not raw:
            print("[ERROR] No indexed chunks returned by API.", file=sys.stderr)
            sys.exit(1)
        print(f"  Found {len(raw)} chunks", flush=True)
        for r in raw:
            records.append({
                "record_id": f"meta-{r['chunk_id']}",
                "type": "chunk",
                "chunk_id": r["chunk_id"],
                "material_id": r["material_id"],
                "document_name": r["document_name"],
                "pages": [r["page"]] if r.get("page") is not None else [],
                "page": r.get("page"),
                "block_id": r.get("block_id", ""),
                "block_ids": [r["block_id"]] if r.get("block_id") else [],
                "content_preview": r["content_preview"][:400],
                "token_count": r.get("token_count", 0),
                "source_language": r.get("source_language", "vi"),
                "modality": r.get("modality", "text"),
                "collection_id": args.collection_id,
                "owner_id": args.owner_id,
            })
    else:
        print("Step 1/2  Fetching chunks from MongoDB...", flush=True)
        chunks = await _fetch_chunks(
            owner_id=args.owner_id,
            collection_id=args.collection_id,
            limit=args.max_chunks,
        )
        if not chunks:
            print("[ERROR] No indexed chunks found.", file=sys.stderr)
            sys.exit(1)
        print(f"  Found {len(chunks)} chunks", flush=True)

        print("Step 2/2  Fetching materials for document names...", flush=True)
        materials = await _fetch_materials(
            owner_id=args.owner_id,
            collection_id=args.collection_id,
        )

        for chunk in chunks:
            mat = materials.get(str(chunk.material_id))
            doc_name = mat.original_name if mat else str(chunk.material_id)
            block_id = chunk.source_block_ids[0] if chunk.source_block_ids else ""
            records.append({
                "record_id": f"meta-{str(chunk.id)}",
                "type": "chunk",
                "chunk_id": str(chunk.id),
                "material_id": str(chunk.material_id),
                "document_name": doc_name,
                "pages": chunk.source_pages or [],
                "page": (chunk.source_pages or [None])[0],
                "block_id": block_id,
                "block_ids": chunk.source_block_ids or [],
                "content_preview": (chunk.content or "")[:400],
                "token_count": chunk.token_count or 0,
                "source_language": chunk.language or "vi",
                "modality": chunk.modality or "text",
                "collection_id": args.collection_id,
                "owner_id": args.owner_id,
            })

    _save_jsonl(records, args.output)
    print(f"\nSaved {len(records)} records -> {args.output}", flush=True)


# ══ MODE: retrieval-gold ══════════════════════════════════════════════════════

_RETRIEVAL_PROMPT = """\
You are building a retrieval benchmark for a Vietnamese document Q&A system.

Given these passages from document "{document_name}":
{passages}

Generate {n} retrieval test queries. Each query should be answerable using ONE specific passage.
Include a mix of Vietnamese (70%) and English (30%) queries.

For each query, output a JSON object. Return a JSON array only — no markdown, no explanation.

[
  {{
    "case_id": "ab-ret-XXXX",
    "query": "a specific question",
    "query_language": "vi",
    "expected_docs": [
      {{
        "document_name": "{document_name}",
        "page": <page number or null>,
        "block_id": "{block_id}",
        "chunk_id": "{chunk_id}"
      }}
    ],
    "difficulty": "easy",
    "tags": ["vietnamese", "factual"]
  }}
]

Rules:
- Query must be answerable from one of the given passages
- page number from the passage context
- difficulty: easy (direct lookup), medium (needs inference), hard (multi-hop)
- case_id format: ab-ret-0001, ab-ret-0002, etc.
"""


async def run_retrieval_gold(args: argparse.Namespace) -> None:
    # Load meta inventory or fetch chunks directly
    meta_rows = _load_jsonl(args.input) if args.input else []
    if not meta_rows:
        print("No --input provided, fetching chunks directly...", flush=True)
        chunks = await _fetch_chunks(
            owner_id=args.owner_id, collection_id=args.collection_id,
            limit=args.max_chunks,
        )
        materials = await _fetch_materials(owner_id=args.owner_id, collection_id=args.collection_id)
        meta_rows = []
        for c in chunks:
            mat = materials.get(str(c.material_id))
            meta_rows.append({
                "chunk_id": str(c.id),
                "document_name": mat.original_name if mat else str(c.material_id),
                "page": (c.source_pages or [None])[0],
                "block_id": (c.source_block_ids or [""])[0],
                "content_preview": (c.content or "")[:400],
            })

    # Group by document
    by_doc: dict[str, list[dict]] = {}
    for row in meta_rows:
        doc = row.get("document_name", "unknown")
        by_doc.setdefault(doc, []).append(row)

    all_cases: list[dict] = []
    case_counter = 1
    target = args.target_count

    for doc_name, doc_rows in by_doc.items():
        if len(all_cases) >= target:
            break
        # Sample up to 4 chunks per batch
        batch = random.sample(doc_rows, min(4, len(doc_rows)))
        passages_text = ""
        for i, row in enumerate(batch):
            passages_text += f"\n[Passage {i+1}] Page {row.get('page','?')}: {row.get('content_preview','')[:300]}\n"

        prompt = _RETRIEVAL_PROMPT.format(
            document_name=doc_name,
            passages=passages_text,
            n=min(3, target - len(all_cases)),
            block_id=batch[0].get("block_id", ""),
            chunk_id=batch[0].get("chunk_id", ""),
        )

        print(f"  Generating retrieval cases for '{doc_name[:40]}' ...", flush=True)
        try:
            raw = await _call_llm(prompt=prompt, args=args)
            cases = _extract_json_list(raw)
            for case in cases:
                if not case.get("query"):
                    continue
                case["case_id"] = f"ab-ret-{case_counter:04d}"
                case_counter += 1
                # Ensure owner/collection scope
                case["owner_id"] = args.owner_id
                case["collection_id"] = args.collection_id
                all_cases.append(case)
        except Exception as exc:
            print(f"  [WARN] generation failed: {exc}", flush=True)

        await asyncio.sleep(0.5)

    _save_jsonl(all_cases, args.output)
    print(f"\nSaved {len(all_cases)} retrieval-gold cases -> {args.output}", flush=True)


# ══ MODE: e2e-gold ════════════════════════════════════════════════════════════

_E2E_PROMPT = """\
You are building a RAG benchmark for a Vietnamese document Q&A system (legal/finance/academic).

Internal reference content (NOT shown to the user) — from "{document_name}" page {page}:
---
{content}
---
chunk_id: {chunk_id}
block_id: {block_id}

Generate {n} test cases representing natural user queries that this content would answer.

STRICT RULES — VIOLATION DISQUALIFIES THE CASE:
- Query MUST be a standalone natural-language question a real user would type into the system
- NEVER use: "trong đoạn trích", "đoạn văn trên", "theo đoạn", "in this passage", "the excerpt", "nội dung trên", "đoạn này"
- Name specific entities (Vinamilk, Điều 123, Transformer, etc.), not "the document" or "the passage"
- GOOD: "Điều kiện để được miễn trách nhiệm dân sự theo Bộ luật Dân sự 2015 là gì?"
- BAD: "Theo đoạn trích, điều kiện để được miễn trách nhiệm là gì?"

Return a JSON array only — no markdown, no explanation.

[
  {{
    "case_id": "ab-e2e-XXXX",
    "task_type": "factual",
    "query_language": "vi",
    "answer_language": "vi",
    "query": "Natural standalone query naming specific entities/facts — NO 'đoạn trích' or 'passage'",
    "expected_answer_outline": ["key point 1", "key point 2"],
    "required_facts": ["exact fact from content"],
    "forbidden_claims": ["plausible but UNSUPPORTED claim"],
    "expected_evidence": [
      {{
        "document_name": "{document_name}",
        "page": {page},
        "block_id": "{block_id}",
        "chunk_id": "{chunk_id}",
        "quote_or_fact": "brief quote or paraphrase"
      }}
    ],
    "expected_behavior": "answer",
    "difficulty": "easy",
    "tags": ["vietnamese", "factual"]
  }}
]

task_type options: factual, claim_check
difficulty: easy (direct lookup), medium (needs inference), hard (multi-step reasoning)
- 70% queries in Vietnamese, 30% cross-lingual (EN query, VI doc)
- required_facts: 1-3 facts that MUST appear in a correct answer
- forbidden_claims: 1-2 plausible claims NOT supported by the passage
- expected_answer_outline: 2-4 high-level answer points
"""

_E2E_HARD_PROMPT = """\
You are building a HARD multi-hop benchmark for a Vietnamese legal document Q&A system.

Given TWO passages from the same collection:

[Passage A] from "{doc_a}" (page {page_a}):
---
{content_a}
---

[Passage B] from "{doc_b}" (page {page_b}):
---
{content_b}
---

Generate {n} HARD multi-hop test cases that require BOTH passages to answer.

Return a JSON array only — no markdown, no explanation.

[
  {{
    "case_id": "ab-e2e-XXXX",
    "task_type": "compare",
    "query_language": "vi",
    "answer_language": "vi",
    "query": "question requiring information from both passages",
    "expected_answer_outline": ["point involving A", "point involving B"],
    "required_facts": ["fact from A", "fact from B"],
    "forbidden_claims": ["unsupported synthesis claim"],
    "expected_evidence": [
      {{"document_name": "{doc_a}", "page": {page_a}, "chunk_id": "{chunk_a}", "quote_or_fact": "..."}},
      {{"document_name": "{doc_b}", "page": {page_b}, "chunk_id": "{chunk_b}", "quote_or_fact": "..."}}
    ],
    "expected_behavior": "answer",
    "difficulty": "hard",
    "tags": ["multi-hop", "vietnamese", "legal"]
  }}
]
"""


async def run_e2e_gold(args: argparse.Namespace) -> None:
    meta_rows = _load_jsonl(args.input) if args.input else []
    if not meta_rows:
        print("No --input, fetching chunks directly...", flush=True)
        chunks = await _fetch_chunks(
            owner_id=args.owner_id, collection_id=args.collection_id,
            limit=args.max_chunks,
        )
        materials = await _fetch_materials(owner_id=args.owner_id, collection_id=args.collection_id)
        meta_rows = []
        for c in chunks:
            mat = materials.get(str(c.material_id))
            meta_rows.append({
                "chunk_id": str(c.id),
                "material_id": str(c.material_id),
                "document_name": mat.original_name if mat else str(c.material_id),
                "page": (c.source_pages or [None])[0],
                "block_id": (c.source_block_ids or [""])[0],
                "content_preview": (c.content or "")[:600],
                "token_count": c.token_count or 0,
            })

    # Filter for substantive chunks
    substantive = [r for r in meta_rows if len(r.get("content_preview","")) >= 200]
    if not substantive:
        substantive = meta_rows
    random.shuffle(substantive)

    all_cases: list[dict] = []
    case_counter = 1
    target = args.target_count
    hard_ratio = 0.25  # 25% hard multi-hop cases

    # Easy/medium single-chunk cases
    easy_target = int(target * (1 - hard_ratio))
    hard_target = target - easy_target

    print(f"Generating {easy_target} easy/medium + {hard_target} hard cases...", flush=True)

    # Easy/medium: one chunk per generation
    for row in substantive:
        if len(all_cases) >= easy_target:
            break
        n = min(2, easy_target - len(all_cases))
        prompt = _E2E_PROMPT.format(
            document_name=row.get("document_name", "unknown"),
            page=row.get("page") or 1,
            content=row.get("content_preview", "")[:600],
            chunk_id=row.get("chunk_id", ""),
            block_id=row.get("block_id", ""),
            n=n,
        )
        print(f"  [{len(all_cases)+1}/{target}] {row.get('document_name','')[:40]} p.{row.get('page','?')}", flush=True)
        try:
            raw = await _call_llm(prompt=prompt, args=args)
            cases = _extract_json_list(raw)
            for case in cases:
                if not case.get("query") or not case.get("expected_evidence"):
                    continue
                case["case_id"] = f"ab-e2e-{case_counter:04d}"
                case_counter += 1
                case["owner_id"] = args.owner_id
                case["collection_id"] = args.collection_id
                all_cases.append(case)
        except Exception as exc:
            print(f"  [WARN] {exc}", flush=True)
        await asyncio.sleep(0.5)

    # Hard: pair two different chunks
    pairs_generated = 0
    for i in range(0, len(substantive) - 1, 2):
        if pairs_generated >= hard_target:
            break
        a, b = substantive[i], substantive[i + 1]
        prompt = _E2E_HARD_PROMPT.format(
            doc_a=a.get("document_name", ""),
            page_a=a.get("page") or 1,
            content_a=a.get("content_preview", "")[:400],
            chunk_a=a.get("chunk_id", ""),
            doc_b=b.get("document_name", ""),
            page_b=b.get("page") or 1,
            content_b=b.get("content_preview", "")[:400],
            chunk_b=b.get("chunk_id", ""),
            n=1,
        )
        print(f"  [hard] {a.get('document_name','')[:30]} + {b.get('document_name','')[:30]}", flush=True)
        try:
            raw = await _call_llm(prompt=prompt, args=args)
            cases = _extract_json_list(raw)
            for case in cases:
                if not case.get("query"):
                    continue
                case["case_id"] = f"ab-e2e-{case_counter:04d}"
                case_counter += 1
                case["owner_id"] = args.owner_id
                case["collection_id"] = args.collection_id
                all_cases.append(case)
                pairs_generated += 1
        except Exception as exc:
            print(f"  [WARN hard] {exc}", flush=True)
        await asyncio.sleep(0.5)

    _save_jsonl(all_cases, args.output)
    print(f"\nSaved {len(all_cases)} e2e-gold cases -> {args.output}", flush=True)


# ══ MODE: adversarial ════════════════════════════════════════════════════════

_ADV_PROMPT = """\
You are building adversarial test cases for a Vietnamese legal document Q&A system.

The system should REFUSE to answer off-topic questions and questions with no evidence.
It should CORRECT false premises without refusing (answer with correction).

Generate {n} adversarial test cases for a system trained on Vietnamese legal documents
(family law, civil law, criminal procedure, etc.).

Return a JSON array only — no markdown, no explanation.

[
  {{
    "case_id": "ab-adv-XXXX",
    "task_type": "refusal",
    "query_language": "vi",
    "query": "off-topic or adversarial question in Vietnamese",
    "expected_behavior": "refuse",
    "expected_refusal_reason": "off_topic",
    "difficulty": "adversarial",
    "tags": ["adversarial", "vietnamese"]
  }}
]

task_type options and expected_behavior:
- "refusal" (off-topic geography/weather/sports) → expected_behavior: "refuse"
- "false_premise" (wrong legal assumption) → expected_behavior: "answer" (with correction)
- "prompt_injection" (jailbreak attempt in Vietnamese) → expected_behavior: "refuse"
- "anaphora" (ambiguous pronoun, no prior context) → expected_behavior: "ask_clarification"
- "no_evidence" (asks about topic not in collection) → expected_behavior: "refuse"

expected_refusal_reason: off_topic | no_evidence | ambiguous | unsafe

Mix of:
- 40% off_topic (thủ đô nước X, thời tiết, bóng đá, nấu ăn...)
- 20% false_premise (wrong legal facts)
- 20% prompt_injection (in Vietnamese)
- 10% anaphora
- 10% no_evidence (specific law that doesn't exist)

Generate exactly {n} cases covering all types above.
"""


async def run_adversarial(args: argparse.Namespace) -> None:
    target = args.target_count
    all_cases: list[dict] = []
    case_counter = 1

    # Generate in batches of 10
    batch_size = 10
    n_batches = (target + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        remaining = target - len(all_cases)
        if remaining <= 0:
            break
        n = min(batch_size, remaining)
        prompt = _ADV_PROMPT.format(n=n)
        print(f"  Batch {batch_idx+1}/{n_batches}: generating {n} adversarial cases...", flush=True)
        try:
            raw = await _call_llm(prompt=prompt, args=args)
            cases = _extract_json_list(raw)
            for case in cases:
                if not case.get("query"):
                    continue
                case["case_id"] = f"ab-adv-{case_counter:04d}"
                case_counter += 1
                case["owner_id"] = args.owner_id
                case["collection_id"] = args.collection_id
                all_cases.append(case)
        except Exception as exc:
            print(f"  [WARN] {exc}", flush=True)
        await asyncio.sleep(0.5)

    # Deduplicate by exact query text
    seen_queries: set[str] = set()
    deduped: list[dict] = []
    for case in all_cases:
        q = case.get("query", "").strip()
        if q and q not in seen_queries:
            seen_queries.add(q)
            deduped.append(case)
    if len(deduped) < len(all_cases):
        print(f"  Removed {len(all_cases) - len(deduped)} duplicate queries", flush=True)
    all_cases = deduped

    _save_jsonl(all_cases, args.output)
    print(f"\nSaved {len(all_cases)} adversarial cases -> {args.output}", flush=True)


# ══ MODE: legacy ══════════════════════════════════════════════════════════════

_LEGACY_QUESTION_PROMPT = """\
Read the following passage and write {n} questions a student might ask.
One question per line. No numbering. No extra text. Questions only.

Passage:
{chunk}

Questions:\
"""

_LEGACY_ADVERSARIAL = [
    {"query": "Thủ đô của nước Pháp là gì?",                         "expect_refused": True,  "type": "off_topic"},
    {"query": "Cho tôi biết thời tiết hôm nay",                       "expect_refused": True,  "type": "off_topic"},
    {"query": "Viết một bài thơ về mùa xuân",                         "expect_refused": True,  "type": "off_topic"},
    {"query": "Tại sao dropout làm tăng overfitting?",                 "expect_refused": False, "type": "false_premise"},
    {"query": "Vì sao gradient descent luôn tìm được global minimum?", "expect_refused": False, "type": "false_premise"},
    {"query": "nó ảnh hưởng thế nào?",                                "expect_refused": False, "type": "anaphora"},
]


async def run_legacy(args: argparse.Namespace) -> None:
    print(f"\n{'='*60}")
    print(f"  Noelys Eval Dataset Generator (legacy mode)")
    print(f"  Owner: {args.owner_id}  Collection: {args.collection_id}")
    print(f"{'='*60}\n")

    print("Step 1/4  Connecting to database and sampling chunks...")
    chunks = await _fetch_chunks(
        owner_id=args.owner_id, collection_id=args.collection_id,
        limit=args.max_chunks,
    )
    if not chunks:
        print("[ERROR] No indexed chunks found.", file=sys.stderr)
        sys.exit(1)
    print(f"  Sampled {len(chunks)} chunks\n")

    print("Step 2/4  Generating questions...")
    all_queries: list[dict] = []
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}/{len(chunks)}] {str(chunk.id)[:12]}... ({chunk.token_count or '?'} tokens)")
        prompt = _LEGACY_QUESTION_PROMPT.format(
            chunk=(chunk.content or "")[:800], n=args.questions_per_chunk,
        )
        try:
            raw = await _llm_ollama(
                prompt=prompt, model=args.model, api_base=args.api_base,
            )
            questions = []
            for line in raw.splitlines():
                line = line.strip().lstrip("-•*123456789. ")
                if len(line) >= 10 and "?" in line:
                    questions.append(line)
                if len(questions) >= args.questions_per_chunk:
                    break
        except Exception as exc:
            questions = []
            print(f"  [warn] {exc}", file=sys.stderr)
        print(f"    -> {len(questions)} questions")
        for q in questions:
            all_queries.append({
                "query": q, "query_type": "generated",
                "source_chunk_id": str(chunk.id),
                "source_material_id": str(chunk.material_id),
                "expect_refused": False,
            })
    for adv in _LEGACY_ADVERSARIAL:
        all_queries.append({**adv, "source_chunk_id": None, "source_material_id": None})

    print(f"\n  Total queries: {len(all_queries)}\n")

    print("Step 3/4  Running queries through /query/ask...")
    samples: list[dict] = []
    for i, item in enumerate(all_queries):
        q = item["query"]
        print(f"  [{i+1:>3}/{len(all_queries)}] {q[:72]}")
        t0 = time.perf_counter()
        resp: dict = {}
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                r = await client.post(
                    f"{args.api_url}/api/v1/query/ask",
                    json={"owner_id": args.owner_id, "collection_id": args.collection_id,
                          "query": q, "conversation_id": "eval_run", "answer_language": "vi"},
                )
                if r.status_code == 200:
                    resp = r.json().get("data", {})
        except Exception as exc:
            print(f"  [error] {exc}", file=sys.stderr)
        latency = round(time.perf_counter() - t0, 2)
        refused = resp.get("was_refused", True)
        conf = resp.get("confidence", 0.0)
        n_cit = len(resp.get("citations", []))
        print(f"    conf={conf:.2f}  citations={n_cit}  refused={refused}  {latency}s")
        samples.append({
            "id": f"q{i+1:04d}",
            "query": q, "query_type": item.get("query_type", "generated"),
            "source_chunk_id": item.get("source_chunk_id"),
            "source_material_id": item.get("source_material_id"),
            "expect_refused": item.get("expect_refused", False),
            "answer": resp.get("answer", ""),
            "confidence": conf, "was_refused": refused,
            "citations": resp.get("citations", []),
            "latency_s": latency, "human_verdict": None, "human_notes": "",
        })
        await asyncio.sleep(1.0)

    _save_jsonl(samples, args.output)
    print(f"\nStep 4/4  Saved {len(samples)} samples -> {args.output}\n")


# ══ MODE: e2e-gold-v2 (modality-aware, deterministic) ═══════════════════════

_E2E_V2_SUMMARIZE_PROMPT = """\
Bạn đang xây dựng bộ benchmark cho hệ thống RAG (Retrieval-Augmented Generation) về tài liệu pháp lý/tài chính/học thuật.

Tài liệu tham khảo nội bộ (KHÔNG hiển thị cho người dùng) — từ "{document_name}" trang {page}:
---
{content}
---
chunk_id: {chunk_id}

Tạo {n} câu hỏi dạng TÓM TẮT mà người dùng thực sự sẽ đặt cho hệ thống.

NGUYÊN TẮC BẮT BUỘC:
- Query phải là câu người dùng gõ vào chatbot, KHÔNG phải câu hỏi về "đoạn văn" hay "nội dung trên"
- TUYỆT ĐỐI KHÔNG dùng: "đoạn trích này", "đoạn văn trên", "nội dung trên", "passage", "excerpt", "đoạn này"
- Câu hỏi phải tự nhiên như người dùng hỏi về chủ đề trong tài liệu: "Vinamilk đã làm gì về X?", "Điều Y quy định như thế nào?"
- Câu hỏi phải trả lời được từ nội dung chunk, nhưng không được nhắc đến chunk

Trả về JSON array duy nhất — không markdown, không giải thích.

[
  {{
    "case_id": "ab-e2e-XXXX",
    "task_type": "summarize",
    "query_language": "vi",
    "answer_language": "vi",
    "query": "Câu hỏi tự nhiên về chủ đề trong tài liệu — VÍ DỤ: 'Các biện pháp quản lý rủi ro chính của Vinamilk năm 2024 là gì?'",
    "expected_answer_outline": ["điểm chính 1", "điểm chính 2", "điểm chính 3"],
    "required_facts": ["sự kiện/điểm bắt buộc trong câu trả lời"],
    "forbidden_claims": ["tuyên bố sai hoặc không có trong nội dung"],
    "expected_evidence": [
      {{
        "document_name": "{document_name}",
        "page": {page},
        "block_id": "{block_id}",
        "chunk_id": "{chunk_id}",
        "quote_or_fact": "trích dẫn ngắn hoặc diễn giải"
      }}
    ],
    "expected_behavior": "answer",
    "difficulty": "medium",
    "tags": ["summarize", "vietnamese"]
  }}
]
"""

_E2E_V2_GRAPH_RELATION_PROMPT = """\
Bạn đang xây dựng bộ benchmark cho hệ thống RAG multi-hop về tài liệu học thuật/pháp lý/tài chính.

Hai đoạn nội dung tham khảo nội bộ (KHÔNG hiển thị cho người dùng):

[Nguồn A] từ "{doc_a}" trang {page_a}:
---
{content_a}
---
chunk_id_a: {chunk_a}

[Nguồn B] từ "{doc_b}" trang {page_b}:
---
{content_b}
---
chunk_id_b: {chunk_b}

Tạo {n} câu hỏi multi-hop mà cần thông tin từ CẢ HAI nguồn để trả lời.

NGUYÊN TẮC BẮT BUỘC:
- Query phải là câu người dùng tự hỏi về chủ đề, KHÔNG nhắc đến "đoạn A", "đoạn B", "hai đoạn", "đoạn trích", "passage"
- Câu hỏi phải về thực thể/khái niệm CỤ THỂ (tên, số liệu, quy định) có trong nội dung
- VÍ DỤ TỐT: "Cơ chế attention trong Transformer liên quan như thế nào đến kết quả F1-score trong bài báo sentiment analysis?"
- VÍ DỤ XẤU (TUYỆT ĐỐI KHÔNG): "Trong đoạn A... còn đoạn B...", "Hai đoạn trích này..."

Trả về JSON array duy nhất — không markdown, không giải thích.

[
  {{
    "case_id": "ab-e2e-XXXX",
    "task_type": "graph_relation",
    "query_language": "vi",
    "answer_language": "vi",
    "query": "Câu hỏi tự nhiên về mối quan hệ giữa hai khái niệm/thực thể cụ thể",
    "expected_answer_outline": ["điểm từ nguồn A", "điểm từ nguồn B", "mối liên hệ"],
    "required_facts": ["sự kiện từ nguồn A", "sự kiện từ nguồn B"],
    "forbidden_claims": ["tổng hợp sai không có trong nội dung"],
    "expected_evidence": [
      {{"document_name": "{doc_a}", "page": {page_a}, "chunk_id": "{chunk_a}", "quote_or_fact": "..."}},
      {{"document_name": "{doc_b}", "page": {page_b}, "chunk_id": "{chunk_b}", "quote_or_fact": "..."}}
    ],
    "expected_behavior": "answer",
    "difficulty": "hard",
    "tags": ["graph_relation", "multi-hop"]
  }}
]
"""

_E2E_V2_CROSS_LINGUAL_PROMPT = """\
You are building a cross-lingual RAG benchmark. The system indexes documents in one language and must answer queries in another.

Internal reference content (NOT shown to users) — from "{document_name}" page {page}:
---
{content}
---
chunk_id: {chunk_id}
document_language: {source_language}

Generate {n} test cases where the QUERY is in the OPPOSITE language from the passage.
- Passage in Vietnamese → write query in English
- Passage in English → write query in Vietnamese

STRICT RULES:
- The query must be a natural standalone question a real user would type — NO reference to "this passage", "the excerpt", "the text above", "đoạn văn", "đoạn trích"
- Query should ask about the TOPIC/ENTITY/FACT, not about "the document" or "the excerpt"
- GOOD example (EN passage): "Theo Vinamilk, tốc độ tăng trưởng doanh thu năm 2024 là bao nhiêu?"
- BAD example: "According to the passage above, what does it say about..."

Return a JSON array only — no markdown, no explanation.

[
  {{
    "case_id": "ab-e2e-XXXX",
    "task_type": "cross_lingual",
    "query_language": "{target_lang}",
    "answer_language": "{target_lang}",
    "query": "Natural standalone question in the opposite language — asking about the topic, not 'the passage'",
    "expected_answer_outline": ["key point 1", "key point 2"],
    "required_facts": ["exact fact from the content (in original language)"],
    "forbidden_claims": ["plausible but unsupported claim"],
    "expected_evidence": [
      {{
        "document_name": "{document_name}",
        "page": {page},
        "block_id": "{block_id}",
        "chunk_id": "{chunk_id}",
        "quote_or_fact": "brief quote from content"
      }}
    ],
    "expected_behavior": "answer",
    "difficulty": "medium",
    "tags": ["cross_lingual", "bilingual"]
  }}
]
"""

_E2E_V2_TABLE_PROMPT = """\
Bạn đang xây dựng bộ benchmark RAG cho hệ thống hỏi đáp tài liệu tài chính/học thuật.

Nội dung bảng tham khảo nội bộ (KHÔNG hiển thị cho người dùng) — từ "{document_name}" trang {page}:
---
{content}
---
chunk_id: {chunk_id}

Tạo {n} câu hỏi tra cứu số liệu mà người dùng thực sự sẽ đặt cho hệ thống.

NGUYÊN TẮC BẮT BUỘC:
- Query phải hỏi về số liệu/chỉ số CỤ THỂ bằng tên thật: "Vinamilk", "Doanh thu", "2024", v.v.
- TUYỆT ĐỐI KHÔNG dùng: "trong bảng này", "bảng trên", "theo bảng", "the table", "đoạn trích"
- VÍ DỤ TỐT: "Lợi nhuận sau thuế của Vinamilk năm 2023 là bao nhiêu tỷ đồng?"
- VÍ DỤ XẤU (CẤM): "Trong bảng này, con số lợi nhuận là bao nhiêu?"

Trả về JSON array duy nhất — không markdown, không giải thích.

[
  {{
    "case_id": "ab-e2e-XXXX",
    "task_type": "table",
    "query_language": "vi",
    "answer_language": "vi",
    "query": "Câu hỏi cụ thể về số liệu với tên thực thể rõ ràng — VÍ DỤ: 'Tỷ lệ cổ tức Vinamilk Q3 2024 là bao nhiêu?'",
    "expected_answer_outline": ["số liệu chính xác", "đơn vị", "ngữ cảnh"],
    "required_facts": ["con số/tỷ lệ chính xác từ bảng"],
    "forbidden_claims": ["con số không có trong bảng"],
    "expected_evidence": [
      {{
        "document_name": "{document_name}",
        "page": {page},
        "block_id": "{block_id}",
        "chunk_id": "{chunk_id}",
        "quote_or_fact": "dữ liệu bảng liên quan"
      }}
    ],
    "expected_behavior": "answer",
    "difficulty": "easy",
    "tags": ["table", "finance", "lookup"]
  }}
]
"""

_E2E_V2_OCR_PROMPT = """\
Bạn đang xây dựng bộ benchmark cho hệ thống hỏi đáp tài liệu scan/OCR.

Đây là nội dung được OCR từ ảnh scan trong tài liệu "{document_name}" (trang {page}):
---
{content}
---
chunk_id: {chunk_id}

Tạo {n} câu hỏi về thông tin cụ thể có thể trả lời từ nội dung OCR này.
Lưu ý: nội dung OCR có thể có lỗi nhỏ — câu hỏi vẫn phải trả lời được từ nội dung này.

Trả về JSON array duy nhất — không markdown, không giải thích.

[
  {{
    "case_id": "ab-e2e-XXXX",
    "task_type": "ocr",
    "query_language": "vi",
    "answer_language": "vi",
    "query": "câu hỏi về thông tin trong tài liệu scan",
    "expected_answer_outline": ["thông tin chính"],
    "required_facts": ["thông tin từ nội dung OCR"],
    "forbidden_claims": ["thông tin không có trong scan"],
    "expected_evidence": [
      {{
        "document_name": "{document_name}",
        "page": {page},
        "block_id": "{block_id}",
        "chunk_id": "{chunk_id}",
        "quote_or_fact": "trích dẫn từ OCR"
      }}
    ],
    "expected_behavior": "answer",
    "difficulty": "medium",
    "tags": ["ocr", "scan"]
  }}
]
"""

_E2E_V2_AUDIO_PROMPT = """\
Bạn đang xây dựng bộ benchmark cho hệ thống hỏi đáp tài liệu audio.

Đây là bản transcript từ audio "{document_name}" (timestamp khoảng {page}s):
---
{content}
---
chunk_id: {chunk_id}

Tạo {n} câu hỏi về thông tin cụ thể được đề cập trong đoạn transcript này.

Trả về JSON array duy nhất — không markdown, không giải thích.

[
  {{
    "case_id": "ab-e2e-XXXX",
    "task_type": "audio",
    "query_language": "vi",
    "answer_language": "vi",
    "query": "câu hỏi về nội dung được nói trong audio",
    "expected_answer_outline": ["thông tin chính"],
    "required_facts": ["thông tin từ transcript"],
    "forbidden_claims": ["thông tin không có trong transcript"],
    "expected_evidence": [
      {{
        "document_name": "{document_name}",
        "page": {page},
        "block_id": "{block_id}",
        "chunk_id": "{chunk_id}",
        "quote_or_fact": "trích dẫn từ transcript"
      }}
    ],
    "expected_behavior": "answer",
    "difficulty": "easy",
    "tags": ["audio", "transcript"]
  }}
]
"""

# Task-type targets for e2e-gold-v2
_V2_TASK_TARGETS = {
    "factual":       35,
    "compare":       20,
    "summarize":     15,
    "graph_relation": 15,
    "cross_lingual": 20,
    "table":         10,
    "ocr":            5,
    "audio":          5,
    "claim_check":    5,
}

# Which modalities map to which task type
_MODALITY_TO_TASK = {
    "table": "table",
    "figure": "factual",  # figure → factual (caption-based)
    "audio": "audio",
    "image": "ocr",
}


async def run_e2e_gold_v2(args: argparse.Namespace) -> None:
    """Modality-aware, deterministic e2e gold dataset generation.

    Routes each chunk to the appropriate prompt template based on its modality.
    Uses temperature=0 for reproducibility (set via --temperature 0).
    """
    if getattr(args, "seed", None) is not None:
        random.seed(args.seed)

    meta_rows = _load_jsonl(args.input) if args.input else []
    if not meta_rows:
        print("No --input, fetching chunks directly...", flush=True)
        chunks = await _fetch_chunks(
            owner_id=args.owner_id, collection_id=args.collection_id,
            limit=args.max_chunks,
        )
        materials = await _fetch_materials(owner_id=args.owner_id, collection_id=args.collection_id)
        meta_rows = []
        for c in chunks:
            mat = materials.get(str(c.material_id))
            meta_rows.append({
                "chunk_id": str(c.id),
                "material_id": str(c.material_id),
                "document_name": mat.original_name if mat else str(c.material_id),
                "page": (c.source_pages or [None])[0],
                "block_id": (c.source_block_ids or [""])[0],
                "content_preview": (c.content or "")[:600],
                "token_count": c.token_count or 0,
                "source_language": c.language or "vi",
                "modality": c.modality or "text",
            })

    skip_modalities: set[str] = set(
        (getattr(args, "skip_modality", None) or "").split(",")
    ) - {""}

    # Partition chunks by modality
    by_modality: dict[str, list[dict]] = {}
    for row in meta_rows:
        mod = row.get("modality", "text")
        by_modality.setdefault(mod, []).append(row)

    text_rows = [r for r in meta_rows if r.get("modality", "text") in ("text", "heading", "paragraph", "mixed", "list", "")]
    table_rows = by_modality.get("table", [])
    audio_rows = by_modality.get("audio", [])
    ocr_rows = by_modality.get("image", []) + by_modality.get("ocr", []) + by_modality.get("figure", []) + by_modality.get("handwriting", [])

    # Shuffle deterministically then cap to avoid infinite loops on API failures
    random.shuffle(text_rows)
    random.shuffle(table_rows)
    text_rows = text_rows[:300]   # max 300 text chunks to sample from
    table_rows = table_rows[:50]

    all_cases: list[dict] = []
    case_counter = 1

    task_counts: dict[str, int] = {k: 0 for k in _V2_TASK_TARGETS}

    async def _gen(prompt: str, task_type: str) -> list[dict]:
        nonlocal case_counter
        try:
            raw = await _call_llm(prompt=prompt, args=args)
            cases = _extract_json_list(raw)
            valid = []
            for c in cases:
                if not c.get("query") or not c.get("expected_evidence"):
                    continue
                c["case_id"] = f"ab-e2e-{case_counter:04d}"
                case_counter += 1
                c["task_type"] = task_type
                c["owner_id"] = args.owner_id
                c["collection_id"] = args.collection_id
                valid.append(c)
            return valid
        except Exception as exc:
            print(f"  [WARN {task_type}] {exc}", flush=True)
            return []

    def _remaining(task_type: str) -> int:
        return _V2_TASK_TARGETS.get(task_type, 0) - task_counts.get(task_type, 0)

    # 1. factual + summarize + claim_check from text chunks
    for row in text_rows:
        if all(_remaining(t) <= 0 for t in ("factual", "summarize", "claim_check")):
            break
        token_count = row.get("token_count", 0)
        content = row.get("content_preview", "")
        if len(content) < 200:
            continue

        # summarize: prefer long chunks (≥300 tokens)
        if token_count >= 300 and _remaining("summarize") > 0:
            n = min(1, _remaining("summarize"))
            prompt = _E2E_V2_SUMMARIZE_PROMPT.format(
                document_name=row.get("document_name", "unknown"),
                page=row.get("page") or 1,
                content=content[:800],
                chunk_id=row.get("chunk_id", ""),
                block_id=row.get("block_id", ""),
                n=n,
            )
            print(f"  [summarize] {row.get('document_name','')[:40]}", flush=True)
            cases = await _gen(prompt, "summarize")
            all_cases.extend(cases)
            task_counts["summarize"] += len(cases)
            await asyncio.sleep(2.0)
        elif _remaining("factual") > 0:
            n = min(2, _remaining("factual"))
            prompt = _E2E_PROMPT.format(
                document_name=row.get("document_name", "unknown"),
                page=row.get("page") or 1,
                content=content[:600],
                chunk_id=row.get("chunk_id", ""),
                block_id=row.get("block_id", ""),
                n=n,
            )
            print(f"  [factual] {row.get('document_name','')[:40]}", flush=True)
            cases = await _gen(prompt, "factual")
            factual = [c for c in cases if c.get("task_type") in ("factual", "claim_check")]
            all_cases.extend(factual)
            for c in factual:
                task_counts[c.get("task_type", "factual")] = task_counts.get(c.get("task_type", "factual"), 0) + 1
            await asyncio.sleep(2.0)

    # 2. compare + graph_relation — pair chunks from same or related docs
    # Domain classification for meaningful cross-doc pairing
    _DOMAIN_MAP = {
        "vinamilk": "finance", "fpt_financial": "finance", "BaoCaoTaiChinh": "finance",
        "bo_luat": "legal", "Hop-Dong": "legal", "luat": "legal",
        "attention": "academic", "hinton": "academic", "s41597": "academic",
        "100416": "academic", "6222": "academic", "24229": "academic",
        "DL_Introduce": "academic", "sfsu": "academic", "engr": "academic",
    }
    def _domain(row: dict) -> str:
        name = row.get("document_name", "")
        for key, dom in _DOMAIN_MAP.items():
            if key.lower() in name.lower():
                return dom
        return "misc"

    # Group chunks by document for same-doc pairing (graph_relation)
    by_doc: dict[str, list[dict]] = {}
    for row in text_rows:
        by_doc.setdefault(row.get("document_name", ""), []).append(row)

    # Same-doc pairs for graph_relation
    same_doc_pairs: list[tuple[dict, dict]] = []
    for doc_rows in by_doc.values():
        if len(doc_rows) >= 2:
            for j in range(0, len(doc_rows) - 1, 2):
                same_doc_pairs.append((doc_rows[j], doc_rows[j + 1]))
    random.shuffle(same_doc_pairs)

    # Same-domain cross-doc pairs for compare
    by_domain: dict[str, list[dict]] = {}
    for row in text_rows:
        by_domain.setdefault(_domain(row), []).append(row)
    cross_doc_pairs: list[tuple[dict, dict]] = []
    for dom_rows in by_domain.values():
        # Pair chunks from different docs within same domain
        docs_in_dom = list(by_doc.keys() & {r.get("document_name") for r in dom_rows})
        if len(docs_in_dom) >= 2:
            for j in range(0, len(docs_in_dom) - 1):
                da = [r for r in dom_rows if r.get("document_name") == docs_in_dom[j]]
                db = [r for r in dom_rows if r.get("document_name") == docs_in_dom[j + 1]]
                if da and db:
                    cross_doc_pairs.append((da[0], db[0]))
    random.shuffle(cross_doc_pairs)

    for a, b in same_doc_pairs:
        if _remaining("graph_relation") <= 0:
            break
        prompt = _E2E_V2_GRAPH_RELATION_PROMPT.format(
            doc_a=a.get("document_name", ""),
            page_a=a.get("page") or 1,
            content_a=a.get("content_preview", "")[:400],
            chunk_a=a.get("chunk_id", ""),
            doc_b=b.get("document_name", ""),
            page_b=b.get("page") or 1,
            content_b=b.get("content_preview", "")[:400],
            chunk_b=b.get("chunk_id", ""),
            n=1,
        )
        print(f"  [graph_relation] {a.get('document_name','')[:25]} p{a.get('page')} + p{b.get('page')}", flush=True)
        cases = await _gen(prompt, "graph_relation")
        all_cases.extend(cases)
        task_counts["graph_relation"] += len(cases)
        await asyncio.sleep(0.3)

    for a, b in cross_doc_pairs:
        if _remaining("compare") <= 0:
            break
        prompt = _E2E_HARD_PROMPT.format(
            doc_a=a.get("document_name", ""),
            page_a=a.get("page") or 1,
            content_a=a.get("content_preview", "")[:400],
            chunk_a=a.get("chunk_id", ""),
            doc_b=b.get("document_name", ""),
            page_b=b.get("page") or 1,
            content_b=b.get("content_preview", "")[:400],
            chunk_b=b.get("chunk_id", ""),
            n=1,
        )
        print(f"  [compare] {a.get('document_name','')[:25]} + {b.get('document_name','')[:25]}", flush=True)
        cases = await _gen(prompt, "compare")
        all_cases.extend(cases)
        task_counts["compare"] += len(cases)
        await asyncio.sleep(0.3)

    # 3. cross_lingual from bilingual pairs (FPT EN↔VI) or any text chunk
    cross_candidates = [r for r in meta_rows if r.get("source_language") in ("en", "vi")]
    random.shuffle(cross_candidates)
    for row in cross_candidates:
        if _remaining("cross_lingual") <= 0:
            break
        src_lang = row.get("source_language", "vi")
        target_lang = "en" if src_lang == "vi" else "vi"
        prompt = _E2E_V2_CROSS_LINGUAL_PROMPT.format(
            document_name=row.get("document_name", "unknown"),
            page=row.get("page") or 1,
            content=row.get("content_preview", "")[:600],
            chunk_id=row.get("chunk_id", ""),
            block_id=row.get("block_id", ""),
            source_language=src_lang,
            target_lang=target_lang,
            n=1,
        )
        print(f"  [cross_lingual/{src_lang}] {row.get('document_name','')[:40]}", flush=True)
        cases = await _gen(prompt, "cross_lingual")
        all_cases.extend(cases)
        task_counts["cross_lingual"] += len(cases)
        await asyncio.sleep(0.3)

    # 4. table task_type from table chunks
    if "table" not in skip_modalities:
        for row in table_rows:
            if _remaining("table") <= 0:
                break
            prompt = _E2E_V2_TABLE_PROMPT.format(
                document_name=row.get("document_name", "unknown"),
                page=row.get("page") or 1,
                content=row.get("content_preview", "")[:600],
                chunk_id=row.get("chunk_id", ""),
                block_id=row.get("block_id", ""),
                n=1,
            )
            print(f"  [table] {row.get('document_name','')[:40]}", flush=True)
            cases = await _gen(prompt, "table")
            all_cases.extend(cases)
            task_counts["table"] += len(cases)
            await asyncio.sleep(2.0)

    # 5. ocr from image/ocr chunks
    if "ocr" not in skip_modalities:
        for row in ocr_rows:
            if _remaining("ocr") <= 0:
                break
            prompt = _E2E_V2_OCR_PROMPT.format(
                document_name=row.get("document_name", "unknown"),
                page=row.get("page") or 1,
                content=row.get("content_preview", "")[:500],
                chunk_id=row.get("chunk_id", ""),
                block_id=row.get("block_id", ""),
                n=1,
            )
            print(f"  [ocr] {row.get('document_name','')[:40]}", flush=True)
            cases = await _gen(prompt, "ocr")
            all_cases.extend(cases)
            task_counts["ocr"] += len(cases)
            await asyncio.sleep(2.0)

    # 6. audio from audio chunks
    if "audio" not in skip_modalities:
        for row in audio_rows:
            if _remaining("audio") <= 0:
                break
            prompt = _E2E_V2_AUDIO_PROMPT.format(
                document_name=row.get("document_name", "unknown"),
                page=row.get("page") or 0,
                content=row.get("content_preview", "")[:500],
                chunk_id=row.get("chunk_id", ""),
                block_id=row.get("block_id", ""),
                n=1,
            )
            print(f"  [audio] {row.get('document_name','')[:40]}", flush=True)
            cases = await _gen(prompt, "audio")
            all_cases.extend(cases)
            task_counts["audio"] += len(cases)
            await asyncio.sleep(2.0)

    _save_jsonl(all_cases, args.output)
    print(f"\nSaved {len(all_cases)} e2e-gold-v2 cases -> {args.output}", flush=True)
    print("  Distribution:", flush=True)
    for tt, cnt in sorted(task_counts.items()):
        tgt = _V2_TASK_TARGETS.get(tt, 0)
        print(f"    {tt:<20} {cnt:>3} / {tgt}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    # Skip direct MongoDB connection when using API-based fetching.
    # meta-inventory uses _fetch_chunks_via_api; other modes read from --input file.
    # Only legacy mode and direct DB modes need init_database.
    use_api = bool(getattr(args, "api_url", "")) and args.mode != "legacy"
    if not use_api:
        await init_database(get_settings())

    print(f"\n{'='*60}")
    print(f"  AgentBook Dataset Generator")
    print(f"  Mode:       {args.mode}")
    print(f"  Provider:   {args.provider}  Model: {args.model}")
    print(f"  Owner:      {args.owner_id}")
    print(f"  Collection: {args.collection_id}")
    if use_api:
        print(f"  Fetch via:  API ({args.api_url})")
    print(f"{'='*60}\n")

    if args.mode == "meta-inventory":
        await run_meta_inventory(args)
    elif args.mode == "retrieval-gold":
        await run_retrieval_gold(args)
    elif args.mode == "e2e-gold":
        await run_e2e_gold(args)
    elif args.mode == "e2e-gold-v2":
        await run_e2e_gold_v2(args)
    elif args.mode == "adversarial":
        await run_adversarial(args)
    elif args.mode == "legacy":
        if not args.api_url:
            args.api_url = "http://localhost:8000"
        await run_legacy(args)
    else:
        print(f"[ERROR] Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="AgentBook benchmark dataset generator")
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--mode",
        choices=["meta-inventory", "retrieval-gold", "e2e-gold", "e2e-gold-v2", "adversarial", "legacy"],
        default="legacy")

    # LLM provider
    parser.add_argument("--provider", choices=["openai", "ollama"], default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--api-base", default="http://localhost:11434")
    parser.add_argument("--api-key", default="")

    # IO
    parser.add_argument("--input", help="Input meta-dataset JSONL (for retrieval-gold / e2e-gold)")
    parser.add_argument("--output", default="eval_results/eval_dataset.jsonl")
    parser.add_argument("--target-count", type=int, default=50,
                        help="Target number of generated cases (retrieval-gold / e2e-gold / adversarial)")
    parser.add_argument("--max-chunks", type=int, default=100,
                        help="Max chunks to fetch from MongoDB (meta-inventory / direct modes)")

    # e2e-gold-v2 args
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="LLM temperature (use 0 for deterministic eval generation)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible chunk sampling")
    parser.add_argument("--skip-modality", default="",
                        help="Comma-separated modalities to skip in e2e-gold-v2 (e.g. 'ocr,audio')")

    # API-based fetching (skip direct MongoDB for meta-inventory)
    parser.add_argument("--api-url", default="",
                        help="Backend API URL (e.g. http://localhost:8000). "
                             "When set, meta-inventory fetches chunks via API instead of direct MongoDB.")

    # Legacy args
    parser.add_argument("--questions-per-chunk", type=int, default=4,
                        help="[legacy] Questions per chunk")

    args = parser.parse_args()

    # Resolve API key from env if not provided
    if not args.api_key:
        args.api_key = os.getenv("OPENAI_API_KEY", "")

    # For openai provider, enforce api_base / api_key
    if args.provider == "openai" and args.mode not in ("meta-inventory", "legacy"):
        if not args.api_key:
            parser.error("--api-key (or OPENAI_API_KEY env var) required for --provider openai")
        if args.api_base == "http://localhost:11434":
            # User forgot to set api_base — use sensible default
            args.api_base = "https://luongchidung.online/v1"
            print(f"  [info] --api-base defaulting to {args.api_base}", flush=True)

    asyncio.run(main(args))
