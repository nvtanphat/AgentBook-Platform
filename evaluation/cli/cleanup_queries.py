"""
Final cleanup for e2e_gold_v2.jsonl — make every query a clean, standalone,
collection-faithful RAG query.

Pipeline per case:
1. DROP if the query is a text-layout / position / OCR-garbage question
   (these come from junk chunks and have no real-world answer).
2. DROP if query empty.
3. If self-referential ("đoạn trích / đoạn A/B / passage / ..."): rewrite it
   into a natural standalone query, feeding the real document/entity names so
   the model can replace "đoạn A/B" with the actual sources.
4. Validate the rewrite; if still self-referential, drop it.
"""
import asyncio, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import _llm_openai, _load_jsonl, _save_jsonl

API_KEY  = "sk-xJmkCstnBHbBzLkCPbCpKl2glSSrZrpZZW9OvjtY3iTUGR4B"
API_BASE = "https://luongchidung.online/v1"
MODEL    = "gpt-5.4-mini"

GOLD = "evaluation/datasets/gold/e2e_gold_v2.jsonl"

# Self-reference: query talks about "the excerpt / passage / đoạn A/B" itself.
_SELF_REF = re.compile(
    r"(đoạn\s+trích|đoạn\s+văn|trích\s+đoạn|đoạn\s+này|đoạn\s+trên|đoạn\s+a\b|đoạn\s+b\b"
    r"|hai\s+đoạn|đoạn\s+nào|nội\s+dung\s+trên|văn\s+bản\s+nào\s+(nêu|chứa|cho)"
    r"|in\s+this\s+passage|the\s+passage|this\s+passage|the\s+excerpt|this\s+excerpt"
    r"|in\s+the\s+text\s+above|shown\s+in\s+the\s+passage)",
    re.IGNORECASE,
)

# Layout / position / OCR-garbage: not a real information-seeking query. DROP.
_GARBAGE = re.compile(
    r"(dòng\s+(chữ|nào)|cụm\s+từ\s+nào\s+(được\s+)?lặp|chuỗi\s+(văn\s+bản|giá\s+trị)\s+(bị\s+)?lặp"
    r"|xuất\s+hiện\s+(ngay\s+)?(trước|sau|ở\s+cuối|ở\s+đầu)|ở\s+cuối\s+(dãy|đoạn|bảng)"
    r"|lặp\s+lại\s+nhiều\s+lần|ngay\s+trước\s+(số|dòng)|ngay\s+sau\s+(dòng|mục)"
    r"|có\s+phải\s+là\s+một\s+con\s+số|nhãn.*lần\s+thứ)",
    re.IGNORECASE,
)


def _doc_names(case: dict) -> list[str]:
    seen, out = set(), []
    for ev in case.get("expected_evidence") or []:
        name = (ev.get("document_name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


_REWRITE_PROMPT = """\
You are fixing a RAG evaluation benchmark. The query below is invalid because it refers to \
"the passage / excerpt / đoạn A / đoạn B" — but a real user never sees the document; they just \
type a question to FIND information.

Rewrite it into ONE natural, standalone question that:
- Names the real subject (company, law article, model, year, metric). Use the source document names below if helpful.
- Has NO words like: "đoạn trích", "đoạn văn", "đoạn A/B", "trích đoạn", "passage", "excerpt", "the text above", "văn bản nào".
- Keeps the same information need and the same language as the original.
- For comparison questions: name BOTH real subjects (e.g. "So sánh X của Vinamilk và FPT...").

Source documents: {docs}
Original (invalid): "{query}"

Return ONLY the rewritten question — no quotes, no explanation, no prefix."""


def _is_self_ref(q: str) -> bool:
    return bool(_SELF_REF.search(q or ""))


def _is_garbage(q: str) -> bool:
    return bool(_GARBAGE.search(q or ""))


async def _rewrite(query: str, docs: list[str]) -> str:
    prompt = _REWRITE_PROMPT.format(query=query, docs=", ".join(docs) or "(unknown)")
    out = await _llm_openai(
        prompt=prompt, model=MODEL, api_base=API_BASE, api_key=API_KEY,
        temperature=0, max_tokens=220, retries=4,
    )
    return out.strip().strip('"').strip()


async def main() -> None:
    cases = _load_jsonl(GOLD)
    print(f"Loaded {len(cases)} cases", flush=True)

    kept, rewritten, dropped_garbage, dropped_badrewrite, dropped_empty = [], 0, 0, 0, 0

    for c in cases:
        q = (c.get("query") or "").strip()
        if not q:
            dropped_empty += 1
            continue
        if _is_garbage(q):
            dropped_garbage += 1
            print(f"  [DROP garbage/{c.get('task_type')}] {q[:75]}", flush=True)
            continue
        if _is_self_ref(q):
            try:
                new_q = await _rewrite(q, _doc_names(c))
            except Exception as exc:
                print(f"  [WARN rewrite failed -> drop] {exc}", flush=True)
                dropped_badrewrite += 1
                continue
            if new_q and not _is_self_ref(new_q) and not _is_garbage(new_q):
                print(f"  [FIX] {q[:65]!r}\n     -> {new_q[:65]!r}", flush=True)
                c["query"] = new_q
                rewritten += 1
                kept.append(c)
            else:
                print(f"  [DROP still-bad] {new_q[:75]!r}", flush=True)
                dropped_badrewrite += 1
            await asyncio.sleep(1.5)
        else:
            kept.append(c)

    _save_jsonl(kept, GOLD)
    print(
        f"\nDone. kept={len(kept)} rewritten={rewritten} "
        f"dropped(garbage={dropped_garbage}, badrewrite={dropped_badrewrite}, empty={dropped_empty})\n"
        f"Saved -> {GOLD}",
        flush=True,
    )


asyncio.run(main())
