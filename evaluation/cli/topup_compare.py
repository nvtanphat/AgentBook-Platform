"""Top-up compare cases in e2e_gold_v2.jsonl to reach target count."""
import asyncio, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import _llm_openai, _extract_json_list, _load_jsonl, _save_jsonl, _E2E_HARD_PROMPT, _DOMAIN_MAP

API_KEY  = "sk-xJmkCstnBHbBzLkCPbCpKl2glSSrZrpZZW9OvjtY3iTUGR4B"
API_BASE = "https://luongchidung.online/v1"
MODEL    = "gpt-5.4-mini"
TARGET   = 20
SEED     = 42
GOLD     = "evaluation/datasets/gold/e2e_gold_v2.jsonl"
META     = "evaluation/datasets/gold/meta_dataset.jsonl"

def _domain(name: str) -> str:
    for key, dom in _DOMAIN_MAP.items():
        if key.lower() in name.lower():
            return dom
    return "misc"

async def main():
    random.seed(SEED + 100)
    existing = _load_jsonl(GOLD)
    have_compare = [c for c in existing if c.get("task_type") == "compare"]
    need = TARGET - len(have_compare)
    if need <= 0:
        print(f"Already have {len(have_compare)} compare cases, nothing to do.")
        return

    meta = _load_jsonl(META)
    by_doc: dict[str, list[dict]] = {}
    for r in meta:
        if r.get("modality") in ("paragraph", "mixed", "heading", "list"):
            by_doc.setdefault(r.get("document_name", ""), []).append(r)

    by_domain: dict[str, list[str]] = {}
    for doc in by_doc:
        by_domain.setdefault(_domain(doc), []).append(doc)

    pairs: list[tuple[dict, dict]] = []
    for docs in by_domain.values():
        if len(docs) >= 2:
            docs_list = list(docs)
            random.shuffle(docs_list)
            for i in range(len(docs_list) - 1):
                a_rows = by_doc[docs_list[i]]
                b_rows = by_doc[docs_list[i + 1]]
                if a_rows and b_rows:
                    random.shuffle(a_rows)
                    random.shuffle(b_rows)
                    for j in range(min(5, len(a_rows), len(b_rows))):
                        pairs.append((a_rows[j], b_rows[j]))
    random.shuffle(pairs)

    counter = max((int(c["case_id"].split("-")[-1]) for c in existing if "case_id" in c), default=0) + 1
    new_cases = []

    for a, b in pairs:
        if len(new_cases) >= need:
            break
        prompt = _E2E_HARD_PROMPT.format(
            doc_a=a.get("document_name", ""), page_a=a.get("page") or 1,
            content_a=a.get("content_preview", "")[:400], chunk_a=a.get("chunk_id", ""),
            doc_b=b.get("document_name", ""), page_b=b.get("page") or 1,
            content_b=b.get("content_preview", "")[:400], chunk_b=b.get("chunk_id", ""),
            n=1,
        )
        print(f"  [compare] {a.get('document_name','')[:30]} + {b.get('document_name','')[:30]}", flush=True)
        try:
            raw = await _llm_openai(prompt=prompt, model=MODEL, api_base=API_BASE, api_key=API_KEY, temperature=0)
            cases = _extract_json_list(raw)
            for c in cases:
                if not c.get("query") or not c.get("expected_evidence"):
                    continue
                c["case_id"] = f"ab-e2e-{counter:04d}"
                counter += 1
                c["task_type"] = "compare"
                c["owner_id"] = "nvtanphat69_gmail_com"
                c["collection_id"] = "6a3569119a31a28f07578964"
                new_cases.append(c)
                print(f"    +1 compare (total new: {len(new_cases)}/{need})", flush=True)
        except Exception as exc:
            print(f"  [WARN] {exc}", flush=True)
        await asyncio.sleep(3.0)

    all_cases = existing + new_cases
    _save_jsonl(all_cases, GOLD)
    print(f"\nSaved {len(all_cases)} total cases ({len(new_cases)} new compare) -> {GOLD}", flush=True)

asyncio.run(main())
