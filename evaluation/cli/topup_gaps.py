"""
Top up the gaps in e2e_gold_v2.jsonl with HIGH-QUALITY, collection-faithful cases:
  - compare      -> target 20  (meaningful same-axis comparisons; SKIP if none)
  - claim_check  -> target  5  (a verifiable claim grounded in one chunk)
  - factual      -> target 35

Quality guarantees:
  * every generated query is validated (no self-reference / no layout-garbage); bad -> retry, then skip
  * compare uses an explicit SKIP escape so forced/meaningless pairs are discarded
  * garbage source chunks (repetitive OCR) are filtered out before use
  * domain-aware pairing (finance/legal/academic) so compares stay on-topic
"""
import asyncio, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import _llm_openai, _extract_json_list, _load_jsonl, _save_jsonl

API_KEY  = "sk-xJmkCstnBHbBzLkCPbCpKl2glSSrZrpZZW9OvjtY3iTUGR4B"
API_BASE = "https://luongchidung.online/v1"
MODEL    = "gpt-5.4-mini"
OWNER    = "nvtanphat69_gmail_com"
COLL     = "6a3569119a31a28f07578964"

GOLD = "D:/GenAI/DoAn01/evaluation/datasets/gold/e2e_gold_v2.jsonl"
META = "D:/GenAI/DoAn01/evaluation/datasets/gold/meta_dataset.jsonl"

TARGETS = {"compare": 20, "claim_check": 5, "factual": 35}

_DOMAIN_MAP = {
    "vinamilk": "finance", "fpt": "finance", "baocaotaichinh": "finance",
    "bo_luat": "legal", "hop-dong": "legal", "luat": "legal",
    "s41597": "academic", "attention": "academic", "24229": "academic",
    "6222": "academic", "dl_introduce": "academic", "100416": "academic",
    "hinton": "academic", "sfsu": "academic", "engr": "academic",
}

# Validation: query must be a standalone, natural information-seeking question.
_SELF_REF = re.compile(
    r"(đoạn\s+trích|đoạn\s+văn|trích\s+đoạn|đoạn\s+này|đoạn\s+trên|đoạn\s+a\b|đoạn\s+b\b"
    r"|hai\s+đoạn|đoạn\s+nào|tài\s+liệu\s+1|tài\s+liệu\s+2|nội\s+dung\s+trên"
    r"|in\s+this\s+passage|the\s+passage|this\s+passage|the\s+excerpt|in\s+the\s+text\s+above)",
    re.IGNORECASE,
)
_GARBAGE = re.compile(
    r"(dòng\s+(chữ|nào)|cụm\s+từ\s+nào\s+(được\s+)?lặp|chuỗi\s+.*lặp"
    r"|xuất\s+hiện\s+(ngay\s+)?(trước|sau|ở\s+cuối)|lặp\s+lại\s+nhiều\s+lần"
    r"|ngay\s+(trước|sau)\s+(số|dòng|mục)|có\s+phải\s+là\s+một\s+con\s+số)",
    re.IGNORECASE,
)


def _domain(name: str) -> str:
    n = (name or "").lower()
    for key, dom in _DOMAIN_MAP.items():
        if key in n:
            return dom
    return "misc"


def _is_repetitive(text: str) -> bool:
    """Detect OCR-garbage chunks: same token repeated a lot, or very low variety."""
    toks = re.findall(r"\w+", (text or "").lower())
    if len(toks) < 12:
        return True
    uniq_ratio = len(set(toks)) / len(toks)
    most = Counter(toks).most_common(1)[0][1] if toks else 0
    return uniq_ratio < 0.35 or most > len(toks) * 0.25


def _valid_query(q: str) -> bool:
    return bool(q) and not _SELF_REF.search(q) and not _GARBAGE.search(q)


def _ev(row: dict) -> dict:
    return {
        "document_name": row.get("document_name", ""),
        "page": row.get("page") or 1,
        "chunk_id": row.get("chunk_id", ""),
        "quote_or_fact": (row.get("content_preview", "") or "")[:200],
    }


_COMPARE_PROMPT = """\
Bạn tạo MỘT câu hỏi SO SÁNH cho bộ benchmark đánh giá hệ thống RAG (người dùng KHÔNG nhìn thấy tài liệu).

Nội dung tham khảo (chỉ để bạn đọc, KHÔNG được nhắc tới trong câu hỏi):

[Nguồn 1] {doc_a} — trang {page_a}:
{content_a}

[Nguồn 2] {doc_b} — trang {page_b}:
{content_b}

Nếu hai nguồn có MỘT TRỤC SO SÁNH HỢP LÝ (cùng loại chỉ số, cùng phương pháp, cùng chủ đề, cùng đại lượng),
hãy viết 1 câu hỏi so sánh tự nhiên, NÊU RÕ TÊN THẬT của cả hai chủ thể.
Nếu KHÔNG có trục so sánh hợp lý (hai nội dung không liên quan), trả về đúng một chữ: SKIP

CẤM tuyệt đối: "đoạn A/B", "nguồn 1/2", "tài liệu 1/2", "hai đoạn", "trong hai tài liệu", "passage".
PHẢI nêu tên thật (Vinamilk, Transformer, bài báo phân loại u não, Bộ luật Dân sự...).

VÍ DỤ TỐT: "So sánh điểm F1-score của mô hình EfficientNetB0 trong bài báo phân loại u não với kết quả của Transformer trong Attention Is All You Need."

Nếu viết được, trả về JSON (mảng 1 phần tử), KHÔNG markdown:
[{{"query":"...","expected_answer_outline":["...","..."],"required_facts":["sự kiện nguồn 1","sự kiện nguồn 2"],"forbidden_claims":["suy diễn sai"]}}]
Nếu không, chỉ trả về: SKIP"""


_CLAIM_PROMPT = """\
Bạn tạo MỘT case KIỂM CHỨNG TUYÊN BỐ (claim verification) cho benchmark RAG.

Nội dung tham khảo (KHÔNG nhắc tới trong câu hỏi) — {doc} trang {page}:
{content}

Viết 1 câu hỏi yêu cầu hệ thống xác minh một tuyên bố CỤ THỂ có thể kiểm chứng từ nội dung trên.
Tuyên bố phải nêu rõ chủ thể thật (tên công ty/luật/mô hình/số liệu/năm), KHÔNG dùng "đoạn trích / đoạn văn / theo nội dung trên".

VÍ DỤ TỐT: "Có đúng là lợi nhuận sau thuế của Vinamilk năm 2024 đạt trên 9.000 tỷ đồng không?"

Trả về JSON (mảng 1 phần tử), KHÔNG markdown:
[{{"query":"...","expected_answer_outline":["..."],"required_facts":["sự kiện đúng từ nội dung"],"forbidden_claims":["phiên bản sai của tuyên bố"]}}]"""


_FACTUAL_PROMPT = """\
Bạn tạo MỘT câu hỏi FACTUAL cho benchmark RAG (người dùng KHÔNG nhìn thấy tài liệu).

Nội dung tham khảo (KHÔNG nhắc tới trong câu hỏi) — {doc} trang {page}:
{content}

Viết 1 câu hỏi tra cứu thông tin tự nhiên, NÊU RÕ chủ thể thật (tên công ty/luật/điều khoản/mô hình/số liệu/năm).
CẤM: "đoạn trích / đoạn văn / đoạn này / theo nội dung trên / passage". CẤM hỏi về vị trí dòng/cụm từ.

VÍ DỤ TỐT: "Điều 154 Bộ luật Dân sự 2015 quy định thời hiệu khởi kiện vụ án dân sự được tính từ thời điểm nào?"

Trả về JSON (mảng 1 phần tử), KHÔNG markdown:
[{{"query":"...","expected_answer_outline":["..."],"required_facts":["sự kiện đúng"],"forbidden_claims":["tuyên bố không có trong nội dung"]}}]"""


async def _call(prompt: str) -> str:
    return await _llm_openai(
        prompt=prompt, model=MODEL, api_base=API_BASE, api_key=API_KEY,
        temperature=0.2, max_tokens=600, retries=4,
    )


async def main() -> None:
    random.seed(123)
    gold = _load_jsonl(GOLD)
    have = Counter(c.get("task_type") for c in gold)
    print("Current:", dict(have), flush=True)

    meta = _load_jsonl(META)
    good = [r for r in meta
            if len(r.get("content_preview", "") or "") >= 200
            and not _is_repetitive(r.get("content_preview", ""))]
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in good:
        by_doc[r.get("document_name", "")].append(r)
    by_domain: dict[str, list[str]] = defaultdict(list)
    for doc in by_doc:
        by_domain[_domain(doc)].append(doc)

    counter = max((int(c["case_id"].split("-")[-1]) for c in gold if "case_id" in c), default=0) + 1
    new_cases: list[dict] = []

    def _attach(c: dict, task: str, evidence: list[dict], difficulty: str) -> None:
        nonlocal counter
        c["case_id"] = f"ab-e2e-{counter:04d}"; counter += 1
        c["task_type"] = task
        c.setdefault("query_language", "vi"); c.setdefault("answer_language", "vi")
        c["expected_evidence"] = evidence
        c["expected_behavior"] = "answer"
        c.setdefault("difficulty", difficulty)
        c["tags"] = [task]
        c["owner_id"] = OWNER; c["collection_id"] = COLL

    # ---- compare ----
    need = TARGETS["compare"] - have["compare"]
    pairs: list[tuple[dict, dict]] = []
    for dom, docs in by_domain.items():
        if dom == "misc" or len(docs) < 2:
            continue
        random.shuffle(docs)
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                a, b = by_doc[docs[i]], by_doc[docs[j]]
                for k in range(min(4, len(a), len(b))):
                    pairs.append((random.choice(a), random.choice(b)))
    random.shuffle(pairs)
    print(f"\n[compare] need {need}, {len(pairs)} candidate pairs", flush=True)
    for a, b in pairs:
        if len([c for c in new_cases if c["task_type"] == "compare"]) >= need:
            break
        prompt = _COMPARE_PROMPT.format(
            doc_a=a.get("document_name", ""), page_a=a.get("page") or 1, content_a=a.get("content_preview", "")[:450],
            doc_b=b.get("document_name", ""), page_b=b.get("page") or 1, content_b=b.get("content_preview", "")[:450])
        try:
            raw = (await _call(prompt)).strip()
            if raw.upper().startswith("SKIP") or "SKIP" in raw[:10].upper():
                print(f"  skip: {a['document_name'][:20]} x {b['document_name'][:20]}", flush=True)
                await asyncio.sleep(1.2); continue
            cases = _extract_json_list(raw)
            for c in cases:
                if _valid_query(c.get("query", "")):
                    _attach(c, "compare", [_ev(a), _ev(b)], "hard")
                    new_cases.append(c)
                    print(f"  +compare ({len([x for x in new_cases if x['task_type']=='compare'])}/{need}): {c['query'][:70]}", flush=True)
                else:
                    print(f"  reject(invalid): {c.get('query','')[:60]}", flush=True)
        except Exception as exc:
            print(f"  [WARN] {exc}", flush=True)
        await asyncio.sleep(1.5)

    # ---- claim_check ----
    need = TARGETS["claim_check"] - have["claim_check"]
    cands = [r for r in good if _domain(r["document_name"]) in ("finance", "legal", "academic")]
    random.shuffle(cands)
    print(f"\n[claim_check] need {need}", flush=True)
    for r in cands:
        if len([c for c in new_cases if c["task_type"] == "claim_check"]) >= need:
            break
        prompt = _CLAIM_PROMPT.format(doc=r["document_name"], page=r.get("page") or 1, content=r["content_preview"][:550])
        try:
            cases = _extract_json_list(await _call(prompt))
            for c in cases:
                if _valid_query(c.get("query", "")):
                    _attach(c, "claim_check", [_ev(r)], "medium")
                    new_cases.append(c)
                    print(f"  +claim_check: {c['query'][:70]}", flush=True)
        except Exception as exc:
            print(f"  [WARN] {exc}", flush=True)
        await asyncio.sleep(1.5)

    # ---- factual ----
    need = TARGETS["factual"] - have["factual"]
    random.shuffle(cands)
    print(f"\n[factual] need {need}", flush=True)
    for r in cands:
        if len([c for c in new_cases if c["task_type"] == "factual"]) >= need:
            break
        prompt = _FACTUAL_PROMPT.format(doc=r["document_name"], page=r.get("page") or 1, content=r["content_preview"][:550])
        try:
            cases = _extract_json_list(await _call(prompt))
            for c in cases:
                if _valid_query(c.get("query", "")):
                    _attach(c, "factual", [_ev(r)], "easy")
                    new_cases.append(c)
                    print(f"  +factual: {c['query'][:70]}", flush=True)
        except Exception as exc:
            print(f"  [WARN] {exc}", flush=True)
        await asyncio.sleep(1.5)

    _save_jsonl(gold + new_cases, GOLD)
    final = Counter(c.get("task_type") for c in gold + new_cases)
    print(f"\nAdded {len(new_cases)} cases. Final: {dict(final)}\nSaved -> {GOLD}", flush=True)


asyncio.run(main())
