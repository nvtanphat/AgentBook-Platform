"""
build_benchmark.py — Rebuild the AgentBook RAG evaluation benchmark from scratch.

Design follows established QA / RAG benchmarks:
  * Natural-Questions / MS-MARCO : queries are standalone natural information needs
                                   (NO "đoạn trích / passage / in the excerpt" self-reference)
  * SQuAD / RAGAS                : each case carries a concise `reference_answer` (gold)
                                   plus `required_facts` / `forbidden_claims` for LLM-judge
  * HotpotQA                     : multi-hop `graph_relation` with 2 supporting evidence chunks
  * RGB / NQ-unanswerable        : `off_topic_should_refuse` (must refuse) + `false_premise`
                                   (must correct) to measure false-accept / false-refuse rates

Source: evaluation/datasets/gold/meta_dataset.jsonl (3145 chunks already fetched).
Output: evaluation/datasets/gold/e2e_gold_v2.jsonl (overwritten).

Every generated query passes an inline validator (no self-reference, no layout-garbage);
failing cases are retried then dropped. Garbage/repetitive source chunks are filtered out.
"""
from __future__ import annotations

import asyncio
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import _llm_openai, _extract_json_list, _load_jsonl, _save_jsonl

# ── config ────────────────────────────────────────────────────────────────────
API_KEY  = "sk-xJmkCstnBHbBzLkCPbCpKl2glSSrZrpZZW9OvjtY3iTUGR4B"
API_BASE = "https://luongchidung.online/v1"
MODEL    = "gpt-5.4-mini"
OWNER    = "nvtanphat69_gmail_com"
COLL     = "6a3569119a31a28f07578964"
SEED     = 42
DELAY    = 1.4  # seconds between LLM calls (proxy rate-limit safety)

META = "D:/GenAI/DoAn01/evaluation/datasets/gold/meta_dataset.jsonl"
OUT  = "D:/GenAI/DoAn01/evaluation/datasets/gold/e2e_gold_v2.jsonl"

TARGETS = {
    "factual": 90,
    "compare": 25,
    "summarize": 25,
    "graph_relation": 30,
    "cross_lingual": 35,
    "table": 20,
    "ocr": 12,
    "claim_check": 20,   # split 50/50 true vs false claims
    "off_topic_should_refuse": 25,
    "false_premise": 18,
}  # total ≈ 300

_DOMAIN_MAP = {
    "vinamilk": "finance", "fpt": "finance", "baocaotaichinh": "finance",
    "bo_luat": "legal", "hop-dong": "legal",
    "s41597": "academic", "attention": "academic", "24229": "academic",
    "6222": "academic", "dl_introduce": "academic", "100416": "academic",
    "hinton": "academic", "sfsu": "academic", "engr": "academic",
}

# Description of what the collection DOES contain — used to craft out-of-scope queries.
COLLECTION_SCOPE = """\
Bộ tài liệu chỉ gồm:
- Tài chính: Báo cáo thường niên Vinamilk 2024; một trang báo cáo tài chính FPT.
- Pháp lý: Bộ luật Dân sự 2015 của Việt Nam; một hợp đồng viết tay.
- Học thuật (ML/AI): bài báo "Attention Is All You Need" (Transformer); bài báo phân loại u não (EfficientNet);
  bài giảng mạng nơ-ron của Hinton; tài liệu giới thiệu Deep Learning; bài báo phân tích cảm xúc (mô hình CD-AHAL);
  bài báo về bộ dữ liệu (s41597); đề cương môn phần cứng cho ML."""

# ── validators ─────────────────────────────────────────────────────────────────
_SELF_REF = re.compile(
    r"(đoạn\s+trích|đoạn\s+văn|trích\s+đoạn|đoạn\s+này|đoạn\s+trên|đoạn\s+a\b|đoạn\s+b\b"
    r"|hai\s+đoạn|đoạn\s+nào|tài\s+liệu\s+1|tài\s+liệu\s+2|nguồn\s+1|nguồn\s+2|nội\s+dung\s+trên"
    # "[artifact] này/trên/đó/ấy" = self-reference without naming which one
    r"|(bài\s+báo|bài\s+giảng|bài\s+viết|tài\s+liệu|văn\s+bản|báo\s+cáo|học\s+phần|đề\s+cương"
    r"|công\s+văn|nghiên\s+cứu|chương|mục|phần|bảng|hình|biểu\s+đồ|slide|trang)\s+(này|trên|đó|ấy|nói\s+trên)\b"
    r"|in\s+this\s+passage|the\s+passage|this\s+passage|the\s+excerpt|in\s+the\s+text\s+above"
    r"|\bthis\s+(paper|document|report|study|article|text|excerpt|passage|table|chart|figure|slide|lecture)\b)",
    re.IGNORECASE,
)
_GARBAGE = re.compile(
    r"(dòng\s+(chữ|nào)|cụm\s+từ\s+nào\s+(được\s+)?lặp|chuỗi\s+.*lặp"
    r"|xuất\s+hiện\s+(ngay\s+)?(trước|sau|ở\s+cuối)|lặp\s+lại\s+nhiều\s+lần"
    r"|ngay\s+(trước|sau)\s+(số|dòng|mục)|có\s+phải\s+là\s+một\s+con\s+số)",
    re.IGNORECASE,
)
# A real user never types raw file names or "trang N của tài liệu X.pdf".
_FILENAME = re.compile(r"(\.pdf|\.docx|\.pptx|\.png|\.jpe?g|\.xlsx?|\.csv|\.mp3|\.wav|\bvietstock_real\b)", re.IGNORECASE)
# Raw filename codes of the academic papers (non-semantic) leaking into queries.
_FILECODE = re.compile(
    r"\b(100416|209536|20240801|6222|20774|20240905|24229|s41597|06753|vietstock)\b"
    r"|article\s+text|huong_danh_bui|_lecture\d|_real\b",
    re.IGNORECASE,
)
# English self-reference ("in the document/table", "shown in the table", "the above text").
_EN_SELF = re.compile(
    r"\b(in|from|on|of|within)\s+the\s+(document|text|passage|excerpt|table|paper|report|article|chart|figure)\b"
    r"|\b(shown|listed|mentioned|described|presented|given|provided)\s+in\s+the\s+(document|text|table|passage|figure|chart|report|paper)\b"
    r"|\baccording\s+to\s+the\s+(document|text|passage|table)\b"
    r"|\bthe\s+(above|following)\s+(document|text|table|passage)\b",
    re.IGNORECASE,
)


def _valid_query(q: str) -> bool:
    return (bool(q and q.strip())
            and not _SELF_REF.search(q)
            and not _GARBAGE.search(q)
            and not _FILENAME.search(q)
            and not _EN_SELF.search(q)
            and not _FILECODE.search(q))


def _domain(name: str) -> str:
    n = (name or "").lower()
    for key, dom in _DOMAIN_MAP.items():
        if key in n:
            return dom
    return "misc"


def _is_repetitive(text: str) -> bool:
    toks = re.findall(r"\w+", (text or "").lower())
    if len(toks) < 12:
        return True
    uniq = len(set(toks)) / len(toks)
    top = Counter(toks).most_common(1)[0][1]
    return uniq < 0.38 or top > len(toks) * 0.22


def _ev(row: dict) -> dict:
    return {
        "document_name": row.get("document_name", ""),
        "page": row.get("page") or 1,
        "block_id": row.get("block_id", ""),
        "chunk_id": row.get("chunk_id", ""),
        "quote_or_fact": (row.get("content_preview", "") or "")[:220],
    }


async def _call(prompt: str, *, temperature: float = 0.2, max_tokens: int = 700) -> str:
    return await _llm_openai(
        prompt=prompt, model=MODEL, api_base=API_BASE, api_key=API_KEY,
        temperature=temperature, max_tokens=max_tokens, retries=4,
    )


# ── shared rules block injected into every grounded prompt ─────────────────────
_RULES = """\
QUY TẮC BẮT BUỘC (vi phạm -> câu hỏi bị loại):
- Người dùng KHÔNG nhìn thấy tài liệu. Câu hỏi phải ĐỘC LẬP, tự nhiên như gõ vào Google.
- CẤM: "đoạn trích", "đoạn văn", "đoạn này/A/B", "nguồn 1/2", "theo nội dung trên", "passage", "excerpt".
- CẤM "bài báo này / tài liệu này / báo cáo này / bài giảng này / bảng này / this paper / this document" — PHẢI gọi TÊN cụ thể (vd "bài báo Attention Is All You Need", "báo cáo thường niên 2024 của Vinamilk").
- CẤM dùng MÃ SỐ/TÊN FILE tài liệu (vd "100416-Article Text", "6222", "24229_Huong_Danh_Bui", "s41597"). Hãy suy ra TIÊU ĐỀ/CHỦ ĐỀ THẬT từ nội dung (vd "bài báo về phân đoạn u phổi", "bài báo phân tích cảm xúc"). Nếu không rõ tiêu đề, hỏi theo CHỦ ĐỀ/khái niệm trong nội dung, không nhắc tên tài liệu.
- CẤM hỏi về vị trí văn bản: "dòng nào", "cụm từ nào xuất hiện trước/sau", "ở cuối đoạn".
- PHẢI nêu rõ chủ thể thật: tên công ty / luật / điều khoản / mô hình / chỉ số / năm.
- "reference_answer" là câu trả lời đúng, ngắn gọn, bám sát nội dung (gold answer)."""

_JSON_SHAPE = """\
Trả về JSON (mảng 1 phần tử), KHÔNG markdown, KHÔNG giải thích:
[{{"query":"...","reference_answer":"...","expected_answer_outline":["..."],"required_facts":["..."],"forbidden_claims":["..."]}}]"""


def _grounded_prompt(kind: str, row: dict) -> str:
    head = {
        "factual": "Tạo MỘT câu hỏi FACTUAL (tra cứu một thông tin cụ thể).",
        "summarize": "Tạo MỘT câu hỏi yêu cầu TÓM TẮT nội dung chính (câu trả lời bao quát nhiều ý).",
        "table": "Tạo MỘT câu hỏi tra cứu/so sánh SỐ LIỆU trong bảng (nêu rõ tên chỉ tiêu, năm, đơn vị).",
        "ocr": "Tạo MỘT câu hỏi về thông tin trích từ ảnh/tài liệu scan (nêu rõ thực thể, con số, tên).",
        "claim_check": "Tạo MỘT câu hỏi KIỂM CHỨNG một tuyên bố cụ thể (ví dụ: 'Có đúng là ... không?').",
    }[kind]
    return f"""Bạn xây dựng benchmark RAG cho hệ thống hỏi đáp tài liệu (pháp lý/tài chính/học thuật).

{head}

Nội dung tham khảo (CHỈ để bạn đọc) — {row.get('document_name','')} trang {row.get('page') or 1}:
\"\"\"{(row.get('content_preview','') or '')[:600]}\"\"\"

{_RULES}

{_JSON_SHAPE}"""


def _cross_prompt(row: dict, target_lang: str) -> str:
    return f"""You are building a cross-lingual RAG benchmark. The user query must be in {target_lang.upper()} \
while the source document is in the other language; cross-lingual retrieval must still find it.

Reference content (FOR YOUR EYES ONLY) — {row.get('document_name','')} page {row.get('page') or 1}:
\"\"\"{(row.get('content_preview','') or '')[:600]}\"\"\"

RULES (violation -> case rejected):
- The query MUST be a natural standalone question in {target_lang.upper()} that names the real subject
  (company / law article / model / metric / year). NO "this passage / the excerpt / đoạn trích / theo nội dung trên".
- "reference_answer" is a concise correct answer in {target_lang.upper()}.

Return JSON (array of 1), no markdown:
[{{"query":"...","reference_answer":"...","expected_answer_outline":["..."],"required_facts":["..."],"forbidden_claims":["..."]}}]"""


def _compare_prompt(a: dict, b: dict) -> str:
    return f"""Bạn tạo MỘT câu hỏi SO SÁNH cho benchmark RAG (người dùng KHÔNG nhìn thấy tài liệu).

Nội dung 1 — {a.get('document_name','')} trang {a.get('page') or 1}:
\"\"\"{(a.get('content_preview','') or '')[:430]}\"\"\"

Nội dung 2 — {b.get('document_name','')} trang {b.get('page') or 1}:
\"\"\"{(b.get('content_preview','') or '')[:430]}\"\"\"

Nếu hai nội dung có MỘT TRỤC SO SÁNH HỢP LÝ (cùng loại chỉ số/phương pháp/đại lượng/chủ đề), viết 1 câu hỏi
so sánh tự nhiên NÊU RÕ TÊN THẬT của cả hai chủ thể. Nếu KHÔNG có trục so sánh hợp lý, trả về đúng chữ: SKIP

{_RULES}

Nếu viết được, trả JSON (mảng 1 phần tử):
[{{"query":"...","reference_answer":"...","expected_answer_outline":["..."],"required_facts":["sự kiện nội dung 1","sự kiện nội dung 2"],"forbidden_claims":["suy diễn sai"]}}]
Nếu không: chỉ trả về SKIP"""


def _graph_prompt(a: dict, b: dict) -> str:
    return f"""Bạn tạo MỘT câu hỏi MULTI-HOP (cần ghép thông tin từ HAI nơi) cho benchmark RAG.

Nội dung 1 — {a.get('document_name','')} trang {a.get('page') or 1}:
\"\"\"{(a.get('content_preview','') or '')[:430]}\"\"\"

Nội dung 2 — {b.get('document_name','')} trang {b.get('page') or 1}:
\"\"\"{(b.get('content_preview','') or '')[:430]}\"\"\"

Viết 1 câu hỏi mà để trả lời PHẢI dùng thông tin từ cả hai nơi, NÊU RÕ tên thực thể/khái niệm thật.
Nếu hai nội dung không có liên hệ ghép nối được, trả về đúng chữ: SKIP

{_RULES}

Nếu viết được, trả JSON (mảng 1 phần tử):
[{{"query":"...","reference_answer":"...","expected_answer_outline":["..."],"required_facts":["mắt xích 1","mắt xích 2"],"forbidden_claims":["suy diễn sai"]}}]
Nếu không: chỉ trả về SKIP"""


_OFF_TOPIC_PROMPT = f"""Bạn tạo các câu hỏi NGOÀI PHẠM VI cho benchmark RAG — để kiểm tra hệ thống có biết TỪ CHỐI
khi thông tin KHÔNG nằm trong bộ tài liệu hay không.

{COLLECTION_SCOPE}

Tạo {{n}} câu hỏi NGHE CÓ VẺ liên quan (cùng lĩnh vực tài chính/pháp lý/ML) NHƯNG chắc chắn KHÔNG trả lời được
từ bộ tài liệu trên (hỏi về công ty khác, luật khác, mô hình khác, năm khác, chủ đề khác).
Câu hỏi phải tự nhiên, cụ thể, có chủ thể rõ ràng. KHÔNG nhắc tới "tài liệu/đoạn trích".

VÍ DỤ TỐT: "Doanh thu thuần của Tập đoàn Hòa Phát năm 2024 là bao nhiêu?" (Hòa Phát không có trong bộ tài liệu);
"Bộ luật Hình sự 2015 quy định khung hình phạt cho tội trộm cắp tài sản như thế nào?" (chỉ có Bộ luật Dân sự).

Trả JSON (mảng {{n}} phần tử), KHÔNG markdown:
[{{{{"query":"...","reason_out_of_scope":"vì sao không có trong bộ tài liệu"}}}}]"""


def _claim_false_prompt(row: dict) -> str:
    return f"""Bạn tạo MỘT case KIỂM CHỨNG TUYÊN BỐ SAI cho benchmark RAG (hệ thống phải BÁC BỎ).

Nội dung tham khảo (CHỈ để bạn đọc) — {row.get('document_name','')} trang {row.get('page') or 1}:
\"\"\"{(row.get('content_preview','') or '')[:550]}\"\"\"

Viết 1 câu hỏi dạng "Có đúng là ... không?" trong đó tuyên bố CHỨA THÔNG TIN SAI so với nội dung trên
(sai số liệu / sai tên / sai quan hệ), nhưng vẫn nghe tự nhiên. Nêu rõ chủ thể thật.
"reference_answer" PHẢI bắt đầu bằng "Sai." rồi nêu thông tin ĐÚNG.
CẤM "đoạn trích / theo nội dung trên / in the document".

VÍ DỤ: "Có đúng là lợi nhuận sau thuế Vinamilk 2024 đạt 50.000 tỷ đồng không?" -> "Sai. Con số đúng là ~9.392 tỷ đồng."

Trả JSON (mảng 1 phần tử), KHÔNG markdown:
[{{"query":"Có đúng là ...?","reference_answer":"Sai. ...","required_facts":["thông tin đúng từ nội dung"],"forbidden_claims":["khẳng định tuyên bố sai là đúng"]}}]"""


def _false_premise_prompt(row: dict) -> str:
    return f"""Bạn tạo MỘT câu hỏi CÓ TIỀN ĐỀ SAI (false premise) cho benchmark RAG — hệ thống phải PHÁT HIỆN và SỬA
tiền đề sai dựa trên tài liệu, thay vì trả lời theo giả định sai.

Sự thật tham khảo — {row.get('document_name','')} trang {row.get('page') or 1}:
\"\"\"{(row.get('content_preview','') or '')[:550]}\"\"\"

Viết 1 câu hỏi GÀI một thông tin SAI (sai số liệu / sai tên / sai quan hệ) so với sự thật trên, nhưng vẫn nghe tự nhiên.
"reference_answer" phải nêu rõ tiền đề sai ở đâu và đưa ra thông tin ĐÚNG.
KHÔNG dùng "đoạn trích / theo nội dung trên". Nêu rõ chủ thể thật.

VÍ DỤ: query "Vì sao Vinamilk báo lỗ trong năm 2024?" (nếu thực tế có lãi) -> reference_answer chỉ ra Vinamilk
KHÔNG lỗ mà có lãi, kèm con số đúng.

Trả JSON (mảng 1 phần tử), KHÔNG markdown:
[{{"query":"...","reference_answer":"...","required_facts":["thông tin đúng từ tài liệu"],"forbidden_claims":["khẳng định theo tiền đề sai"]}}]"""


# ── case assembly ──────────────────────────────────────────────────────────────
class Builder:
    def __init__(self) -> None:
        self.n = 0
        self.cases: list[dict] = []

    def add(self, *, task: str, query: str, evidence: list[dict], difficulty: str,
            reference_answer: str = "", outline=None, required=None, forbidden=None,
            query_language="vi", answer_language="vi", expected_behavior="answer",
            extra: dict | None = None) -> None:
        self.n += 1
        case = {
            "case_id": f"ab-bench-{self.n:04d}",
            "task_type": task,
            "query_type": task,
            "query_language": query_language,
            "answer_language": answer_language,
            "query": query.strip(),
            "reference_answer": (reference_answer or "").strip(),
            "expected_answer_outline": outline or [],
            "required_facts": required or [],
            "forbidden_claims": forbidden or [],
            "expected_evidence": evidence,
            "expected_behavior": expected_behavior,
            "difficulty": difficulty,
            "tags": [task],
            "owner_id": OWNER,
            "collection_id": COLL,
        }
        if extra:
            case.update(extra)
        self.cases.append(case)


async def main() -> None:
    random.seed(SEED)
    meta = _load_jsonl(META)

    text_mods = {"paragraph", "mixed", "heading", "list"}
    good = [r for r in meta
            if len(r.get("content_preview", "") or "") >= 200
            and not _is_repetitive(r.get("content_preview", ""))]
    text_rows = [r for r in good if r.get("modality") in text_mods]
    table_rows = [r for r in good if r.get("modality") == "table"]
    ocr_rows = [r for r in meta if (
        r.get("modality") in ("figure", "handwriting")
        or _domain(r.get("document_name", "")) == "legal" and r.get("modality") == "mixed"
        or "hop-dong" in r.get("document_name", "").lower()
        or ".jpg" in r.get("document_name", "").lower()
        or ".png" in r.get("document_name", "").lower()
    ) and len(r.get("content_preview", "") or "") >= 60]

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in text_rows:
        by_doc[r["document_name"]].append(r)
    by_domain: dict[str, list[str]] = defaultdict(list)
    for doc in by_doc:
        by_domain[_domain(doc)].append(doc)

    for lst in (text_rows, table_rows, ocr_rows):
        random.shuffle(lst)

    def _interleave_by_doc(rows: list[dict]) -> list[dict]:
        """Round-robin across documents so EVERY file is covered and the
        1810-chunk Vinamilk report doesn't crowd out small docs (sfsu, fpt, Hop-Dong)."""
        by: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by[r.get("document_name", "")].append(r)
        for v in by.values():
            random.shuffle(v)
        docs = list(by.keys())
        random.shuffle(docs)
        out, i = [], 0
        while any(by[d] for d in docs):
            d = docs[i % len(docs)]
            if by[d]:
                out.append(by[d].pop())
            i += 1
        return out

    b = Builder()
    docs_present = sorted({r["document_name"] for r in text_rows} | {r["document_name"] for r in ocr_rows})
    print(f"Pools: text={len(text_rows)} table={len(table_rows)} ocr={len(ocr_rows)} | docs={len(docs_present)}", flush=True)

    async def gen_single(kind: str, rows: list[dict], target: int, difficulty: str) -> None:
        made = 0
        for row in rows:
            if made >= target:
                break
            try:
                cases = _extract_json_list(await _call(_grounded_prompt(kind, row)))
            except Exception as exc:
                print(f"  [WARN {kind}] {exc}", flush=True)
                await asyncio.sleep(DELAY); continue
            for c in cases:
                if _valid_query(c.get("query", "")):
                    b.add(task=kind, query=c["query"], evidence=[_ev(row)], difficulty=difficulty,
                          reference_answer=c.get("reference_answer", ""),
                          outline=c.get("expected_answer_outline"),
                          required=c.get("required_facts"), forbidden=c.get("forbidden_claims"))
                    made += 1
                    print(f"  +{kind} ({made}/{target}): {c['query'][:66]}", flush=True)
                    break
            await asyncio.sleep(DELAY)
        print(f"[{kind}] done {made}/{target}", flush=True)

    # 1. factual / 2. summarize / 3. table / 4. ocr — round-robin across all docs
    await gen_single("factual", _interleave_by_doc(text_rows), TARGETS["factual"], "easy")
    await gen_single("summarize", _interleave_by_doc([r for r in text_rows if len(r["content_preview"]) >= 350]),
                     TARGETS["summarize"], "medium")
    await gen_single("table", _interleave_by_doc(table_rows), TARGETS["table"], "easy")
    await gen_single("ocr", _interleave_by_doc(ocr_rows), TARGETS["ocr"], "medium")

    # 5. claim_check — balanced: half TRUE claims, half FALSE claims (test refutation too)
    n_true = TARGETS["claim_check"] // 2
    n_false = TARGETS["claim_check"] - n_true
    claim_pool = _interleave_by_doc(text_rows)
    made = 0
    for row in claim_pool:
        if made >= n_true:
            break
        try:
            cases = _extract_json_list(await _call(_grounded_prompt("claim_check", row)))
        except Exception as exc:
            print(f"  [WARN claim_true] {exc}", flush=True); await asyncio.sleep(DELAY); continue
        for c in cases:
            if _valid_query(c.get("query", "")):
                b.add(task="claim_check", query=c["query"], evidence=[_ev(row)], difficulty="medium",
                      reference_answer=c.get("reference_answer", ""), outline=c.get("expected_answer_outline"),
                      required=c.get("required_facts"), forbidden=c.get("forbidden_claims"),
                      extra={"claim_truth": "true"})
                made += 1; print(f"  +claim_check[T] ({made}/{n_true}): {c['query'][:55]}", flush=True); break
        await asyncio.sleep(DELAY)
    made = 0
    for row in reversed(claim_pool):
        if made >= n_false:
            break
        try:
            cases = _extract_json_list(await _call(_claim_false_prompt(row)))
        except Exception as exc:
            print(f"  [WARN claim_false] {exc}", flush=True); await asyncio.sleep(DELAY); continue
        for c in cases:
            ref = (c.get("reference_answer") or "").strip().lower()
            if _valid_query(c.get("query", "")) and ref.startswith("sai"):
                b.add(task="claim_check", query=c["query"], evidence=[_ev(row)], difficulty="hard",
                      reference_answer=c.get("reference_answer", ""),
                      required=c.get("required_facts"), forbidden=c.get("forbidden_claims"),
                      extra={"claim_truth": "false"})
                made += 1; print(f"  +claim_check[F] ({made}/{n_false}): {c['query'][:55]}", flush=True); break
        await asyncio.sleep(DELAY)
    print(f"[claim_check] done (true+false)", flush=True)

    # 6. cross_lingual
    made = 0
    cross_pool = _interleave_by_doc([r for r in text_rows if r.get("source_language") in ("en", "vi", "mixed")])
    for row in cross_pool:
        if made >= TARGETS["cross_lingual"]:
            break
        src = row.get("source_language", "vi")
        target_lang = "en" if src in ("vi", "mixed") else "vi"
        try:
            cases = _extract_json_list(await _call(_cross_prompt(row, target_lang)))
        except Exception as exc:
            print(f"  [WARN cross] {exc}", flush=True)
            await asyncio.sleep(DELAY); continue
        for c in cases:
            if _valid_query(c.get("query", "")):
                b.add(task="cross_lingual", query=c["query"], evidence=[_ev(row)], difficulty="medium",
                      reference_answer=c.get("reference_answer", ""),
                      outline=c.get("expected_answer_outline"),
                      required=c.get("required_facts"), forbidden=c.get("forbidden_claims"),
                      query_language=target_lang, answer_language=target_lang)
                made += 1
                print(f"  +cross_lingual ({made}/{TARGETS['cross_lingual']}) [{target_lang}]: {c['query'][:54]}", flush=True)
                break
        await asyncio.sleep(DELAY)
    print(f"[cross_lingual] done {made}/{TARGETS['cross_lingual']}", flush=True)

    # 7. compare (same-domain cross-doc, SKIP gate)
    pairs: list[tuple[dict, dict]] = []
    for dom, docs in by_domain.items():
        if dom == "misc" or len(docs) < 2:
            continue
        random.shuffle(docs)
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                a, bb = by_doc[docs[i]], by_doc[docs[j]]
                for _ in range(min(5, len(a), len(bb))):
                    pairs.append((random.choice(a), random.choice(bb)))
    random.shuffle(pairs)
    await gen_pairwise(b, "compare", _compare_prompt, pairs, TARGETS["compare"], "hard")

    # 8. graph_relation (same-doc multi-hop, different pages)
    gpairs: list[tuple[dict, dict]] = []
    for doc, rows in by_doc.items():
        if len(rows) < 2:
            continue
        rr = sorted(rows, key=lambda r: r.get("page") or 0)
        for k in range(0, len(rr) - 1, 2):
            if (rr[k].get("page") or 0) != (rr[k + 1].get("page") or 0):
                gpairs.append((rr[k], rr[k + 1]))
    # Interleave pairs by source doc so Vinamilk doesn't dominate graph_relation.
    _gp_by_doc: dict[str, list] = defaultdict(list)
    for pa in gpairs:
        _gp_by_doc[pa[0].get("document_name", "")].append(pa)
    for v in _gp_by_doc.values():
        random.shuffle(v)
    _gp_docs = list(_gp_by_doc.keys()); random.shuffle(_gp_docs)
    gpairs, _gi = [], 0
    while any(_gp_by_doc[d] for d in _gp_docs):
        d = _gp_docs[_gi % len(_gp_docs)]
        if _gp_by_doc[d]:
            gpairs.append(_gp_by_doc[d].pop())
        _gi += 1
    await gen_pairwise(b, "graph_relation", _graph_prompt, gpairs, TARGETS["graph_relation"], "hard")

    # 9. off_topic_should_refuse (ungrounded; must refuse)
    made = 0
    target = TARGETS["off_topic_should_refuse"]
    seen_off: set[str] = set()
    stall = 0
    while made < target and stall < 4:
        batch = min(8, target - made)
        try:
            cases = _extract_json_list(await _call(_OFF_TOPIC_PROMPT.format(n=batch), temperature=0.7))
        except Exception as exc:
            print(f"  [WARN off_topic] {exc}", flush=True)
            await asyncio.sleep(DELAY); break
        progressed = False
        for c in cases:
            if made >= target:
                break
            q = (c.get("query", "") or "").strip()
            key = q.lower()
            if _valid_query(q) and key not in seen_off:
                seen_off.add(key)
                b.add(task="off_topic_should_refuse", query=q, evidence=[], difficulty="medium",
                      reference_answer="Hệ thống nên từ chối: thông tin này không có trong bộ tài liệu đã tải lên.",
                      forbidden=["Bịa số liệu/thông tin không có trong tài liệu"],
                      expected_behavior="refuse",
                      extra={"reason_out_of_scope": c.get("reason_out_of_scope", "")})
                made += 1; progressed = True
                print(f"  +off_topic ({made}/{target}): {q[:60]}", flush=True)
        await asyncio.sleep(DELAY)
        stall = 0 if progressed else stall + 1
    print(f"[off_topic_should_refuse] done {made}/{target}", flush=True)

    # 10. false_premise (must correct)
    made = 0
    target = TARGETS["false_premise"]
    fp_pool = _interleave_by_doc([r for r in text_rows if any(ch.isdigit() for ch in r.get("content_preview", ""))])
    for row in fp_pool:
        if made >= target:
            break
        try:
            cases = _extract_json_list(await _call(_false_premise_prompt(row), temperature=0.4))
        except Exception as exc:
            print(f"  [WARN false_premise] {exc}", flush=True)
            await asyncio.sleep(DELAY); continue
        for c in cases:
            if _valid_query(c.get("query", "")):
                b.add(task="false_premise", query=c["query"], evidence=[_ev(row)], difficulty="hard",
                      reference_answer=c.get("reference_answer", ""),
                      required=c.get("required_facts"), forbidden=c.get("forbidden_claims"),
                      expected_behavior="answer")
                made += 1
                print(f"  +false_premise ({made}/{target}): {c['query'][:58]}", flush=True)
                break
        await asyncio.sleep(DELAY)
    print(f"[false_premise] done {made}/{target}", flush=True)

    _save_jsonl(b.cases, OUT)
    dist = Counter(c["task_type"] for c in b.cases)
    print(f"\n=== DONE: {len(b.cases)} cases ===", flush=True)
    for k in TARGETS:
        print(f"  {k:26s} {dist.get(k,0):2d}/{TARGETS[k]}", flush=True)
    print(f"Saved -> {OUT}", flush=True)


async def gen_pairwise(b: "Builder", kind: str, prompt_fn, pairs, target: int, difficulty: str) -> None:
    made = 0
    for a, bb in pairs:
        if made >= target:
            break
        try:
            raw = (await _call(prompt_fn(a, bb))).strip()
        except Exception as exc:
            print(f"  [WARN {kind}] {exc}", flush=True)
            await asyncio.sleep(DELAY); continue
        if raw.upper().startswith("SKIP") or raw[:8].upper().count("SKIP"):
            await asyncio.sleep(DELAY); continue
        try:
            cases = _extract_json_list(raw)
        except Exception:
            await asyncio.sleep(DELAY); continue
        for c in cases:
            if _valid_query(c.get("query", "")):
                b.add(task=kind, query=c["query"], evidence=[_ev(a), _ev(bb)], difficulty=difficulty,
                      reference_answer=c.get("reference_answer", ""),
                      outline=c.get("expected_answer_outline"),
                      required=c.get("required_facts"), forbidden=c.get("forbidden_claims"))
                made += 1
                print(f"  +{kind} ({made}/{target}): {c['query'][:64]}", flush=True)
                break
        await asyncio.sleep(DELAY)
    print(f"[{kind}] done {made}/{target}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
