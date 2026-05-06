from __future__ import annotations

import re
import unicodedata

# Tier 1A: instant replies for unambiguous conversational queries.
_INSTANT_REPLY_MAP: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"^(xin chào|chào\s*(bạn|noelys|mọi người)?|hello(\s+there)?|hi(\s+there)?|hey(\s+there)?|alo+|helo)\s*[!.]*$", re.IGNORECASE),
        "Xin chào! Tôi là Noelys, trợ lý học tập của Noelys. Bạn muốn hỏi gì về tài liệu hôm nay?",
    ),
    (
        re.compile(r"^chào buổi sáng\s*[!.]*$", re.IGNORECASE),
        "Chào buổi sáng! Hôm nay bạn muốn ôn tập hay tìm hiểu chủ đề gì?",
    ),
    (
        re.compile(r"^good\s*morning\s*[!.]*$", re.IGNORECASE),
        "Good morning! What would you like to study today?",
    ),
    (
        re.compile(r"^chào buổi (trưa|chiều|tối)\s*[!.]*$", re.IGNORECASE),
        "Xin chào! Tôi có thể giúp gì cho bạn?",
    ),
    (
        re.compile(r"^good\s*(afternoon|evening)\s*[!.]*$", re.IGNORECASE),
        "Hello! How can I help you today?",
    ),
    (
        re.compile(r"^(tạm biệt|bye+|goodbye|hẹn gặp lại|see you)\s*[!.]*$", re.IGNORECASE),
        "Tạm biệt! Chúc bạn học tập hiệu quả. Hẹn gặp lại!",
    ),
    (
        re.compile(r"^good\s*night\s*[!.]*$", re.IGNORECASE),
        "Good night! See you next time.",
    ),
    (
        re.compile(r"^chúc ngủ ngon\s*[!.]*$", re.IGNORECASE),
        "Chúc bạn ngủ ngon! Hẹn gặp lại bạn lần sau nhé.",
    ),
    (
        re.compile(r"^(cảm ơn|cám ơn|thanks?|thank you|xin cảm ơn)(.*)(rồi|nhé|nha|thôi|vậy|ok|oke)?\s*[!.]*$", re.IGNORECASE),
        "Không có gì! Nếu bạn còn câu hỏi nào, cứ hỏi tôi nhé.",
    ),
    (
        re.compile(r"^(ok|okay|oke|được|rồi|được rồi|hiểu rồi|alright|got it|sure|vâng|dạ|ừ+)\s*[!.]*$", re.IGNORECASE),
        "Tốt! Bạn cần thêm thông tin gì không?",
    ),
    (
        re.compile(r"^(không có gì|you'?re welcome|de gi|không dám)\s*[!.]*$", re.IGNORECASE),
        "Nếu cần gì thêm, bạn cứ hỏi nhé!",
    ),
]


_CHITCHAT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(bạn là ai|mày là ai|you are who|who are you)\b", re.IGNORECASE),
    re.compile(r"\b(bạn tên gì|tên của bạn|what('s| is) your name)\b", re.IGNORECASE),
    re.compile(r"\b(bạn làm được gì|bạn có thể làm gì|what can you do|bạn có thể giúp gì)\b", re.IGNORECASE),
    re.compile(r"\b(bạn là (gì|loại gì|AI|robot)|what (are|kind of) (are )?you)\b", re.IGNORECASE),
    re.compile(r"\b(noelys là gì|bạn hoạt động như thế nào)\b", re.IGNORECASE),
    re.compile(r"\b(bạn có khỏe không|bạn khỏe không|bạn ổn không|how are you|how('s| is) it going|you okay)\b", re.IGNORECASE),
    re.compile(r"\b(bạn đang làm gì|bạn đang nghĩ gì|what are you doing|what('re| are) you thinking)\b", re.IGNORECASE),
    re.compile(r"^(xin lỗi|xin lỗi bạn|sorry|pardon|thông cảm)\b", re.IGNORECASE),
    re.compile(r"\b(bạn (giỏi|thông minh|tuyệt|hay|xịn|thật sự tốt)|giỏi lắm|hay quá|tuyệt (quá|vời)|quá xịn|bạn thật tuyệt)\b", re.IGNORECASE),
    re.compile(r"^(wow|wao|ồ|ôi|oa+|tuyệt|hay|xịn|ngon)\s*[!.]*$", re.IGNORECASE),
    re.compile(r"^(kể chuyện cười|kể joke|tell (me )?a joke|joke|kể cho tôi nghe)\b", re.IGNORECASE),
    re.compile(r"^(good\s*(morning|afternoon|evening|night))\b", re.IGNORECASE),
    re.compile(r"^(chào buổi (sáng|trưa|chiều|tối))\b", re.IGNORECASE),
    re.compile(r"\b(tạm biệt|hẹn gặp lại|see you (later|soon|tomorrow))\b", re.IGNORECASE),
    re.compile(r"^(cảm ơn|cám ơn|thanks?|thank you|xin cảm ơn)\b", re.IGNORECASE),
    re.compile(r"^(không có gì|you'?re welcome)\b", re.IGNORECASE),
    re.compile(r"^(ok|okay|oke|được|rồi|được rồi|hiểu rồi|alright|got it|sure)\s*[.!]*$", re.IGNORECASE),
    re.compile(r"^(vâng|dạ|ừ+|uh[ -]?huh)\s*[.!]*$", re.IGNORECASE),
    re.compile(r"^(cảm ơn|thanks?).{0,40}(rồi|nhé|nha|thôi|vậy|ok|oke)\s*[.!]*$", re.IGNORECASE),
]


def get_instant_reply(query: str) -> str | None:
    """Return a pre-written reply for common unambiguous chitchat, or None to fall through to LLM."""
    text = query.strip()
    for pattern, reply in _INSTANT_REPLY_MAP:
        if pattern.search(text):
            return reply
    return None


def _ascii_fold(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("\u0111", "d").replace("\u0110", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\w\s'?!.-]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _is_chitchat_normalized(text: str) -> bool:
    normalized = _ascii_fold(text)
    if not normalized:
        return False

    instant_patterns = [
        r"^(xin chao|chao( ban| noelys| moi nguoi)?|hello( there)?|hi( there)?|hey( there)?|alo+|helo)[!.]*$",
        r"^chao buoi (sang|trua|chieu|toi)[!.]*$",
        r"^good\s*(morning|afternoon|evening|night)[!.]*$",
        r"^(tam biet|bye+|goodbye|hen gap lai|see you)[!.]*$",
        r"^(cam on|thanks?|thank you|xin cam on).*$",
        r"^(ok|okay|oke|duoc|roi|duoc roi|hieu roi|alright|got it|sure|vang|da|u+)[!.]*$",
        r"^(khong co gi|you're welcome|youre welcome|de gi|khong dam)[!.]*$",
    ]
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in instant_patterns):
        return True

    conversational_patterns = [
        r"\b(ban la ai|may la ai|who are you|you are who)\b",
        r"\b(ban ten gi|ten cua ban|what('s| is) your name)\b",
        r"\b(ban lam duoc gi|ban co the lam gi|what can you do|ban co the giup gi)\b",
        r"\b(ban la gi|ban la loai gi|ban la ai|what are you)\b",
        r"\b(noelys la gi|ban hoat dong nhu the nao)\b",
        r"\b(ban co khoe khong|ban khoe khong|ban on khong|how are you|how's it going|how is it going|you okay)\b",
        r"\b(ban dang lam gi|ban dang nghi gi|what are you doing|what are you thinking)\b",
        r"^(xin loi|sorry|pardon|thong cam)\b",
        r"\b(ban (gioi|thong minh|tuyet|hay|xin|that su tot)|gioi lam|hay qua|tuyet qua|tuyet voi|qua xin|ban that tuyet)\b",
        r"^(wow|wao|o|oi|oa+|tuyet|hay|xin|ngon)[!.]*$",
        r"^(ke chuyen cuoi|ke joke|tell (me )?a joke|joke|ke cho toi nghe)\b",
    ]
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in conversational_patterns)


def is_chitchat(query: str) -> bool:
    """Return True when the query is conversational and does not require RAG retrieval."""
    text = query.strip()
    if not text:
        return False
    if _is_chitchat_normalized(text):
        return True
    return any(pattern.search(text) for pattern in _CHITCHAT_PATTERNS)
