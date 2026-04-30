from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Tier 1A — Instant reply map (no LLM, sub-millisecond)
# Match against the query and return a pre-written response directly.
# Only include patterns where the intent is completely unambiguous and the
# reply is equally appropriate for all surface variants.
# ---------------------------------------------------------------------------

_INSTANT_REPLY_MAP: list[tuple[re.Pattern[str], str]] = [
    # Greetings
    (
        re.compile(r"^(xin chào|chào\s*(bạn|prism|mọi người)?|hello(\s+there)?|hi(\s+there)?|hey(\s+there)?|alo+|helo)\s*[!.]*$", re.IGNORECASE),
        "Xin chào! Tôi là Prism, trợ lý học tập của AgentBook. Bạn muốn hỏi gì về tài liệu hôm nay?",
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
    # Farewells
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
    # Gratitude
    (
        re.compile(r"^(cảm ơn|cám ơn|thanks?|thank you|xin cảm ơn)(.*)(rồi|nhé|nha|thôi|vậy|ok|oke)?\s*[!.]*$", re.IGNORECASE),
        "Không có gì! Nếu bạn còn câu hỏi nào, cứ hỏi tôi nhé.",
    ),
    (
        re.compile(r"^(cảm ơn|cám ơn|thanks?|thank you|xin cảm ơn)\s*[!.]*$", re.IGNORECASE),
        "Không có gì! Nếu bạn còn câu hỏi nào, cứ hỏi tôi nhé.",
    ),
    # Simple acknowledgements
    (
        re.compile(r"^(ok|okay|oke|được|rồi|được rồi|hiểu rồi|alright|got it|sure|vâng|dạ|ừ+)\s*[!.]*$", re.IGNORECASE),
        "Tốt! Bạn cần thêm thông tin gì không?",
    ),
    # You're welcome
    (
        re.compile(r"^(không có gì|you'?re welcome|de gi|không dám)\s*[!.]*$", re.IGNORECASE),
        "Nếu cần gì thêm, bạn cứ hỏi nhé!",
    ),
]

# ---------------------------------------------------------------------------
# Tier 1B — Chitchat signal patterns (needs LLM response from chitchat.txt)
# These identify conversational intent but require LLM to craft a good reply.
# ---------------------------------------------------------------------------

_CHITCHAT_PATTERNS: list[re.Pattern[str]] = [
    # Identity / capability questions
    re.compile(r"\b(bạn là ai|mày là ai|you are who|who are you)\b", re.IGNORECASE),
    re.compile(r"\b(bạn tên gì|tên của bạn|what('s| is) your name)\b", re.IGNORECASE),
    re.compile(r"\b(bạn làm được gì|bạn có thể làm gì|what can you do|bạn có thể giúp gì)\b", re.IGNORECASE),
    re.compile(r"\b(bạn là (gì|loại gì|AI|robot)|what (are|kind of) (are )?you)\b", re.IGNORECASE),
    re.compile(r"\b(agentbook là gì|prism là gì|bạn hoạt động như thế nào)\b", re.IGNORECASE),
    # How are you / feelings
    re.compile(r"\b(bạn có khỏe không|bạn khỏe không|bạn ổn không|how are you|how('s| is) it going|you okay)\b", re.IGNORECASE),
    re.compile(r"\b(bạn đang làm gì|bạn đang nghĩ gì|what are you doing|what('re| are) you thinking)\b", re.IGNORECASE),
    # Apologies
    re.compile(r"^(xin lỗi|xin lỗi bạn|sorry|pardon|thông cảm)\b", re.IGNORECASE),
    # Compliments / reactions
    re.compile(r"\b(bạn (giỏi|thông minh|tuyệt|hay|xịn|thật sự tốt)|giỏi lắm|hay quá|tuyệt (quá|vời)|quá xịn|bạn thật tuyệt)\b", re.IGNORECASE),
    re.compile(r"^(wow|wao|ồ|ôi|oa+|tuyệt|hay|xịn|ngon)\s*[!.]*$", re.IGNORECASE),
    # Jokes / fun
    re.compile(r"^(kể chuyện cười|kể joke|tell (me )?a joke|joke|kể cho tôi nghe)\b", re.IGNORECASE),
    # Greetings (need LLM for non-trivial variants)
    re.compile(r"^(good\s*(morning|afternoon|evening|night))\b", re.IGNORECASE),
    re.compile(r"^(chào buổi (sáng|trưa|chiều|tối))\b", re.IGNORECASE),
    # Farewells (non-trivial variants)
    re.compile(r"\b(tạm biệt|hẹn gặp lại|see you (later|soon|tomorrow))\b", re.IGNORECASE),
    # Gratitude (non-trivial variants — longer thank-you messages)
    re.compile(r"^(cảm ơn|cám ơn|thanks?|thank you|xin cảm ơn)\b", re.IGNORECASE),
    re.compile(r"^(không có gì|you'?re welcome)\b", re.IGNORECASE),
    # Affirmations that are context-dependent
    re.compile(r"^(ok|okay|oke|được|rồi|được rồi|hiểu rồi|alright|got it|sure)\s*[.!]*$", re.IGNORECASE),
    re.compile(r"^(vâng|dạ|ừ+|uh[ -]?huh)\s*[.!]*$", re.IGNORECASE),
    # Thanks + completion
    re.compile(r"^(cảm ơn|thanks?).{0,40}(rồi|nhé|nha|thôi|vậy|ok|oke)\s*[.!]*$", re.IGNORECASE),
]

# Domain signals — presence means the query is NOT pure chitchat.
_DOMAIN_SIGNALS = re.compile(
    r"\b(là gì|nghĩa là|định nghĩa|giải thích|tại sao|vì sao|như thế nào|how|what|why|when|where|which|define|explain|describe|compare)\b",
    re.IGNORECASE,
)


def get_instant_reply(query: str) -> str | None:
    """Return a pre-written reply for common unambiguous chitchat, or None to fall through to LLM."""
    text = query.strip()
    for pattern, reply in _INSTANT_REPLY_MAP:
        if pattern.search(text):
            return reply
    return None


def is_chitchat(query: str) -> bool:
    """Return True when the query is conversational and does not require RAG retrieval."""
    text = query.strip()
    if not text:
        return False

    # Fast path: instant reply map implies chitchat
    for pattern, _ in _INSTANT_REPLY_MAP:
        if pattern.search(text):
            return True

    for pattern in _CHITCHAT_PATTERNS:
        if pattern.search(text):
            return True

    return False
