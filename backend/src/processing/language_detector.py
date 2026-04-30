from __future__ import annotations

import re
import unicodedata
from collections import Counter

_MIN_CHARS = 8
_VIETNAMESE_SPECIALS = {chr(codepoint) for codepoint in (0x0111, 0x0110, 0x01A1, 0x01A0, 0x01B0, 0x01AF)}
_VIETNAMESE_WORDS = {
    "ban",
    "bai",
    "cac",
    "cho",
    "cua",
    "duoc",
    "dung",
    "du",
    "giai",
    "hoc",
    "khong",
    "la",
    "mot",
    "nguoi",
    "nhieu",
    "nhung",
    "phan",
    "qua",
    "quy",
    "sinh",
    "thuc",
    "trinh",
    "trong",
    "voi",
}


def _has_combining_mark(char: str) -> bool:
    return any(unicodedata.category(part) == "Mn" for part in unicodedata.normalize("NFD", char))


def _has_vietnamese_signal(text: str) -> bool:
    lowered = text.lower()
    if any(char in _VIETNAMESE_SPECIALS for char in lowered):
        return True
    if sum(_has_combining_mark(char) for char in lowered) >= 2:
        return True
    ascii_words = re.findall(r"[a-z]+", unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode("ascii"))
    if not ascii_words:
        return False
    vi_hits = sum(word in _VIETNAMESE_WORDS for word in ascii_words)
    return vi_hits >= 3 and vi_hits / len(ascii_words) >= 0.12


def detect_block_language(text: str, fallback: str = "unknown") -> str:
    """Detect the dominant source language for one parsed block."""
    stripped = text.strip()
    if len(stripped) < _MIN_CHARS:
        return fallback
    if _has_vietnamese_signal(stripped):
        return "vi"
    try:
        from langdetect import detect  # type: ignore

        lang = detect(stripped)
    except Exception:
        return "en" if re.search(r"[A-Za-z]", stripped) else fallback
    return lang if lang in {"vi", "en"} else fallback


def detect_document_language(texts: list[str], fallback: str = "unknown") -> str:
    """Aggregate block-level detections; return mixed when no language dominates."""
    counts = Counter(detect_block_language(text, fallback="skip") for text in texts)
    counts.pop("skip", None)
    if not counts:
        return fallback
    if len(counts) > 1:
        total = sum(counts.values())
        _, top_count = counts.most_common(1)[0]
        if top_count / total < 0.8:
            return "mixed"
    return counts.most_common(1)[0][0]

