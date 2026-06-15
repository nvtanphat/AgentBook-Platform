"""Shared, Unicode-safe slug utilities for knowledge-graph node IDs.

Every component that produces an entity node ID (event_extractor,
cross_modal_linker, semantic_relation_extractor, graph_quality_gate) and the
query-time graph endpoint MUST use these helpers so relation source_id/target_id
always match the entity slug derived at read time.

Vietnamese is ASCII-folded first ("Trí tuệ nhân tạo" → "tri-tue-nhan-tao")
instead of being shredded by a naive ``[^a-z0-9]`` substitution
("tr-tu-nh-n-t-o"), which previously produced unreadable, collision-prone IDs.
"""

from __future__ import annotations

import re
import unicodedata

# Vietnamese đ/Đ are single code points (U+0111/U+0110) that do NOT decompose
# under NFD, so map them explicitly before stripping combining marks.
_BASE_LETTER_MAP = str.maketrans({"đ": "d", "Đ": "D", "ð": "d", "Ð": "D"})


def ascii_fold(text: str) -> str:
    """Fold diacritics to ASCII: đ→d, NFD-decompose, drop combining marks."""
    text = text.translate(_BASE_LETTER_MAP)
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def slugify(name: str) -> str:
    """Stable bare slug: ASCII-folded, lowercase, hyphen-separated.

    "Trí tuệ nhân tạo" → "tri-tue-nhan-tao"; empty/garbage → "unknown".
    """
    folded = ascii_fold(name).lower()
    return re.sub(r"[^a-z0-9]+", "-", folded).strip("-") or "unknown"


def entity_node_id(name: str) -> str:
    """Canonical entity node ID used as relation source_id/target_id."""
    return f"entity:{slugify(name)}"
