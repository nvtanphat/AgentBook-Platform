"""Structure-adaptive visualization selector.

Picks the visualization mode from MEASURED document structure, never from the
domain name. A legal code and a contract both score high "hierarchy" → tree;
a research paper scores high "semantic" → concept graph. No per-domain code —
ladders/patterns/thresholds live in config/viz_config.yaml.

Signals (all derived from already-extracted data):
  - hierarchy : fraction of headings that fit a level ladder or numbering
  - reference : internal cross-references ("Điều 5", "[3]") per text block
  - semantic  : typed concept relations per displayed entity
  - temporal  : time/date/event entities per total entities

Public API:
  detect_structure(...) -> StructureSignals
  build_hierarchy_tree(...) -> list[MindmapNode]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.schemas.graph import GraphEdge, GraphNode
from src.schemas.mindmap import MindmapNode

# ── Built-in defaults (overridden by viz_config.yaml) ─────────────────────────

_DEFAULT_THRESHOLDS = {
    "hierarchy_min": 0.30,
    "reference_min": 0.15,
    "semantic_min": 0.20,
    "temporal_min": 0.20,
}

_DEFAULT_LADDERS = {
    "vi_legal": ["phần", "chương", "mục", "tiểu mục", "điều", "khoản", "điểm"],
    "generic_doc": ["part", "chapter", "section", "subsection", "article", "clause"],
}

_DEFAULT_REFERENCE_PATTERNS = [
    r"điều\s+\d+",
    r"khoản\s+\d+",
    r"article\s+\d+",
    r"section\s+\d+(?:\.\d+)*",
    r"\[\d+\]",
]

_TEMPORAL_ENTITY_TYPES = frozenset({"time", "date", "event"})

# Leading enumeration like "1.", "1.2", "1.2.3", "IV.", "a)" → numbering depth.
_NUMBERING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+")


@dataclass
class StructureSignals:
    hierarchy: float = 0.0
    reference: float = 0.0
    semantic: float = 0.0
    temporal: float = 0.0
    recommended_mode: str = "concept_graph"
    # Raw counts for transparency / debugging in the API response.
    counts: dict = field(default_factory=dict)


def _thresholds(viz_config: dict) -> dict:
    cfg = dict(_DEFAULT_THRESHOLDS)
    cfg.update(viz_config.get("thresholds", {}) or {})
    return cfg


def _ladders(viz_config: dict) -> list[list[str]]:
    raw = viz_config.get("hierarchy_ladders") or _DEFAULT_LADDERS
    return [[w.lower() for w in ladder] for ladder in raw.values()]


def _reference_patterns(viz_config: dict) -> list[re.Pattern]:
    raw = viz_config.get("reference_patterns") or _DEFAULT_REFERENCE_PATTERNS
    out: list[re.Pattern] = []
    for pat in raw:
        try:
            out.append(re.compile(pat, re.IGNORECASE))
        except re.error:
            continue
    return out


def _heading_level(text: str, ladders: list[list[str]]) -> int | None:
    """Infer a 0-based hierarchy level for a heading, or None if it doesn't fit.

    Strategy: ladder keyword rank first (most reliable for structured docs),
    then numbering depth (generic), else None (free-form heading).
    """
    lower = text.strip().lower()
    if not lower:
        return None
    for ladder in ladders:
        for rank, keyword in enumerate(ladder):
            # match keyword at the start, optionally followed by a number
            if re.match(rf"^{re.escape(keyword)}\b", lower):
                return rank
    m = _NUMBERING_RE.match(text)
    if m:
        return m.group(1).count(".")  # "1"→0, "1.2"→1, "1.2.3"→2
    return None


def detect_structure(
    *,
    headings: list[str],
    total_text_blocks: int,
    block_texts: list[str],
    entities: list,
    relations: list,
    viz_config: dict,
) -> StructureSignals:
    """Compute the 4 structural signals and pick the dominant viz mode."""
    ladders = _ladders(viz_config)
    ref_patterns = _reference_patterns(viz_config)
    thresholds = _thresholds(viz_config)

    # ── hierarchy: fraction of headings that fit a level ladder/numbering ──────
    fitted = sum(1 for h in headings if _heading_level(h, ladders) is not None)
    hierarchy = (fitted / len(headings)) if headings else 0.0

    # ── reference: internal cross-refs per text block ─────────────────────────
    ref_hits = 0
    for text in block_texts:
        for pat in ref_patterns:
            ref_hits += len(pat.findall(text or ""))
    reference = (ref_hits / total_text_blocks) if total_text_blocks else 0.0

    # ── semantic: typed concept relations per displayed entity ────────────────
    # Co-occurrence relations don't count — they exist for any corpus and aren't
    # a sign of a genuine concept web.
    structural_rel = {
        "co_occurs_in_block", "co_occurs_on_page", "section_contains",
        "mentioned_in_block", "mentioned_in_event", "has_caption", "caption_of",
    }
    typed_rel = sum(
        1 for r in relations
        if (getattr(r, "relation_type", "") or "") not in structural_rel
    )
    semantic = (typed_rel / len(entities)) if entities else 0.0

    # ── temporal: time/date/event entities per total entities ─────────────────
    temporal_entities = sum(
        1 for e in entities
        if (getattr(e, "entity_type", "") or "").lower() in _TEMPORAL_ENTITY_TYPES
    )
    temporal = (temporal_entities / len(entities)) if entities else 0.0

    # ── mode selection: dominant signal above its floor ───────────────────────
    candidates = [
        ("hierarchy", hierarchy, thresholds["hierarchy_min"]),
        ("citation_network", reference, thresholds["reference_min"]),
        ("concept_graph", semantic, thresholds["semantic_min"]),
        ("timeline", temporal, thresholds["temporal_min"]),
    ]
    eligible = [(mode, score) for mode, score, floor in candidates if score >= floor]
    if eligible:
        recommended = max(eligible, key=lambda kv: kv[1])[0]
    else:
        recommended = viz_config.get("default_mode", "concept_graph")

    return StructureSignals(
        hierarchy=round(hierarchy, 3),
        reference=round(reference, 3),
        semantic=round(semantic, 3),
        temporal=round(temporal, 3),
        recommended_mode=recommended,
        counts={
            "headings": len(headings),
            "headings_fitted": fitted,
            "text_blocks": total_text_blocks,
            "reference_hits": ref_hits,
            "typed_relations": typed_rel,
            "entities": len(entities),
            "temporal_entities": temporal_entities,
        },
    )


@dataclass
class HeadingItem:
    """A heading block to place in the hierarchy tree."""
    text: str
    material_id: str
    page: int | None
    block_id: str | None


def _short(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def _article_label(text: str, max_words: int = 6) -> str:
    """Truncate article heading to ≤ max_words so the frontend label filter passes.

    Frontend cleanGraphLabel rejects labels with > 6 words. Article headings
    like "Điều 59. Nguyên tắc giải quyết tài sản..." are 14 words and get
    dropped. We keep the article number ("Điều N.") + first few content words.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def build_hierarchy_tree(
    *,
    root_topic: str,
    headings: list[HeadingItem],
    viz_config: dict,
) -> list[MindmapNode]:
    """Build a nested section tree from ordered heading blocks.

    Levels are inferred per heading (ladder keyword rank → numbering depth →
    nest-under-current fallback). A stack tracks open ancestors so each heading
    attaches under the nearest shallower one.
    """
    ladders = _ladders(viz_config)
    limits = viz_config.get("limits", {}) or {}
    max_nodes = int(limits.get("max_tree_nodes", 300))
    max_depth = int(limits.get("max_tree_depth", 6))

    root = MindmapNode(id="root", label=_short(root_topic, 60), entity_type="root")
    # stack of (level, node); root sits at sentinel level -1
    stack: list[tuple[int, MindmapNode]] = [(-1, root)]
    emitted = 0
    last_level = 0

    for idx, item in enumerate(headings):
        if emitted >= max_nodes:
            break
        label = _short(item.text)
        if len(label) < 2:
            continue
        level = _heading_level(item.text, ladders)
        if level is None:
            level = last_level + 1  # free-form heading nests under current section
        level = min(level, max_depth - 1)
        last_level = level

        # Pop ancestors that are at the same or deeper level than this heading.
        while len(stack) > 1 and stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][1]

        citations = []
        if item.block_id:
            citations = [{
                "material_id": item.material_id,
                "page": item.page or 0,
                "block_id": item.block_id,
            }]
        node = MindmapNode(
            id=f"section:{item.material_id}:{idx}",
            label=label,
            entity_type="section",
            citations=citations,
        )
        parent.children.append(node)
        stack.append((level, node))
        emitted += 1

    return root.children


def prune_tree_to_focus(nodes: list[MindmapNode], focus_block_ids: set[str]) -> list[MindmapNode]:
    """Keep only branches that contain a focused (cited) heading.

    A node survives if its own heading block_id is in `focus_block_ids` (then it
    is marked as cited) OR any descendant survives (kept as ancestor context).
    Used so the "verify-by-graph" view shows just the Điều backing the answer
    plus their parent chapters, instead of the whole document tree.
    """

    def _visit(node: MindmapNode) -> MindmapNode | None:
        own_block_id = node.citations[0].get("block_id") if node.citations else None
        is_cited = bool(own_block_id) and own_block_id in focus_block_ids
        kept_children = [c for c in (_visit(ch) for ch in node.children) if c is not None]
        if not is_cited and not kept_children:
            return None
        return node.model_copy(update={
            "children": kept_children,
            "summary": "✓ được trích dẫn" if is_cited else node.summary,
        })

    return [n for n in (_visit(node) for node in nodes) if n is not None]


# Article number from a heading like "Điều 33. Tài sản chung…" or a reference
# "…quy định tại Điều 5" → captures "33" / "5".
_ARTICLE_RE = re.compile(r"điều\s+(\d+)", re.IGNORECASE)


def _query_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from query text for article heading match.

    Keeps Vietnamese diacritics intact — "kết hôn" must match "kết hôn" in
    headings. Folding to ASCII loses distinction between "hôn" (marriage) and
    generic syllables, making the match useless.
    """
    _VN_STOPS = {
        # Vietnamese function words (with diacritics) — too generic to help
        "của", "và", "là", "có", "được", "không", "theo", "trong", "với",
        "khi", "đã", "này", "đó", "các", "để", "trên", "tại", "về", "một",
        "hai", "ba", "đúng", "phải", "luôn", "rằng", "đến", "từ", "cho",
        "bằng", "nhau", "thì", "hay", "nếu", "vì", "do", "mà",
        # English stopwords
        "the", "and", "or", "is", "are", "not", "this", "that",
    }
    words = re.split(r"[^\wÀ-ỹĐđ]+", text, flags=re.UNICODE)
    return {w.lower() for w in words if len(w) >= 3 and w.lower() not in _VN_STOPS}


def build_citation_network(
    *,
    sections: list[tuple[HeadingItem, str]],
    viz_config: dict,
    focus_block_ids: set[str] | None = None,
    focus_query_text: str | None = None,
    use_query_text_signal: bool = True,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Build a real node-edge graph for legal docs: one node per Điều, an edge
    "dẫn chiếu" whenever an article's text references another article.

    Focus is determined by TWO signals (OR-combined):
    1. Citation block_ids → map to the article the cited block belongs to.
    2. Query-text keyword match → articles whose title contains query keywords
       (catches "Điều 8 Điều kiện kết hôn" when query asks about "tuổi kết hôn"
        but retriever didn't find it directly).
    """
    focus_block_ids = focus_block_ids or set()
    query_kws = _query_keywords(focus_query_text) if focus_query_text else set()
    limits = viz_config.get("limits", {}) or {}
    max_edges = int(limits.get("max_reference_edges", 200))

    # One node per distinct article number (first heading wins as the label).
    node_by_art: dict[str, HeadingItem] = {}
    order: list[str] = []
    for heading, _text in sections:
        m = _ARTICLE_RE.search(heading.text)
        if not m:
            continue
        art = m.group(1)
        if art not in node_by_art:
            node_by_art[art] = heading
            order.append(art)

    # Edges from references inside each article's own body text.
    raw_edges: list[tuple[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()
    for heading, text in sections:
        m = _ARTICLE_RE.search(heading.text)
        if not m:
            continue
        src = m.group(1)
        for rm in _ARTICLE_RE.finditer(text or ""):
            tgt = rm.group(1)
            if tgt == src or tgt not in node_by_art:
                continue
            key = (src, tgt)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            raw_edges.append(key)
            if len(raw_edges) >= max_edges:
                break
        if len(raw_edges) >= max_edges:
            break

    # Gather all focused articles from BOTH signals before building keep_arts.
    # (keep_arts must be built after all signal collection — order matters.)

    # Signal 1: citation block_ids → article containing that block.
    cited_arts: set[str] = set()
    if focus_block_ids:
        for art, heading in node_by_art.items():
            if heading.block_id and heading.block_id in focus_block_ids:
                cited_arts.add(art)

    # Signal 2: query keyword match against article heading text.
    # Only active in explore mode (use_query_text_signal=True). In verify mode,
    # Signal 1 (citation blocks) is sufficient — the graph shows exactly the
    # Điều the answer cited, nothing more.
    if use_query_text_signal and query_kws:
        scored: list[tuple[int, str]] = []
        for art, heading in node_by_art.items():
            head_kws = {w.lower() for w in re.split(r"[^\wÀ-ỹĐđ]+", heading.text, flags=re.UNICODE) if len(w) >= 3}
            score = len(query_kws & head_kws)
            if score >= 2:
                scored.append((score, art))
        # Dynamic threshold: include all articles scoring ≥ max(2, top_score//2)
        # so the MOST relevant articles always appear even when globally rare.
        # Hard cap at 25 to avoid flooding when many headings mention all keywords.
        # Build 2-gram phrases from query for phrase-level heading match.
        # "kết hôn" as a phrase in the query must match "kết hôn" in a heading
        # even when the unigram score is low (short heading = few overlapping words).
        query_words_list = [w.lower() for w in re.split(r"[^\wÀ-ỹĐđ]+", focus_query_text or "", flags=re.UNICODE) if len(w) >= 2]
        query_bigrams = {f"{query_words_list[i]} {query_words_list[i+1]}" for i in range(len(query_words_list)-1)}
        if scored:
            # Rank by unigram overlap first, cap at 20.
            top20 = {art for _, art in sorted(scored, reverse=True)[:20]}
            for art in top20:
                cited_arts.add(art)
        # Bigram match: always include if heading contains a query bigram (≥6 chars).
        for art, heading in node_by_art.items():
            head_lower = heading.text.lower()
            if any(bg in head_lower and len(bg) >= 6 for bg in query_bigrams):
                cited_arts.add(art)

    # Build keep_arts AFTER both signals are collected: cited + 1-hop neighbours.
    keep_arts: set[str] | None = None
    if cited_arts:
        keep_arts = set(cited_arts)
        for src, tgt in raw_edges:
            if src in cited_arts:
                keep_arts.add(tgt)
            if tgt in cited_arts:
                keep_arts.add(src)

    def _ref(h: HeadingItem) -> list[dict]:
        return [{"material_id": h.material_id, "page": h.page or 0, "block_id": h.block_id or ""}]

    nodes: list[GraphNode] = []
    for art in order:
        if keep_arts is not None and art not in keep_arts:
            continue
        h = node_by_art[art]
        nodes.append(GraphNode(
            id=f"dieu:{art}",
            label=_article_label(h.text),
            type="article",
            confidence=None,
            is_focused=art in cited_arts,
            evidence_refs=_ref(h),
        ))

    visible = {n.id for n in nodes}
    edges: list[GraphEdge] = []
    for src, tgt in raw_edges:
        sid, tid = f"dieu:{src}", f"dieu:{tgt}"
        if sid not in visible or tid not in visible:
            continue
        edges.append(GraphEdge(
            source=sid,
            target=tid,
            relation_type="dẫn chiếu",
            source_label=_article_label(node_by_art[src].text),
            target_label=_article_label(node_by_art[tgt].text),
            confidence=None,
            evidence_count=1,
            evidence_refs=_ref(node_by_art[src]),
        ))
    return nodes, edges
