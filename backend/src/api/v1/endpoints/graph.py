from __future__ import annotations

import asyncio
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict

from beanie import PydanticObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.core.config import Settings, get_settings
from src.core.model_factory import build_llm
from src.dependencies import get_app_settings, verify_owner_access
from src.models.chunk import Chunk
from src.models.knowledge_graph import Entity, Relation
from src.models.material import Material, MaterialPageDocument
from src.rag.graph_builder import build_digraph, compute_communities, compute_degrees, compute_pagerank
from src.rag.structure_detector import (
    HeadingItem,
    build_citation_network,
    build_hierarchy_tree,
    detect_structure,
    prune_tree_to_focus,
)
from src.schemas.common import APIResponse
from src.schemas.graph import AutoVizResponse, GraphEdge, GraphNode, GraphResponse, MindmapRequest, VizSignals
from src.schemas.mindmap import MindmapNode, MindmapResponse

router = APIRouter(prefix="/graph", tags=["graph"])

_CROSS_MODAL_TYPES = frozenset({"table", "figure", "equation"})

# Relation types that carry no concept-to-concept information and
# pollute the concept-graph viz. Section nesting and entity-to-block
# mentions are dropped; co_located_with / adjacent_context survive
# because they at least connect two concept entities and are the only
# signal available until LLM-based semantic relation extraction lands.
_STRUCTURAL_RELATION_TYPES = frozenset({
    "section_contains",      # block hierarchy, not concept relation
    "mentioned_in_block",    # entity → chunk, not entity → entity
    "mentioned_in_event",    # entity → event scaffold
    "has_caption",           # figure ↔ caption text
    "caption_of",            # caption text ↔ figure
})
_NOISY_ENTITY_LABELS = frozenset(
    {
        "bang",
        "bảng",
        "caption",
        "cau",
        "câu",
        "chart",
        "checklist",
        "converted",
        "docx",
        "fail",
        "file word",
        "hinh",
        "hình",
        "increases",
        "jpg",
        "jpeg",
        "llm",
        "metadata",
        "nguon",
        "nguồn",
        "ocr",
        "ocr engine",
        "ocr engine png",
        "parser",
        "pass",
        "pdf",
        "png",
        "png ocr",
        "pptx",
        "randomly",
        "section",
        "slide",
        "test",
        "text",
        "trong",
        "txt",
        "vlm",
        "word",
        "xlsx",
    }
)
_NOISY_ENTITY_WORDS = frozenset(
    {
        "adds",
        "description",
        "fail",
        "file",
        "pass",
        "source",
        "sources",
        "stabilizes",
        "stops",
        "technique",
    }
)
_FORMAT_ENTITY_WORDS = frozenset({"docx", "jpg", "jpeg", "llm", "ocr", "pdf", "png", "pptx", "text", "vlm", "xlsx"})
_TECHNICAL_GRAPH_LABELS = frozenset(
    {
        "answer chunk",
        "chunk",
        "evidence",
        "key points",
        "metadata",
        "source",
        "sources",
    }
)
_BAD_ENTITY_LABEL_RE = re.compile(
    r"(?:jocaled|dalch|uon|nornlalizal|regulariza|techniq|trace viewer question)",
    re.IGNORECASE,
)
_BAD_MINDMAP_LABEL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:pdf|pptx|docx|png|jpg|jpeg|xlsx)\b.*\b(?:nguá»“n|nguon|source)\b",
        r"\b(?:docx|pdf|png|pptx|xlsx){2,}\b",
        r"\b(?:jocaled|dalch|uon|nornlalizal|regulariza)\b",
        r"\b(?:key points|metadata)\b$",
        r"^[^\W\d_]{10,}(?:\s+[^\W\d_]+)*$",
    )
)
_THEME_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Dataset & preprocessing", ("dataset", "data", "preprocess", "clean", "normalize", "split", "sample", "missing")),
    ("Feature engineering", ("feature", "embedding", "token", "topology", "weather", "time", "attribute", "signal")),
    ("Model architecture", ("model", "architecture", "layer", "network", "gru", "lstm", "kan", "transformer", "fusion")),
    ("Training strategy", ("train", "training", "loss", "optimizer", "epoch", "learning", "regularization", "fine-tune")),
    ("Evaluation metrics", ("evaluate", "evaluation", "metric", "accuracy", "mae", "rmse", "wape", "f1", "precision", "recall")),
    ("Retrieval & indexing", ("retrieval", "retrieve", "rerank", "rank", "index", "chunk", "vector", "qdrant", "hybrid")),
    ("Parsing & OCR", ("parse", "parser", "docling", "ocr", "scan", "image", "table", "figure", "layout")),
    ("Evidence & citation", ("evidence", "citation", "trace", "source", "grounded", "answer", "question")),
    ("Knowledge graph", ("graph", "entity", "relation", "node", "edge", "mindmap")),
    ("Results & findings", ("result", "finding", "performance", "forecast", "prediction", "improve", "comparison")),
    ("Ablation & limitations", ("ablation", "limitation", "constraint", "risk", "error", "failure", "weakness")),
    ("Methods & workflow", ("method", "pipeline", "workflow", "algorithm", "process", "approach", "technique")),
)


def _entity_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unknown"
    return f"entity:{slug}"


# Import synonym map from extraction layer for cross-document entity merging.
# This catches duplicates that survive per-chunk extraction (e.g. "ML" extracted
# from doc A while "Machine Learning" extracted from doc B).
try:
    from src.processing.entity_resolution import _SYNONYM_MAP as _GRAPH_SYNONYM_MAP
except Exception:
    _GRAPH_SYNONYM_MAP = {}

# Fuzzy string matching for cross-document dedup (non-hardcoded).
# Uses rapidfuzz when available, falls back to difflib. Handles cases like
# "Transformer" / "Transformers", "Neural Net" / "Neural Network" without
# requiring per-entity synonym entries.
try:
    from rapidfuzz import fuzz as _graph_fuzz  # type: ignore[import-not-found]
    def _graph_string_similarity(a: str, b: str) -> float:
        return _graph_fuzz.token_set_ratio(a, b) / 100.0
except Exception:
    import difflib as _graph_difflib
    def _graph_string_similarity(a: str, b: str) -> float:
        return _graph_difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Dedup thresholds live in retrieval_config.yaml → graph.*
# ── Embedding-based semantic clustering (BGE-M3) ────────────────────────────
# Captures cross-lingual + paraphrase duplicates that string match misses
# (e.g. "Học máy" ≡ "Machine Learning", "thực tế dương" ≡ "True Positive").
# Domain-agnostic: works for any topic, no hardcoded synonyms required.
_graph_embedder = None  # lazy-loaded BGE-M3 singleton
_graph_embedding_cache: dict[str, list[float]] = {}  # entity_name → dense vector


def _get_graph_embedder():
    global _graph_embedder
    if _graph_embedder is None:
        try:
            from src.core.config import get_settings
            from src.rag.embedder import BGEM3Embedder
            _graph_embedder = BGEM3Embedder(get_settings())
        except Exception:
            _graph_embedder = False  # signal: unavailable, don't retry
    return _graph_embedder if _graph_embedder else None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


def _union_find_clusters(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Group indices [0..n) into clusters connected by `edges`."""
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    clusters: dict[int, list[int]] = {}
    for idx in range(n):
        clusters.setdefault(find(idx), []).append(idx)
    return list(clusters.values())


def _merge_entity_cluster(members: list, *, display_name: str | None = None) -> object:
    """Pick representative entity from cluster, merge mentions/aliases/chunk_ids."""
    rep = max(members, key=lambda e: (len(e.mention_refs), e.confidence, -len(e.canonical_name)))
    all_refs: list = []
    seen_keys: set[tuple] = set()
    all_aliases: set[str] = set(rep.aliases or [])
    all_chunks: set[str] = set(rep.chunk_ids or [])
    for entity in members:
        for ref in entity.mention_refs:
            key_ref = (ref.material_id, ref.page, ref.block_id)
            if key_ref not in seen_keys:
                seen_keys.add(key_ref)
                all_refs.append(ref)
        all_aliases.update(entity.aliases or [])
        all_aliases.add(entity.canonical_name)
        all_chunks.update(entity.chunk_ids or [])
    rep.canonical_name = display_name or rep.canonical_name
    rep.mention_refs = all_refs
    rep.aliases = sorted(a for a in all_aliases if a != rep.canonical_name)
    rep.chunk_ids = sorted(all_chunks)
    return rep


def _cluster_and_merge(entities: list, similarity_fn) -> list:
    """Union-find on pairwise similarity, then merge each cluster's mentions."""
    if len(entities) <= 1:
        return entities
    edges: list[tuple[int, int]] = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if similarity_fn(entities[i], entities[j]):
                edges.append((i, j))
    clusters = _union_find_clusters(len(entities), edges)
    merged: list = []
    for idx_list in clusters:
        if len(idx_list) == 1:
            merged.append(entities[idx_list[0]])
        else:
            merged.append(_merge_entity_cluster([entities[i] for i in idx_list]))
    return merged


def _semantic_cluster_entities(entities: list, threshold: float) -> list[list[int]]:
    """Return clusters of entity indices grouped by BGE-M3 cosine ≥ threshold.
    No-op when the embedder is unavailable.
    """
    if len(entities) < 2:
        return [[i] for i in range(len(entities))]
    embedder = _get_graph_embedder()
    if embedder is None:
        return [[i] for i in range(len(entities))]

    names = [e.canonical_name for e in entities]
    to_encode = [n for n in names if n.lower() not in _graph_embedding_cache]
    if to_encode:
        try:
            encoded = embedder.encode(to_encode)
            for name, item in zip(to_encode, encoded):
                _graph_embedding_cache[name.lower()] = list(item.dense)
        except Exception:
            return [[i] for i in range(len(entities))]

    vectors = [_graph_embedding_cache.get(n.lower(), []) for n in names]
    edges = [
        (i, j)
        for i in range(len(entities))
        for j in range(i + 1, len(entities))
        if _cosine(vectors[i], vectors[j]) >= threshold
    ]
    return _union_find_clusters(len(entities), edges)


def _canonical_dedup_key(name: str) -> str:
    """Stable key for cross-document dedup. Applies synonym lookup + diacritic fold."""
    base = (_GRAPH_SYNONYM_MAP.get(name.lower().strip()) or name).lower().strip()
    folded = _ascii_fold(base)
    return re.sub(r"[^a-z0-9]+", "", folded)


def _dedupe_and_merge_entities(entities: list) -> list:
    """Three-pass dedup: synonym/canonical key → fuzzy string → semantic embedding.

    Thresholds live in retrieval_config.yaml → graph.* and are read via Settings.
    """
    settings = get_settings()
    fuzzy_threshold = settings.graph_fuzzy_dedup_threshold
    semantic_threshold = settings.graph_semantic_dedup_threshold

    # ── Pass 1: synonym + canonical-key grouping ─────────────────────────────
    groups: dict[str, list] = {}
    for entity in entities:
        key = _canonical_dedup_key(entity.canonical_name)
        if not key:
            continue
        groups.setdefault(key, []).append(entity)

    merged: list = []
    for group in groups.values():
        synonym_canonical = next(
            (_GRAPH_SYNONYM_MAP.get(e.canonical_name.lower().strip()) for e in group
             if _GRAPH_SYNONYM_MAP.get(e.canonical_name.lower().strip())),
            None,
        )
        if len(group) == 1 and not synonym_canonical:
            merged.append(group[0])
            continue
        canonical_name = synonym_canonical or max(
            group, key=lambda e: (len(e.mention_refs), e.confidence, -len(e.canonical_name)),
        ).canonical_name
        merged.append(_merge_entity_cluster(group, display_name=canonical_name))

    # ── Pass 2: fuzzy string similarity ──────────────────────────────────────
    merged = _cluster_and_merge(
        merged,
        lambda a, b: _graph_string_similarity(a.canonical_name, b.canonical_name) >= fuzzy_threshold,
    )

    # ── Pass 3: BGE-M3 embedding similarity (cross-lingual + paraphrase) ─────
    if len(merged) <= 1:
        return merged
    sem_clusters = _semantic_cluster_entities(merged, semantic_threshold)
    semantic_merged: list = []
    for idx_list in sem_clusters:
        if len(idx_list) == 1:
            semantic_merged.append(merged[idx_list[0]])
        else:
            semantic_merged.append(_merge_entity_cluster([merged[i] for i in idx_list]))
    return semantic_merged


# Patterns that mark an entity as low-quality / extraction noise.
# Goal: visualization should show concepts a learner would recognise, not OCR fragments.
_VN_DIACRITIC_RE = re.compile(r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", re.IGNORECASE)
_HAS_DIGIT_RE = re.compile(r"\d")
_UPPERCASE_TOKEN_RE = re.compile(r"\b[A-Z]{2,}\b")


_NOISE_PREFIX_RE = re.compile(r"^(mô hình|dữ liệu|chỉ số|thực tế|tỷ lệ|độ chính|hệ số)\s", re.IGNORECASE)
_GENERIC_NOUN_SET = frozenset({
    # Standalone generic nouns that shouldn't be entities by themselves
    "mô hình", "dữ liệu", "chỉ số", "tỷ lệ", "hệ số", "công thức",
    "kết quả", "phương pháp", "kỹ thuật", "thuật toán",
    "model", "data", "metric", "method", "technique", "result",
    "value", "score", "weight", "feature", "label", "class",
})


def _is_quality_entity_label(label: str) -> bool:
    """Reject entities that look like extraction noise.

    Heuristics tuned for ML/AI corpus in Vietnamese + English:
       - Multi-token + digits inside → OCR fragment
       - Mixed VN diacritics + multiple uppercase English → slide title
       - All-caps Vietnamese multi-token → slide title
       - Starts with generic noun ("mô hình", "chỉ số") → incomplete phrase
       - Pure generic noun → too generic
    """
    if not label:
        return False
    stripped = label.strip()
    if len(stripped) < 3:
        return False

    tokens = stripped.split()
    n_tokens = len(tokens)
    lower = stripped.lower()

    # Reject pure generic nouns
    if lower in _GENERIC_NOUN_SET:
        return False

    # NB: removed the blanket "1-token all-caps ≤4 chars" reject. It was
    # killing legitimate domain acronyms (KAN, GRU, MLP, RAG, BGE, ...).
    # Filtering noise acronyms is the extractor's job — high-confidence
    # entities are gated separately in `_is_display_entity`.

    # Multi-token + contains digit → likely OCR fragment
    if n_tokens >= 2 and _HAS_DIGIT_RE.search(stripped):
        digit_tokens = [t for t in tokens if _HAS_DIGIT_RE.search(t)]
        if len(digit_tokens) >= 2 or (n_tokens >= 3 and digit_tokens):
            return False

    has_vn = bool(_VN_DIACRITIC_RE.search(stripped))
    upper_tokens = _UPPERCASE_TOKEN_RE.findall(stripped)

    # Mixed VN diacritics + 2+ uppercase English tokens
    if has_vn and len(upper_tokens) >= 2:
        return False

    # All-uppercase Vietnamese multi-token (slide titles like "CHỈ SỐ ĐÁNH GIÁ")
    if has_vn and stripped.upper() == stripped and n_tokens >= 2:
        return False

    # Starts with a generic noun → likely fragment (e.g. "mô hình Giảm", "dữ liệu TIỀN")
    if n_tokens >= 2 and _NOISE_PREFIX_RE.match(stripped):
        # But only reject if remainder is short/garbled
        remainder = " ".join(tokens[1:]).strip()
        if len(remainder) < 4 or remainder.isupper():
            return False

    return True


def _mindmap_slug(prefix: str, name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "node"
    return f"{prefix}:{slug[:64]}"


def _short_label(text: str, *, limit: int = 90) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1].rstrip()}..."


def _repair_text_encoding(text: str) -> str:
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text
    return repaired if repaired else text


def _ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii").lower()


def _clean_entity_label(text: str) -> str | None:
    """Return a human-readable graph node label, or None for OCR/layout artifacts."""
    compact = _repair_text_encoding(text)
    compact = re.sub(r"\s*[,;:/]\s*", " ", compact)
    compact = " ".join(compact.split()).strip(" \t\r\n,;:|()[]{}'\"")
    if not compact:
        return None
    if "|" in compact or "_" in compact:
        return None

    compact = re.sub(r"^[\W\d_]+", "", compact, flags=re.UNICODE).strip()
    compact = compact.rstrip(".")
    compact = re.sub(r"^(?:\d+(?:[.,]\d+)?\s+){1,3}", "", compact).strip()
    compact = re.sub(r"\s+", " ", compact)
    if re.search(r"[.!?]\s+\S+", compact):
        first_clause = re.split(r"[.!?]\s+", compact, maxsplit=1)[0].strip()
        if len(first_clause) >= 3:
            compact = first_clause
    if len(compact) < 3:
        return None
    if any(marker in compact for marker in ("\u00c3", "\u00c2", "\u00e2", "\ufffd")):
        return None
    if "…" in compact or "..." in compact:
        return None

    lower = compact.lower()
    folded = _ascii_fold(compact)
    if _BAD_ENTITY_LABEL_RE.search(lower):
        return None
    if re.match(r"^(?:hinh|bang|cau)\s+\d+$", folded):
        return None
    if "docxipdfipng" in folded or "docx pdf png" in folded:
        return None
    if any(marker in lower for marker in ("phiếu ghi chú ocr", "phieu ghi chu ocr", "ghi chú ocr", "ghi chu ocr")):
        return None
    if lower in _NOISY_ENTITY_LABELS:
        return None
    if lower in _TECHNICAL_GRAPH_LABELS:
        return None
    words = compact.split()
    if lower in _NOISY_ENTITY_WORDS:
        return None
    if words[0].lower() in _NOISY_ENTITY_WORDS:
        return None
    format_word_count = sum(1 for word in words if word.lower() in _FORMAT_ENTITY_WORDS)
    if format_word_count > 1 or format_word_count == len(words):
        return None
    if words[-1].lower() in _FORMAT_ENTITY_WORDS and len(words) <= 3:
        return None
    if len(words) > 4:
        return None
    # No length-based rejection for single-word labels here. Domain acronyms
    # (KAN, GRU, RAG, MLP, etc.) are 2-5 chars; rejecting them lost legitimate
    # concept nodes. The extractor's confidence + mention_count gate (in
    # `_is_display_entity`) already filters genuine noise.
    if re.match(r"^(?:hình|hinh|bảng|bang|câu|cau)\s+\d+$", lower):
        return None
    if len({word.lower() for word in words}) < len(words):
        return None
    if len(words) >= 3 and all(word.isascii() and word.isupper() and len(word) >= 2 for word in words):
        return None
    if words[-1].lower() in _NOISY_ENTITY_WORDS:
        return None
    if re.fullmatch(r"[\d\W_]+", compact, flags=re.UNICODE):
        return None
    if re.search(r"\.(?:png|jpe?g|pdf|docx|pptx|xlsx)$", lower):
        return None

    chars = compact.replace(" ", "")
    digits = sum(ch.isdigit() for ch in chars)
    letters = sum(ch.isalpha() for ch in chars)
    symbols = max(0, len(chars) - digits - letters)
    if digits >= 3 and digits >= letters:
        return None
    if symbols > letters and symbols > 1:
        return None
    if len(re.findall(r"\d+", compact)) >= 3:
        return None

    return compact


def _is_display_entity(entity: Entity) -> bool:
    label = _clean_entity_label(entity.canonical_name)
    if label is None:
        return False
    settings = get_settings()
    if (
        entity.confidence < settings.graph_display_min_confidence
        and len(entity.mention_refs) < settings.graph_display_min_mentions
    ):
        return False
    return True


def _scope_query(request: MindmapRequest) -> dict:
    if not request.collection_id and not request.material_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="collection_id or material_ids is required for scoped graph retrieval",
        )
    query = {"owner_id": request.owner_id}
    if request.collection_id:
        query["collection_id"] = PydanticObjectId(request.collection_id)
    return query


def _evidence_refs(refs) -> list[dict[str, str | int]]:
    return [
        {"material_id": str(ref.material_id), "page": ref.page or 0, "block_id": ref.block_id or ""}
        for ref in refs[:3]
    ]


def _fallback_node(node_id: str) -> GraphNode:
    prefix, _, raw = node_id.partition(":")
    label = raw.replace("-", " ") if raw else node_id
    node_type = prefix if prefix in {"block", "entity", "event", "relation"} else "entity"
    return GraphNode(id=node_id, label=label, type=node_type, confidence=None)


async def _block_nodes_from_relations(relations) -> list[GraphNode]:
    refs = [
        ref
        for relation in relations
        for ref in getattr(relation, "evidence_refs", [])
        if getattr(ref, "block_id", None)
    ]
    if not refs:
        return []

    material_ids = list({ref.material_id for ref in refs})
    materials = await Material.find({"_id": {"$in": material_ids}}).to_list()
    ref_keys = {(str(ref.material_id), ref.page, ref.block_id) for ref in refs}
    nodes: list[GraphNode] = []
    seen: set[str] = set()

    for material in materials:
        pages = getattr(material, "pages", []) or []
        for page in pages:
            page_number = getattr(page, "page_number", None)
            for block in getattr(page, "blocks", []) or []:
                block_id = getattr(block, "block_id", None)
                key = (str(getattr(material, "id", material_ids[0])), page_number, block_id)
                if key not in ref_keys or not block_id:
                    continue
                node_id = f"block:{block_id}"
                if node_id in seen:
                    continue
                seen.add(node_id)
                content = " ".join(str(getattr(block, "content", "")).split())
                label = f"p.{page_number}: {_short_label(content, limit=72)}" if page_number else _short_label(content, limit=80)
                nodes.append(
                    GraphNode(
                        id=node_id,
                        label=label,
                        type=getattr(block, "block_type", None) or "block",
                        confidence=getattr(block, "ocr_confidence", None),
                        source_docs=[getattr(material, "original_name", "")] if getattr(material, "original_name", "") else [],
                        evidence_refs=[{"material_id": str(getattr(material, "id", material_ids[0])), "page": page_number or 0, "block_id": block_id}],
                    )
                )
    return nodes


def _entity_weight(entity: Entity) -> float:
    return entity.confidence * 3 + min(len(entity.mention_refs), 8) * 0.55


def _entity_ref_keys(entity: Entity, *, include_block: bool = False) -> set[tuple[str, int | None, str | None]]:
    return {
        (str(ref.material_id), ref.page, ref.block_id if include_block else None)
        for ref in entity.mention_refs
    }


def _relatedness(a: Entity, b: Entity) -> float:
    a_pages = _entity_ref_keys(a)
    b_pages = _entity_ref_keys(b)
    a_blocks = _entity_ref_keys(a, include_block=True)
    b_blocks = _entity_ref_keys(b, include_block=True)
    shared_pages = len(a_pages & b_pages)
    shared_blocks = len(a_blocks & b_blocks)
    shared_materials = len({key[0] for key in a_pages} & {key[0] for key in b_pages})
    return shared_blocks * 3.0 + shared_pages * 1.7 + shared_materials * 0.45


def _pick_topic_seeds(entities: list[Entity], *, max_topics: int = 8) -> list[Entity]:
    degree: Counter[str] = Counter()
    for i, source in enumerate(entities):
        for target in entities[i + 1 :]:
            score = _relatedness(source, target)
            if score <= 0:
                continue
            degree[str(source.id)] += score
            degree[str(target.id)] += score

    ranked = sorted(
        entities,
        key=lambda item: (degree[str(item.id)] + _entity_weight(item), len(item.mention_refs), item.confidence),
        reverse=True,
    )
    seeds: list[Entity] = []
    seed_tokens: list[set[str]] = []
    for entity in ranked:
        label = _clean_entity_label(entity.canonical_name) or entity.canonical_name
        if not _is_mindmap_label(label):
            continue
        tokens = {token for token in re.split(r"[^a-z0-9]+", label.lower()) if len(token) > 2}
        if any(tokens and len(tokens & existing) / max(len(tokens), 1) > 0.65 for existing in seed_tokens):
            continue
        seeds.append(entity)
        seed_tokens.append(tokens)
        if len(seeds) >= max_topics:
            break
    return seeds


def _semantic_bucket(label: str, entity_type: str) -> str:
    normalized = label.lower()
    for bucket, keywords in _THEME_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return bucket
    type_key = entity_type.lower()
    if type_key in {"metric"}:
        return "Evaluation metrics"
    if type_key in {"method", "technology"}:
        return "Methods & workflow"
    if type_key in {"event", "date"}:
        return "Timeline & milestones"
    if type_key in {"organization", "org", "person", "location"}:
        return "Actors & context"
    return "Key concepts"


def _is_mindmap_label(label: str) -> bool:
    normalized = label.strip()
    if not normalized:
        return False
    lower = normalized.lower()
    if any(pattern.search(normalized) for pattern in _BAD_MINDMAP_LABEL_PATTERNS):
        return False
    if "nguá»“n" in lower or "nguon" in lower:
        return False
    format_hits = sum(1 for word in _FORMAT_ENTITY_WORDS if re.search(rf"\b{re.escape(word)}\b", lower))
    if format_hits >= 2:
        return False
    if re.search(r"(?:[A-Z]{3,}I){2,}", normalized):
        return False
    return True


def _mindmap_display_label(label: str) -> str:
    compact = " ".join(label.split()).strip()
    if re.fullmatch(r"[a-z][a-z0-9-]{3,}", compact):
        return compact[:1].upper() + compact[1:]
    return compact


def _chunk_scope_query(request: MindmapRequest) -> dict:
    query = _scope_query(request)
    if request.material_ids:
        query["material_id"] = {"$in": [PydanticObjectId(material_id) for material_id in request.material_ids]}
    return query


def _chunk_citations(chunk: Chunk) -> list[dict[str, str | int]]:
    page = chunk.source_pages[0] if chunk.source_pages else 0
    block_id = chunk.source_block_ids[0] if chunk.source_block_ids else ""
    return [{"material_id": str(chunk.material_id), "page": page, "block_id": block_id}]


def _quality_label(label: str) -> str | None:
    cleaned = _clean_entity_label(label)
    if cleaned is None:
        return None
    cleaned = _mindmap_display_label(_short_label(cleaned, limit=52))
    words = cleaned.split()
    if words and words[-1].lower() in {"a", "an", "the", "uses", "adds", "increases", "decreases"}:
        return None
    if cleaned.lower().endswith(("regularizat", "normalizat")):
        return None
    return cleaned if _is_mindmap_label(cleaned) else None


def _extract_chunk_concepts(chunks: list[Chunk], entities: list[Entity], *, detail_level: str) -> list[tuple[str, str, float, list[dict[str, str | int]]]]:
    candidates: dict[str, tuple[str, str, float, list[dict[str, str | int]]]] = {}

    for entity in entities:
        label = _quality_label(entity.canonical_name)
        if not label:
            continue
        bucket = _semantic_bucket(label, entity.entity_type)
        if bucket == "Key concepts":
            bucket = "Core concepts"
        score = _entity_weight(entity)
        key = label.casefold()
        old = candidates.get(key)
        if old is None or score > old[2]:
            candidates[key] = (label, bucket, score, _evidence_refs(entity.mention_refs))

    phrase_re = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9+-]*(?:[- ][A-Z0-9][A-Za-z0-9+-]*){0,3}|"
        r"(?:hybrid|semantic|vector|graph|evidence|retrieval|reranker|chunking|embedding|ocr|parser|grounding|regularization|dropout|normalization)(?:\s+[A-Za-z0-9+-]+){0,2})\b",
        flags=re.IGNORECASE,
    )
    # Brief: small slice, fewer phrases. Overview: medium. Detailed: full.
    max_chunk_chars = (
        1600 if detail_level == "detailed"
        else 500 if detail_level == "brief"
        else 900
    )
    for chunk in chunks[:40]:
        text = _repair_text_encoding(chunk.content or "")
        for match in phrase_re.finditer(text[:max_chunk_chars]):
            raw = match.group(0).strip(" .,:;()[]{}")
            label = _quality_label(raw)
            if not label:
                continue
            bucket = _semantic_bucket(label, "concept")
            if bucket == "Key concepts":
                bucket = "Core concepts"
            score = 1.0 + min(len(label.split()), 4) * 0.2
            key = label.casefold()
            old = candidates.get(key)
            if old is None:
                candidates[key] = (label, bucket, score, _chunk_citations(chunk))
            else:
                candidates[key] = (old[0], old[1], old[2] + score, old[3] or _chunk_citations(chunk))

    return sorted(candidates.values(), key=lambda item: item[2], reverse=True)


def _build_mindmap_from_concepts(root_topic: str, concepts: list[tuple[str, str, float, list[dict[str, str | int]]]], *, detail_level: str) -> MindmapResponse:
    grouped: dict[str, list[tuple[str, str, float, list[dict[str, str | int]]]]] = defaultdict(list)
    for item in concepts:
        grouped[item[1]].append(item)

    # Brief: 4 topics × 3 concepts (compact). Overview: 6×5. Detailed: 8×8.
    if detail_level == "detailed":
        max_topics, max_concepts = 8, 8
    elif detail_level == "brief":
        max_topics, max_concepts = 4, 3
    else:
        max_topics, max_concepts = 6, 5
    ranked_groups = sorted(grouped.items(), key=lambda item: (sum(concept[2] for concept in item[1]), len(item[1])), reverse=True)

    nodes: list[MindmapNode] = []
    used: set[str] = set()
    for topic, items in ranked_groups[:max_topics]:
        children: list[MindmapNode] = []
        for label, _bucket, score, citations in sorted(items, key=lambda item: item[2], reverse=True):
            key = label.casefold()
            if key in used:
                continue
            used.add(key)
            children.append(
                MindmapNode(
                    id=_mindmap_slug("concept", f"{topic}-{label}"),
                    label=label,
                    entity_type="concept",
                    summary=f"score={score:.2f}",
                    citations=citations,
                )
            )
            if len(children) >= max_concepts:
                break
        if children:
            nodes.append(
                MindmapNode(
                    id=_mindmap_slug("topic", topic),
                    label=topic,
                    entity_type="topic",
                    children=children,
                    citations=children[0].citations,
                )
            )

    return MindmapResponse(root_topic=root_topic, nodes=nodes)


def _parse_json_object(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _build_llm_mindmap(root_topic: str, chunks: list[Chunk], *, settings: Settings, detail_level: str) -> MindmapResponse | None:
    if not chunks:
        return None
    sample = []
    for idx, chunk in enumerate(chunks[:10], 1):
        text = _repair_text_encoding(" ".join((chunk.content or "").split()))
        if len(text) < 80:
            continue
        sample.append(f"[S{idx}] {text[:900]}")
    if not sample:
        return None

    # Brief: tighter than overview. Detailed: full breadth.
    if detail_level == "brief":
        topic_count, concept_count = 4, 3
    elif detail_level == "detailed":
        topic_count, concept_count = 8, 8
    else:
        topic_count, concept_count = 6, 5
    prompt = f"""Create a NotebookLM-style mind map from the provided document excerpts.
Return ONLY valid JSON with this schema:
{{"root_topic":"...","topics":[{{"label":"short natural topic label","children":[{{"label":"short natural concept label","source_index":1}}]}}]}}

Rules:
- Use natural labels, not raw OCR tokens.
- Do not include file names, page labels, "source", "pass/fail", or broken encoding.
- Prefer 4-{topic_count} broad topics and 2-{concept_count} children per topic.
- Labels must be <= 6 words.
- Root topic: {root_topic}

Excerpts:
{chr(10).join(sample)}
"""
    try:
        llm = build_llm(settings)
        raw = await asyncio.wait_for(llm.generate(prompt=prompt), timeout=min(settings.llm_timeout_seconds, 30.0))
        close = getattr(llm, "close", None)
        if close is not None:
            await close()
    except Exception:
        return None

    parsed = _parse_json_object(raw)
    if not parsed:
        return None
    topics = parsed.get("topics")
    if not isinstance(topics, list):
        return None

    nodes: list[MindmapNode] = []
    for topic_item in topics[:topic_count]:
        if not isinstance(topic_item, dict):
            continue
        topic_label = _quality_label(str(topic_item.get("label") or ""))
        if not topic_label:
            continue
        children: list[MindmapNode] = []
        raw_children = topic_item.get("children")
        if not isinstance(raw_children, list):
            continue
        for child_item in raw_children[:concept_count]:
            if not isinstance(child_item, dict):
                continue
            label = _quality_label(str(child_item.get("label") or ""))
            if not label:
                continue
            source_index = child_item.get("source_index")
            chunk = chunks[int(source_index) - 1] if isinstance(source_index, int) and 1 <= source_index <= len(chunks) else chunks[0]
            children.append(
                MindmapNode(
                    id=_mindmap_slug("concept", f"{topic_label}-{label}"),
                    label=label,
                    entity_type="concept",
                    citations=_chunk_citations(chunk),
                )
            )
        if children:
            nodes.append(
                MindmapNode(
                    id=_mindmap_slug("topic", topic_label),
                    label=topic_label,
                    entity_type="topic",
                    children=children,
                    citations=children[0].citations,
                )
            )
    return MindmapResponse(root_topic=root_topic, nodes=nodes) if nodes else None


def _build_thematic_mindmap(root_topic: str, entities: list[Entity]) -> MindmapResponse:
    if not entities:
        return MindmapResponse(root_topic=root_topic, nodes=[])

    buckets: dict[str, list[Entity]] = defaultdict(list)
    for entity in entities:
        label = _clean_entity_label(entity.canonical_name) or entity.canonical_name
        if not _is_mindmap_label(label):
            continue
        bucket_name = _semantic_bucket(label, entity.entity_type)
        if bucket_name == "Key concepts":
            bucket_name = "Core concepts"
        buckets[bucket_name].append(entity)

    nodes: list[MindmapNode] = []
    used_concepts: set[str] = set()
    ranked_buckets = sorted(
        buckets.items(),
        key=lambda item: (sum(_entity_weight(entity) for entity in item[1]), len(item[1])),
        reverse=True,
    )

    for bucket_name, bucket_entities in ranked_buckets[:8]:
        concept_nodes: list[MindmapNode] = []
        for member in sorted(bucket_entities, key=_entity_weight, reverse=True):
            label = _mindmap_display_label(_short_label(_clean_entity_label(member.canonical_name) or member.canonical_name, limit=48))
            if not _is_mindmap_label(label):
                continue
            key = label.casefold()
            if key in used_concepts:
                continue
            used_concepts.add(key)
            concept_nodes.append(
                MindmapNode(
                    id=_mindmap_slug("concept", f"{bucket_name}-{label}"),
                    label=label,
                    entity_type=member.entity_type or "concept",
                    summary=f"mentions={len(member.mention_refs)}; confidence={member.confidence:.2f}",
                    citations=_evidence_refs(member.mention_refs),
                )
            )
            if len(concept_nodes) >= 8:
                break

        if not concept_nodes:
            continue

        nodes.append(
            MindmapNode(
                id=_mindmap_slug("topic", bucket_name),
                label=bucket_name,
                entity_type="topic",
                children=concept_nodes,
                citations=concept_nodes[0].citations,
            )
        )

    if not nodes:
        fallback_entities = sorted(entities, key=_entity_weight, reverse=True)
        fallback_concepts: list[MindmapNode] = []
        for entity in fallback_entities:
            label = _mindmap_display_label(_short_label(_clean_entity_label(entity.canonical_name) or entity.canonical_name, limit=48))
            if not _is_mindmap_label(label):
                continue
            fallback_concepts.append(
                MindmapNode(
                    id=_mindmap_slug("concept", label),
                    label=label,
                    entity_type=entity.entity_type or "concept",
                    citations=_evidence_refs(entity.mention_refs),
                )
            )
            if len(fallback_concepts) >= 8:
                break
        if fallback_concepts:
            nodes.append(
                MindmapNode(
                    id=_mindmap_slug("topic", "Core concepts"),
                    label="Core concepts",
                    entity_type="topic",
                    children=fallback_concepts,
                    citations=fallback_concepts[0].citations,
                )
            )

    return MindmapResponse(root_topic=root_topic, nodes=nodes)


def _entity_cooccurrence_edges(entities: list[Entity], *, limit: int = 80) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    seen: set[tuple[str, str]] = set()
    for index, source_entity in enumerate(entities):
        source_label = _clean_entity_label(source_entity.canonical_name)
        if not source_label:
            continue
        source_refs = source_entity.mention_refs
        for target_entity in entities[index + 1 :]:
            target_label = _clean_entity_label(target_entity.canonical_name)
            if not target_label:
                continue
            target_refs = target_entity.mention_refs
            shared_block = _shared_ref(source_refs, target_refs, require_same_block=True)
            shared_page = shared_block or _shared_ref(source_refs, target_refs, require_same_block=False)
            if shared_page is None:
                continue
            source_id = _entity_slug(source_entity.canonical_name)
            target_id = _entity_slug(target_entity.canonical_name)
            key = tuple(sorted((source_id, target_id)))
            if key in seen:
                continue
            seen.add(key)
            relation_type = "co_occurs_in_block" if shared_block is not None else "co_occurs_on_page"
            confidence = min(source_entity.confidence, target_entity.confidence, 0.58 if shared_block is not None else 0.52)
            edges.append(
                GraphEdge(
                    source=source_id,
                    target=target_id,
                    relation_type=relation_type,
                    source_label=source_label,
                    target_label=target_label,
                    confidence=confidence,
                    evidence_count=1,
                    evidence_refs=_evidence_refs([shared_page]),
                )
            )
            if len(edges) >= limit:
                return edges
    return edges


def _shared_ref(source_refs, target_refs, *, require_same_block: bool):
    target_keys = {
        (
            str(ref.material_id),
            ref.page,
            ref.block_id if require_same_block else None,
        )
        for ref in target_refs
    }
    for ref in source_refs:
        key = (str(ref.material_id), ref.page, ref.block_id if require_same_block else None)
        if key in target_keys:
            return ref
    return None


@router.post("", response_model=APIResponse[GraphResponse])
async def graph(
    request: Request,
    body: MindmapRequest,
    settings: Settings = Depends(get_app_settings),
) -> APIResponse[GraphResponse]:
    verify_owner_access(request, body.owner_id)
    query = _scope_query(body)
    entity_query = dict(query)
    relation_query = dict(query)
    if body.material_ids:
        material_ids = [PydanticObjectId(material_id) for material_id in body.material_ids]
        entity_query["mention_refs.material_id"] = {"$in": material_ids}
        relation_query["evidence_refs.material_id"] = {"$in": material_ids}

    # Only named entities, high-confidence first; no events/block nodes (too noisy for viz)
    text_entity_query = {**entity_query, "entity_type": {"$nin": list(_CROSS_MODAL_TYPES)}}
    raw_entities = (
        await Entity.find(text_entity_query)
        .sort("-confidence")
        .limit(settings.graph_max_entities_fetch)
        .to_list()
    )
    display_entities = [entity for entity in raw_entities if _is_display_entity(entity)]
    # Cross-document dedup: merge "ML" + "Machine Learning" + "Học máy" → 1 entity
    deduped = _dedupe_and_merge_entities(display_entities)
    # Fetch relations early so focus expansion can use them for 1-hop neighbors
    relations = await Relation.find(relation_query).limit(settings.graph_max_relations_fetch).to_list()

    # ── Focus mode: filter to entities backing the last answer ──────────────
    # Strategy: ENTITY NAME MATCHING against query + answer text.
    # This is more precise than block_id/material filtering because:
    #   - Block_ids in citations often don't align with entity mention_refs (different chunking)
    #   - Material-level fallback grabs the entire document's entities (50+ entities)
    #   - Name matching directly captures what the answer discusses
    focus_block_set: set[str] = set(body.focus_block_ids or [])
    focus_material_set: set[str] = set(body.focus_material_ids or [])
    focus_page_set: set[str] = set(body.focus_pages or [])
    focus_text = " ".join(filter(None, [body.focus_query_text or "", body.focus_answer_text or ""])).strip()
    is_focus_mode = bool(focus_block_set or focus_material_set or focus_page_set or focus_text)
    primary_ids: set[str] = set()
    if is_focus_mode:
        primary: list = []
        # Tier 1 (preferred): name match in query + answer text
        if focus_text:
            folded_text = _ascii_fold(focus_text)
            for entity in deduped:
                names = [entity.canonical_name, *(entity.aliases or [])]
                for name in names:
                    folded_name = _ascii_fold(name).strip()
                    if len(folded_name) >= 3 and folded_name in folded_text:
                        primary.append(entity)
                        break

        # Tier 2 (fallback): block_id match — only when no name matches found
        if not primary and focus_block_set:
            primary = [
                e for e in deduped
                if any(r.block_id and r.block_id in focus_block_set for r in e.mention_refs)
            ]

        # Tier 3 (last resort): page match
        if not primary and focus_page_set:
            primary = [
                e for e in deduped
                if any(
                    r.page is not None and f"{r.material_id}:{r.page}" in focus_page_set
                    for r in e.mention_refs
                )
            ]
        # NOTE: material-level fallback intentionally removed — it pulls every entity
        # in the cited document and defeats the purpose of "focused" view.

        # Cap primary by mention_count (most central entities first). Earlier
        # caps of 10/6 made focused views feel empty (e.g. KAN-GRU → 2-node graph).
        primary = sorted(primary, key=lambda e: len(e.mention_refs), reverse=True)[: settings.graph_focus_primary_cap]
        primary_ids = {_entity_slug(e.canonical_name) for e in primary}

        # 1-hop expansion: top-N neighbors most connected to primary set
        if primary_ids:
            neighbor_counts: dict[str, int] = {}
            for rel in relations:
                if rel.source_id in primary_ids and rel.target_id not in primary_ids:
                    neighbor_counts[rel.target_id] = neighbor_counts.get(rel.target_id, 0) + 1
                elif rel.target_id in primary_ids and rel.source_id not in primary_ids:
                    neighbor_counts[rel.source_id] = neighbor_counts.get(rel.source_id, 0) + 1
            top_neighbor_ids = {
                slug
                for slug, _ in sorted(neighbor_counts.items(), key=lambda kv: kv[1], reverse=True)[
                    : settings.graph_focus_neighbor_cap
                ]
            }
            for entity in deduped:
                slug = _entity_slug(entity.canonical_name)
                if slug in top_neighbor_ids and slug not in primary_ids:
                    primary.append(entity)

        # If focus matching produced nothing, fall back to showing the most
        # central concept entities globally rather than an empty canvas.
        if not primary:
            primary = sorted(deduped, key=lambda e: (len(e.mention_refs), e.confidence), reverse=True)[
                : settings.graph_focus_fallback_cap
            ]

        deduped = primary

    entities = sorted(deduped, key=lambda e: (len(e.mention_refs), e.confidence), reverse=True)[
        : settings.graph_max_visible_nodes
    ]
    all_entities = entities

    # Collect all material_ids referenced by entities to batch-load names
    all_material_ids: set[PydanticObjectId] = set()
    for entity in all_entities:
        for ref in entity.mention_refs:
            all_material_ids.add(ref.material_id)
    material_name_map: dict[str, str] = {}
    if all_material_ids:
        mats = await Material.find({"_id": {"$in": list(all_material_ids)}}).to_list()
        material_name_map = {str(m.id): m.original_name for m in mats}

    nodes_by_id: dict[str, GraphNode] = {}
    for entity in all_entities:
        cleaned = _clean_entity_label(entity.canonical_name) or entity.canonical_name
        if not _is_quality_entity_label(cleaned):
            continue
        slug = _entity_slug(entity.canonical_name)
        # Deduplicate: keep entity with higher mention_count for identical slug
        existing = nodes_by_id.get(slug)
        if existing and existing.mention_count >= len(entity.mention_refs):
            continue
        nodes_by_id[slug] = GraphNode(
            id=slug,
            label=_short_label(cleaned, limit=40),
            type=entity.entity_type,
            confidence=entity.confidence,
            mention_count=len(entity.mention_refs),
            source_docs=list(dict.fromkeys(
                material_name_map[str(ref.material_id)]
                for ref in entity.mention_refs
                if str(ref.material_id) in material_name_map
            ))[:5],
            evidence_refs=_evidence_refs(entity.mention_refs),
        )

    node_ids = set(nodes_by_id.keys())
    filtered_relations = [
        r for r in relations
        if r.source_id in node_ids
        and r.target_id in node_ids
        and r.evidence_refs
        and r.relation_type not in _STRUCTURAL_RELATION_TYPES
    ]

    # Batch-load chunk text for relations that don't already have evidence_text_chunk
    evidence_text_map: dict[tuple[str, str], str] = {}
    refs_by_material: dict[str, set[str]] = defaultdict(set)
    for relation in filtered_relations:
        if relation.evidence_text_chunk:
            continue
        first_ref = relation.evidence_refs[0]
        if first_ref.block_id:
            refs_by_material[str(first_ref.material_id)].add(first_ref.block_id)
    if refs_by_material:
        for mat_id_str, block_ids in refs_by_material.items():
            try:
                mat_chunks = await Chunk.find(
                    {"material_id": PydanticObjectId(mat_id_str), "source_block_ids": {"$in": list(block_ids)}}
                ).to_list()
                for chunk in mat_chunks:
                    for bid in chunk.source_block_ids:
                        if bid in block_ids:
                            evidence_text_map[(mat_id_str, bid)] = chunk.content[:600]
            except Exception:
                pass

    edges = []
    for relation in filtered_relations:
        source_node = nodes_by_id.get(relation.source_id)
        target_node = nodes_by_id.get(relation.target_id)
        evidence_text = relation.evidence_text_chunk
        if not evidence_text:
            first_ref = relation.evidence_refs[0]
            evidence_text = evidence_text_map.get((str(first_ref.material_id), first_ref.block_id or ""))
        edges.append(
            GraphEdge(
                source=relation.source_id,
                target=relation.target_id,
                relation_type=relation.relation_type,
                source_label=source_node.label if source_node else None,
                target_label=target_node.label if target_node else None,
                confidence=relation.confidence,
                evidence_count=len(relation.evidence_refs),
                evidence_refs=_evidence_refs(relation.evidence_refs),
                evidence_text_chunk=evidence_text,
            )
        )
    if not edges:
        edges = _entity_cooccurrence_edges(all_entities)
    edges.sort(key=lambda edge: ((edge.evidence_count or 0), edge.confidence or 0), reverse=True)

    # ── Enrich nodes with centrality + community structure ──────────────────
    # Pipeline:
    #   1. PageRank → importance score (hub detection)
    #   2. Louvain communities → grouping by topic
    #   3. Filter orphan nodes (degree = 0) for visual clarity
    G = build_digraph(list(nodes_by_id.values()), edges)
    degree_map = compute_degrees(G)
    pagerank_map = compute_pagerank(G)
    community_map = compute_communities(G)

    # Normalize PageRank to [0, 1] for easier UI consumption
    max_pr = max(pagerank_map.values()) if pagerank_map else 1.0
    hub_threshold = sorted(pagerank_map.values(), reverse=True)[: max(1, len(pagerank_map) // 10)][-1] if pagerank_map else 0.0

    nodes = []
    for node in nodes_by_id.values():
        deg = degree_map.get(node.id, 0)
        is_primary = is_focus_mode and node.id in primary_ids
        # In focus mode, keep primary entities even if degree=0
        if deg == 0 and not is_primary:
            continue
        pr = pagerank_map.get(node.id, 0.0)
        # In focus mode, suppress hub status — user wants Dropout-centric view,
        # not global "accuracy" hub. Hubs are misleading after subgraph filter.
        is_hub_global = pr >= hub_threshold and len(pagerank_map) >= 3
        nodes.append(
            node.model_copy(update={
                "degree": deg,
                "importance": pr / max_pr if max_pr > 0 else 0.0,
                "community": community_map.get(node.id, 0),
                "is_hub": is_primary if is_focus_mode else is_hub_global,
                "is_focused": is_primary,
            })
        )
    # Keep only edges connecting visible nodes
    visible_ids = {n.id for n in nodes}
    edges = [e for e in edges if e.source in visible_ids and e.target in visible_ids]

    result = GraphResponse(nodes=nodes, edges=edges)
    return APIResponse(success=True, message="Graph loaded successfully", data=result, error=None)


@router.get("/entity/{entity_id}/subgraph", response_model=APIResponse[GraphResponse])
async def entity_subgraph(
    entity_id: str,
    request: Request,
    owner_id: str,
    collection_id: str | None = None,
    hops: int = 2,
    limit_nodes: int = 40,
) -> APIResponse[GraphResponse]:
    """G1 — K-hop subgraph around a single entity.

    The `entity_id` argument is the slug-form id the frontend already holds
    (e.g. `entity:dropout`). Returns the same `GraphResponse` shape as `POST
    /graph` so the existing GraphCanvas can render it without any new node
    types. Used by the frontend "expand around this node" interaction and as
    a preview after a graph-anchored query.
    """
    verify_owner_access(request, owner_id)
    if not collection_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="collection_id is required for entity subgraph retrieval",
        )
    if hops not in (1, 2):
        hops = 2
    from src.rag.graph_retriever import GraphRetriever
    from src.rag.types import RetrievalScope

    settings = get_app_settings()
    retriever = GraphRetriever(settings)
    scope = RetrievalScope(owner_id=owner_id, collection_id=collection_id)
    entities, relations = await retriever.subgraph_around_entities(
        entity_slugs=[entity_id], scope=scope, hops=hops,
    )
    if not entities:
        return APIResponse(
            success=True,
            message="No subgraph found for this entity",
            data=GraphResponse(nodes=[], edges=[]),
            error=None,
        )

    # Dedupe entities by slug — extraction sometimes emits multiple entries with
    # the same canonical_name across chunks/docs. Keep the one with the most
    # mentions (most informative for the user).
    by_slug: dict[str, Entity] = {}
    for e in entities:
        nid = _entity_slug(e.canonical_name)  # "entity:slug-form"
        existing = by_slug.get(nid)
        if existing is None or len(e.mention_refs or []) > len(existing.mention_refs or []):
            by_slug[nid] = e

    # Render with the same conventions as POST /graph: slug-style ids, label
    # from canonical_name, basic node metadata. Relations turn into edges
    # using `source_id` / `target_id` already in slug form.
    nodes: list[GraphNode] = []
    target_seed = entity_id if entity_id.startswith("entity:") else f"entity:{entity_id}"
    for nid, e in list(by_slug.items())[:limit_nodes]:
        is_seed = nid == target_seed
        nodes.append(
            GraphNode(
                id=nid,
                label=e.canonical_name,
                type=e.entity_type,
                confidence=e.confidence,
                mention_count=len(e.mention_refs or []),
                is_focused=is_seed,
                is_hub=is_seed,
                source_docs=[],
                evidence_refs=_evidence_refs(e.mention_refs[:3]),
            )
        )

    edges: list[GraphEdge] = []
    visible_ids = {n.id for n in nodes}
    for r in relations:
        if r.source_id in visible_ids and r.target_id in visible_ids:
            edges.append(
                GraphEdge(
                    source=r.source_id,
                    target=r.target_id,
                    relation_type=r.relation_type,
                    confidence=r.confidence,
                    evidence_count=len(r.evidence_refs or []),
                    evidence_refs=_evidence_refs(r.evidence_refs or []),
                    evidence_text_chunk=r.evidence_text_chunk,
                )
            )

    return APIResponse(
        success=True,
        message="Entity subgraph loaded successfully",
        data=GraphResponse(nodes=nodes, edges=edges),
        error=None,
    )


@router.post("/auto", response_model=APIResponse[AutoVizResponse])
async def auto_viz(
    request: Request,
    body: MindmapRequest,
    settings: Settings = Depends(get_app_settings),
) -> APIResponse[AutoVizResponse]:
    """Structure-adaptive visualization: pick the viz mode from MEASURED document
    structure (not domain name) and return the matching payload.

    - hierarchy / citation_network → nested section `tree` (good for legal/manuals)
    - concept_graph → `signals` only; the client then calls POST /graph
      (avoids duplicating the 200-line graph builder here)
    - timeline → falls back to tree/graph until a dedicated builder lands
    """
    verify_owner_access(request, body.owner_id)
    scope = _scope_query(body)

    entity_query = dict(scope)
    relation_query = dict(scope)
    page_query = dict(scope)
    if body.material_ids:
        material_oids = [PydanticObjectId(m) for m in body.material_ids]
        entity_query["mention_refs.material_id"] = {"$in": material_oids}
        relation_query["evidence_refs.material_id"] = {"$in": material_oids}
        page_query["material_id"] = {"$in": material_oids}

    entities = await Entity.find(entity_query).limit(settings.graph_max_entities_fetch).to_list()
    relations = await Relation.find(relation_query).limit(settings.graph_max_relations_fetch).to_list()
    pages = await MaterialPageDocument.find(page_query).to_list()

    # Collect headings (ordered) + all block texts for the signal computations.
    heading_items: list[HeadingItem] = []
    block_texts: list[str] = []
    text_block_count = 0
    # Focus = which sections back the last answer. Citations give CONTENT block
    # ids / pages; we map each cited block to the heading section that owns it
    # (nearest preceding heading in reading order) so the tree can be pruned to
    # just the cited Điều + their ancestor chapters.
    focus_block_ids = set(body.focus_block_ids or [])
    focus_pages = set(body.focus_pages or [])
    is_focus = bool(focus_block_ids or focus_pages)
    relevant_heading_ids: set[str] = set()

    # Per-section body text (heading → concatenated content) for citation edges.
    sections: list[tuple[HeadingItem, list[str]]] = []
    current_section: tuple[HeadingItem, list[str]] | None = None

    ordered_pages = sorted(pages, key=lambda p: (str(p.material_id), p.page_number))
    last_heading_by_material: dict[str, str] = {}
    for page in ordered_pages:
        mat = str(page.material_id)
        for block in sorted(page.blocks, key=lambda b: b.reading_order):
            content = (block.content or "").strip()
            if not content:
                continue
            block_texts.append(content)
            text_block_count += 1
            if block.block_type == "heading":
                hi = HeadingItem(
                    text=content, material_id=mat, page=page.page_number, block_id=block.block_id,
                )
                heading_items.append(hi)
                current_section = (hi, [])
                sections.append(current_section)
                last_heading_by_material[mat] = block.block_id
            elif current_section is not None:
                current_section[1].append(content)
            if is_focus:
                page_key = f"{mat}:{page.page_number}"
                cited = block.block_id in focus_block_ids or page_key in focus_pages
                if cited:
                    if block.block_type == "heading":
                        relevant_heading_ids.add(block.block_id)
                    elif mat in last_heading_by_material:
                        relevant_heading_ids.add(last_heading_by_material[mat])

    signals = detect_structure(
        headings=[h.text for h in heading_items],
        total_text_blocks=text_block_count,
        block_texts=block_texts,
        entities=entities,
        relations=relations,
        viz_config=settings.viz_config,
    )

    mode = signals.recommended_mode
    tree: list[MindmapNode] = []
    graph: GraphResponse | None = None

    if mode in {"hierarchy", "citation_network"} and heading_items:
        root_topic = body.root_topic or "Cấu trúc tài liệu"
        # Citation network = a real node-edge graph (Điều + cross-references) so
        # the Knowledge Graph tab stays a graph, not a tree. The hierarchy tree
        # is still returned (for the Mindmap tab / outline use).
        section_texts = [(hi, " ".join(texts)) for hi, texts in sections]
        graph_mode = getattr(body, "graph_mode", "auto")
        if graph_mode == "auto":
            use_query_text = not bool(relevant_heading_ids)
        elif graph_mode == "verify":
            use_query_text = False
        else:  # explore
            use_query_text = True
        c_nodes, c_edges = build_citation_network(
            sections=section_texts,
            viz_config=settings.viz_config,
            focus_block_ids=relevant_heading_ids if is_focus else None,
            focus_query_text=(body.focus_query_text or None) if is_focus else None,
            use_query_text_signal=use_query_text,
        )
        if c_nodes:
            graph = GraphResponse(nodes=c_nodes, edges=c_edges)
        tree = build_hierarchy_tree(
            root_topic=root_topic, headings=heading_items, viz_config=settings.viz_config,
        )
        if is_focus and relevant_heading_ids:
            tree = prune_tree_to_focus(tree, relevant_heading_ids)
        # Nothing usable from structure → let the client fall back to concept graph.
        if not graph and not tree:
            mode = "concept_graph"

    viz_signals = VizSignals(
        hierarchy=signals.hierarchy,
        reference=signals.reference,
        semantic=signals.semantic,
        temporal=signals.temporal,
        counts=signals.counts,
    )
    return APIResponse(
        success=True,
        message=f"Visualization mode '{mode}' selected from document structure",
        data=AutoVizResponse(viz_mode=mode, signals=viz_signals, tree=tree, graph=graph),
        error=None,
    )


@router.post("/mindmap", response_model=APIResponse[MindmapResponse])
async def mindmap(
    request: Request,
    body: MindmapRequest,
    settings: Settings = Depends(get_app_settings),
) -> APIResponse[MindmapResponse]:
    verify_owner_access(request, body.owner_id)
    query = _scope_query(body)
    if body.material_ids:
        query["mention_refs.material_id"] = {"$in": [PydanticObjectId(material_id) for material_id in body.material_ids]}
    raw_entities = (
        await Entity.find(query)
        .sort("-confidence")
        .limit(settings.graph_mindmap_entity_fetch)
        .to_list()
    )
    entities: list[Entity] = []
    seen_labels: set[str] = set()
    for entity in raw_entities:
        label = _clean_entity_label(entity.canonical_name)
        if label is None or not _is_display_entity(entity):
            continue
        label_key = label.casefold()
        if label_key in seen_labels:
            continue
        seen_labels.add(label_key)
        entities.append(entity)
        if len(entities) >= settings.graph_mindmap_entity_cap:
            break
    root_topic = body.root_topic or "Noelys Knowledge Map"
    chunks = await Chunk.find(_chunk_scope_query(body)).sort("-indexed_at").limit(80).to_list()
    result = None
    if body.use_llm:
        result = await _build_llm_mindmap(root_topic, chunks, settings=settings, detail_level=body.detail_level)
    if result is None:
        concepts = _extract_chunk_concepts(chunks, entities, detail_level=body.detail_level)
        result = _build_mindmap_from_concepts(root_topic, concepts, detail_level=body.detail_level)
    if not result.nodes:
        result = _build_thematic_mindmap(root_topic, entities)
    return APIResponse(success=True, message="Mindmap generated successfully", data=result, error=None)


