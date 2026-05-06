from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from collections import Counter, defaultdict

from beanie import PydanticObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.core.config import Settings
from src.core.model_factory import build_llm
from src.dependencies import get_app_settings, verify_owner_access
from src.models.chunk import Chunk
from src.models.knowledge_graph import Entity, Relation
from src.models.material import Material
from src.schemas.common import APIResponse
from src.schemas.graph import GraphEdge, GraphNode, GraphResponse, MindmapRequest
from src.schemas.mindmap import MindmapNode, MindmapResponse

router = APIRouter(prefix="/graph", tags=["graph"])

_CROSS_MODAL_TYPES = frozenset({"table", "figure", "equation"})
_NOISY_ENTITY_LABELS = frozenset(
    {
        "caption",
        "chart",
        "converted",
        "metadata",
        "docx",
        "file word",
        "jpg",
        "jpeg",
        "llm",
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
        "vlm",
        "word",
        "xlsx",
        "txt",
        "trong",
        "increases",
        "fail",
        "pass",
        "checklist",
        "hinh",
        "hình",
        "bang",
        "bảng",
        "cau",
        "câu",
        "nguon",
        "nguồn",
    }
)
_NOISY_ENTITY_WORDS = frozenset(
    {
        "adds",
        "description",
        "file",
        "source",
        "sources",
        "stabilizes",
        "stops",
        "technique",
        "fail",
        "pass",
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
    if len(words) == 1 and len(compact) <= 5 and lower not in {"chunk", "graph", "query", "ocr", "rag", "kan", "gru"}:
        return None
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
    if entity.confidence < 0.5 and len(entity.mention_refs) < 2:
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
    max_chunk_chars = 1600 if detail_level == "detailed" else 900
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

    max_topics = 8 if detail_level == "detailed" else 6
    max_concepts = 8 if detail_level == "detailed" else 5
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

    topic_count = 6 if detail_level == "overview" else 8
    concept_count = 5 if detail_level == "overview" else 8
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


def _relation_display_label(relation_type: str) -> str:
    labels = {
        "co_occurs_in_block": "cùng đoạn bằng chứng",
        "co_occurs_on_page": "cùng trang bằng chứng",
        "related_to": "liên quan",
        "part_of": "thuộc về",
        "causes": "gây ra",
        "affects": "ảnh hưởng",
        "includes": "bao gồm",
        "uses": "sử dụng",
        "compares_with": "so sánh với",
    }
    return labels.get(relation_type, relation_type.replace("_", " "))


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
async def graph(request: Request, body: MindmapRequest) -> APIResponse[GraphResponse]:
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
    raw_entities = await Entity.find(text_entity_query).sort("-confidence").limit(160).to_list()
    entities     = [entity for entity in raw_entities if _is_display_entity(entity)][:50]
    relations    = await Relation.find(relation_query).limit(120).to_list()
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

    nodes_by_id = {
        _entity_slug(entity.canonical_name): GraphNode(
            id=_entity_slug(entity.canonical_name),
            label=_short_label(_clean_entity_label(entity.canonical_name) or entity.canonical_name, limit=40),
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
        for entity in all_entities
    }

    nodes = list(nodes_by_id.values())
    node_ids = {node.id for node in nodes}
    edges = []
    for relation in relations:
        if relation.source_id not in node_ids or relation.target_id not in node_ids:
            continue
        if not relation.evidence_refs:
            continue
        source_node = nodes_by_id.get(relation.source_id)
        target_node = nodes_by_id.get(relation.target_id)
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
            )
        )
    if not edges:
        edges = _entity_cooccurrence_edges(all_entities)
    edges.sort(key=lambda edge: ((edge.evidence_count or 0), edge.confidence or 0), reverse=True)
    result = GraphResponse(nodes=nodes, edges=edges)
    return APIResponse(success=True, message="Graph loaded successfully", data=result, error=None)


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
    raw_entities = await Entity.find(query).sort("-confidence").limit(160).to_list()
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
        if len(entities) >= 60:
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


