"""
Validate AgentBook benchmark datasets for schema compliance, anchor integrity,
and data quality (no duplicates, no split leakage).

Usage:
    cd backend
    # Validate a single artifact
    python scripts/validate_benchmark_dataset.py \
        --e2e ../evaluation/datasets/agentbook_e2e_gold.jsonl \
        --owner-id user_demo \
        --collection-id <ID>

    # Validate all artifacts together
    python scripts/validate_benchmark_dataset.py \
        --meta   ../evaluation/datasets/agentbook_meta_dataset.jsonl \
        --retrieval ../evaluation/datasets/agentbook_retrieval_gold.jsonl \
        --e2e    ../evaluation/datasets/agentbook_e2e_gold.jsonl \
        --adversarial ../evaluation/datasets/agentbook_adversarial.jsonl \
        --owner-id user_demo \
        --collection-id <ID> \
        --check-anchors
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Schema definitions ─────────────────────────────────────────────────────────

_META_REQUIRED = {"record_id", "type", "chunk_id", "material_id", "document_name", "collection_id", "owner_id"}
_RETRIEVAL_REQUIRED = {"case_id", "query", "expected_docs"}
_E2E_REQUIRED = {"case_id", "task_type", "query", "expected_behavior"}
_ADV_REQUIRED = {"case_id", "task_type", "query", "expected_behavior"}

_E2E_TASK_TYPES = {
    "factual", "compare", "comparison", "summarize", "summarization",
    "study_guide", "table", "graph_relation", "ocr", "audio",
    "refusal", "cross_lingual", "false_premise", "prompt_injection", "claim_check",
}
_E2E_BEHAVIORS = {"answer", "refuse", "ask_clarification"}
_E2E_DIFFICULTIES = {"easy", "medium", "hard", "adversarial"}


def _load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] {p.name}:{lineno} invalid JSON: {exc}", flush=True)
    return rows


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


# ── Validators ─────────────────────────────────────────────────────────────────

def validate_schema(rows: list[dict], required: set[str], label: str) -> list[str]:
    """Return list of error messages for missing required fields."""
    errors: list[str] = []
    for i, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            cid = row.get("case_id") or row.get("record_id") or f"row#{i+1}"
            errors.append(f"  [{label}] {cid}: missing fields {missing}")
    return errors


def validate_e2e_values(rows: list[dict]) -> list[str]:
    """Check enum values for e2e gold rows."""
    errors: list[str] = []
    for row in rows:
        cid = row.get("case_id", "?")
        tt = row.get("task_type", "")
        if tt and tt not in _E2E_TASK_TYPES:
            errors.append(f"  [e2e] {cid}: unknown task_type '{tt}'")
        eb = row.get("expected_behavior", "")
        if eb and eb not in _E2E_BEHAVIORS:
            errors.append(f"  [e2e] {cid}: unknown expected_behavior '{eb}'")
        diff = row.get("difficulty", "")
        if diff and diff not in _E2E_DIFFICULTIES:
            errors.append(f"  [e2e] {cid}: unknown difficulty '{diff}'")
        ee = row.get("expected_evidence", [])
        if row.get("expected_behavior") == "answer" and not ee:
            errors.append(f"  [e2e] {cid}: expected_behavior=answer but no expected_evidence")
    return errors


def validate_evidence_anchors(rows: list[dict], label: str) -> list[str]:
    """Check evidence anchors have document_name + page/chunk_id."""
    errors: list[str] = []
    for row in rows:
        cid = row.get("case_id") or row.get("record_id") or "?"
        evidences = row.get("expected_evidence") or []
        for i, ev in enumerate(evidences):
            if not ev.get("document_name"):
                errors.append(f"  [{label}] {cid}: evidence[{i}] missing document_name")
            if not ev.get("page") and not ev.get("chunk_id") and not ev.get("block_id"):
                errors.append(f"  [{label}] {cid}: evidence[{i}] missing page/chunk_id/block_id anchor")
    return errors


def validate_no_duplicates(rows: list[dict], label: str, threshold: float = 0.92) -> list[str]:
    """Check for exact and near-duplicate queries."""
    errors: list[str] = []
    queries = [row.get("query", "") for row in rows]
    seen_exact: set[str] = set()
    duplicates: list[str] = []
    for q in queries:
        if q in seen_exact:
            duplicates.append(q)
        seen_exact.add(q)
    if duplicates:
        errors.append(f"  [{label}] {len(duplicates)} exact duplicate queries found")

    near_dups = 0
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            if _jaccard(queries[i], queries[j]) >= threshold:
                near_dups += 1
    if near_dups:
        errors.append(f"  [{label}] {near_dups} near-duplicate query pairs (Jaccard≥{threshold})")
    return errors


def validate_unique_case_ids(rows: list[dict], label: str) -> list[str]:
    """Check case_id / record_id uniqueness."""
    id_field = "case_id" if rows and "case_id" in rows[0] else "record_id"
    ids = [r.get(id_field) for r in rows]
    seen: set = set()
    dups: list = []
    for cid in ids:
        if cid in seen:
            dups.append(cid)
        seen.add(cid)
    if dups:
        return [f"  [{label}] duplicate IDs: {dups[:5]}{'...' if len(dups)>5 else ''}"]
    return []


def validate_owner_scope(rows: list[dict], owner_id: str, collection_id: str, label: str) -> list[str]:
    """Check all rows belong to the stated owner/collection (when field present)."""
    errors: list[str] = []
    for row in rows:
        ro = row.get("owner_id")
        rc = row.get("collection_id")
        if ro and ro != owner_id:
            errors.append(f"  [{label}] owner mismatch: {ro!r} != {owner_id!r}")
        if rc and rc != collection_id:
            errors.append(f"  [{label}] collection mismatch: {rc!r} != {collection_id!r}")
    return errors


async def validate_mongo_anchors(rows: list[dict], owner_id: str, collection_id: str, label: str) -> list[str]:
    """Optional: verify chunk_ids exist in MongoDB."""
    errors: list[str] = []
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
        from src.core.config import get_settings
        from src.database import init_database
        from src.models.chunk import Chunk
        from beanie import PydanticObjectId

        settings = get_settings()
        await init_database(settings)

        col_oid = PydanticObjectId(collection_id)
        missing = 0
        checked = 0
        for row in rows:
            for ev in row.get("expected_evidence", []):
                chunk_id = ev.get("chunk_id")
                if not chunk_id:
                    continue
                checked += 1
                try:
                    exists = await Chunk.find_one(
                        Chunk.id == PydanticObjectId(chunk_id),
                        Chunk.owner_id == owner_id,
                        Chunk.collection_id == col_oid,
                    )
                    if not exists:
                        missing += 1
                except Exception:
                    missing += 1
        if missing:
            errors.append(f"  [{label}] {missing}/{checked} chunk_id anchors not found in MongoDB")
    except Exception as exc:
        errors.append(f"  [{label}] anchor DB check failed: {type(exc).__name__}: {exc}")
    return errors


# ── Report ─────────────────────────────────────────────────────────────────────

def _print_section(title: str, rows: list[dict], errors: list[str]) -> bool:
    ok = not errors
    badge = "✅" if ok else "❌"
    print(f"\n{badge} {title}: {len(rows)} records")
    if errors:
        for e in errors[:20]:
            print(e, flush=True)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
    return ok


async def main(args: argparse.Namespace) -> None:
    all_pass = True
    w = 60

    print(f"\n{'='*w}")
    print(f"  AgentBook Benchmark Dataset Validator")
    print(f"  Owner:      {args.owner_id}")
    print(f"  Collection: {args.collection_id}")
    print(f"{'='*w}\n")

    # ── meta dataset ──────────────────────────────────────────────────────────
    if args.meta:
        meta_rows = _load_jsonl(args.meta)
        errs = []
        errs += validate_schema(meta_rows, _META_REQUIRED, "meta")
        errs += validate_unique_case_ids(meta_rows, "meta")
        errs += validate_owner_scope(meta_rows, args.owner_id, args.collection_id, "meta")
        ok = _print_section(f"Meta inventory ({Path(args.meta).name})", meta_rows, errs)
        all_pass = all_pass and ok

    # ── retrieval gold ─────────────────────────────────────────────────────────
    if args.retrieval:
        ret_rows = _load_jsonl(args.retrieval)
        errs = []
        errs += validate_schema(ret_rows, _RETRIEVAL_REQUIRED, "retrieval")
        errs += validate_unique_case_ids(ret_rows, "retrieval")
        errs += validate_no_duplicates(ret_rows, "retrieval")
        ok = _print_section(f"Retrieval gold ({Path(args.retrieval).name})", ret_rows, errs)
        all_pass = all_pass and ok

    # ── e2e gold ───────────────────────────────────────────────────────────────
    if args.e2e:
        e2e_rows = _load_jsonl(args.e2e)
        errs = []
        errs += validate_schema(e2e_rows, _E2E_REQUIRED, "e2e")
        errs += validate_e2e_values(e2e_rows)
        errs += validate_evidence_anchors(e2e_rows, "e2e")
        errs += validate_unique_case_ids(e2e_rows, "e2e")
        errs += validate_no_duplicates(e2e_rows, "e2e")
        if args.check_anchors:
            errs += await validate_mongo_anchors(e2e_rows, args.owner_id, args.collection_id, "e2e")
        ok = _print_section(f"E2E gold ({Path(args.e2e).name})", e2e_rows, errs)
        all_pass = all_pass and ok

        # Coverage breakdown
        if e2e_rows:
            from collections import Counter
            tt_counts = Counter(r.get("task_type", "?") for r in e2e_rows)
            diff_counts = Counter(r.get("difficulty", "?") for r in e2e_rows)
            lang_counts = Counter(r.get("query_language", "?") for r in e2e_rows)
            print(f"  Task types:   {dict(tt_counts)}")
            print(f"  Difficulties: {dict(diff_counts)}")
            print(f"  Languages:    {dict(lang_counts)}")

    # ── adversarial ────────────────────────────────────────────────────────────
    if args.adversarial:
        adv_rows = _load_jsonl(args.adversarial)
        errs = []
        errs += validate_schema(adv_rows, _ADV_REQUIRED, "adversarial")
        errs += validate_unique_case_ids(adv_rows, "adversarial")
        errs += validate_no_duplicates(adv_rows, "adversarial")
        ok = _print_section(f"Adversarial ({Path(args.adversarial).name})", adv_rows, errs)
        all_pass = all_pass and ok

    # ── cross-split leakage check ──────────────────────────────────────────────
    artifacts = {}
    if args.retrieval:
        artifacts["retrieval"] = [r.get("query","") for r in _load_jsonl(args.retrieval)]
    if args.e2e:
        artifacts["e2e"] = [r.get("query","") for r in _load_jsonl(args.e2e)]
    if args.adversarial:
        artifacts["adversarial"] = [r.get("query","") for r in _load_jsonl(args.adversarial)]

    leaks = 0
    artifact_names = list(artifacts.keys())
    for i in range(len(artifact_names)):
        for j in range(i+1, len(artifact_names)):
            a, b = artifact_names[i], artifact_names[j]
            shared = set(artifacts[a]) & set(artifacts[b])
            if shared:
                leaks += len(shared)
                print(f"\n  [WARN] {leaks} exact query overlap between {a} and {b}")
    if leaks == 0 and len(artifacts) > 1:
        print(f"\n  ✅ No exact query overlap between artifacts")

    # ── final verdict ─────────────────────────────────────────────────────────
    print(f"\n{'='*w}")
    if all_pass:
        print("  ✅ ALL CHECKS PASSED")
    else:
        print("  ❌ VALIDATION FAILED — fix errors above before running eval")
    print(f"{'='*w}\n")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate AgentBook benchmark datasets")
    parser.add_argument("--meta", help="Path to agentbook_meta_dataset.jsonl")
    parser.add_argument("--retrieval", help="Path to agentbook_retrieval_gold.jsonl")
    parser.add_argument("--e2e", help="Path to agentbook_e2e_gold.jsonl")
    parser.add_argument("--adversarial", help="Path to agentbook_adversarial.jsonl")
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--check-anchors", action="store_true",
                        help="Verify chunk_id anchors exist in MongoDB (requires DB connection)")
    args = parser.parse_args()

    if not any([args.meta, args.retrieval, args.e2e, args.adversarial]):
        parser.error("At least one of --meta / --retrieval / --e2e / --adversarial is required")

    asyncio.run(main(args))
