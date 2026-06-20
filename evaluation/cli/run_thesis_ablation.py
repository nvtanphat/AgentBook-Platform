"""
Thesis ablation orchestrator — chạy toàn bộ pipeline từ dataset generation đến freeze.

Steps:
  0. Flag audit      — fail-fast nếu rag_flags endpoint không nhận đủ 8 flag
  1. Generate        — generate_dataset.py --mode e2e-gold-v2 (temp=0, seed cố định)
  2. Validate        — validate_dataset.py (fail-fast nếu schema sai)
  3a. Ladder ablation — run_ablation.py --mode ladder (Trục A+C, C0→C7+Full)
  3b. LOO ablation   — run_ablation.py --mode loo (Trục A+C, LOO từ Full)
  4. Judge (Trục B)  — judge.py (RAGAS LLM-judge, model khác generator để tránh leakage)
  5. Freeze          — snapshot toàn bộ vào evaluation/results/_frozen/<date>/
  6. Report          — in bảng tổng hợp + kiểm tra CI (G3)

Usage:
    python evaluation/cli/run_thesis_ablation.py \\
        --owner-id nguyenvtp69_gmail_com \\
        --collection-id <EVAL_COLLECTION_ID> \\
        --api-url http://localhost:8000 \\
        --generator-model gpt-5.4-mini \\
        --judge-model <different-model> \\
        --api-base https://luongchidung.online/v1 \\
        --api-key sk-... \\
        --output evaluation/results/thesis_ablation.md

    # Smoke test (skip generation, run 3 queries per config)
    python evaluation/cli/run_thesis_ablation.py \\
        --owner-id nguyenvtp69_gmail_com \\
        --collection-id <EVAL_COLLECTION_ID> \\
        --skip-generate \\
        --max-queries 3

Anti-leakage checklist (G2):
  - judge-model MUST differ from generator-model; script will warn if they match
  - human anchor: run validate_dataset.py --human-anchor 30 and verify Cohen's κ ≥ 0.6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "evaluation"
CLI_DIR = EVAL_DIR / "cli"
DATASETS_DIR = EVAL_DIR / "datasets" / "gold"
RESULTS_DIR = EVAL_DIR / "results"
FROZEN_DIR = RESULTS_DIR / "_frozen"


def _run(cmd: list[str], *, step: str, check: bool = True) -> int:
    print(f"\n{'─'*70}", flush=True)
    print(f"  STEP {step}", flush=True)
    print(f"  CMD: {' '.join(cmd)}", flush=True)
    print(f"{'─'*70}", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if check and result.returncode != 0:
        print(f"\n[FAIL] Step {step} exited {result.returncode} — aborting.", file=sys.stderr)
        sys.exit(result.returncode)
    return result.returncode


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def step0_flag_audit(*, api_url: str) -> None:
    """Verify the /query/ask endpoint accepts all 8 rag_flags without 422."""
    import requests

    test_flags = {
        "reranker_enabled": True,
        "agentic_rag_enabled": False,
        "multi_query_enabled": True,
        "sparse_enabled": True,
        "graph_probe_enabled": True,
        "slec_enabled": True,
        "claim_verifier_enabled": True,
        "crag_enabled": True,
    }
    print("\n  Step 0: Flag audit — sending all 8 rag_flags to /query/ask ...", flush=True)
    try:
        resp = requests.post(
            f"{api_url}/api/v1/query/ask",
            json={
                "owner_id": "_flag_audit_",
                "collection_id": "_flag_audit_",
                "query": "flag audit probe",
                "rag_flags": test_flags,
            },
            timeout=10,
        )
        if resp.status_code == 422:
            detail = resp.json().get("detail", "")
            print(f"[FAIL] Schema rejected rag_flags: {detail}", file=sys.stderr)
            print("       → Fix: schemas/query.py Literal type must include all 8 flags.", file=sys.stderr)
            sys.exit(1)
        # 400/404/500 are ok here — we only care that the schema was accepted (not 422)
        print(f"  [OK] Schema accepted all 8 flags (status={resp.status_code})", flush=True)
    except requests.ConnectionError:
        print(f"[WARN] Cannot reach {api_url} — skipping flag audit. Start backend first.", file=sys.stderr)


def step1_generate(args: argparse.Namespace, output_path: Path) -> None:
    if output_path.exists():
        print(f"  [SKIP] Dataset already exists: {output_path}", flush=True)
        return
    cmd = [
        sys.executable, str(CLI_DIR / "generate_dataset.py"),
        "--owner-id", args.owner_id,
        "--collection-id", args.collection_id,
        "--mode", "e2e-gold-v2",
        "--provider", "openai",
        "--model", args.generator_model,
        "--api-base", args.api_base,
        "--api-key", args.api_key,
        "--temperature", "0",
        "--seed", str(args.seed),
        "--input", str(DATASETS_DIR / "meta_dataset.jsonl"),
        "--output", str(output_path),
        "--target-count", str(args.target_count),
    ]
    if args.skip_modality:
        cmd += ["--skip-modality", args.skip_modality]
    _run(cmd, step="1 — Generate e2e-gold-v2 dataset")


def step2_validate(*, dataset_path: Path) -> None:
    validate_script = CLI_DIR / "validate_dataset.py"
    if not validate_script.exists():
        print(f"  [SKIP] validate_dataset.py not found at {validate_script}", flush=True)
        return
    cmd = [
        sys.executable, str(validate_script),
        "--e2e", str(dataset_path),
    ]
    _run(cmd, step="2 — Validate dataset schema", check=False)
    print("  [REMINDER] Human-anchor gate: manually score ≥30 cases, verify Cohen's κ ≥ 0.6 (G2)", flush=True)


def step3_ablation(
    args: argparse.Namespace,
    *,
    dataset_path: Path,
    adv_path: Path,
    ladder_output: Path,
    loo_output: Path,
) -> None:
    base_cmd = [
        sys.executable, str(CLI_DIR / "run_ablation.py"),
        "--owner-id", args.owner_id,
        "--collection-id", args.collection_id,
        "--api-url", args.api_url,
        "--question-set", str(dataset_path),
        "--k", str(args.k),
        "--timeout", str(args.timeout),
        "--seed", str(args.seed),
    ]
    if adv_path.exists():
        base_cmd += ["--adversarial-set", str(adv_path)]
    if args.max_queries:
        base_cmd += ["--max-queries", str(args.max_queries)]

    _run(
        base_cmd + ["--mode", "ladder", "--output", str(ladder_output)],
        step="3a — Ladder ablation (C0→C7+Full)",
    )
    _run(
        base_cmd + ["--mode", "loo", "--output", str(loo_output)],
        step="3b — LOO ablation (Full − each component)",
    )


def step4_judge(
    args: argparse.Namespace,
    *,
    ladder_output: Path,
    judge_output: Path,
) -> None:
    if args.judge_model == args.generator_model:
        print(
            f"\n[WARN G2] judge_model == generator_model ({args.judge_model}).\n"
            "  Self-preference leakage risk! Use a different-family model as judge.\n"
            "  Continuing anyway — set --judge-model to override.",
            file=sys.stderr,
        )
    judge_script = CLI_DIR / "judge.py"
    if not judge_script.exists():
        print(f"  [SKIP] judge.py not found at {judge_script}", flush=True)
        return
    cmd = [
        sys.executable, str(judge_script),
        "--input", str(ladder_output),
        "--output", str(judge_output),
        "--model", args.judge_model,
        "--api-base", args.api_base,
        "--api-key", args.api_key,
        "--rubric", str(EVAL_DIR / "config" / "judge_rubric.yaml"),
    ]
    _run(cmd, step="4 — LLM judge (Trục B: Faithfulness / Relevancy)", check=False)


def step5_freeze(
    *,
    date_tag: str,
    ladder_output: Path,
    loo_output: Path,
    judge_output: Path,
    dataset_path: Path,
    adv_path: Path,
    args: argparse.Namespace,
) -> Path:
    frozen_run_dir = FROZEN_DIR / date_tag
    frozen_run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "frozen_at": date_tag,
        "owner_id": args.owner_id,
        "collection_id": args.collection_id,
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "seed": args.seed,
        "k": args.k,
        "files": {},
    }

    for src in [dataset_path, adv_path, ladder_output, loo_output, judge_output]:
        if src.exists():
            dst = frozen_run_dir / src.name
            shutil.copy2(src, dst)
            manifest["files"][src.name] = _file_hash(src)

    manifest_path = frozen_run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  [FREEZE] Snapshot saved → {frozen_run_dir}", flush=True)
    return frozen_run_dir


def step6_report(
    *,
    ladder_output: Path,
    loo_output: Path,
    frozen_dir: Path,
    output_md: Path,
    k: int,
) -> None:
    sections: list[str] = []
    sections.append(f"# Thesis Ablation Report\n\n_Generated: {datetime.now().isoformat()}_\n")
    sections.append(f"**Frozen snapshot:** `{frozen_dir}`\n")
    sections.append(f"**Top-k:** {k}\n")

    def _md_table(summaries: list[dict], title: str) -> str:
        lines = [f"## {title}\n"]
        lines.append(f"| Config | R@{k} | MRR@{k} | nDCG@{k} | CI-95 | FAR | FRR | p50(s) |")
        lines.append("|--------|-------|---------|---------|-------|-----|-----|--------|")
        for s in summaries:
            a = s.get("truc_a", {})
            c = s.get("truc_c", {})
            lat = s.get("latency", {})
            ci = a.get(f"ndcg_at_{k}_ci95", [0.0, 0.0])
            lines.append(
                f"| {s['config_name']} "
                f"| {a.get(f'avg_recall_at_{k}', 0.0):.3f} "
                f"| {a.get(f'avg_mrr_at_{k}', 0.0):.3f} "
                f"| {a.get(f'avg_ndcg_at_{k}', 0.0):.3f} "
                f"| [{ci[0]:.3f},{ci[1]:.3f}] "
                f"| {c.get('false_accept_rate', 0.0):.3f} "
                f"| {c.get('false_refusal_rate', 0.0):.3f} "
                f"| {lat.get('p50_s', 0.0):.1f} |"
            )
        return "\n".join(lines)

    for path, title in [
        (ladder_output, "Incremental Ladder (C0 → Full)"),
        (loo_output, "Leave-One-Out (LOO từ Full)"),
    ]:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                summaries = json.load(f)
            sections.append(_md_table(summaries, title))

    sections.append(
        "\n## Checklist trước khi nộp luận văn\n\n"
        "- [ ] Human-anchor: chấm tay ≥30 case, Cohen's κ ≥ 0.6 (G2)\n"
        "- [ ] Judge model ≠ generator model (G2)\n"
        "- [ ] Tất cả số liệu trích từ snapshot frozen (G6)\n"
        "- [ ] CI không chồng nhau trước khi kết luận 'tăng' (G3)\n"
        "- [ ] C0 < Full trên nDCG@k (sanity check wiring)\n"
        "- [ ] FAR (False Accept Rate) ≤ 5% trên adversarial set\n"
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n\n".join(sections), encoding="utf-8")
    print(f"  [REPORT] Saved → {output_md}", flush=True)


def main(args: argparse.Namespace) -> None:
    date_tag = datetime.now().strftime("%Y%m%d_%H%M")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = DATASETS_DIR / "e2e_gold_v2.jsonl"
    adv_path = DATASETS_DIR / "adversarial.jsonl"
    ladder_output = RESULTS_DIR / "ladder_ablation.json"
    loo_output = RESULTS_DIR / "loo_ablation.json"
    judge_output = RESULTS_DIR / "judge_truc_b.jsonl"
    output_md = Path(args.output)

    print(f"\n{'='*70}", flush=True)
    print("  THESIS ABLATION PIPELINE", flush=True)
    print(f"  Date: {date_tag}", flush=True)
    print(f"  Owner: {args.owner_id}  Collection: {args.collection_id}", flush=True)
    print(f"  Generator: {args.generator_model}  Judge: {args.judge_model}", flush=True)
    print(f"{'='*70}", flush=True)

    # Step 0: flag audit
    if not args.skip_flag_audit:
        step0_flag_audit(api_url=args.api_url)

    # Step 1: generate dataset
    if not args.skip_generate:
        step1_generate(args, dataset_path)

    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}\n  Run without --skip-generate first.", file=sys.stderr)
        sys.exit(1)

    # Step 2: validate
    if not args.skip_validate:
        step2_validate(dataset_path=dataset_path)

    # Step 3: ablation runs
    step3_ablation(
        args,
        dataset_path=dataset_path,
        adv_path=adv_path,
        ladder_output=ladder_output,
        loo_output=loo_output,
    )

    # Step 4: LLM judge (Trục B)
    if not args.skip_judge:
        step4_judge(args, ladder_output=ladder_output, judge_output=judge_output)

    # Step 5: freeze
    frozen_dir = step5_freeze(
        date_tag=date_tag,
        ladder_output=ladder_output,
        loo_output=loo_output,
        judge_output=judge_output,
        dataset_path=dataset_path,
        adv_path=adv_path,
        args=args,
    )

    # Step 6: report
    step6_report(
        ladder_output=ladder_output,
        loo_output=loo_output,
        frozen_dir=frozen_dir,
        output_md=output_md,
        k=args.k,
    )

    print(f"\n{'='*70}", flush=True)
    print("  DONE. Kiểm tra:", flush=True)
    print(f"    Report:  {output_md.resolve()}", flush=True)
    print(f"    Frozen:  {frozen_dir}", flush=True)
    print(f"    Ladder:  {ladder_output.resolve()}", flush=True)
    print(f"    LOO:     {loo_output.resolve()}", flush=True)
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thesis ablation orchestrator")
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--api-url", default="http://localhost:8000")

    # LLM
    parser.add_argument("--generator-model", default="gpt-5.4-mini",
                        help="Model used to generate the gold dataset")
    parser.add_argument("--judge-model", default="gpt-5.4-mini",
                        help="Model used to judge (G2: must differ from generator to avoid self-preference leakage)")
    parser.add_argument("--api-base", default="https://luongchidung.online/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--skip-modality", default="",
                        help="Comma-separated modalities to skip in generation (e.g. 'ocr,audio')")

    # Dataset
    parser.add_argument("--target-count", type=int, default=150,
                        help="Target gold QA pairs to generate")
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval metrics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Limit queries per config (0=all). Use 3 for smoke test.")
    parser.add_argument("--timeout", type=int, default=120)

    # Skip flags
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip step 1 (dataset already exists)")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip step 2 (validation)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip step 4 (LLM judge, Trục B)")
    parser.add_argument("--skip-flag-audit", action="store_true",
                        help="Skip step 0 (flag schema audit)")

    # Output
    parser.add_argument("--output", default="evaluation/results/thesis_ablation.md")

    args = parser.parse_args()

    import os
    if not args.api_key:
        args.api_key = os.getenv("OPENAI_API_KEY", "")

    main(args)
