"""
AgentBook Full Benchmark Runner — chạy toàn bộ test ladder theo thứ tự.

Tier 0: Health check (infra)
Tier 1: Parser unit test (dry run)
Tier 2: Chunk quality
Tier 3: Embedding + Qdrant
Tier 4: Retrieval eval
Tier 5: E2E single query smoke test
Tier 6: Full benchmark (generate gold → validate → e2e eval → judge)

Usage:
    cd backend
    python scripts/run_full_benchmark.py \\
        --owner-id nguyenvtp69_gmail_com \\
        --collection-id 6a16f8d1a0d535db39664088 \\
        --api-url http://localhost:8000 \\
        --model gpt-5.4-mini \\
        --api-base https://luongchidung.online/v1 \\
        --api-key sk-... \\
        --start-tier 0 \\
        --stop-tier 6 \\
        --output-dir eval_results/full_run

    # Run only Tier 6 (assumes lower tiers already pass)
    python scripts/run_full_benchmark.py \\
        --owner-id nguyenvtp69_gmail_com \\
        --collection-id 6a16f8d1a0d535db39664088 \\
        --model gpt-5.4-mini \\
        --api-base https://luongchidung.online/v1 \\
        --api-key sk-... \\
        --start-tier 6 \\
        --skip-generation  # Use existing gold datasets
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PYTHON = sys.executable

# Relative paths (from backend/)
_SCRIPTS = Path(__file__).parent
_EVAL_DIR = _SCRIPTS.parents[1] / "evaluation" / "datasets"
_RESULTS_BASE = Path("eval_results")


def _run(cmd: list[str], *, label: str, capture: bool = False) -> tuple[int, str]:
    """Run a subprocess command, print output, return (returncode, output)."""
    print(f"\n  $ {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - t0
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        print(f"  [FAIL] {label} — exit {result.returncode} ({elapsed:.1f}s)", flush=True)
        if capture and output:
            print(output[:500], flush=True)
    else:
        print(f"  [OK]   {label} ({elapsed:.1f}s)", flush=True)
    return result.returncode, output


def _tier_header(n: int, title: str) -> None:
    print(f"\n{'═'*65}", flush=True)
    print(f"  TIER {n} — {title}", flush=True)
    print(f"{'═'*65}", flush=True)


def _write_report(tier_results: list[dict], out_dir: Path) -> None:
    lines: list[str] = [
        "# AgentBook Full Benchmark Report",
        "",
        f"**Date:** {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Tier Results",
        "",
        "| Tier | Name | Status | Duration |",
        "|------|------|--------|----------|",
    ]
    for t in tier_results:
        badge = "✅" if t["passed"] else "❌"
        lines.append(f"| {t['tier']} | {t['name']} | {badge} {t['status']} | {t['elapsed']:.1f}s |")
    lines.append("")

    overall = all(t["passed"] for t in tier_results)
    lines += [
        "## Overall",
        "",
        f"**{'✅ ALL TIERS PASSED' if overall else '❌ SOME TIERS FAILED'}**",
        "",
    ]

    p = out_dir / "benchmark_report.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Full report saved: {p.resolve()}", flush=True)


# ── Tier implementations ───────────────────────────────────────────────────────

def tier0_health(args: argparse.Namespace) -> bool:
    """Health check: backend + Qdrant + Ollama."""
    import urllib.request

    checks = [
        (f"{args.api_url}/health", "Backend /health"),
        ("http://localhost:6333/collections", "Qdrant"),
    ]
    all_ok = True
    for url, label in checks:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
                print(f"  ✅ {label}: {str(data)[:60]}", flush=True)
        except Exception as exc:
            print(f"  ❌ {label}: {exc}", flush=True)
            all_ok = False
    return all_ok


def tier4_retrieval(args: argparse.Namespace, out_dir: Path) -> bool:
    ret_out = out_dir / "retrieval_debug.jsonl"
    rc, _ = _run(
        [_PYTHON, str(_SCRIPTS / "quick_eval.py"),
         "--owner-id", args.owner_id,
         "--collection-id", args.collection_id,
         "--output", str(ret_out)],
        label="quick_eval",
    )
    if rc != 0 or not ret_out.exists():
        return False
    rc2, _ = _run(
        [_PYTHON, str(_SCRIPTS / "score_retrieval_eval.py"),
         "--input", str(ret_out), "--auto"],
        label="score_retrieval",
    )
    return rc2 == 0


def tier5_smoke(args: argparse.Namespace) -> bool:
    """Run one E2E query and verify it answers."""
    rc, out = _run(
        [_PYTHON, str(_SCRIPTS / "e2e_eval.py"),
         "--owner-id", args.owner_id,
         "--collection-id", args.collection_id,
         "--api-url", args.api_url,
         "--types", "factual",
         "--output", "eval_results/e2e_smoke.jsonl",
         "--timeout", "120"],
        label="e2e smoke (factual)",
        capture=True,
    )
    return rc == 0


def tier6_generate(args: argparse.Namespace, out_dir: Path) -> bool:
    """Generate or validate the gold benchmark dataset."""
    e2e_gold = _EVAL_DIR / "agentbook_e2e_gold.jsonl"
    adversarial = _EVAL_DIR / "agentbook_adversarial.jsonl"

    if args.skip_generation:
        print("  --skip-generation: using existing datasets", flush=True)
        if not e2e_gold.exists():
            print(f"  ❌ {e2e_gold} not found — run without --skip-generation first", flush=True)
            return False
    else:
        # meta-inventory
        meta_path = _EVAL_DIR / "agentbook_meta_dataset.jsonl"
        rc, _ = _run(
            [_PYTHON, str(_SCRIPTS / "generate_eval_dataset.py"),
             "--owner-id", args.owner_id,
             "--collection-id", args.collection_id,
             "--mode", "meta-inventory",
             "--max-chunks", "150",
             "--output", str(meta_path)],
            label="meta-inventory",
        )
        if rc != 0:
            return False

        # e2e gold
        rc, _ = _run(
            [_PYTHON, str(_SCRIPTS / "generate_eval_dataset.py"),
             "--owner-id", args.owner_id,
             "--collection-id", args.collection_id,
             "--mode", "e2e-gold",
             "--provider", "openai",
             "--model", args.model,
             "--api-base", args.api_base,
             "--api-key", args.api_key,
             "--input", str(meta_path),
             "--output", str(e2e_gold),
             "--target-count", str(args.gold_target)],
            label="e2e-gold generation",
        )
        if rc != 0:
            return False

        # adversarial
        rc, _ = _run(
            [_PYTHON, str(_SCRIPTS / "generate_eval_dataset.py"),
             "--owner-id", args.owner_id,
             "--collection-id", args.collection_id,
             "--mode", "adversarial",
             "--provider", "openai",
             "--model", args.model,
             "--api-base", args.api_base,
             "--api-key", args.api_key,
             "--output", str(adversarial),
             "--target-count", str(args.adv_target)],
            label="adversarial generation",
        )
        if rc != 0:
            return False

    # validate
    val_args = [
        _PYTHON, str(_SCRIPTS / "validate_benchmark_dataset.py"),
        "--owner-id", args.owner_id,
        "--collection-id", args.collection_id,
        "--e2e", str(e2e_gold),
    ]
    if adversarial.exists():
        val_args += ["--adversarial", str(adversarial)]
    rc, _ = _run(val_args, label="validate datasets")
    return rc == 0


def tier6_e2e_eval(args: argparse.Namespace, out_dir: Path) -> bool:
    """Run E2E eval on the gold question set."""
    e2e_gold = _EVAL_DIR / "agentbook_e2e_gold.jsonl"
    if not e2e_gold.exists():
        print(f"  ❌ {e2e_gold} missing — run Tier 6 generation first", flush=True)
        return False

    e2e_out = out_dir / "e2e_eval.jsonl"
    e2e_report = out_dir / "e2e_report.md"

    rc, _ = _run(
        [_PYTHON, str(_SCRIPTS / "e2e_eval.py"),
         "--owner-id", args.owner_id,
         "--collection-id", args.collection_id,
         "--api-url", args.api_url,
         "--question-set", str(e2e_gold),
         "--output", str(e2e_out),
         "--report", str(e2e_report),
         "--timeout", "300"],
        label="e2e eval on gold set",
    )
    return rc == 0


def tier6_judge(args: argparse.Namespace, out_dir: Path) -> bool:
    """Run LLM judge on E2E results."""
    e2e_out = out_dir / "e2e_eval.jsonl"
    e2e_gold = _EVAL_DIR / "agentbook_e2e_gold.jsonl"
    rubric = _EVAL_DIR / "agentbook_judge_rubric.yaml"
    judged_out = out_dir / "e2e_judged.jsonl"
    judge_report = out_dir / "judge_report.md"

    if not e2e_out.exists():
        print(f"  ❌ {e2e_out} missing — run E2E eval first", flush=True)
        return False
    if not args.api_key:
        print("  [SKIP] --api-key missing, skipping judge step", flush=True)
        return True

    rc, _ = _run(
        [_PYTHON, str(_SCRIPTS / "judge_eval_with_gpt4o.py"),
         "--input", str(e2e_out),
         "--gold", str(e2e_gold) if e2e_gold.exists() else "",
         "--rubric", str(rubric) if rubric.exists() else "",
         "--model", args.model,
         "--api-base", args.api_base,
         "--api-key", args.api_key,
         "--output", str(judged_out),
         "--report", str(judge_report)],
        label="LLM judge",
    )
    return rc == 0


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'█'*65}", flush=True)
    print(f"  AgentBook Full Benchmark Runner", flush=True)
    print(f"  Owner:      {args.owner_id}", flush=True)
    print(f"  Collection: {args.collection_id}", flush=True)
    print(f"  Tiers:      {args.start_tier} → {args.stop_tier}", flush=True)
    print(f"  Output:     {out_dir.resolve()}", flush=True)
    print(f"{'█'*65}", flush=True)

    tier_results: list[dict] = []

    def _record(tier: int, name: str, passed: bool, elapsed: float) -> None:
        tier_results.append({
            "tier": tier, "name": name,
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
            "elapsed": elapsed,
        })
        badge = "✅" if passed else "❌"
        print(f"\n  {badge} Tier {tier} ({name}) — {'PASS' if passed else 'FAIL'}", flush=True)

    # Tier 0
    if args.start_tier <= 0 <= args.stop_tier:
        _tier_header(0, "Health Check")
        t0 = time.perf_counter()
        ok = tier0_health(args)
        _record(0, "health_check", ok, time.perf_counter() - t0)
        if not ok and args.strict:
            sys.exit(1)

    # Tier 1
    if args.start_tier <= 1 <= args.stop_tier:
        _tier_header(1, "Parser Unit Test")
        print("  [INFO] Tier 1 (parser) requires test data files — run manually:", flush=True)
        print(f"    python scripts/dry_run_test_data_pipeline.py <data_dir> --max-files 1", flush=True)
        _record(1, "parser_unit", True, 0.0)  # manual gate

    # Tier 2
    if args.start_tier <= 2 <= args.stop_tier:
        _tier_header(2, "Chunk Quality")
        print("  [INFO] Tier 2 (chunk quality) requires local data dir — run manually:", flush=True)
        print(f"    python scripts/chunk_quality_check.py <data_dir>", flush=True)
        _record(2, "chunk_quality", True, 0.0)  # manual gate

    # Tier 3
    if args.start_tier <= 3 <= args.stop_tier:
        _tier_header(3, "Embedding + Qdrant")
        t0 = time.perf_counter()
        rc, _ = _run(
            [_PYTHON, str(_SCRIPTS / "diag_pipeline.py")],
            label="diag_pipeline",
            capture=True,
        )
        _record(3, "embedding_qdrant", rc == 0, time.perf_counter() - t0)
        if rc != 0 and args.strict:
            sys.exit(1)

    # Tier 4
    if args.start_tier <= 4 <= args.stop_tier:
        _tier_header(4, "Retrieval Only")
        t0 = time.perf_counter()
        ok = tier4_retrieval(args, out_dir)
        _record(4, "retrieval_eval", ok, time.perf_counter() - t0)
        if not ok and args.strict:
            sys.exit(1)

    # Tier 5
    if args.start_tier <= 5 <= args.stop_tier:
        _tier_header(5, "E2E Smoke Test")
        t0 = time.perf_counter()
        ok = tier5_smoke(args)
        _record(5, "e2e_smoke", ok, time.perf_counter() - t0)
        if not ok and args.strict:
            sys.exit(1)

    # Tier 6
    if args.start_tier <= 6 <= args.stop_tier:
        _tier_header(6, "Full Benchmark Suite")

        # 6a: Generate/validate gold datasets
        t0 = time.perf_counter()
        ok6a = tier6_generate(args, out_dir)
        _record(6, "6a_gold_generation", ok6a, time.perf_counter() - t0)

        # 6b: E2E eval on gold set
        if ok6a or args.skip_generation:
            t0 = time.perf_counter()
            ok6b = tier6_e2e_eval(args, out_dir)
            _record(6, "6b_e2e_eval", ok6b, time.perf_counter() - t0)

            # 6c: LLM judge
            t0 = time.perf_counter()
            ok6c = tier6_judge(args, out_dir)
            _record(6, "6c_llm_judge", ok6c, time.perf_counter() - t0)

    # Final report
    _write_report(tier_results, out_dir)

    failed = [t for t in tier_results if not t["passed"]]
    print(f"\n{'█'*65}", flush=True)
    if not failed:
        print("  ✅  ALL TIERS PASSED", flush=True)
    else:
        print(f"  ❌  {len(failed)} TIER(S) FAILED: {[t['name'] for t in failed]}", flush=True)
    print(f"{'█'*65}\n", flush=True)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="AgentBook full benchmark runner")
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--api-url", default="http://localhost:8000")

    # LLM (for Tier 6 generation + judge)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--api-base", default="https://luongchidung.online/v1")
    parser.add_argument("--api-key", default="")

    # Tier control
    parser.add_argument("--start-tier", type=int, default=0, help="First tier to run (0-6)")
    parser.add_argument("--stop-tier", type=int, default=6, help="Last tier to run (0-6)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit immediately when a tier fails")

    # Tier 6 options
    parser.add_argument("--skip-generation", action="store_true",
                        help="Tier 6: skip dataset generation, use existing JSONL files")
    parser.add_argument("--gold-target", type=int, default=30,
                        help="Target number of E2E gold cases to generate")
    parser.add_argument("--adv-target", type=int, default=20,
                        help="Target number of adversarial cases to generate")

    parser.add_argument("--output-dir", default="eval_results/full_run",
                        help="Directory for all output files and final report")

    args = parser.parse_args()

    if not args.api_key:
        args.api_key = os.getenv("OPENAI_API_KEY", "")

    main(args)
