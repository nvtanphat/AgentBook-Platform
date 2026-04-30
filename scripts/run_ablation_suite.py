from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ABLATIONS = [
    "a1_hybrid_vs_vector.yaml",
    "a2_flat_vs_layout.yaml",
    "a3_no_rerank_vs_rerank.yaml",
    "a4_no_refusal_vs_refusal.yaml",
    "a5_hybrid_vs_graph.yaml",
    "a6_no_claim_verifier_vs_claim_verifier.yaml",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AgentBook ablation configs with the evaluation runner.")
    parser.add_argument("--configs-dir", type=Path, default=Path("evaluation/ablation_configs"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for config_name in ABLATIONS:
        config_path = args.configs_dir / config_name
        print(f"Running {config_path}")
        subprocess.run([sys.executable, "evaluation/run_eval.py", "--config", str(config_path)], check=True)


if __name__ == "__main__":
    main()
