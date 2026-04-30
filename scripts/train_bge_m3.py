from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate readiness for BGE-M3 fine-tuning.")
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/results/model_adaptation/manifest.json"))
    parser.add_argument("--min-pairs", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs = int(manifest.get("retrieval_pairs", 0))
    if pairs < args.min_pairs:
        raise SystemExit(f"Not enough clean retrieval pairs for BGE-M3 fine-tuning: {pairs}/{args.min_pairs}. Use calibration path.")
    print(json.dumps({"status": "ready", "retrieval_pairs": pairs, "training_entrypoint": "sentence-transformers MultipleNegativesRankingLoss"}, indent=2))


if __name__ == "__main__":
    main()
