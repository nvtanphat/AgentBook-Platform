from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate readiness for QLoRA generation adaptation.")
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/results/model_adaptation/manifest.json"))
    parser.add_argument("--min-examples", type=int, default=500)
    parser.add_argument("--available-vram-gb", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    examples = int(manifest.get("instruction_examples", 0))
    if examples < args.min_examples:
        raise SystemExit(f"Not enough QA instruction examples for QLoRA: {examples}/{args.min_examples}.")
    if args.available_vram_gb < 16:
        raise SystemExit(f"Insufficient VRAM for QLoRA 7B target: {args.available_vram_gb}GB/16GB.")
    print(json.dumps({"status": "ready", "instruction_examples": examples, "available_vram_gb": args.available_vram_gb}, indent=2))


if __name__ == "__main__":
    main()
