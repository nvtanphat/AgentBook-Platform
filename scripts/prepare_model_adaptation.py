from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.adaptation.dataset_builder import ModelAdaptationDatasetBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build AgentBook model-adaptation JSONL datasets.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "model_adaptation_config.yaml")
    parser.add_argument("--dataset", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    dataset_config = config.get("dataset", {})
    datasets = args.dataset or [
        ROOT / "evaluation" / "datasets" / "gold_qa_pairs.json",
        ROOT / "evaluation" / "datasets" / "cross_lingual.json",
        ROOT / "evaluation" / "datasets" / "false_premise.json",
    ]
    output_dir = args.output_dir or ROOT / dataset_config.get("output_dir", "evaluation/results/model_adaptation")
    builder = ModelAdaptationDatasetBuilder(hard_negatives_per_query=dataset_config.get("hard_negatives_per_query", 2))
    examples = builder.load_examples(datasets)
    retrieval_pairs = builder.build_retrieval_pairs(examples)
    instructions = builder.build_instruction_examples(examples)
    retrieval_path = output_dir / "retrieval_pairs.jsonl"
    instruction_path = output_dir / "qa_instruction.jsonl"
    retrieval_count = builder.write_jsonl(retrieval_path, retrieval_pairs)
    instruction_count = builder.write_jsonl(instruction_path, instructions)
    manifest = {
        "examples": len(examples),
        "retrieval_pairs": retrieval_count,
        "instruction_examples": instruction_count,
        "retrieval_pairs_path": str(retrieval_path),
        "instruction_path": str(instruction_path),
        "ready_for_embedding_finetune": retrieval_count >= dataset_config.get("min_retrieval_pairs_for_finetune", 1000),
        "ready_for_qlora": instruction_count >= dataset_config.get("min_instruction_examples_for_qlora", 500),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
