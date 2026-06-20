from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.harness.metrics import EvidenceKey, citation_accuracy, mrr_at_k, ndcg_at_k, precision_at_k, ragas_stub, recall_at_k


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AgentBook retrieval/evidence evaluation over saved predictions.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--predictions", type=Path, default=None, help="Optional JSON predictions file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    dataset_path = Path(config["dataset"])
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    predictions = _load_predictions(args.predictions)
    k_values = config.get("metrics", {}).get("k_values", [5])

    rows = []
    for item in dataset:
        expected = [EvidenceKey.from_mapping(value) for value in item.get("expected_evidence", [])]
        prediction = predictions.get(item["id"], {})
        retrieved = [EvidenceKey.from_mapping(value) for value in prediction.get("retrieved_evidence", [])]
        citations = [EvidenceKey.from_mapping(value) for value in prediction.get("citations", [])]
        row = {"id": item["id"], "citation_accuracy": citation_accuracy(expected, citations)}
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(expected, retrieved, k)
            row[f"precision@{k}"] = precision_at_k(expected, retrieved, k)
            row[f"mrr@{k}"] = mrr_at_k(expected, retrieved, k)
            row[f"ndcg@{k}"] = ndcg_at_k(expected, retrieved, k)
        rows.append(row)

    summary = {
        "config": config.get("name", args.config.stem),
        "num_examples": len(rows),
        "metrics": _average_rows(rows),
        "ragas": ragas_stub(),
    }
    print(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))


def _load_predictions(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in data}


def _average_rows(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [key for key in rows[0] if key != "id"]
    return {key: round(mean(float(row[key]) for row in rows), 4) for key in keys}


if __name__ == "__main__":
    main()
