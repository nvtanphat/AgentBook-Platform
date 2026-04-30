from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.adaptation.calibration import CalibrationPoint, ThresholdCalibrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate AgentBook retrieval/refusal thresholds from scored examples.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "model_adaptation_config.yaml")
    parser.add_argument("--scores", type=Path, required=True, help="JSON list with score and is_relevant fields")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "results" / "calibration_report.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    thresholds = [float(value) for value in config.get("calibration", {}).get("candidate_thresholds", [])]
    if not thresholds:
        thresholds = [0.55]
    data = json.loads(args.scores.read_text(encoding="utf-8"))
    points = [CalibrationPoint(score=float(item["score"]), is_relevant=bool(item["is_relevant"])) for item in data]
    report = ThresholdCalibrator().calibrate(points, thresholds)
    payload = report.__dict__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
