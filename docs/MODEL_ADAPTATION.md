# AgentBook Model Adaptation Path

Phase 5 uses calibration as the default MVP path. Fine-tuning is enabled only after the dataset quality and size gates pass.

## Generated Artifacts

- `evaluation/results/model_adaptation/retrieval_pairs.jsonl`
- `evaluation/results/model_adaptation/qa_instruction.jsonl`
- `evaluation/results/model_adaptation/manifest.json`
- `evaluation/results/calibration_report.json`

## Commands

```bash
python scripts/prepare_model_adaptation.py
python scripts/calibrate_thresholds.py --scores evaluation/results/sample_scores.json
python scripts/train_bge_m3.py --manifest evaluation/results/model_adaptation/manifest.json
python scripts/train_qlora_generation.py --manifest evaluation/results/model_adaptation/manifest.json --available-vram-gb 16
```

If the clean retrieval pair count is below the configured gate, keep BGE-M3 base and use calibrated thresholds instead of fine-tuning.
