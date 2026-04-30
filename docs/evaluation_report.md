# AgentBook Evaluation Report

## Current Status

The evaluation framework is implemented with small seeded examples. Full metrics are placeholders until the project corpus reaches the planned evaluation scale.

## Implemented Datasets

- Gold QA: `evaluation/datasets/gold_qa_pairs.json`
- Cross-lingual: `evaluation/datasets/cross_lingual.json`
- False premise: `evaluation/datasets/false_premise.json`

## Metrics

Implemented:

- Recall@k
- Precision@k
- MRR@k
- nDCG@k
- Citation accuracy
- RAGAS integration stub

## Ablation Configs

- A1 Dense-only vs Hybrid Retrieval
- A2 Flat chunking vs Layout-aware chunking
- A3 No reranking vs Cross-encoder reranking
- A4 No refusal gate vs refusal gate
- A5 Hybrid RAG vs Hybrid + Graph
- A6 No claim verifier vs claim verifier

## Placeholder Results

The seeded examples run through `evaluation/run_eval.py`; without real predictions, metrics are expected to be `0.0`. This is intentional. Once a representative corpus is indexed, save predictions with `retrieved_evidence[]` and `citations[]`, then rerun:

```bash
python scripts/run_ablation_suite.py
```

## Model Adaptation

Phase 5 generated:

- `retrieval_pairs.jsonl`
- `qa_instruction.jsonl`
- `manifest.json`

The current sample dataset is too small for fine-tuning, so AgentBook uses calibrated thresholds over base BGE-M3 until enough clean data exists.
