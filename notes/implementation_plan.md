# Implementation Plan - Fuzzy Logic System for Response Gating

We will replace the hardcoded threshold (`min_reranker_score = 0.35`) with a multi-variable **Fuzzy Logic System** to decide whether to refuse or answer.

The system will evaluate four inputs:
1. **Reranker Score** ($x_1$): Sigmoid-normalized maximum rerank score of retrieved chunks.
2. **SLEC Coverage Ratio** ($x_2$): Sentence-level evidence coverage ratio from the SLEC gate.
3. **OCR Quality Score** ($x_3$): Average OCR confidence of the retrieved blocks (falling back to 1.0 for digital PDFs).
4. **CRAG Correctness Ratio** ($x_4$): Ratio of chunks classified as CORRECT by the CRAG evaluator.

## User Review Required

> [!IMPORTANT]
> - The Fuzzy Logic System introduces a smoother gating mechanism. Instead of sharp rejection boundaries, borderline cases will trigger a **Partial Confidence** warning (`uncertain` state, score in `[0.35, 0.65)`) which appends an advisory banner, while low-scoring cases below `0.35` will be refused outright.
> - We will run the Fuzzy Logic System **twice**: once pre-generation with `slec_coverage = 1.0` (acting as a conservative early-refusal gate to save LLM tokens/latency), and once post-generation using the actual `slec_coverage`.

## Open Questions

> [!NOTE]
> 1. Do you approve the default threshold ranges for fuzzy outputs: Accept ($\ge 0.65$), Uncertain ($[0.35, 0.65)$), and Refuse ($< 0.35$)?
> 2. For routes that skip or disable the SLEC gate (e.g. `CLAIM_CHECK`), we plan to default the SLEC input to `1.0` so the overall score isn't artificially penalized. Does that make sense?


---

## Proposed Changes

### Guardrails Component

#### [NEW] [fuzzy_refusal.py](file:///d:/GenAI/DoAn01/backend/src/guardrails/fuzzy_refusal.py)
- Create a pure Python Mamdani/Sugeno-style fuzzy logic evaluator `FuzzyRefusalEvaluator`.
- Define triangular/trapezoidal membership functions (`trimf`, `trapmf`) for `Low`, `Medium`, and `High` partitions of the four inputs.
- Define a rule base mapping combinations of input linguistic states to three output singleton values:
  - `0.0` (Refuse)
  - `0.5` (Uncertain / Partial Confidence)
  - `1.0` (Accept / High Confidence)
- Implement `evaluate(x1, x2, x3, x4) -> tuple[float, bool, str | None]` returning:
  - `fuzzy_score`: The defuzzified output in `[0, 1]`.
  - `should_refuse`: `True` if `fuzzy_score < 0.35`, otherwise `False`.
  - `reason`: `"partial_confidence"` if `0.35 <= fuzzy_score < 0.65`, else `"refusal_reason"` if refusing, else `None`.

#### [MODIFY] [refusal_policy.py](file:///d:/GenAI/DoAn01/backend/src/guardrails/refusal_policy.py)
- Integrate `FuzzyRefusalEvaluator` into `RefusalPolicy`.
- Add a new method `check_evidence_fuzzy` or update `check_evidence` to support fuzzy inputs.
- To prevent breaking route relaxation logic, preserve NLI and graph-fallback refusal override methods.

### Inference Component

#### [MODIFY] [inference_engine.py](file:///d:/GenAI/DoAn01/backend/src/inference/inference_engine.py)
- Update `InferenceEngine.answer` and `InferenceEngine.answer_stream` to:
  1. Retrieve OCR quality score (average of `confidence` on chunk evidence blocks) and CRAG correctness ratio (fraction of chunks above `crag_correct_threshold`).
  2. Perform a pre-generation fuzzy check using `slec_coverage = 1.0` as a placeholder. If it recommends refusal, abort and refuse early to save LLM tokens.
  3. Perform a post-generation/post-SLEC check using the actual `slec_coverage` returned by the sentence coverage gate.
  4. If the post-SLEC fuzzy check recommends refusal, refuse the answer.
  5. If the post-SLEC fuzzy check decides "partial_confidence", append the warning banner `"\n\n> ⚠️ Câu trả lời dựa trên bằng chứng có độ tin cậy hạn chế. Vui lòng kiểm tra lại nguồn gốc."`.

---

## Verification Plan

### Automated Tests
- Run existing retrieval/generation validation checks.
- Add unit tests for `FuzzyRefusalEvaluator` covering:
  - High quality signals (Reranker High, SLEC High, OCR High, CRAG High) -> Accept
  - Low quality signals (Reranker Low) -> Refuse
  - Conflicted signals (Reranker High, SLEC Low) -> Refuse (due to hallucination risk)
  - Poor OCR signals (OCR Low, other factors High) -> Uncertain/Refuse

### Manual Verification
- Test queries through the backend streaming/non-streaming endpoints with documents of different qualities (e.g. low OCR confidence vs clean PDFs).
- Verify refusal responses in cases where evidence is poor.
