# Evaluation: Plain RAG vs Agentic RAG

Collection: Pháp Luật | Questions: 4 | Top-K: 6

## Per-question results

| ID | Type | Plain latency | Agentic latency | Plain SLEC | Agentic guardrail | Winner | Reason |
|---|---|---|---|---|---|---|---|
| Q2 | multi_hop_death | 115.9s | 141.9s | 100% | supported(0.97) | **B** | Cả hai đều trả lời đúng và đủ các ý chính, nhưng B… |
| Q3 | comparison | 1.0s | 126.8s | 100% | supported(0.84) | **A** | Cả hai đều đầy đủ và có căn cứ, nhưng A bám sát th… |
| Q4 | chained_consequence | 2.3s | 102.3s | 100% | supported(0.78) | **B** | B đáp ứng đủ hơn hai hop chính: căn cứ/hậu quả phá… |
| Q5 | complex_factual | 0.7s | 105.1s | 100% | supported(1.00) | **tie** | Hai câu trả lời gần như giống nhau, đều đầy đủ và … |

## LLM Judge scores (0-10)

| ID | Plain complete | Plain grounding | Plain legal | Agentic complete | Agentic grounding | Agentic legal | Winner |
|---|---|---|---|---|---|---|---|
| Q2 | 9 | 7 | 8 | 9 | 8 | 9 | **B** |
| Q3 | 9 | 8 | 9 | 9 | 8 | 8 | **A** |
| Q4 | 4 | 7 | 4 | 6 | 6 | 5 | **B** |
| Q5 | 10 | 7 | 9 | 10 | 7 | 9 | **tie** |

## Aggregate

| Metric | Plain RAG | Agentic RAG | Delta |
|---|---|---|---|
| Completeness (avg) | 8.0 | 8.5 | ▲0.5 |
| Grounding (avg) | 7.25 | 7.25 | —0.0 |
| Legal precision (avg) | 7.5 | 7.75 | ▲0.25 |
| **Total score** | **22.75** | **23.5** | **0.75** |

**Win/Tie/Loss (Agentic vs Plain):** 2 / 1 / 1

**Avg latency:** Plain 29.975s | Agentic 119.025s | Overhead 89.1s