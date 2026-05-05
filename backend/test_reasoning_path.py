"""
Test script để verify reasoning path feature.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.schemas.query import QueryResponse, ReasoningStep

# Test 1: Create mock response with reasoning path
print("=" * 70)
print("TEST 1: Schema Validation")
print("=" * 70)

response = QueryResponse(
    answer="Dropout là kỹ thuật regularization giúp giảm overfitting.",
    answer_language="vi",
    query_language="vi",
    translated_query=None,
    source_languages=["en"],
    citations=[],
    confidence=0.85,
    was_refused=False,
    refusal_reason=None,
    reasoning_path=[
        ReasoningStep(
            step_type="retrieve",
            entities=["Dropout", "Regularization"],
            relations=[],
            confidence=0.90,
            description="Retrieved 15 relevant chunks from 3 documents: ML_Techniques.pdf, Deep_Learning.pdf, Neural_Networks.pdf"
        ),
        ReasoningStep(
            step_type="traverse",
            entities=["Dropout", "Overfitting", "Regularization"],
            relations=["prevents", "is_type_of"],
            confidence=0.78,
            description="Traversed knowledge graph: Dropout --prevents--> Overfitting, Dropout --is_type_of--> Regularization"
        ),
        ReasoningStep(
            step_type="synthesize",
            entities=["Dropout", "Neural Networks", "Training"],
            relations=[],
            confidence=0.92,
            description="Synthesized answer from top 5 most relevant chunks with high confidence"
        ),
    ]
)

print(f"✓ QueryResponse created successfully")
print(f"✓ Answer: {response.answer[:50]}...")
print(f"✓ Reasoning path steps: {len(response.reasoning_path)}")
print()

# Test 2: Show reasoning path details
print("=" * 70)
print("TEST 2: Reasoning Path Details")
print("=" * 70)

for i, step in enumerate(response.reasoning_path, 1):
    print(f"\nStep {i}: {step.step_type.upper()}")
    print(f"  Description: {step.description}")
    print(f"  Entities: {', '.join(step.entities)}")
    if step.relations:
        print(f"  Relations: {', '.join(step.relations)}")
    print(f"  Confidence: {step.confidence * 100:.0f}%")

print()

# Test 3: JSON serialization
print("=" * 70)
print("TEST 3: JSON Serialization (API Response)")
print("=" * 70)

import json
response_json = response.model_dump()
print(json.dumps(response_json, indent=2, ensure_ascii=False))

print()
print("=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print()
print("Next: Test with real query via API")
print("  1. Open: http://127.0.0.1:8000/docs")
print("  2. Try: POST /api/v1/query/ask")
print("  3. Check response has 'reasoning_path' field")
