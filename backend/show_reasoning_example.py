"""
Script để test reasoning path với mock data.
"""
import json

# Mock QueryResponse với reasoning_path
response = {
    "answer": "Dropout là kỹ thuật regularization được sử dụng trong deep learning để giảm overfitting[1][2]. Kỹ thuật này hoạt động bằng cách ngẫu nhiên tắt một số neurons trong quá trình training[3].",
    "answer_language": "vi",
    "query_language": "vi",
    "translated_query": None,
    "source_languages": ["en"],
    "citations": [
        {
            "doc_id": "doc1",
            "doc_name": "ML_Techniques.pdf",
            "page": 5,
            "pages": [5],
            "block_id": "blk-001",
            "block_type": "text",
            "snippet_original": "Dropout is a regularization technique...",
            "snippet_translated": None,
            "bbox": None,
            "role": "evidence",
            "source_language": "en",
            "confidence": 0.95
        }
    ],
    "confidence": 0.85,
    "was_refused": False,
    "refusal_reason": None,
    "reasoning_path": [
        {
            "step_type": "retrieve",
            "entities": ["Dropout", "Regularization"],
            "relations": [],
            "confidence": 0.90,
            "description": "Retrieved 15 relevant chunks from 3 documents: ML_Techniques.pdf, Deep_Learning.pdf, Neural_Networks.pdf"
        },
        {
            "step_type": "traverse",
            "entities": ["Dropout", "Overfitting", "Regularization"],
            "relations": ["prevents", "is_type_of"],
            "confidence": 0.78,
            "description": "Traversed knowledge graph: Dropout --prevents--> Overfitting, Dropout --is_type_of--> Regularization"
        },
        {
            "step_type": "synthesize",
            "entities": ["Dropout", "Neural Networks", "Training"],
            "relations": [],
            "confidence": 0.92,
            "description": "Synthesized answer from top 5 most relevant chunks with high confidence"
        }
    ]
}

print("=" * 80)
print("MOCK API RESPONSE WITH REASONING PATH")
print("=" * 80)
print()
print(json.dumps(response, indent=2, ensure_ascii=False))
print()
print("=" * 80)
print("REASONING PATH VISUALIZATION (Frontend will render this as):")
print("=" * 80)
print()
print("Answer:")
print(response["answer"])
print()
print("💡 How I found this answer:")
print()

for i, step in enumerate(response["reasoning_path"], 1):
    icon = {"retrieve": "📈", "traverse": "✨", "synthesize": "💡"}[step["step_type"]]
    print(f"{i}. {icon} {step['description']}")
    if step["entities"]:
        print(f"   Entities: {', '.join(step['entities'])}")
    if step["relations"]:
        print(f"   Relations: {', '.join(step['relations'])}")
    print(f"   Confidence: {int(step['confidence'] * 100)}%")
    print()

print("=" * 80)
print("TO SEE THIS IN FRONTEND:")
print("=" * 80)
print("1. Make sure you have documents uploaded")
print("2. Ask any question")
print("3. Look below the answer for the reasoning trace")
print()
print("If you don't see it:")
print("- Check browser console (F12) for errors")
print("- Hard refresh (Ctrl+Shift+R)")
print("- Check Network tab to see if reasoning_path is in API response")
