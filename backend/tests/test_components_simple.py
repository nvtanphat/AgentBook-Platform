"""
Simple test để kiểm tra các component chính.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("AGENTBOOK COMPONENT TEST")
print("=" * 60)

# Test 1: Config
print("\n1. Testing Config...")
try:
    from src.core.config import get_settings
    settings = get_settings()
    print(f"   ✓ Config loaded")
    print(f"   - LLM: {settings.llm_local_model}")
    print(f"   - Embedding: {settings.embedding_model}")
    print(f"   - Chunk strategy: {settings.chunk_strategy}")
except Exception as e:
    print(f"   ✗ Config failed: {e}")

# Test 2: Docling Parser
print("\n2. Testing Docling Parser...")
try:
    from src.processing.docling_parser import DoclingParser
    parser = DoclingParser()

    test_file = Path("D:/GenAI/DoAn01/data/test data/Machine_Learning_Regularization_Techniques.docx")
    if test_file.exists():
        doc = parser.parse(test_file, language="en")
        print(f"   ✓ Parsed: {test_file.name}")
        print(f"   - Pages: {len(doc.pages)}")
        print(f"   - Blocks: {len(doc.blocks)}")
        if doc.blocks:
            print(f"   - First block: {doc.blocks[0].snippet_original[:80]}...")
    else:
        print(f"   ⚠ Test file not found")
except Exception as e:
    print(f"   ✗ Parser failed: {e}")

# Test 3: EasyOCR
print("\n3. Testing EasyOCR...")
try:
    from src.processing.ocr_engine import EasyOCREngine
    ocr = EasyOCREngine(lang="vi", gpu=False)
    print(f"   ✓ EasyOCR initialized")
    print(f"   - Language: vi+en")
    print(f"   - Device: CPU")
except Exception as e:
    print(f"   ✗ OCR failed: {e}")

# Test 4: BGE-M3 Embedder
print("\n4. Testing BGE-M3 Embedder...")
try:
    from src.rag.embedder import BGEM3Embedder
    embedder = BGEM3Embedder(settings)

    test_texts = ["Machine learning is a subset of AI"]
    embeddings = embedder.encode(test_texts)

    print(f"   ✓ Embedding completed")
    print(f"   - Dense dim: {len(embeddings[0].dense)}")
    print(f"   - Sparse nnz: {len(embeddings[0].sparse.indices)}")
except Exception as e:
    print(f"   ✗ Embedding failed: {e}")

# Test 5: Reranker
print("\n5. Testing CrossEncoder Reranker...")
try:
    from src.rag.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker(settings)
    print(f"   ✓ Reranker loaded")
    print(f"   - Model: {settings.reranker_model_name}")
    print(f"   - Device: {settings.reranker_device}")
except Exception as e:
    print(f"   ✗ Reranker failed: {e}")

# Test 6: Chunking
print("\n6. Testing Semantic Chunker...")
try:
    from src.processing.chunking import build_chunker
    chunker = build_chunker(settings, embedder=embedder)
    print(f"   ✓ Chunker initialized")
    print(f"   - Strategy: {settings.chunk_strategy}")
    print(f"   - Target tokens: {settings.chunk_target_token_count}")
except Exception as e:
    print(f"   ✗ Chunker failed: {e}")

# Test 7: LLM
print("\n7. Testing Qwen2.5 3B (Ollama)...")
try:
    from src.core.local_llm import OllamaLLM
    import asyncio

    llm = OllamaLLM(settings)

    async def run_llm():
        prompt = "What is 2+2? Answer in one word."
        answer = await llm.generate(prompt=prompt)
        return answer

    answer = asyncio.run(run_llm())
    print(f"   ✓ LLM generation completed")
    print(f"   - Model: {settings.llm_local_model}")
    print(f"   - Answer: {answer[:100]}")
except Exception as e:
    print(f"   ✗ LLM failed: {e}")

# Test 8: Redis Cache
print("\n8. Testing Redis Cache...")
try:
    from src.rag.embedding_cache import RedisEmbeddingCache
    cache = RedisEmbeddingCache(settings.redis_url)

    if cache.enabled:
        stats = cache.stats()
        print(f"   ✓ Redis cache connected")
        print(f"   - Hit rate: {stats.get('hit_rate', 0):.2%}")
    else:
        print(f"   ⚠ Redis not available (fallback to in-memory)")
except Exception as e:
    print(f"   ✗ Cache failed: {e}")

# Test 9: Smart Reranker
print("\n9. Testing Smart Reranker...")
try:
    from src.rag.smart_reranker import SmartReranker
    smart_reranker = SmartReranker(reranker, confidence_threshold=0.7)
    print(f"   ✓ Smart reranker initialized")
    print(f"   - Confidence threshold: 0.7")
except Exception as e:
    print(f"   ✗ Smart reranker failed: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
