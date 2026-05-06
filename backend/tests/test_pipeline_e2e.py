"""
Test script để kiểm tra pipeline xử lý documents.

Test cases:
1. PDF parsing (Docling)
2. DOCX parsing (Docling)
3. PPTX parsing (Docling)
4. Image OCR (EasyOCR)
5. Chunking (Semantic)
6. Embedding (BGE-M3)
7. Indexing (Qdrant)
8. Retrieval (Hybrid)
9. Reranking (CrossEncoder)
10. Answer generation (Qwen2.5 3B)
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_settings
from src.processing.docling_parser import DoclingParser
from src.processing.ocr_engine import EasyOCREngine
from src.processing.chunking import build_chunker
from src.rag.embedder import BGEM3Embedder
from src.rag.retriever import HybridRetriever
from src.rag.reranker import CrossEncoderReranker
from src.rag.vector_store import get_qdrant_client_for_settings
from src.rag.types import RetrievalScope
from src.core.local_llm import OllamaLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_pdf_parsing():
    """Test PDF parsing với Docling."""
    logger.info("=" * 60)
    logger.info("TEST 1: PDF Parsing")
    logger.info("=" * 60)

    test_file = Path("D:/GenAI/DoAn01/data/test data/rag_mau_hoc_tap.pdf")
    if not test_file.exists():
        logger.warning(f"File not found: {test_file}")
        return None

    try:
        parser = DoclingParser()
        doc = parser.parse(test_file, language="vi")

        logger.info(f"✓ Parsed PDF: {doc.document_name}")
        logger.info(f"  Pages: {len(doc.pages)}")
        logger.info(f"  Total blocks: {sum(len(p.blocks) for p in doc.pages)}")

        # Show first few blocks
        if doc.pages and doc.pages[0].blocks:
            logger.info(f"  First block: {doc.pages[0].blocks[0].snippet_original[:100]}...")

        return doc
    except Exception as exc:
        logger.error(f"✗ PDF parsing failed: {exc}", exc_info=True)
        return None


async def run_docx_parsing():
    """Test DOCX parsing với Docling."""
    logger.info("=" * 60)
    logger.info("TEST 2: DOCX Parsing")
    logger.info("=" * 60)

    test_file = Path("D:/GenAI/DoAn01/data/test data/Machine_Learning_Regularization_Techniques.docx")
    if not test_file.exists():
        logger.warning(f"File not found: {test_file}")
        return None

    try:
        parser = DoclingParser()
        doc = parser.parse(test_file, language="en")

        logger.info(f"✓ Parsed DOCX: {doc.document_name}")
        logger.info(f"  Pages: {len(doc.pages)}")
        logger.info(f"  Total blocks: {sum(len(p.blocks) for p in doc.pages)}")

        return doc
    except Exception as exc:
        logger.error(f"✗ DOCX parsing failed: {exc}", exc_info=True)
        return None


async def run_image_ocr():
    """Test Image OCR với EasyOCR."""
    logger.info("=" * 60)
    logger.info("TEST 3: Image OCR")
    logger.info("=" * 60)

    test_file = Path("D:/GenAI/DoAn01/data/test data/rag_scan_mau.png")
    if not test_file.exists():
        logger.warning(f"File not found: {test_file}")
        return None

    try:
        ocr = EasyOCREngine(lang="vi", gpu=False)
        doc = ocr.parse_image(test_file, language="vi")

        logger.info(f"✓ OCR completed: {doc.document_name}")
        logger.info(f"  Blocks: {len(doc.pages[0].blocks)}")
        logger.info(f"  OCR confidence: {doc.pages[0].ocr_confidence:.2f}")

        # Show first block
        if doc.pages[0].blocks:
            logger.info(f"  First text: {doc.pages[0].blocks[0].snippet_original[:100]}...")

        return doc
    except Exception as exc:
        logger.error(f"✗ OCR failed: {exc}", exc_info=True)
        return None


async def run_chunking(doc):
    """Test Semantic Chunking."""
    logger.info("=" * 60)
    logger.info("TEST 4: Semantic Chunking")
    logger.info("=" * 60)

    if not doc:
        logger.warning("No document to chunk")
        return None

    try:
        settings = get_settings()
        embedder = BGEM3Embedder(settings)
        chunker = build_chunker(settings, embedder=embedder)

        # Build evidence map
        from src.processing.types import EvidenceMap, EvidenceBlock
        blocks = []
        for page in doc.pages:
            for block in page.blocks:
                blocks.append(EvidenceBlock(
                    owner_id="test_user",
                    collection_id="test_collection",
                    material_id="test_material",
                    document_name=doc.document_name,
                    page=page.page_number,
                    block_id=block.block_id,
                    block_type=block.block_type,
                    snippet_original=block.snippet_original,
                    source_language=block.source_language or "unknown",
                    bbox=block.bbox,
                    confidence=block.ocr_confidence,
                ))

        evidence_map = EvidenceMap(
            owner_id="test_user",
            collection_id="test_collection",
            material_id="test_material",
            document_name=doc.document_name,
            blocks=blocks,
        )

        chunks = chunker.build_chunks(evidence_map)

        logger.info(f"✓ Chunking completed")
        logger.info(f"  Input blocks: {len(blocks)}")
        logger.info(f"  Output chunks: {len(chunks)}")
        logger.info(f"  Avg chunk size: {sum(c.token_count for c in chunks) / len(chunks):.0f} tokens")
        logger.info(f"  Strategy: {chunks[0].chunk_strategy if chunks else 'N/A'}")

        # Show first chunk
        if chunks:
            logger.info(f"  First chunk: {chunks[0].content[:150]}...")

        return chunks
    except Exception as exc:
        logger.error(f"✗ Chunking failed: {exc}", exc_info=True)
        return None


async def run_embedding(chunks):
    """Test BGE-M3 Embedding."""
    logger.info("=" * 60)
    logger.info("TEST 5: BGE-M3 Embedding")
    logger.info("=" * 60)

    if not chunks:
        logger.warning("No chunks to embed")
        return None

    try:
        settings = get_settings()
        embedder = BGEM3Embedder(settings)

        # Embed first 3 chunks
        test_chunks = chunks[:3]
        texts = [c.content for c in test_chunks]

        embeddings = embedder.encode(texts)

        logger.info(f"✓ Embedding completed")
        logger.info(f"  Chunks embedded: {len(embeddings)}")
        logger.info(f"  Dense dim: {len(embeddings[0].dense)}")
        logger.info(f"  Sparse nnz: {len(embeddings[0].sparse.indices)}")

        return embeddings
    except Exception as exc:
        logger.error(f"✗ Embedding failed: {exc}", exc_info=True)
        return None


async def run_retrieval():
    """Test Hybrid Retrieval."""
    logger.info("=" * 60)
    logger.info("TEST 6: Hybrid Retrieval")
    logger.info("=" * 60)

    try:
        settings = get_settings()
        qdrant = get_qdrant_client_for_settings(settings)
        retriever = HybridRetriever(settings=settings, qdrant_client=qdrant)

        # Test query
        query = "Regularization techniques in machine learning"
        scope = RetrievalScope(
            owner_id="test_user",
            collection_id="test_collection",
        )

        results = await retriever.retrieve(query=query, scope=scope, limit=5)

        logger.info(f"✓ Retrieval completed")
        logger.info(f"  Query: {query}")
        logger.info(f"  Results: {len(results)}")

        if results:
            logger.info(f"  Top result score: {results[0].fused_score:.4f}")
            logger.info(f"  Top result: {results[0].content[:100]}...")

        return results
    except Exception as exc:
        logger.error(f"✗ Retrieval failed: {exc}", exc_info=True)
        return None


async def run_reranking(results):
    """Test CrossEncoder Reranking."""
    logger.info("=" * 60)
    logger.info("TEST 7: CrossEncoder Reranking")
    logger.info("=" * 60)

    if not results:
        logger.warning("No results to rerank")
        return None

    try:
        settings = get_settings()
        reranker = CrossEncoderReranker(settings)

        query = "Regularization techniques"
        reranked = reranker.rerank(query=query, chunks=results, limit=5)

        logger.info(f"✓ Reranking completed")
        logger.info(f"  Input: {len(results)}")
        logger.info(f"  Output: {len(reranked)}")

        if reranked:
            logger.info(f"  Top rerank score: {reranked[0].rerank_score:.4f}")
            logger.info(f"  Top result: {reranked[0].content[:100]}...")

        return reranked
    except Exception as exc:
        logger.error(f"✗ Reranking failed: {exc}", exc_info=True)
        return None


async def run_llm_generation(chunks):
    """Test LLM Answer Generation."""
    logger.info("=" * 60)
    logger.info("TEST 8: LLM Generation (Qwen2.5 3B)")
    logger.info("=" * 60)

    if not chunks:
        logger.warning("No chunks for generation")
        return None

    try:
        settings = get_settings()
        llm = OllamaLLM(settings)

        # Build simple prompt
        evidence = "\n\n".join([f"[{i+1}] {c.content[:200]}..." for i, c in enumerate(chunks[:3])])
        prompt = f"""Evidence:
{evidence}

Question: What are regularization techniques in machine learning?

Answer in English. Cite sources as [N].
"""

        logger.info("Generating answer (this may take 5-10s)...")
        answer = await llm.generate(prompt=prompt)

        logger.info(f"✓ Generation completed")
        logger.info(f"  Answer length: {len(answer)} chars")
        logger.info(f"  Answer: {answer[:300]}...")

        return answer
    except Exception as exc:
        logger.error(f"✗ Generation failed: {exc}", exc_info=True)
        return None


async def main():
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("AGENTBOOK PIPELINE TEST")
    logger.info("=" * 60 + "\n")

    # Test 1-3: Parsing
    pdf_doc = await run_pdf_parsing()
    docx_doc = await run_docx_parsing()
    ocr_doc = await run_image_ocr()

    # Test 4: Chunking (use DOCX doc if available)
    test_doc = docx_doc or pdf_doc or ocr_doc
    chunks = await run_chunking(test_doc)

    # Test 5: Embedding
    embeddings = await run_embedding(chunks)

    # Test 6-7: Retrieval & Reranking (requires indexed data)
    results = await run_retrieval()
    reranked = await run_reranking(results)

    # Test 8: LLM Generation
    test_chunks = reranked or results or chunks
    answer = await run_llm_generation(test_chunks)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"PDF Parsing:     {'✓' if pdf_doc else '✗'}")
    logger.info(f"DOCX Parsing:    {'✓' if docx_doc else '✗'}")
    logger.info(f"Image OCR:       {'✓' if ocr_doc else '✗'}")
    logger.info(f"Chunking:        {'✓' if chunks else '✗'}")
    logger.info(f"Embedding:       {'✓' if embeddings else '✗'}")
    logger.info(f"Retrieval:       {'✓' if results else '✗'}")
    logger.info(f"Reranking:       {'✓' if reranked else '✗'}")
    logger.info(f"LLM Generation:  {'✓' if answer else '✗'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
