"""
Test chunking và OCR với data thực tế.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_settings
from src.processing.docling_parser import DoclingParser
from src.processing.ocr_engine import EasyOCREngine
from src.processing.chunking import build_chunker
from src.rag.embedder import BGEM3Embedder
from src.processing.types import EvidenceMap, EvidenceBlock

settings = get_settings()

print('=' * 70)
print('TEST CHUNKING & OCR WITH REAL DATA')
print('=' * 70)

# Test 1: DOCX Chunking
print('\n[TEST 1] DOCX Parsing + Chunking')
print('-' * 70)

docx_file = Path('D:/GenAI/DoAn01/data/test data/Machine_Learning_Regularization_Techniques.docx')

try:
    parser = DoclingParser()
    doc = parser.parse(docx_file, language='en')

    print(f'File: {docx_file.name}')
    print(f'Pages: {len(doc.pages)}')
    print(f'Total blocks: {len(doc.blocks)}')

    # Build evidence map (use correct field names)
    blocks = []
    for block in doc.blocks:
        blocks.append(EvidenceBlock(
            owner_id='test',
            collection_id='test',
            material_id='test',
            document_name=docx_file.name,
            page=block.page_number,
            block_id=block.block_id,
            block_type=block.block_type,
            snippet_original=block.content,  # content -> snippet_original
            source_language=block.language,
            bbox=block.bbox,
            confidence=block.ocr_confidence,
        ))

    evidence_map = EvidenceMap(
        owner_id='test',
        collection_id='test',
        material_id='test',
        document_name=docx_file.name,
        blocks=blocks,
    )

    # Chunk with semantic chunker
    print('\nChunking with Semantic Chunker...')
    embedder = BGEM3Embedder(settings)
    chunker = build_chunker(settings, embedder=embedder)
    chunks = chunker.build_chunks(evidence_map)

    print(f'[OK] Chunks created: {len(chunks)}')
    print(f'     Avg chunk size: {sum(c.token_count for c in chunks) / len(chunks):.0f} tokens')
    print(f'     Strategy: {chunks[0].chunk_strategy}')
    print(f'     Min tokens: {min(c.token_count for c in chunks)}')
    print(f'     Max tokens: {max(c.token_count for c in chunks)}')

    # Show first chunk
    print(f'\n     First chunk:')
    print(f'       Pages: {chunks[0].source_pages}')
    print(f'       Blocks: {len(chunks[0].source_block_ids)}')
    print(f'       Tokens: {chunks[0].token_count}')
    print(f'       Content: {chunks[0].content[:150]}...')

    # Show chunk distribution
    print(f'\n     Chunk size distribution:')
    for i, chunk in enumerate(chunks[:5], 1):
        print(f'       Chunk {i}: {chunk.token_count} tokens, {len(chunk.source_block_ids)} blocks')

except Exception as e:
    print(f'[FAIL] DOCX test failed: {e}')
    import traceback
    traceback.print_exc()

# Test 2: Image OCR
print('\n[TEST 2] Image OCR + Chunking')
print('-' * 70)

img_file = Path('D:/GenAI/DoAn01/data/test data/rag_scan_mau.png')

try:
    ocr = EasyOCREngine(lang='vi', gpu=False)
    doc = ocr.parse_image(img_file, language='vi')

    print(f'File: {img_file.name}')
    print(f'Blocks detected: {len(doc.blocks)}')
    print(f'OCR confidence: {doc.pages[0].ocr_confidence:.2f}')

    # Build evidence map
    blocks = []
    for block in doc.blocks:
        blocks.append(EvidenceBlock(
            owner_id='test',
            collection_id='test',
            material_id='test',
            document_name=img_file.name,
            page=block.page_number,
            block_id=block.block_id,
            block_type=block.block_type,
            snippet_original=block.content,
            source_language=block.language,
            bbox=block.bbox,
            confidence=block.ocr_confidence,
        ))

    evidence_map = EvidenceMap(
        owner_id='test',
        collection_id='test',
        material_id='test',
        document_name=img_file.name,
        blocks=blocks,
    )

    # Chunk
    print('\nChunking OCR results...')
    chunks = chunker.build_chunks(evidence_map)

    print(f'[OK] Chunks created: {len(chunks)}')
    if chunks:
        print(f'     Avg chunk size: {sum(c.token_count for c in chunks) / len(chunks):.0f} tokens')
        print(f'     First chunk: {chunks[0].content[:150]}...')

except Exception as e:
    print(f'[FAIL] OCR test failed: {e}')
    import traceback
    traceback.print_exc()

# Test 3: PDF Chunking
print('\n[TEST 3] PDF Parsing + Chunking')
print('-' * 70)

pdf_file = Path('D:/GenAI/DoAn01/data/test data/rag_mau_hoc_tap.pdf')

try:
    parser = DoclingParser()
    doc = parser.parse(pdf_file, language='vi')

    print(f'File: {pdf_file.name}')
    print(f'Pages: {len(doc.pages)}')
    print(f'Total blocks: {len(doc.blocks)}')

    # Build evidence map
    blocks = []
    for block in doc.blocks[:50]:  # Limit to first 50 blocks for speed
        blocks.append(EvidenceBlock(
            owner_id='test',
            collection_id='test',
            material_id='test',
            document_name=pdf_file.name,
            page=block.page_number,
            block_id=block.block_id,
            block_type=block.block_type,
            snippet_original=block.content,
            source_language=block.language,
            bbox=block.bbox,
            confidence=block.ocr_confidence,
        ))

    evidence_map = EvidenceMap(
        owner_id='test',
        collection_id='test',
        material_id='test',
        document_name=pdf_file.name,
        blocks=blocks,
    )

    # Chunk
    print('\nChunking PDF (first 50 blocks)...')
    chunks = chunker.build_chunks(evidence_map)

    print(f'[OK] Chunks created: {len(chunks)}')
    print(f'     Avg chunk size: {sum(c.token_count for c in chunks) / len(chunks):.0f} tokens')

except Exception as e:
    print(f'[FAIL] PDF test failed: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 70)
print('TEST SUMMARY')
print('=' * 70)
print('All tests completed. Check results above.')
