# AgentBook Test Pack

Generated on 2026-05-23 14:08.

This pack contains two collections for end-to-end AgentBook testing.

## Collection_A
- `lecture_notes.pdf`: text-based English PDF, 5+ pages, tables and figures.
- `report_vn.pdf`: image-only Vietnamese scanned PDF for OCR.
- `comparison_doc.docx`: DOCX with multiple tables, same topic as the PDF.
- `slides_mixed.pptx`: 10-slide mixed-modality deck with text, charts and images.

## Collection_B
- `data_table.csv`: structured product table with text and numeric columns.
- `workbook_multi.xlsx`: multi-sheet workbook for spreadsheet parsing.
- `scan_clear.png`: clear printed Vietnamese scan.
- `scan_low_quality.png`: intentionally poor image for OCR quality refusal.
- `handwriting_good.png` / `handwriting_ok.png`: readable handwriting sample.
- `handwriting_bad.png` / `handwriting_blur.png`: low-quality handwriting refusal sample.
- `lecture_vi.wav`: short Vietnamese audio sample generated with eSpeak.

## Quick tests
1. Ask in Vietnamese over `lecture_notes.pdf`: “Hệ thống AgentBook gồm những thành phần nào?”
2. Compare `lecture_notes.pdf` with `comparison_doc.docx`.
3. Ask `data_table.csv`: “Sản phẩm nào có giá cao nhất?”
4. Upload `scan_clear.png` and check OCR confidence.
5. Upload `scan_low_quality.png` or `handwriting_blur.png` and expect refusal.
