# Kịch bản test đề xuất

| Tính năng | File | Collection | Câu hỏi mẫu |
|---|---|---|---|
| VN→EN RAG | lecture_notes.pdf | A | AgentBook xử lý cross-lingual query như thế nào? |
| Summarize | lecture_notes.pdf | A | Tóm tắt tài liệu này trong 5 ý. |
| Study-guide | lecture_notes.pdf | A | Tạo study guide cho tài liệu này. |
| Compare | lecture_notes.pdf + comparison_doc.docx | A | Hai tài liệu giống và khác nhau ở đâu? |
| OCR pass | report_vn.pdf / scan_clear.png | A/B | Pipeline OCR gồm những bước nào? |
| OCR fail | scan_low_quality.png | B | Hệ thống nên từ chối vì sao? |
| Handwriting OK | handwriting_good.png | B | Ghi chú viết tay nói gì? |
| Handwriting fail | handwriting_bad.png | B | Kỳ vọng status failed hoặc error quality. |
| Table retrieval | data_table.csv | B | Liệt kê sản phẩm đánh giá trên 4.5. |
| Multi-sheet | workbook_multi.xlsx | B | Sheet Inventory có bao nhiêu sản phẩm? |
| Audio transcription | lecture_vi.wav | B | Bài nói nhắc đến hệ thống nào? |
