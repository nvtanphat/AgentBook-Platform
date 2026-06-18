# AgentBook Real-world Multimodal Dataset

Bản này loại bỏ file mẫu/synthetic khỏi bộ trước. Các PDF/DOCX/PPTX là tài liệu thật từ nguồn công khai hoặc nguồn tổ chức; ảnh OCR được lấy trực tiếp từ trang/ảnh thật hoặc được render từ PDF thật. Không có audio synthetic.

Điểm cần biết:
- Vinamilk 2023: link chính thức được giữ trong `02_finance/vinamilk_annual_report_2023_source_link.md`, nhưng file PDF 2023 bị chặn tải tự động trong sandbox. Dataset kèm một báo cáo thường niên Vinamilk thật tải được từ Vietstock để vẫn có tài liệu tài chính thực tế.
- Audio: không đưa file giả. `05_audio_real_sources/README_audio_sources.md` chứa link nguồn audio tiếng Việt thật; sandbox chặn tải MIME audio/mpeg nên cần tải thủ công nếu muốn đóng gói audio vật lý.
- Manifest có SHA-256 để kiểm tra toàn vẹn file.

Gợi ý test RAG:
1. Ingestion đa định dạng: PDF scan, PDF 2 cột, DOCX, PPTX, PNG/JPEG.
2. Table extraction: FPT financial statement + annual reports.
3. OCR/layout: ảnh render từ luật, tài chính, paper + ảnh viết tay tiếng Việt thật.
4. Cross-lingual: hỏi tiếng Việt trên FPT Annual Report EN và paper tiếng Anh.
5. Citation grounding: kiểm tra trích dẫn theo trang/vùng/nguồn.
