# Hướng Dẫn Kiểm Tra Hệ Thống (AgentBook Testing Guide)

Tài liệu này hướng dẫn chi tiết cách kiểm tra (test) từng component trong dự án AgentBook, bao gồm cả cách kiểm tra xem dữ liệu OCR đã được chunk và index chính xác vào Qdrant như thế nào.

## 1. Cách Kiểm Tra Index Chunk OCR (Qdrant & Metadata)

Để đảm bảo các đoạn văn bản (text chunk) được bóc tách bằng OCR từ tài liệu (định dạng ảnh/pdf) được lưu trữ đúng cách cùng với `evidence trace`, hãy làm theo các cách sau:

### Phương pháp 1: Sử dụng Qdrant Dashboard (Giao diện trực quan)
Dự án chạy Qdrant qua Docker (cổng 6333), Qdrant có sẵn một Web UI để bạn có thể xem trực tiếp Vector và Payload (metadata) được index.
1. Mở trình duyệt và truy cập: `http://localhost:6333/dashboard`
2. Chọn Collection tương ứng (ví dụ: `agentbook_chunks` hoặc tên collection bạn cấu hình).
3. Tìm kiếm (Search) hoặc duyệt (Browse) các điểm dữ liệu (points).
4. **Kiểm tra Payload:** Ở mỗi point, bạn sẽ thấy phần Payload chứa thông tin của `TextChunk` (dựa trên class `TextChunk` trong file `types.py`).
   - Tìm kiếm trường `evidence` (kiểu mảng).
   - Kiểm tra xem trong các phần tử `EvidenceBlock` có chứa các thuộc tính: `block_type: "ocr_text"`, `bbox`, `confidence`, `page`, và `snippet_original` hay không.
   - Kiểm tra xem bắt buộc có `owner_id`, `collection_id`, `material_id` hay không (để tuân thủ nghiêm ngặt quy định về Isolation trong `CLAUDE.md`).

### Phương pháp 2: Gọi API trực tiếp của Qdrant
Bạn có thể dùng curl (hoặc Postman) để gọi Qdrant và lấy thử một vài vector đang lưu trữ trong cơ sở dữ liệu:
```bash
curl -X POST "http://localhost:6333/collections/agentbook_chunks/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 5,
    "with_payload": true,
    "with_vector": false
  }'
```
Hãy quan sát JSON trả về xem cấu trúc Chunk và các bằng chứng OCR có đủ độ tin cậy (confidence scores) không.

### Phương pháp 3: Kiểm tra thông qua Unit Test / Script
Bạn có thể tham khảo viết một bài test sử dụng thư viện `qdrant-client` để fetch thử payload của một `material_id` bất kỳ và `assert` rằng nó thỏa mãn schema quy định.

---

## 2. Quy Trình Kiểm Tra Các Component Khác

### 2.1 Backend (FastAPI + Beanie)
Thực hiện chạy Unit Tests với Pytest:
- **Lệnh thực thi:** 
  ```bash
  cd backend
  pytest tests/
  ```
- **Các vùng kiểm tra trọng tâm:**
  - **Upload Safety:** Kiểm tra logic từ chối file có kích thước quá lớn, sai định dạng (MIME validation), phòng chống path traversal.
  - **Scope Isolation:** Đảm bảo mọi truy vấn dữ liệu từ API đều đi kèm `owner_id`.
  - **API Error Handling:** Trả về đúng mã lỗi 400/422 nếu thiếu dữ liệu.

### 2.2 Background Processing (Celery Worker + Pipeline OCR)
Celery (worker container) là nơi thực thi quá trình parsing và OCR. 
- **Cách kiểm tra logs:**
  Dùng lệnh sau trong terminal để theo dõi log của worker khi bạn upload file:
  ```bash
  docker logs -f <tên-container-worker>  # Ví dụ: doan01-worker-1
  ```
- **Nội dung cần theo dõi:**
  - Xem job có nhận đúng tiến trình với `material_id` không.
  - Khi xử lý ảnh, hãy kiểm tra hệ thống in ra confidence scores của OCR/Handwriting. Theo quy định, nếu điểm quá thấp, hệ thống phải kích hoạt cơ chế `refusal` và cảnh báo chất lượng ảnh tồi tệ, không được phép "hallucinate" tạo ra text giả mạo.

### 2.3 RAG Pipeline & Evaluation Metrics
Dự án được đánh giá độ chuẩn xác bằng các bộ dataset chuyên biệt.
- **Thực thi:**
  ```bash
  python evaluation/run_eval.py
  # Hoặc kiểm tra sự so sánh giữa Hybrid vs Dense, Reranking vs Không Reranking:
  python scripts/run_ablation_suite.py
  ```
- **Chỉ số:** Hệ thống sẽ quét qua thư mục `evaluation/datasets/` để trả về kết quả Recall@k, Precision@k, MRR@k, và kiểm tra xem hệ thống có trả lời đúng theo `cross-lingual` hay `false premise` hay không.

### 2.4 Frontend (UI/UX)
- Chạy hệ thống local và kiểm tra trực tiếp giao diện.
- Khi gửi tin nhắn hỏi đáp, câu trả lời hiển thị bắt buộc phải kèm theo các Trích dẫn (Citations) cho biết nguồn lấy thông tin ở trang số mấy, dòng nào, dựa trên metadata từ Qdrant/Backend.
