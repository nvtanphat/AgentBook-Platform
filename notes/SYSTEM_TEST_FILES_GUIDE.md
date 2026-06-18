# Hướng Dẫn Tải Học Liệu & Kiểm Thử Toàn Diện Hệ Thống AgentBook-PME
> **Bộ tài liệu chuẩn hóa dùng để xác thực 100% tính năng SOTA của hệ thống**
> *Tài liệu hướng dẫn thực hành - Phục vụ hội đồng chấm và viết báo khoa học*

---

## 🎯 1. Tại sao cần bộ 6 file học liệu dị cấu trúc này?
Để chứng minh với hội đồng chấm đồ án và phản biện báo khoa học rằng **AgentBook-PME** thực sự là một hệ thống **Universal Multi-format RAG** hoạt động bền bỉ, chúng ta không thể chỉ sử dụng tài liệu văn bản thuần (PDF). 

Hệ thống của bạn sở hữu các cấu hình phân tích hạt mịn đặc trưng (BBox pixel, timestamp audio, trích xuất bảng Excel, nhận diện chữ viết tay, multi-agent planner). Do đó, bộ 6 file học liệu dưới đây được tuyển chọn thiết kế để **kích hoạt và kiểm duyệt 100% các tính năng nâng cao** này.

---

## 📊 2. Bảng Danh Sách Học Liệu Kiểm Thử Toàn Diện

Bạn hãy mở Google / YouTube, gõ chính xác từ khóa để tải 6 file sau về máy và đưa vào hệ thống:

| # | Loại Học Liệu | Định Dạng | Lĩnh Vực | Từ Khóa Tìm Kiếm Google / YouTube | Tính Năng Hệ Thống Được Kiểm Thử (Features Tested) |
|---|---|---|---|---|---|
| **1** | **Báo cáo Bài tập lớn Đại học** | `.docx` | STEM / Kinh tế | `Báo cáo bài tập lớn hệ điều hành filetype:docx` <br>hoặc `Báo cáo bài tập lớn Mạng máy tính filetype:docx` | **Docling Word Parser & Reading-Order Context Fallback:** Kiểm thử khả năng dùng thứ tự đọc (Reading Order) để lấy ngữ cảnh bao quanh ảnh đồ thị/sơ đồ khi tọa độ pixel bị khuyết (`bbox=None`). |
| **2** | **Paper Học thuật Tiếng Anh** | `.pdf` | STEM | `attention is all you need pdf arxiv` | **Bilingual Cross-Lingual RAG & BBox Citation:** Kiểm thử khả năng gõ câu hỏi tiếng Việt truy vấn trên tài liệu tiếng Anh, dịch nghĩa chéo, định hạng lại bằng BGE Reranker và vẽ khung đỏ Bounding Box (`bbox`) dẫn chứng đè lên PDF ở frontend. |
| **3** | **Tài liệu Scanned / Viết tay** | `.pdf` | Khoa học Xã hội | `giáo trình triết học mác lênin scanned pdf` <br>*(Hoặc tự chụp ảnh tài liệu có viết chữ tay của bạn rồi chuyển sang PDF)* | **Image Quality Checker & Handwriting Reader:** Kích hoạt module đánh giá chất lượng scan ảnh. Nếu độ tin cậy thấp, chuyển dịch sang bộ đọc chữ viết tay để trích xuất và chặn an toàn ảo giác nếu độ tự tin dưới ngưỡng. |
| **4** | **Slide Bài Giảng** | `.pptx` | STEM / Kinh tế | `slide bài giảng Học máy HUST filetype:pptx` <br>hoặc `slide Deep Learning UET filetype:pptx` | **Per-Slide Chunking:** Đánh giá thuật toán gom cụm (cluster) toàn bộ các hộp chữ (text boxes) rời rạc trong cùng một slide $N$ về thành một phân mảnh ngữ nghĩa hợp nhất. |
| **5** | **Báo Cáo Tài Chính** | `.xlsx` | Kinh tế / Kế toán | `báo cáo tài chính FPT excel filetype:xlsx` <br>hoặc `Báo cáo tài chính Vinamilk excel cafef` | **Spreadsheet Parser & Row-Level Citation:** Trích xuất bảng tính Excel, đồng hóa dòng thành văn bản tự nhiên để bảo toàn logic hàng/cột, hiển thị trích dẫn chính xác dạng `[Dòng N]` trên giao diện. |
| **6** | **Audio Ghi Âm Bài Giảng** | `.mp3` | STEM / Xã hội | Lên YouTube gõ: `overfitting học máy` hoặc `bài giảng triết học` <br>➜ Tải MP3 qua công cụ convert. | **Whisper Speech Transcription & Audio Seek-to-Second:** Kiểm thử Whisper chia utterance theo giọng nói (VAD). Trên UI, click vào trích dẫn `[Audio @ 12:34]` bắt buộc audio player phải tự tua đến giây thứ 754. |

---

## 🛠️ 3. Quy Trình Chạy Thực Nghiệm (Step-by-Step Verification)

### 📥 Bước 1: Tải học liệu và Upload lập chỉ mục
1.  Tải đủ 6 định dạng file học liệu theo từ khóa gợi ý ở trên.
2.  Tải chúng lên cùng một **Collection** mới trên giao diện người dùng của AgentBook.
3.  Theo dõi terminal chạy Celery worker để đảm bảo tất cả 6 tài liệu đều được parse thành công và chuyển sang trạng thái xanh **`INDEXED`**.

### 📝 Bước 2: Thực hiện 6 câu hỏi kiểm thử vàng (Gold Queries)
Sau khi dữ liệu được index, bạn hãy đặt chính xác 6 câu hỏi sau vào chatbox để xác thực chất lượng phản hồi:

1.  **Test Word (.docx):** *"Mô tả quy trình hoạt động hoặc kiến trúc hệ thống đề xuất được vẽ trong sơ đồ của báo cáo bài tập lớn?"*
    *   *Kỳ vọng:* Hệ thống trích xuất đúng lý thuyết mô tả sơ đồ, mặc dù ảnh trong Word không có tọa độ BBox nhưng vẫn định vị được nhờ thuật toán đọc thứ tự tuần tự liền kề.
2.  **Test Bilingual & BBox PDF:** *"Cơ chế Multi-head attention hoạt động thế nào trong paper Attention?"*
    *   *Kỳ vọng:* Câu trả lời sinh ra bằng tiếng Việt trôi chảy. Khung tài liệu PDF bên phải tự động mở ra, cuộn đến đúng trang và vẽ **khung viền đỏ Bounding Box** đè lên đoạn văn chứa dẫn chứng.
3.  **Test OCR Scanned & Handwriting:** *"Định nghĩa vật chất của V.I.Lênin trong giáo trình Triết học là gì?"*
    *   *Kỳ vọng:* Hệ thống quét qua ảnh scan chất lượng thấp (hoặc phần chữ viết tay phụ lục), trả lời chính xác định nghĩa học thuyết mà không bị ảo giác.
4.  **Test Slide PPTX:** *"Quy trình huấn luyện mô hình học sâu gồm những bước cơ bản nào được tóm tắt trên Slide?"*
    *   *Kỳ vọng:* Câu trả lời gán nhãn dẫn chứng trỏ chính xác vào trang Slide bài giảng `slide_machine_learning.pptx`.
5.  **Test Excel XLSX:** *"Doanh thu tài chính hoặc lợi nhuận của doanh nghiệp đạt bao nhiêu tỷ đồng trong file Excel?"*
    *   *Kỳ vọng:* Câu trả lời chính xác con số. Phía dưới câu trả lời xuất hiện dẫn chứng dạng hạt mịn: `[Bảng số liệu - Dòng N]`. Click vào sẽ highlight đúng dòng đó.
6.  **Test Audio MP3:** *"Giảng viên giải thích và hướng dẫn cách khắc phục hiện tượng quá khớp (overfitting) như thế nào trong audio?"*
    *   *Kỳ vọng:* Trả lời đúng nội dung giảng viên nói. Xuất hiện dẫn chứng dạng thời gian: `[Audio @ Phút:Giây]`. Click vào dẫn chứng, trình phát audio tự động tua phát đúng đoạn âm thanh đó.

### 🧬 Bước 3: Chạy Multi-hop Query (Tột đỉnh Multi-Agent)
Sau khi test đơn lẻ, hãy gõ câu hỏi tổng hợp đa nguồn:
💬 *"So sánh định nghĩa overfitting và các kỹ thuật giải quyết giữa slide PPTX và ghi âm bài giảng audio?"*
*   **Hành vi mong đợi:** Bộ lập lịch Planner của Agentic RAG sẽ nhận diện đây là câu hỏi đa nguồn, Blackboard phân rã thành các truy vấn con, gọi song song 2 công cụ (PPTX reader + Audio transcriber), hợp nhất câu trả lời và cite đồng thời 2 nguồn khác định dạng trong cùng 1 câu trả lời.

---

> [!TIP]
> **Khởi động Benchmark VN-EduRAG-2000:**
> Sau khi đã xác thực 6 file học liệu này hoạt động hoàn hảo trên hệ thống, bạn hãy chạy lệnh dưới đây để bắt đầu tự động hóa quét toàn bộ database MongoDB và sinh bộ dữ liệu vàng 2000 câu hỏi học thuật phục vụ viết báo Q2:
> ```powershell
> python scripts/generate_testset.py
> ```
