# Nguồn audio tiếng Việt thực tế

Thư mục này không chứa audio synthetic. Trong phiên sandbox hiện tại, công cụ tải file bị chặn với MIME `audio/mpeg`, nên không thể lưu trực tiếp MP3/OGG vào ZIP mà không làm giả dữ liệu.

Nguồn thực tế để tải thủ công:

1. Internet Archive — Kinh Tương Ưng Bộ, audio tiếng Việt, nhiều file MP3 dài, phù hợp test ASR long-form / chunking:
   https://archive.org/details/KinhTuongUngBo
   Ví dụ file:
   https://archive.org/download/KinhTuongUngBo/001-Tap1C1ChuThienP1CayLau.mp3
   https://archive.org/download/KinhTuongUngBo/002-Tap1C1ChuThienP2VuonHoanHy.mp3

2. FPT Open Speech Dataset (FOSD) — dữ liệu tiếng Việt công khai, metadata/transcript, CC0/Public Domain theo mô tả trên OpenScience:
   https://openscience.vn/chi-tiet-du-lieu/fpt-open-speech-dataset-fosd-vietnamese-11619
   Nguồn gốc dữ liệu: https://data.mendeley.com/datasets/k9sxg2twv4/4

Khuyến nghị: tải 2–3 file MP3 từ Internet Archive, cắt bằng ffmpeg thành đoạn 5–15 phút nếu cần test giống podcast/phỏng vấn:
ffmpeg -i input.mp3 -ss 00:00:00 -t 00:10:00 -ac 1 -ar 16000 output_10min.wav
