# AgentBook Internet Test Pack - Tieng Viet la chinh

Bo test nay duoc setup theo `docs/test_documents_guide.md`, nhung uu tien tieng Viet cho hau het kich ban. File tieng Anh chi duoc giu rieng de test cross-lingual VN -> EN khi can.

## Collection_A

- `lecture_notes.pdf`: PDF text-based tieng Viet ve tri tue nhan tao, tai tu nguon cong khai, dung lam tai lieu RAG chinh.
- `report_vn.pdf`: PDF image-only tao tu anh scan tieng Viet, dung test OCR.
- `comparison_doc.docx`: DOCX tieng Viet co 3 bang, cung chu de AgentBook/RAG/OCR de test compare va table retrieval.
- `slides_mixed.pptx`: slide bai giang tieng Viet tai tu UET/VNU, 64 slide, co text, bang va hinh.
- `lecture_notes_en.pdf`: PDF tieng Anh phu tro, chi dung cho kich ban hoi tieng Viet tren tai lieu EN.
- `slides_mixed_downloaded_source.pptx`: ban source tai tu web, giong `slides_mixed.pptx`, giu lai de trace nguon.

## Collection_B

- `data_table.csv`: bang san pham tieng Viet, co cot text va so.
- `workbook_multi.xlsx`: workbook 4 sheet tieng Viet: `San_pham`, `Doanh_so_thang`, `Kiem_dinh_OCR`, `Ghi_chu_tai_lieu`.
- `scan_clear.png`: anh scan tieng Viet ro net, ky vong indexed.
- `scan_low_quality.png`: anh tieng Viet bi lam mo/toi, ky vong failed.
- `handwriting_good.png` / `handwriting_ok.png`: mau chu viet tay tieng Viet doc duoc.
- `handwriting_bad.png` / `handwriting_blur.png`: mau chu viet tay mo, tuong phan thap, ky vong failed.
- `lecture.wav` / `lecture_vi_real.wav`: file WAV co giong noi tieng Viet that, dai khoang 70 giay, dung smoke test audio parser.

## Nguon web

- PDF tieng Viet chinh: https://dost.hochiminhcity.gov.vn/documents/1165/Ai_for_Student_Phien_ban_Tieng_Viet_SIHUB.pdf
- PDF EN phu tro: https://arxiv.org/pdf/2506.18027
- DOCX source: https://www2.hu-berlin.de/stadtlabor/wp-content/uploads/2021/12/sample3.docx
- PPTX tieng Viet source: https://uet.vnu.edu.vn/~vietanh/courses/thcs/Part_2.pptx
- Audio tieng Viet that: https://commons.wikimedia.org/wiki/File:B%C3%A0i_th%C6%A1_Ki%E1%BA%BFp_L%C6%B0u_Vong.wav

## Snapshot validate

- `lecture_notes.pdf`: 149 trang, co text layer tieng Viet.
- `report_vn.pdf`: 1 trang, image-only, 0 ky tu text extract.
- `comparison_doc.docx`: 3 bang.
- `slides_mixed.pptx`: 64 slide, 301 text shapes, 9 hinh, 9 bang.
- `data_table.csv`: 23 dong gom header, 5 cot.
- `workbook_multi.xlsx`: 4 sheet.
- `scan_clear.png` va `handwriting_ok.png`: pass image quality checker.
- `scan_low_quality.png` va `handwriting_blur.png`: fail image quality checker dung muc tieu.
- `lecture.wav`: WAV co giong noi tieng Viet, 70.66 giay, 44.1 kHz stereo.
