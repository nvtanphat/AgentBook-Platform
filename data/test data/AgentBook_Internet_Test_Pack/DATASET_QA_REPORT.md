# Dataset QA Report

Ngay kiem tra: 2026-06-12

## Ket luan nhanh

- Dataset da dung tieng Viet lam chinh.
- Audio da duoc thay bang file WAV co giong noi tieng Viet that.
- Anh pass/fail dung voi `ImageQualityChecker`.
- PDF scan `report_vn.pdf` dung image-only, khong co text layer.
- Can luu y: `workbook_multi.xlsx` dang bi Windows khoa nen chua overwrite duoc ban co formula. Ban sua nam o `workbook_multi_fixed.xlsx`.

## Collection_A

| File | Trang thai | Ghi chu |
|---|---|---|
| `lecture_notes.pdf` | PASS | PDF text tieng Viet, 149 trang, co text layer. |
| `lecture_notes_en.pdf` | PASS | PDF EN phu tro, 14 trang, dung cho cross-lingual. |
| `report_vn.pdf` | PASS | PDF image-only, 1 trang, extract text = 0. |
| `comparison_doc.docx` | PASS | DOCX tieng Viet, 3 bang, dung cho compare/table retrieval. |
| `slides_mixed.pptx` | PASS | Slide bai giang tieng Viet tai tu UET/VNU, 64 slide, 301 text shapes, 9 hinh, 9 bang. |
| `slides_mixed_downloaded_source.pptx` | INFO | Ban source tai tu web, giong `slides_mixed.pptx`, giu lai de trace nguon. |

## Collection_B

| File | Trang thai | Ghi chu |
|---|---|---|
| `data_table.csv` | PASS | 23 dong, 5 cot, co 3 cot so. |
| `workbook_multi.xlsx` | WARN | 4 sheet nhung chua co formula; file dang bi process khac khoa luc ghi de. |
| `workbook_multi_fixed.xlsx` | PASS | Ban da them 40 formula va cac bang tong hop; dung ban nay neu can workbook thuc te hon. |
| `scan_clear.png` | PASS | Quality score 1.0, OCR quality pass. |
| `scan_low_quality.png` | PASS | Quality score 0.5532, fail dung muc tieu. |
| `handwriting_ok.png` | PASS | Quality score 1.0, handwriting pass. |
| `handwriting_good.png` | PASS | Quality score 1.0, handwriting pass. |
| `handwriting_blur.png` | PASS | Quality score 0.5486, fail dung muc tieu. |
| `handwriting_bad.png` | PASS | Quality score 0.5486, fail dung muc tieu. |
| `handwriting_good.jpg` | INFO | Anh source goc tai tu web; PNG derivatives la file test chinh. |
| `lecture.wav` | PASS | Giong noi tieng Viet that, 70.66 giay, 44.1 kHz stereo. |
| `lecture_vi_real.wav` | PASS | Ban alias cua audio that de de nhan dien. |

## Viec nen lam tiep

1. Dong Excel/preview neu dang mo `workbook_multi.xlsx`.
2. Thay `workbook_multi.xlsx` bang `workbook_multi_fixed.xlsx`.
3. Neu muon dataset thuc te hon nua, thay CSV san pham tu tao bang mot CSV nghiep vu tieng Viet lay tu he thong/nguon mo thuc.
