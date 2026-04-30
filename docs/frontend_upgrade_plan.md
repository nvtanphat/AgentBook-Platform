# AgentBook — Đánh Giá & Kế Hoạch Nâng Cấp Frontend

Sau khi kiểm tra toàn bộ ~30 file frontend + backend, đây là tổng hợp các vấn đề và kế hoạch sửa, ưu tiên theo mức ảnh hưởng trải nghiệm người dùng.

---

## 🔴 Các vấn đề nghiêm trọng (Must-fix)

### 1. Chat — Citation hiển thị raw IDs thay vì `[1] [2]` dạng đánh số

**Hiện trạng:**
- Câu trả lời hiện chỉ render text thuần túy (line 288 ChatPage.tsx): `<p className="whitespace-pre-wrap">{message.content}</p>`
- LLM prompt yêu cầu trích dẫn dạng `[Nguồn: doc_name, trang X, block Y]` (qa_grounded.txt line 6)
- Nhưng frontend **không parse** chuỗi này → hiển thị nguyên text raw, lẫn doc_id / block_id dài dòng
- Citation chips ở dưới hiện show: `{citation.doc_name} - page {citation.page} - {citation.block_id}` → **in ra cả block_id UUID phiền**

**Kế hoạch sửa:**
- **Backend:** Sửa prompt `qa_grounded.txt` → yêu cầu LLM dùng dạng `[1]`, `[2]` (footnote style) thay vì chuỗi nguồn dài
- **Backend:** `format_evidence_for_prompt()` đã đánh số `[1]`, `[2]` sẵn → chỉ cần prompt bảo LLM dùng đúng số đó
- **Frontend:** Parse `[1]`, `[2]` trong câu trả lời thành clickable links, click → scroll tới citation tương ứng
- **Frontend:** Citation chip: bỏ block_id, chỉ hiện `[1] doc_name - trang X`

---

### 2. Upload Page — Luồng chọn tài liệu không hợp lý

**Hiện trạng:**
- Kéo thả chỉ nhận 1 file → không multi-file upload
- Khu vực metadata quá nhiều field kỹ thuật cho user thường (modality, source_type, language code `vi`/`en`)
- `accept` attribute thiếu trên `<input type="file">` → không filter file type khi browse
- Không preview file đã chọn (PDF/image)
- Upload xong hiện raw job ID + stage trong success message → user bình thường không hiểu

**Kế hoạch sửa:**
- Thêm `accept` attribute lọc file types được hỗ trợ
- Hỗ trợ multi-file upload (chọn nhiều file, queue upload tuần tự)
- Rút gọn metadata form: auto-detect language từ filename/content, ẩn modality/source_type vào "Advanced"
- Success message thân thiện hơn: `"filename.pdf đã upload thành công! Đang xử lý..."` thay vì job ID
- Thêm progress indicator rõ ràng hơn (giai đoạn: uploading → parsing → indexing)

---

### 3. Sidebar — Hiển thị Collection ID raw

**Hiện trạng (AppShell.tsx line 60):**
```tsx
<p className="mt-1 truncate text-xs text-muted">
  {workspace.collectionId || "No active collection yet"}
</p>
```
→ Hiện chuỗi ObjectId `6745a2b3...` rất xấu

**Kế hoạch sửa:**
- Ẩn collection ID, chỉ hiện collection name + số chunks
- Nếu muốn xem ID → tooltip hoặc copy button

---

## 🟡 Tính năng hiện có trên giao diện nhưng chưa hoạt động/chưa đủ

### 4. Summary & Study Guide — Backend có API nhưng Frontend chưa gọi

**Hiện trạng:**
- Backend có `/api/v1/query/summarize` và `/api/v1/query/study-guide` endpoints (query.py lines 55-80)
- Backend có `SummaryService` và `StudyGuideService` đầy đủ
- Frontend `client.ts` **không có** hàm gọi 2 endpoint này
- Không có trang Summary / Study Guide nào trên UI

**Kế hoạch sửa:**
- Thêm 2 API functions vào `client.ts`: `summarizeCollection()` và `buildStudyGuide()`
- Thêm nút "Summarize" và "Study Guide" trên ChatPage hoặc tạo tab riêng

---

### 5. Evidence Page — Yêu cầu nhập Document ID thủ công

**Hiện trạng:**
- User phải paste doc_id vào ô input (EvidencePage.tsx line 62)
- Dropdown chọn material chỉ hiện khi có local materials → nếu user mới mở trang thì trống
- Không có cách từ ChatPage nhảy sang Evidence Page tự động với đúng doc + page

**Kế hoạch sửa:**
- Citation chips trong Chat khi click → navigate tới `/evidence?doc=X&page=Y` tự động
- Bỏ ô nhập Document ID thủ công, thay bằng dropdown liệt kê tất cả materials từ active collection
- Thêm button "View in Evidence" trên EvidencePanel

---

### 6. Graph / Mindmap — Chỉ render cơ bản, thiếu tương tác

**Hiện trạng:**
- Graph/Mindmap dùng ReactFlow nhưng layout vòng tròn đơn giản (`fallbackPosition()`)
- Click node chỉ set text `selectedNode` → không load evidence tương ứng
- Không có search/filter nodes
- Mindmap chỉ render flat 1-level (tất cả nodes nối trực tiếp root)

**Kế hoạch sửa:**
- Khi click node → load evidence/citation tương ứng từ mention_refs
- Thêm search bar filter nodes
- Cải thiện layout: dùng dagre/elkjs cho hierarchical layout
- Mindmap: build tree từ children[] thực sự (MindmapNode đã có field children)

---

### 7. Compare Page — UX cơ bản

**Hiện trạng:**
- Citation chip chỉ hiện `block_id` (ComparePage line 103) → không rõ nguồn
- Không có highlight/diff giữa các sources khác nhau
- Conflict section chỉ hiện text thuần

**Kế hoạch sửa:**
- Citation chip hiện doc_name + page
- Highlight diff giữa các giá trị từ sources khác nhau
- Conflicts hiện kèm evidence snippet

---

## 🟢 Cải thiện UX/UI tổng thể

### 8. Chat UX improvements

| Vấn đề | Sửa |
|---------|-----|
| Không auto-scroll khi có message mới | Thêm `scrollIntoView` khi messages thay đổi |
| Không có markdown rendering | Parse markdown trong answer (bold, list, code block) |
| Casual reply hardcoded client-side | Giữ nguyên (OK cho MVP) |
| Loading message chung chung | Hiện "Đang tìm kiếm trong tài liệu..." |
| Không có nút clear chat | Thêm nút clear history |

### 9. Settings Page — Collection ID vẫn là text input thủ công

**Kế hoạch sửa:**
- Thêm dropdown chọn collection (giống ChatPage đã có) thay vì paste ID
- Hiển thị collection info (số materials, số chunks)

---

## Danh sách file cần thay đổi

### Backend

| File | Thay đổi |
|------|----------|
| `backend/src/prompts/qa_grounded.txt` | Sửa prompt: yêu cầu LLM dùng `[1]`, `[2]` inline footnote thay vì `[Nguồn: ...]` |

### Frontend

| File | Thay đổi |
|------|----------|
| `frontend/src/pages/ChatPage.tsx` | Parse `[1]` `[2]` thành clickable links, citation chips bỏ block_id, auto-scroll, markdown rendering, nút clear chat |
| `frontend/src/pages/UploadPage.tsx` | Thêm `accept` attribute, multi-file, ẩn advanced fields, success message thân thiện |
| `frontend/src/components/AppShell.tsx` | Ẩn Collection ID raw, chỉ hiện name + chunk count |
| `frontend/src/pages/EvidencePage.tsx` | Dropdown chọn material từ collection thay vì paste doc_id |
| `frontend/src/pages/ComparePage.tsx` | Citation chip hiện doc_name thay vì block_id |
| `frontend/src/api/client.ts` | Thêm `summarizeCollection()` và `buildStudyGuide()` API functions |
| `frontend/src/pages/SettingsPage.tsx` | Thêm dropdown chọn collection |

---

## Thứ tự thực hiện đề xuất

| Priority | Task | Impact |
|----------|------|--------|
| **P0** | Sửa citation `[1] [2]` trong Chat + bỏ raw IDs | ⭐⭐⭐⭐⭐ |
| **P0** | Sửa prompt backend cho footnote style | ⭐⭐⭐⭐⭐ |
| **P1** | Upload page UX (accept, success msg, ẩn advanced) | ⭐⭐⭐⭐ |
| **P1** | Sidebar ẩn Collection ID | ⭐⭐⭐⭐ |
| **P2** | Auto-scroll + markdown rendering Chat | ⭐⭐⭐ |
| **P2** | Evidence page dropdown thay vì paste ID | ⭐⭐⭐ |
| **P2** | Compare page citation fix | ⭐⭐⭐ |
| **P3** | Summary + Study Guide frontend integration | ⭐⭐ |
| **P3** | Graph/Mindmap interaction improvements | ⭐⭐ |
| **P3** | Settings page collection dropdown | ⭐⭐ |

---

## Ghi chú

- Tất cả thay đổi đề xuất đều nằm trong phạm vi MVP, không thêm dependency mới
- Markdown rendering trong chat chỉ cần regex đơn giản cho bold/list, không cần library
- Backend đã có đầy đủ services (SummaryService, StudyGuideService, GraphRetriever) — chỉ cần frontend kết nối
