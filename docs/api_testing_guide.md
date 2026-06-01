# Hướng dẫn Kiểm tra Hệ thống AgentBook qua API

> [!NOTE]
> Backend: `http://localhost:8000` · Swagger UI: `http://localhost:8000/docs`  
> Auth mặc định **TẮT** (`api_auth_enabled=false`) ở môi trường development → bỏ qua header `Authorization` nếu chưa bật.

---

## 0. Khởi động hệ thống

```powershell
cd d:\GenAI\DoAn01
.\start_all.ps1
```

Kiểm tra health check:
```powershell
Invoke-RestMethod http://localhost:8000/health
```
Kỳ vọng: `{ "status": "ok", "service": "Noelys" }`

---

## 1. Auth — Đăng ký & Đăng nhập

### 1a. Đăng ký tài khoản mới
```powershell
$body = @{
    email        = "test@example.com"
    password     = "secret123"
    display_name = "Tester"
} | ConvertTo-Json

$resp = Invoke-RestMethod `
    -Uri    "http://localhost:8000/api/v1/auth/register" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body

$resp.data | ConvertTo-Json
```

### 1b. Đăng nhập
```powershell
$body = @{
    email_or_id = "test@example.com"
    password    = "secret123"
} | ConvertTo-Json

$resp = Invoke-RestMethod `
    -Uri    "http://localhost:8000/api/v1/auth/login" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body

# Lưu token và owner_id để dùng ở bước sau
$TOKEN    = $resp.data.access_token
$OWNER_ID = $resp.data.user.user_id
Write-Host "owner_id = $OWNER_ID"
```

> [!IMPORTANT]
> Nếu **auth bị tắt** (development), bạn vẫn cần `owner_id` thực từ MongoDB.  
> Lấy bằng cách đăng ký/login rồi dùng `$OWNER_ID` ở mọi bước tiếp theo.

---

## 2. Collections — Tạo & liệt kê

### 2a. Tạo collection
```powershell
$body = @{
    owner_id    = $OWNER_ID
    name        = "Lập trình Python"
    subject     = "Python"
    description = "Tài liệu học Python cơ bản"
} | ConvertTo-Json

$resp = Invoke-RestMethod `
    -Uri    "http://localhost:8000/api/v1/collections" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body

$COLLECTION_ID = $resp.data.collection_id
Write-Host "collection_id = $COLLECTION_ID"
```

### 2b. Liệt kê collections
```powershell
Invoke-RestMethod `
    "http://localhost:8000/api/v1/collections?owner_id=$OWNER_ID" `
    -Method GET
```

### 2c. Dashboard của collection
```powershell
Invoke-RestMethod `
    "http://localhost:8000/api/v1/collections/$COLLECTION_ID/dashboard?owner_id=$OWNER_ID" `
    -Method GET
```

---

## 3. Materials — Upload tài liệu

### 3a. Upload một file (PDF/DOCX/PPTX/PNG...)
```powershell
$metadata = @{
    owner_id      = $OWNER_ID
    collection_id = $COLLECTION_ID
    subject       = "Python"
    topic         = "Cơ bản"
} | ConvertTo-Json

$filePath = "C:\path\to\your\document.pdf"

$form = @{
    metadata = $metadata
    file     = Get-Item $filePath
}

$resp = Invoke-RestMethod `
    -Uri    "http://localhost:8000/api/v1/materials/upload" `
    -Method POST `
    -Form   $form

$MATERIAL_ID = $resp.data.material_id
Write-Host "material_id = $MATERIAL_ID"
```

### 3b. Theo dõi trạng thái pipeline
```powershell
# Lặp polling đến khi indexed hoặc failed
do {
    $status = Invoke-RestMethod `
        "http://localhost:8000/api/v1/materials/$MATERIAL_ID/status?owner_id=$OWNER_ID"
    
    Write-Host "[$($status.data.stage)] $($status.data.progress_pct)% — $($status.data.status)"
    Start-Sleep 5
} while ($status.data.status -notin @("indexed", "failed"))
```

Các trạng thái pipeline:

| Status | Progress | Ý nghĩa |
|--------|----------|---------|
| `uploaded` | 10% | File nhận xong, chờ parse |
| `parsing` | 30% | Docling đang parse |
| `parsed` | 55% | Parse xong, chờ chunk + embed |
| `indexing` | 80% | Đang đưa vào Qdrant |
| `indexed` | 100% | Sẵn sàng truy vấn ✅ |
| `failed` | 100% | Lỗi — xem `error_message` |

### 3c. Debug một material (xem chunks + vectors)
```powershell
Invoke-RestMethod `
    "http://localhost:8000/api/v1/materials/$MATERIAL_ID/debug?owner_id=$OWNER_ID" `
    -Method GET | ConvertTo-Json -Depth 5
```

### 3d. Retry material bị failed
```powershell
Invoke-RestMethod `
    "http://localhost:8000/api/v1/materials/$MATERIAL_ID/retry?owner_id=$OWNER_ID" `
    -Method POST
```

### 3e. Liệt kê materials trong collection
```powershell
Invoke-RestMethod `
    "http://localhost:8000/api/v1/materials?owner_id=$OWNER_ID&collection_id=$COLLECTION_ID" `
    -Method GET
```

---

## 4. Query — Truy vấn RAG (chính)

> [!IMPORTANT]
> **Phải upload và index xong ít nhất một material** trước khi query.

### 4a. Ask — Hỏi đáp cơ bản (không stream)
```powershell
$body = @{
    owner_id      = $OWNER_ID
    collection_id = $COLLECTION_ID
    question      = "Python là gì? Ứng dụng của Python trong thực tế?"
    conversation_id = "test-session-01"
} | ConvertTo-Json

$resp = Invoke-RestMethod `
    -Uri    "http://localhost:8000/api/v1/query/ask" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body

# Xem câu trả lời
$resp.data.answer
# Xem evidence/citations
$resp.data.evidence | ConvertTo-Json -Depth 4
```

### 4b. Ask-Stream — Trả lời theo SSE (streaming)
```powershell
# Dùng curl cho SSE vì PowerShell không hỗ trợ streaming tốt
curl -N -X POST http://localhost:8000/api/v1/query/ask-stream `
  -H "Content-Type: application/json" `
  -d "{`"owner_id`":`"$OWNER_ID`",`"collection_id`":`"$COLLECTION_ID`",`"question`":`"Giải thích list comprehension trong Python`"}"
```

### 4c. Compare — So sánh hai tài liệu
```powershell
# Lấy material_ids bằng lệnh list ở bước 3e
$body = @{
    owner_id      = $OWNER_ID
    collection_id = $COLLECTION_ID
    material_ids  = @($MATERIAL_ID_1, $MATERIAL_ID_2)  # 2 materials
    question      = "So sánh điểm khác nhau về cú pháp giữa hai tài liệu"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri    "http://localhost:8000/api/v1/query/compare" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

### 4d. Summarize — Tóm tắt tài liệu
```powershell
$body = @{
    owner_id      = $OWNER_ID
    collection_id = $COLLECTION_ID
    material_ids  = @($MATERIAL_ID)
    focus         = "Các khái niệm Python cốt lõi"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri    "http://localhost:8000/api/v1/query/summarize" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

### 4e. Study Guide — Tạo tài liệu học tập
```powershell
$body = @{
    owner_id      = $OWNER_ID
    collection_id = $COLLECTION_ID
    material_ids  = @($MATERIAL_ID)
    focus         = "Python cơ bản"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri    "http://localhost:8000/api/v1/query/study-guide" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

---

## 5. Swagger UI — Cách dễ nhất để test

Mở trình duyệt: **http://localhost:8000/docs**

Tại đây bạn có thể:
1. Xem toàn bộ endpoint và schema
2. Test trực tiếp qua giao diện **"Try it out"**
3. Không cần viết PowerShell

---

## 6. Script tổng hợp — End-to-End Test

```powershell
# ============================================================
# AgentBook — End-to-End API Test Script
# ============================================================
$BASE = "http://localhost:8000/api/v1"

# --- Bước 1: Health check ---
$health = Invoke-RestMethod "$($BASE -replace '/api/v1','')/health"
if ($health.status -ne "ok") { throw "Backend chưa chạy!" }
Write-Host "✅ Backend OK"

# --- Bước 2: Đăng ký / Login ---
$loginBody = @{ email_or_id = "test@example.com"; password = "secret123" } | ConvertTo-Json
try {
    $auth = Invoke-RestMethod -Uri "$BASE/auth/login" -Method POST -ContentType "application/json" -Body $loginBody
} catch {
    $regBody = @{ email = "test@example.com"; password = "secret123"; display_name = "Tester" } | ConvertTo-Json
    $auth = Invoke-RestMethod -Uri "$BASE/auth/register" -Method POST -ContentType "application/json" -Body $regBody
}
$TOKEN    = $auth.data.access_token
$OWNER_ID = $auth.data.user.user_id
Write-Host "✅ Auth OK — owner_id: $OWNER_ID"

# --- Bước 3: Tạo collection ---
$colBody = @{ owner_id = $OWNER_ID; name = "Test Collection"; subject = "Test" } | ConvertTo-Json
$col = Invoke-RestMethod -Uri "$BASE/collections" -Method POST -ContentType "application/json" -Body $colBody
$COLLECTION_ID = $col.data.collection_id
Write-Host "✅ Collection created: $COLLECTION_ID"

# --- Bước 4: Upload file ---
# Thay đường dẫn file thật ở đây
$FILE_PATH = "C:\path\to\test.pdf"
$meta = @{ owner_id = $OWNER_ID; collection_id = $COLLECTION_ID } | ConvertTo-Json
$uploadResp = Invoke-RestMethod -Uri "$BASE/materials/upload" -Method POST `
    -Form @{ metadata = $meta; file = Get-Item $FILE_PATH }
$MATERIAL_ID = $uploadResp.data.material_id
Write-Host "✅ Uploaded: $MATERIAL_ID"

# --- Bước 5: Chờ indexed ---
Write-Host "⏳ Chờ pipeline..."
do {
    Start-Sleep 10
    $st = Invoke-RestMethod "$BASE/materials/$MATERIAL_ID/status?owner_id=$OWNER_ID"
    Write-Host "   → $($st.data.stage) ($($st.data.progress_pct)%)"
} while ($st.data.status -notin @("indexed","failed"))

if ($st.data.status -eq "failed") {
    Write-Host "❌ Pipeline failed: $($st.data.error_message)"
    exit 1
}
Write-Host "✅ Material indexed!"

# --- Bước 6: Query ---
$qBody = @{ owner_id = $OWNER_ID; collection_id = $COLLECTION_ID; question = "Tóm tắt nội dung tài liệu này" } | ConvertTo-Json
$qResp = Invoke-RestMethod -Uri "$BASE/query/ask" -Method POST -ContentType "application/json" -Body $qBody
Write-Host "`n📖 Câu trả lời:`n$($qResp.data.answer)"
Write-Host "`n📌 Evidence count: $($qResp.data.evidence.Count)"
```

---

## 7. Checklist kiểm tra nhanh

| # | Bước | Endpoint | Kết quả mong đợi |
|---|------|----------|-----------------|
| 1 | Health | `GET /health` | `status: ok` |
| 2 | Đăng ký | `POST /auth/register` | Token + user_id |
| 3 | Login | `POST /auth/login` | Token hợp lệ |
| 4 | Tạo collection | `POST /collections` | `collection_id` |
| 5 | Upload file | `POST /materials/upload` | `material_id`, status=`uploaded` |
| 6 | Theo dõi status | `GET /materials/{id}/status` | Tiến đến `indexed` |
| 7 | Query ask | `POST /query/ask` | `answer` + `evidence[]` có citation |
| 8 | Debug chunks | `GET /materials/{id}/debug` | Danh sách chunks + qdrant_vector_count > 0 |

