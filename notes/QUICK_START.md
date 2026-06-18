# Quick Start - Noelys Local (không cần Docker)

## Yêu cầu cài đặt sẵn
- Python 3.11+
- Node.js 18+
- Ollama đang chạy (`ollama serve` + model `qwen2.5:3b`)

> **Không cần Docker, Redis, hay Celery worker riêng.**  
> Qdrant chạy embedded trong process Python (`data/vectordb`).  
> Celery task chạy đồng bộ (eager mode).  
> MongoDB đã trên Atlas (cloud).

---

## Bước 1: Tạo file `.env`

Copy từ example nếu chưa có:
```bash
cp backend/.env.example backend/.env
```

Điền `MONGODB_URI` trong `backend/.env` (Atlas URI của bạn).

---

## Bước 2: Cài Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

---

## Bước 3: Khởi động Backend

```bash
cd backend
python -m uvicorn src.main:app --port 8000
```

Đợi backend load xong (~30-60s để load BGE-M3 model).  
Kiểm tra: http://localhost:8000/health

---

## Bước 4: Khởi động Frontend

Mở terminal mới:
```bash
cd frontend
npm install   # lần đầu
npm run dev
```

Truy cập: http://localhost:5173

---

## Hoặc dùng script tự động (khuyến nghị)

```powershell
powershell -ExecutionPolicy Bypass -File start_all.ps1
```

Script sẽ tự động:
- Kill các process cũ trên port 8000, 5173
- Tạo thư mục Qdrant storage nếu chưa có
- Khởi động Backend (background)
- Khởi động Frontend (background)
- Hiển thị status và links

---

## Cấu hình Qdrant

Mặc định dùng embedded mode (`data/vectordb`).  
Nếu muốn dùng Docker Qdrant standalone thay thế:

```bash
docker compose up -d qdrant
```

Rồi đổi trong `backend/.env`:
```
AGENTBOOK_QDRANT_URL=http://localhost:6333
```

---

## Dừng services

```powershell
# Kill backend/frontend (Windows)
powershell -Command "Get-NetTCPConnection -LocalPort 8000,5173 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"
```

---

## Troubleshooting

### Backend không khởi động
```bash
# Xem log
cat backend.err.log

# Nguyên nhân thường gặp:
# - MongoDB URI sai → kiểm tra backend/.env
# - Port 8000 bị chiếm → kill process hoặc đổi port
# - Model chưa download → chạy: ollama pull qwen2.5:3b
```

### Frontend không khởi động
```bash
# Xem log
cat frontend.err.log

# Nguyên nhân:
# - Chưa npm install → cd frontend && npm install
# - Port 5173 bị chiếm → kill process
```

### Qdrant embedded lỗi
```bash
# Xóa storage và tạo lại
rm -rf data/vectordb
mkdir -p data/vectordb
# Rồi restart backend và re-index lại các materials
```
