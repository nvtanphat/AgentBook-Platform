# 🔄 RESTART BACKEND - Load New Features

## Backend đang chạy (PID: 7888)

Để load **Reasoning Path Visualization** feature mới:

### Option 1: Restart trong terminal hiện tại
```bash
# Trong terminal đang chạy backend:
# 1. Press Ctrl+C để stop
# 2. Chạy lại:
cd D:/GenAI/DoAn01/backend
uvicorn src.main:app --reload
```

### Option 2: Kill process và restart
```bash
# Kill process
taskkill /PID 7888 /F

# Start lại
cd D:/GenAI/DoAn01/backend
uvicorn src.main:app --reload
```

### Option 3: Dùng --reload flag (Recommended)
Nếu bạn đã start với `--reload` flag, backend sẽ **tự động reload** khi detect file changes.

Thử save lại file để trigger reload:
```bash
# Touch file để trigger reload
cd D:/GenAI/DoAn01/backend
touch src/main.py
```

---

## ✅ Verify Backend Loaded New Code

Sau khi restart, test API:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Hoặc mở browser:
```
http://127.0.0.1:8000/docs
```

Check xem `/api/v1/query/ask` endpoint có field `reasoning_path` trong response schema không.

---

## 🎯 Expected Response Schema

```json
{
  "answer": "string",
  "answer_language": "vi",
  "query_language": "vi",
  "citations": [...],
  "confidence": 0.85,
  "reasoning_path": [
    {
      "step_type": "retrieve",
      "entities": ["Dropout"],
      "relations": [],
      "confidence": 0.9,
      "description": "Retrieved 10 chunks from 3 documents"
    }
  ]
}
```

---

## 🚀 After Restart

Frontend sẽ tự động nhận `reasoning_path` từ API response và hiển thị:

```
💡 How I found this answer:
1. 📈 Retrieved X chunks from Y documents (confidence%)
2. ✨ Traversed knowledge graph
3. 💡 Synthesized answer
```

**Restart backend để thấy feature mới!** 🎉
