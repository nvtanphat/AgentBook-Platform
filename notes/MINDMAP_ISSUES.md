# 🗺️ MINDMAP ISSUES - TROUBLESHOOTING GUIDE

## ❓ VẤN ĐỀ CÓ THỂ GẶP

### 1. **Mindmap không hiển thị gì**
**Nguyên nhân:**
- Chưa có entities trong database
- API response rỗng
- Frontend không render

**Giải pháp:**
```bash
# Check API response
curl -X POST http://127.0.0.1:8000/api/v1/graph/mindmap \
  -H "Content-Type: application/json" \
  -d '{
    "owner_id": "your_owner_id",
    "collection_id": "your_collection_id",
    "root_topic": "Machine Learning"
  }'

# Should return:
{
  "success": true,
  "data": {
    "root_topic": "Machine Learning",
    "nodes": [...]  # Should have nodes
  }
}
```

---

### 2. **Mindmap layout lộn xộn**
**Nguyên nhân:**
- Quá nhiều nodes (>50)
- Dagre layout không tối ưu
- Cluster logic không tốt

**Hiện tại:**
```typescript
// toMindmap() groups by entity type
// Root → Clusters (by type) → Entities

Root: "Machine Learning"
  ├─ Cluster: "Concept"
  │   ├─ Dropout
  │   └─ Regularization
  └─ Cluster: "Method"
      ├─ L1
      └─ L2
```

**Vấn đề:**
- Tất cả entities cùng type vào 1 cluster
- Không có hierarchy thực sự
- Không phản ánh document structure

---

### 3. **Mindmap không editable**
**Nguyên nhân:**
- Chưa implement edit mode
- Read-only by design

**Đây là Task #9** - Chưa làm

---

### 4. **Mindmap không có "big picture"**
**Nguyên nhân:**
- Chỉ show entities từ graph
- Không show document structure
- Không có table of contents

**Cần implement:**
- Document structure mindmap
- Chapter/section hierarchy
- Clickable navigation

---

## 🔧 QUICK FIXES

### Fix 1: Improve Clustering Logic
```typescript
// Better clustering by semantic similarity
function toMindmap(response: MindmapResponse) {
  // Group by topic, not just entity type
  const topics = clusterByTopic(response.nodes);
  
  // Create hierarchy
  Root → Topics → Subtopics → Entities
}
```

### Fix 2: Add Document Structure View
```typescript
// New mode: "document" mindmap
function toDocumentMindmap(materials: Material[]) {
  // Show document structure
  Root: "Collection Name"
    ├─ Document 1
    │   ├─ Chapter 1
    │   │   ├─ Section 1.1
    │   │   └─ Section 1.2
    │   └─ Chapter 2
    └─ Document 2
}
```

### Fix 3: Make Editable
```typescript
// Add edit mode
<GraphCanvas
  mode="mindmap"
  editable={editMode}
  onNodeDrag={handleDrag}
  onNodeAdd={handleAdd}
  onNodeDelete={handleDelete}
/>
```

---

## 📊 CURRENT MINDMAP FLOW

```
User clicks "Tạo Mindmap"
  ↓
Frontend calls: POST /api/v1/graph/mindmap
  ↓
Backend:
  1. Query entities from MongoDB
  2. Group by entity_type
  3. Return MindmapResponse
  ↓
Frontend:
  1. toMindmap() transforms to nodes/edges
  2. Groups by type → clusters
  3. Dagre layout (left-to-right)
  4. Render with GraphCanvas
```

---

## 🎯 WHAT DO YOU WANT TO FIX?

**Option A: Mindmap không hiển thị**
→ Tôi sẽ debug API response

**Option B: Layout lộn xộn**
→ Tôi sẽ improve clustering logic

**Option C: Muốn edit được**
→ Tôi sẽ implement editable mode (Task #9)

**Option D: Muốn thấy document structure**
→ Tôi sẽ add document structure view

**Option E: Khác**
→ Mô tả cụ thể vấn đề bạn gặp

---

## 🔍 DEBUG CHECKLIST

```
□ Có documents đã upload?
□ Có entities trong database? (Check MongoDB)
□ API /graph/mindmap có return nodes?
□ Frontend có render GraphCanvas?
□ Browser console có errors?
□ Mindmap tab có active?
```

---

**Cho tôi biết cụ thể vấn đề gì để tôi fix!** 🛠️
