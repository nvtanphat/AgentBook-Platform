# 🐛 MINDMAP BUG FOUND

## VẤN ĐỀ

### Backend trả về:
```python
# backend/src/api/v1/endpoints/graph.py line 225
summary=f"{entity.entity_type} confidence={entity.confidence:.2f}"

# Example:
{
  "id": "123",
  "label": "Dropout",
  "summary": "concept confidence=0.85",  # ← Format này
  "citations": [...]
}
```

### Frontend parse:
```typescript
// frontend/src/components/workspace/studio/GraphTab.tsx line 53
const typeMatch = node.summary?.match(/^(\w+)/);
const entityType = typeMatch?.[1] ?? "concept";

// Regex /^(\w+)/ sẽ match "concept" từ "concept confidence=0.85"
// → OK, không có bug ở đây
```

---

## THỰC SỰ VẤN ĐỀ LÀ GÌ?

Có 3 khả năng:

### 1. **Không có entities trong database**
```bash
# Check MongoDB
# Nếu không có entities → mindmap rỗng
```

### 2. **Mindmap layout không đẹp**
```
Hiện tại:
Root → Cluster (by type) → All entities of that type

Vấn đề:
- Tất cả "concept" entities vào 1 cluster
- Không có semantic grouping
- Flat hierarchy
```

### 3. **Mindmap không interactive**
```
- Không edit được
- Không expand/collapse
- Không drill-down
```

---

## SOLUTION

### Fix 1: Better Clustering (Semantic grouping)
```typescript
// Instead of grouping by entity_type
// Group by semantic similarity or topic

function toMindmap(response: MindmapResponse) {
  // Cluster entities by semantic similarity
  const clusters = semanticClustering(response.nodes);
  
  // Create hierarchy
  Root
    ├─ Topic 1 (e.g., "Regularization Techniques")
    │   ├─ Dropout
    │   ├─ L1 Regularization
    │   └─ L2 Regularization
    └─ Topic 2 (e.g., "Neural Network Architectures")
        ├─ CNN
        └─ RNN
}
```

### Fix 2: Add Entity Type to Node
```python
# Backend: Return entity_type explicitly
MindmapNode(
    id=str(entity.id),
    label=entity.canonical_name,
    entity_type=entity.entity_type,  # ← Add this
    summary=f"confidence={entity.confidence:.2f}",
    citations=_evidence_refs(entity.mention_refs),
)
```

### Fix 3: Improve Layout
```typescript
// Use better Dagre settings
g.setGraph({ 
  rankdir: "TB",  // Top-to-bottom instead of left-to-right
  ranksep: 120,   // More space between levels
  nodesep: 60,    // More space between nodes
  edgesep: 30,
});
```

---

## QUICK TEST

Để xem mindmap có data không:

```bash
# Test API
curl -X POST http://127.0.0.1:8000/api/v1/graph/mindmap \
  -H "Content-Type: application/json" \
  -d '{
    "owner_id": "test_user",
    "collection_id": "your_collection_id",
    "root_topic": "Test"
  }'

# Should return nodes array
# If empty → No entities in database
# If has nodes → Frontend rendering issue
```

---

## WHAT TO FIX?

**Bạn muốn tôi fix cái nào?**

A. Improve clustering logic (semantic grouping)
B. Add entity_type field to backend response
C. Change layout from LR to TB (top-to-bottom)
D. Make mindmap editable
E. Add expand/collapse functionality

**Hoặc cho tôi biết cụ thể mindmap "không ổn" như thế nào?**
- Không hiển thị?
- Layout xấu?
- Quá nhiều nodes?
- Không interactive?
