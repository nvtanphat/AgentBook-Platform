# 🎯 AGENTBOOK UX IMPROVEMENT ROADMAP

## PRIORITY 1: Connect Graph to Answer (CRITICAL)

### Problem
Khi AI trả lời, người dùng KHÔNG thấy graph nodes/edges nào được sử dụng. Graph và Answer như 2 thế giới riêng biệt.

### Solution: Reasoning Path Visualization

#### Backend Changes
```python
# backend/src/schemas/query.py
class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    
    # NEW: Graph reasoning path
    reasoning_path: Optional[List[ReasoningStep]] = None

class ReasoningStep(BaseModel):
    step_type: Literal["retrieve", "traverse", "synthesize"]
    entities: List[str]  # Entity IDs involved
    relations: List[str]  # Relation types used
    confidence: float
    description: str  # Human-readable explanation
```

#### Frontend Changes
```typescript
// components/AnswerWithGraph.tsx
function AnswerWithGraph({ response }: { response: QueryResponse }) {
  const [highlightedPath, setHighlightedPath] = useState<string[]>([]);

  return (
    <div className="grid grid-cols-2 gap-4">
      {/* Left: Answer */}
      <div className="prose">
        <MarkdownRenderer content={response.answer} />
        
        {/* NEW: Reasoning trace */}
        <ReasoningTrace 
          steps={response.reasoning_path}
          onStepHover={(entities) => setHighlightedPath(entities)}
        />
      </div>

      {/* Right: Graph with highlighted path */}
      <GraphCanvas
        nodes={nodes}
        edges={edges}
        highlightedNodes={highlightedPath}
        animateHighlight={true}
      />
    </div>
  );
}

// components/ReasoningTrace.tsx
function ReasoningTrace({ steps, onStepHover }) {
  return (
    <div className="mt-4 space-y-2">
      <p className="text-xs font-semibold text-muted">
        💡 How I found this answer:
      </p>
      {steps.map((step, i) => (
        <div 
          key={i}
          className="flex items-start gap-2 p-2 rounded hover:bg-blue-50 cursor-pointer"
          onMouseEnter={() => onStepHover(step.entities)}
          onMouseLeave={() => onStepHover([])}
        >
          <span className="text-xs font-bold text-primary">{i + 1}</span>
          <div className="flex-1">
            <p className="text-xs text-text">{step.description}</p>
            <div className="flex gap-1 mt-1">
              {step.entities.map(e => (
                <span key={e} className="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 text-blue-700">
                  {e}
                </span>
              ))}
            </div>
          </div>
          <ConfidenceBar value={step.confidence} />
        </div>
      ))}
    </div>
  );
}
```

#### Example Output
```
User: "Dropout giúp giảm overfitting như thế nào?"

AI Answer: "Dropout là kỹ thuật regularization..."

💡 How I found this answer:
1. Retrieved concept [Dropout] from document ML_Techniques.pdf
   Entities: [Dropout] | Confidence: 95%

2. Traversed relation [Dropout] --prevents--> [Overfitting]
   Entities: [Dropout, Overfitting] | Confidence: 87%

3. Found supporting evidence in [Regularization Methods]
   Entities: [Regularization, Dropout] | Confidence: 82%

[Graph highlights these 3 nodes with animated path]
```

---

## PRIORITY 2: Semantic Clustering (HIGH)

### Problem
Với 100+ nodes, graph trở thành "búi tóc". Overwhelming cho người dùng.

### Solution: Progressive Disclosure with Clustering

#### Implementation
```typescript
// components/ClusteredGraph.tsx
function ClusteredGraph({ nodes, edges }) {
  const [expandedClusters, setExpandedClusters] = useState<Set<string>>(new Set());
  
  // Detect communities using Louvain algorithm
  const communities = useMemo(() => 
    detectCommunities(nodes, edges), 
    [nodes, edges]
  );

  // Show clusters as meta-nodes
  const displayNodes = useMemo(() => {
    const result = [];
    
    for (const [clusterId, members] of communities) {
      if (expandedClusters.has(clusterId)) {
        // Show individual nodes
        result.push(...members);
      } else {
        // Show cluster as single meta-node
        result.push({
          id: clusterId,
          type: 'cluster',
          label: `${members[0].type} (${members.length})`,
          size: Math.sqrt(members.length) * 20,
          members: members,
        });
      }
    }
    
    return result;
  }, [communities, expandedClusters]);

  return (
    <GraphCanvas
      nodes={displayNodes}
      edges={edges}
      onNodeDoubleClick={(node) => {
        if (node.type === 'cluster') {
          // Expand cluster with animation
          setExpandedClusters(prev => new Set([...prev, node.id]));
        }
      }}
    />
  );
}
```

#### Visual Design
```
Level 1 (Overview):
┌─────────────────┐
│ ML Concepts (25)│  ← Cluster meta-node
└─────────────────┘
        ↓ Double-click
        
Level 2 (Expanded):
    Dropout ──┐
    L1 Reg ───┼─→ Overfitting
    L2 Reg ───┘
    Early Stop
```

---

## PRIORITY 3: Editable Mindmap (MEDIUM)

### Problem
Người dùng không thể chỉnh sửa mindmap để "dạy ngược lại" AI.

### Solution: Collaborative Mindmap Editor

#### Implementation
```typescript
// components/EditableMindmap.tsx
function EditableMindmap({ initialNodes, initialEdges, onSave }) {
  const [editMode, setEditMode] = useState(false);
  const [customNodes, setCustomNodes] = useState(initialNodes);

  return (
    <div className="relative h-full">
      {/* Edit mode toggle */}
      <div className="absolute top-4 right-4 z-10">
        <button
          onClick={() => setEditMode(!editMode)}
          className="px-3 py-1.5 rounded bg-white border shadow-sm"
        >
          {editMode ? '✓ Done Editing' : '✏️ Edit Mindmap'}
        </button>
      </div>

      <ReactFlow
        nodes={customNodes}
        edges={edges}
        nodesDraggable={editMode}
        nodesConnectable={editMode}
        elementsSelectable={editMode}
        onNodesChange={(changes) => {
          if (editMode) {
            setCustomNodes(applyNodeChanges(changes, customNodes));
          }
        }}
      >
        {editMode && (
          <Panel position="top-left">
            <div className="bg-white p-2 rounded shadow space-y-2">
              <button onClick={addNode}>+ Add Node</button>
              <button onClick={deleteSelected}>🗑️ Delete</button>
              <button onClick={() => onSave(customNodes)}>
                💾 Save My Version
              </button>
            </div>
          </Panel>
        )}
      </ReactFlow>
    </div>
  );
}
```

#### Use Cases
1. **Student**: Reorganize concepts theo cách hiểu riêng
2. **Teacher**: Add annotations và notes
3. **Researcher**: Group related ideas, add hypotheses

---

## PRIORITY 4: Context Menu & Quick Actions (MEDIUM)

### Problem
Click node chỉ show info card, không có actions.

### Solution: Rich Context Menu

#### Implementation
```typescript
// components/NodeContextMenu.tsx
function NodeContextMenu({ node, position, onClose }) {
  return (
    <div 
      className="absolute z-50 bg-white rounded-lg shadow-xl border p-1"
      style={{ left: position.x, top: position.y }}
    >
      <MenuItem 
        icon={<Search />}
        onClick={() => findRelatedConcepts(node.id)}
      >
        Find related concepts
      </MenuItem>
      
      <MenuItem 
        icon={<FileText />}
        onClick={() => viewSourceDocuments(node.id)}
      >
        View source documents ({node.source_docs.length})
      </MenuItem>
      
      <MenuItem 
        icon={<MessageSquare />}
        onClick={() => askAboutConcept(node.label)}
      >
        Ask AI about "{node.label}"
      </MenuItem>
      
      <MenuItem 
        icon={<Share />}
        onClick={() => shareSubgraph(node.id)}
      >
        Share this subgraph
      </MenuItem>
      
      <MenuItem 
        icon={<Eye />}
        onClick={() => highlightInDocuments(node.label)}
      >
        Highlight in documents
      </MenuItem>
    </div>
  );
}
```

---

## PRIORITY 5: Animation & Storytelling (LOW)

### Problem
Graph xuất hiện đột ngột, không có storytelling.

### Solution: Animated Entrance & Path Highlighting

#### Implementation
```typescript
// hooks/useGraphAnimation.ts
function useGraphAnimation(nodes: Node[], edges: Edge[]) {
  useEffect(() => {
    // Stagger node entrance
    nodes.forEach((node, i) => {
      setTimeout(() => {
        node.style = {
          ...node.style,
          opacity: 1,
          transform: 'scale(1)',
        };
      }, i * 50);
    });

    // Draw edges after nodes
    setTimeout(() => {
      edges.forEach((edge, i) => {
        setTimeout(() => {
          edge.animated = true;
        }, i * 30);
      });
    }, nodes.length * 50);
  }, [nodes, edges]);
}

// Highlight path animation
function highlightPath(path: string[], duration: number = 1000) {
  path.forEach((nodeId, i) => {
    setTimeout(() => {
      // Pulse animation
      const node = document.querySelector(`[data-id="${nodeId}"]`);
      node?.classList.add('pulse-highlight');
      
      setTimeout(() => {
        node?.classList.remove('pulse-highlight');
      }, 500);
    }, (duration / path.length) * i);
  });
}
```

#### CSS
```css
@keyframes pulse-highlight {
  0%, 100% { 
    transform: scale(1); 
    box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7);
  }
  50% { 
    transform: scale(1.1); 
    box-shadow: 0 0 0 10px rgba(59, 130, 246, 0);
  }
}

.pulse-highlight {
  animation: pulse-highlight 0.5s ease-out;
}
```

---

## IMPLEMENTATION TIMELINE

### Week 1-2: Foundation
- ✅ Add reasoning_path to QueryResponse schema
- ✅ Implement ReasoningTrace component
- ✅ Add graph highlighting API

### Week 3-4: Clustering
- ✅ Implement community detection (Louvain)
- ✅ Create ClusteredGraph component
- ✅ Add expand/collapse animations

### Week 5-6: Interactivity
- ✅ Editable mindmap mode
- ✅ Context menu with quick actions
- ✅ Save custom mindmap versions

### Week 7-8: Polish
- ✅ Animation & transitions
- ✅ Performance optimization
- ✅ User testing & iteration

---

## SUCCESS METRICS

### Quantitative
- **Time to insight**: 2-8s → 0.5-3s (với reasoning trace)
- **Graph comprehension**: 40% → 80% (với clustering)
- **User engagement**: 30% → 70% (với editable mindmap)

### Qualitative
- "Tôi hiểu AI suy luận như thế nào" (reasoning trace)
- "Graph không còn overwhelming" (clustering)
- "Tôi có thể tổ chức kiến thức theo cách riêng" (editable)

---

**Last Updated:** 2026-05-02
**Status:** Design phase - Ready for implementation
