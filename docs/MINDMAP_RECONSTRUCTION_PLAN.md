# 🗺️ MINDMAP RECONSTRUCTION - IMPLEMENTATION PLAN

## PHASE 1: SMART HIERARCHY (Backend)

### Current Problem:
```python
# backend/src/api/v1/endpoints/graph.py
# Flat list of entities, no hierarchy
entities = await Entity.find(query).sort("-confidence").limit(60).to_list()
```

### Solution: Semantic Clustering + Hierarchy

```python
# backend/src/services/mindmap_builder.py (NEW FILE)
from sklearn.cluster import AgglomerativeClustering
import numpy as np

class MindmapBuilder:
    """
    Build hierarchical mindmap from entities using semantic clustering.
    
    Hierarchy:
    Root Topic
      ├─ Main Topic 1 (semantic cluster)
      │   ├─ Subtopic 1.1
      │   └─ Subtopic 1.2
      └─ Main Topic 2
          └─ Subtopic 2.1
    """
    
    def __init__(self, embedder):
        self.embedder = embedder
    
    async def build_hierarchy(
        self, 
        entities: list[Entity], 
        root_topic: str,
        max_clusters: int = 5
    ) -> MindmapHierarchy:
        """
        Build 3-level hierarchy using semantic clustering.
        
        Algorithm:
        1. Embed all entity names
        2. Cluster into main topics (level 1)
        3. Sub-cluster each topic (level 2)
        4. Assign entities to subtopics (level 3)
        """
        
        # Step 1: Embed entities
        entity_texts = [e.canonical_name for e in entities]
        embeddings = await self.embedder.encode(entity_texts)
        
        # Step 2: Main topic clustering (level 1)
        n_main_clusters = min(max_clusters, len(entities) // 5)
        main_clustering = AgglomerativeClustering(
            n_clusters=n_main_clusters,
            metric='cosine',
            linkage='average'
        )
        main_labels = main_clustering.fit_predict(embeddings)
        
        # Step 3: Build hierarchy
        hierarchy = MindmapHierarchy(root=root_topic, children=[])
        
        for cluster_id in range(n_main_clusters):
            # Get entities in this cluster
            cluster_entities = [
                entities[i] for i, label in enumerate(main_labels) 
                if label == cluster_id
            ]
            
            # Generate topic name from most central entity
            topic_name = self._generate_topic_name(cluster_entities)
            
            # Sub-cluster if needed
            if len(cluster_entities) > 8:
                subtopics = self._create_subtopics(cluster_entities, embeddings)
                hierarchy.children.append(
                    MindmapTopic(name=topic_name, children=subtopics)
                )
            else:
                # Direct children
                hierarchy.children.append(
                    MindmapTopic(
                        name=topic_name,
                        children=[
                            MindmapNode(
                                id=str(e.id),
                                label=e.canonical_name,
                                entity_type=e.entity_type,
                                confidence=e.confidence
                            )
                            for e in cluster_entities
                        ]
                    )
                )
        
        return hierarchy
    
    def _generate_topic_name(self, entities: list[Entity]) -> str:
        """
        Generate topic name from cluster.
        
        Strategy:
        1. Find most central entity (highest confidence)
        2. Use entity_type as category
        3. Combine: "{type}: {central_entity}"
        """
        if not entities:
            return "Miscellaneous"
        
        # Sort by confidence
        sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)
        central = sorted_entities[0]
        
        # Get common type
        types = [e.entity_type for e in entities]
        most_common_type = max(set(types), key=types.count)
        
        return f"{most_common_type.title()}: {central.canonical_name}"
    
    def _create_subtopics(
        self, 
        entities: list[Entity], 
        embeddings: np.ndarray
    ) -> list[MindmapTopic]:
        """Create subtopics for large clusters."""
        n_sub = min(3, len(entities) // 3)
        
        sub_clustering = AgglomerativeClustering(
            n_clusters=n_sub,
            metric='cosine',
            linkage='average'
        )
        sub_labels = sub_clustering.fit_predict(embeddings)
        
        subtopics = []
        for sub_id in range(n_sub):
            sub_entities = [
                entities[i] for i, label in enumerate(sub_labels)
                if label == sub_id
            ]
            
            subtopics.append(
                MindmapTopic(
                    name=self._generate_topic_name(sub_entities),
                    children=[
                        MindmapNode(
                            id=str(e.id),
                            label=e.canonical_name,
                            entity_type=e.entity_type,
                            confidence=e.confidence
                        )
                        for e in sub_entities
                    ]
                )
            )
        
        return subtopics


# New schemas
class MindmapNode(BaseModel):
    id: str
    label: str
    entity_type: str
    confidence: float
    collapsed: bool = False  # NEW: For collapsible branches

class MindmapTopic(BaseModel):
    name: str
    children: list[MindmapNode | MindmapTopic]
    collapsed: bool = False

class MindmapHierarchy(BaseModel):
    root: str
    children: list[MindmapTopic]
```

---

## PHASE 2: PREMIUM UX (Frontend)

### 2.1 Collapsible Branches

```typescript
// frontend/src/components/EnhancedMindmap.tsx
import { useState, useCallback } from 'react';
import ReactFlow, { 
  Node, 
  Edge, 
  useNodesState, 
  useEdgesState,
  NodeProps 
} from 'reactflow';

function CollapsibleNode({ data, id }: NodeProps) {
  const [collapsed, setCollapsed] = useState(data.collapsed || false);
  
  const handleToggle = useCallback(() => {
    // Toggle collapse state
    setCollapsed(!collapsed);
    
    // Hide/show children
    updateChildrenVisibility(id, !collapsed);
  }, [collapsed, id]);
  
  return (
    <div className="mindmap-node">
      <div className="node-content">
        {data.label}
      </div>
      
      {data.hasChildren && (
        <button 
          className="collapse-btn"
          onClick={handleToggle}
        >
          {collapsed ? '+' : '−'}
        </button>
      )}
    </div>
  );
}
```

### 2.2 Context Menu

```typescript
// frontend/src/components/MindmapContextMenu.tsx
import { useCallback } from 'react';
import { MessageSquare, FileText, Link2, Trash2 } from 'lucide-react';

interface ContextMenuProps {
  node: Node;
  position: { x: number; y: number };
  onClose: () => void;
}

function MindmapContextMenu({ node, position, onClose }: ContextMenuProps) {
  const handleAskAI = useCallback(() => {
    // Open chat with pre-filled question about this node
    openChat(`Tell me more about ${node.data.label}`);
    onClose();
  }, [node]);
  
  const handleViewSources = useCallback(() => {
    // Show source documents for this entity
    showSourcePanel(node.id);
    onClose();
  }, [node]);
  
  const handleFindRelated = useCallback(() => {
    // Highlight related nodes in graph
    highlightRelatedNodes(node.id);
    onClose();
  }, [node]);
  
  return (
    <div 
      className="context-menu"
      style={{ left: position.x, top: position.y }}
    >
      <button onClick={handleAskAI}>
        <MessageSquare size={14} />
        Ask AI about this
      </button>
      
      <button onClick={handleViewSources}>
        <FileText size={14} />
        View source documents
      </button>
      
      <button onClick={handleFindRelated}>
        <Link2 size={14} />
        Find related concepts
      </button>
      
      <div className="divider" />
      
      <button onClick={() => deleteNode(node.id)} className="danger">
        <Trash2 size={14} />
        Remove from mindmap
      </button>
    </div>
  );
}
```

### 2.3 Smooth Animations

```typescript
// frontend/src/components/AnimatedMindmap.tsx
import { useSpring, animated } from '@react-spring/web';

function AnimatedMindmapNode({ data, isNew }: NodeProps) {
  // Entrance animation for new nodes
  const style = useSpring({
    from: { 
      opacity: 0, 
      scale: 0.8,
      y: -20 
    },
    to: { 
      opacity: 1, 
      scale: 1,
      y: 0 
    },
    config: { tension: 280, friction: 60 }
  });
  
  return (
    <animated.div style={style} className="mindmap-node">
      {data.label}
    </animated.div>
  );
}

// Smooth edge transitions
const edgeStyle = {
  stroke: '#cbd5e1',
  strokeWidth: 2,
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
};
```

### 2.4 Drag & Drop with Auto-save

```typescript
// frontend/src/components/DraggableMindmap.tsx
import { useCallback } from 'react';
import { NodeDragHandler } from 'reactflow';

function DraggableMindmap() {
  const onNodeDragStop: NodeDragHandler = useCallback(
    async (event, node) => {
      // Auto-save position to backend
      await saveMindmapNodePosition({
        node_id: node.id,
        position: node.position,
        owner_id: workspace.ownerId,
        collection_id: workspace.collectionId,
      });
      
      // Show subtle feedback
      showToast('Position saved', { duration: 1000 });
    },
    [workspace]
  );
  
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodeDragStop={onNodeDragStop}
      nodesDraggable={editMode}
    />
  );
}
```

---

## PHASE 3: AESTHETIC REFINEMENT

### 3.1 Professional Color System

```typescript
// frontend/src/styles/mindmap-theme.ts
export const MINDMAP_THEME = {
  // Node colors (semantic, not random)
  nodes: {
    root: {
      bg: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      border: '#5a67d8',
      text: '#ffffff',
      shadow: '0 10px 40px -10px rgba(102, 126, 234, 0.4)',
    },
    topic: {
      bg: '#ffffff',
      border: '#e2e8f0',
      text: '#1e293b',
      shadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
    },
    concept: {
      bg: '#eff6ff',
      border: '#3b82f6',
      text: '#1e40af',
      shadow: '0 2px 8px rgba(59, 130, 246, 0.15)',
    },
    // ... other types
  },
  
  // Edge styles
  edges: {
    default: {
      stroke: '#cbd5e1',
      strokeWidth: 2,
      type: 'smoothstep', // Smooth bezier curves
    },
    highlighted: {
      stroke: '#3b82f6',
      strokeWidth: 3,
      animated: true,
    },
  },
  
  // Spacing
  layout: {
    nodeSpacing: 80,
    levelSpacing: 150,
    padding: 40,
  },
};
```

### 3.2 Bezier Edges with Gradient

```typescript
// Custom edge component
function GradientEdge({ 
  id, 
  sourceX, 
  sourceY, 
  targetX, 
  targetY,
  style 
}: EdgeProps) {
  const edgePath = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    curvature: 0.25, // Smooth curve
  });
  
  return (
    <>
      <defs>
        <linearGradient id={`gradient-${id}`}>
          <stop offset="0%" stopColor="#cbd5e1" />
          <stop offset="100%" stopColor="#94a3b8" />
        </linearGradient>
      </defs>
      
      <path
        id={id}
        d={edgePath}
        stroke={`url(#gradient-${id})`}
        strokeWidth={2}
        fill="none"
        className="react-flow__edge-path"
        style={{
          ...style,
          filter: 'drop-shadow(0 1px 2px rgba(0, 0, 0, 0.1))',
        }}
      />
    </>
  );
}
```

---

## PHASE 4: EXPORT & SHARE

```typescript
// frontend/src/utils/mindmap-export.ts
import { toPng, toSvg } from 'html-to-image';
import jsPDF from 'jspdf';

export async function exportMindmap(
  format: 'png' | 'svg' | 'pdf',
  reactFlowInstance: ReactFlowInstance
) {
  const viewport = reactFlowInstance.getViewport();
  const nodes = reactFlowInstance.getNodes();
  
  // Fit view before export
  reactFlowInstance.fitView({ padding: 0.2 });
  
  const element = document.querySelector('.react-flow') as HTMLElement;
  
  switch (format) {
    case 'png':
      const dataUrl = await toPng(element, {
        backgroundColor: '#ffffff',
        quality: 1.0,
        pixelRatio: 2, // High DPI
      });
      downloadFile(dataUrl, 'mindmap.png');
      break;
      
    case 'svg':
      const svgData = await toSvg(element);
      downloadFile(svgData, 'mindmap.svg');
      break;
      
    case 'pdf':
      const imgData = await toPng(element, { quality: 1.0 });
      const pdf = new jsPDF('landscape');
      pdf.addImage(imgData, 'PNG', 10, 10, 280, 180);
      pdf.save('mindmap.pdf');
      break;
  }
  
  // Restore viewport
  reactFlowInstance.setViewport(viewport);
}
```

---

## IMPLEMENTATION TIMELINE

### Week 1: Backend Hierarchy
- Day 1-2: Implement MindmapBuilder with semantic clustering
- Day 3-4: Update API endpoint to return hierarchical structure
- Day 5: Test with real data

### Week 2: Frontend Core
- Day 1-2: Collapsible branches
- Day 3: Context menu
- Day 4-5: Drag & drop with auto-save

### Week 3: Polish
- Day 1-2: Animations & transitions
- Day 3: Color system & aesthetics
- Day 4-5: Export functionality

---

## SUCCESS METRICS

**Performance:**
- Render 200+ nodes smoothly (60 FPS)
- Layout calculation < 500ms
- Drag response < 16ms

**UX:**
- Collapse/expand < 200ms
- Context menu appears instantly
- Export completes < 2s

**Quality:**
- Semantic clustering accuracy > 80%
- User satisfaction with hierarchy
- Export quality (high DPI, clean edges)

---

**Ready to implement?** 🚀
