"""Multi-tool layer for the AgenticCoordinatingEngine.

Each tool is a self-describing class with a stable `name`, a one-line
`description`, and an async `run()` that returns typed evidence. Tools are
the *only* mechanism agents use to touch the outside world — this gives the
coordinator a single auditing surface and lets us swap implementations
without touching agent logic.
"""

from src.agentic.tools.base import BaseTool, ToolResult
from src.agentic.tools.graph_relation_search import GraphRelationSearchTool
from src.agentic.tools.hybrid_text_search import HybridTextSearchTool
from src.agentic.tools.nli_verifier import NLIVerifierTool
from src.agentic.tools.text_cleaner import TextCleanerTool
from src.agentic.tools.visual_image_search import VisualImageSearchTool

__all__ = [
    "BaseTool",
    "GraphRelationSearchTool",
    "HybridTextSearchTool",
    "NLIVerifierTool",
    "TextCleanerTool",
    "ToolResult",
    "VisualImageSearchTool",
]
