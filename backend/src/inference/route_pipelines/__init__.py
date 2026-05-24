"""Phase C — True per-route execution pipelines.

The `inference_engine.answer()` orchestrator delegates per-route decisions to
a strategy object selected from this package. Each pipeline implements the
hooks defined in `base.BaseRoutePipeline`. Add new behaviours here, not in
`answer()` — that keeps the orchestrator readable and tests focused.
"""

from src.inference.route_pipelines.base import BaseRoutePipeline, RouteHooks
from src.inference.route_pipelines.claim_check import ClaimCheckPipeline
from src.inference.route_pipelines.factual import FactualPipeline
from src.inference.route_pipelines.general import GeneralPipeline
from src.inference.route_pipelines.summarization import SummarizationPipeline
from src.rag.query_router import RouteType


_REGISTRY: dict[RouteType, type[BaseRoutePipeline]] = {
    RouteType.FACTUAL: FactualPipeline,
    RouteType.SUMMARIZATION: SummarizationPipeline,
    RouteType.CLAIM_CHECK: ClaimCheckPipeline,
    # COMPARISON / GRAPH_RELATION / GENERAL share the relaxed fallback for now.
    RouteType.COMPARISON: GeneralPipeline,
    RouteType.GRAPH_RELATION: GeneralPipeline,
    RouteType.GENERAL: GeneralPipeline,
}


def get_pipeline(route_type: RouteType, **kwargs) -> BaseRoutePipeline:
    """Return the strategy object for a given route_type.

    Pipelines are cheap to construct; the heavyweight dependencies (NLI model,
    claim_verifier) are lazy-loaded inside each pipeline so cold-start stays
    fast even when the route is never exercised.
    """
    cls = _REGISTRY.get(route_type, GeneralPipeline)
    # Tweak override for graph_relation / comparison: relax refusal as before.
    if route_type in (RouteType.COMPARISON, RouteType.GRAPH_RELATION):
        return cls(name=route_type.value, relax_refusal=True, skip_llm_retry_on_refusal=True, **kwargs)
    return cls(**kwargs)


__all__ = [
    "BaseRoutePipeline",
    "ClaimCheckPipeline",
    "FactualPipeline",
    "GeneralPipeline",
    "RouteHooks",
    "SummarizationPipeline",
    "get_pipeline",
]
