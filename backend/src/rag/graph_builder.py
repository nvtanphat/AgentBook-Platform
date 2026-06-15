"""In-memory graph analytics using networkx for knowledge graph enrichment."""
from __future__ import annotations

import networkx as nx

from src.schemas.graph import GraphEdge, GraphNode


def build_digraph(nodes: list[GraphNode], edges: list[GraphEdge]) -> nx.DiGraph:
    G: nx.DiGraph = nx.DiGraph()
    for node in nodes:
        G.add_node(node.id)
    for edge in edges:
        if G.has_node(edge.source) and G.has_node(edge.target):
            G.add_edge(edge.source, edge.target, weight=edge.confidence or 0.5)
    return G


def compute_degrees(G: nx.DiGraph) -> dict[str, int]:
    return dict(G.degree())


def compute_pagerank(G: nx.DiGraph) -> dict[str, float]:
    if G.number_of_nodes() == 0:
        return {}
    try:
        return nx.pagerank(G, alpha=0.85, max_iter=100)
    except nx.PowerIterationFailedConvergence:
        n = G.number_of_nodes()
        return {node: 1.0 / n for node in G.nodes()}


def compute_communities(G: nx.DiGraph) -> dict[str, int]:
    """Detect communities using Louvain on the undirected projection.

    Returns dict[node_id → community_id]. Falls back to single-community when
    the graph is empty or louvain library is unavailable.
    """
    if G.number_of_nodes() == 0:
        return {}
    try:
        import community as community_louvain  # python-louvain
        UG = G.to_undirected()
        partition = community_louvain.best_partition(UG, weight="weight", random_state=42)
        return partition
    except Exception:
        # Fallback: use connected components as communities
        UG = G.to_undirected()
        communities = list(nx.connected_components(UG))
        partition: dict[str, int] = {}
        for idx, component in enumerate(communities):
            for node in component:
                partition[node] = idx
        return partition


def compute_community_labels(
    community_map: dict[str, int],
    label_by_id: dict[str, str],
    importance_by_id: dict[str, float],
    *,
    top_k: int = 3,
) -> dict[int, str]:
    """Derive a human-readable label per community from its most central members.

    GraphRAG generates LLM summaries per community; this is the zero-cost
    deterministic variant — join the top-`top_k` highest-importance member
    labels (e.g. "RAG · Embedding · Retrieval"). Good enough to orient the user
    without an LLM call; callers may override with an LLM summary when desired.
    """
    members: dict[int, list[str]] = {}
    for node_id, comm in community_map.items():
        members.setdefault(comm, []).append(node_id)

    labels: dict[int, str] = {}
    for comm, node_ids in members.items():
        ranked = sorted(
            node_ids,
            key=lambda nid: importance_by_id.get(nid, 0.0),
            reverse=True,
        )
        names = [label_by_id.get(nid, "").strip() for nid in ranked[:top_k]]
        names = [n for n in names if n]
        labels[comm] = " · ".join(names) if names else f"Cụm {comm}"
    return labels


def compute_betweenness(G: nx.DiGraph) -> dict[str, float]:
    """Betweenness centrality — identifies 'bridge' nodes between communities."""
    if G.number_of_nodes() == 0:
        return {}
    try:
        return nx.betweenness_centrality(G, weight="weight", normalized=True)
    except Exception:
        return {n: 0.0 for n in G.nodes()}
