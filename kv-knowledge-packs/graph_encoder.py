"""Graph Geometry Encoder — Encode knowledge graph topology as KV cache.

Three encoding methods for converting graph structure into pre-computed
KV cache blocks that can be injected at inference time via Knowledge Packs.

The model doesn't read about the connections. It attends through them.

Methods:
  1. Adjacency-as-attention: edge weights → attention matrix → KV cache
     that would produce those attention patterns
  2. Spectral encoding: graph Laplacian eigenvectors → positional
     embeddings → KV cache with structural position information
  3. Walk encoding: random walk transition probabilities → soft attention
     distribution → KV cache

Requires: numpy (for linear algebra), networkx (for graph operations)
Integrates with: kv_packs.py (CacheBlock, KVPackBuilder)
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@dataclass
class GraphEncoding:
    """Encoded graph topology ready for KV cache conversion."""
    method: str
    num_nodes: int
    attention_matrix: object = None
    positional_embeddings: object = None
    node_labels: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def encode_adjacency(graph, normalize: bool = True) -> GraphEncoding:
    """Method 1: Adjacency-as-attention.

    Converts edge weights into an attention-like matrix. Each node
    becomes a "token" whose attention to other nodes reflects the
    graph connectivity. The resulting matrix can be used to synthesize
    KV cache that produces these attention patterns.

    For a graph with n nodes, produces an n×n attention matrix where
    A[i,j] = normalized edge weight between node i and node j.
    """
    if not HAS_NUMPY or not HAS_NETWORKX:
        raise ImportError("numpy and networkx required for graph encoding")

    n = graph.number_of_nodes()
    nodes = list(graph.nodes())
    node_idx = {node: i for i, node in enumerate(nodes)}

    adj = np.zeros((n, n), dtype=np.float32)
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)
        i, j = node_idx[u], node_idx[v]
        adj[i, j] = weight
        if not graph.is_directed():
            adj[j, i] = weight

    adj += np.eye(n) * 0.1

    if normalize:
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        adj = adj / row_sums

    labels = [str(n) for n in nodes]
    if all(isinstance(n, str) for n in nodes):
        labels = list(nodes)

    return GraphEncoding(
        method="adjacency",
        num_nodes=n,
        attention_matrix=adj,
        node_labels=labels,
        metadata={
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_directed": graph.is_directed(),
        },
    )


def encode_spectral(graph, k: int = None) -> GraphEncoding:
    """Method 2: Spectral encoding.

    Computes the graph Laplacian eigenvectors and uses them as
    positional embeddings. Each node gets a position in spectral
    space that encodes its structural role in the graph.

    Nodes that are structurally similar (same community, similar
    connectivity pattern) get similar spectral positions — even if
    they're not directly connected.

    k: number of eigenvectors to use. Default: min(n, 16).
    """
    if not HAS_NUMPY or not HAS_NETWORKX:
        raise ImportError("numpy and networkx required for graph encoding")

    n = graph.number_of_nodes()
    if k is None:
        k = min(n, 16)

    L = nx.laplacian_matrix(graph).toarray().astype(np.float32)

    eigenvalues, eigenvectors = np.linalg.eigh(L)

    positional = eigenvectors[:, 1:k+1]

    if positional.shape[1] < k:
        padding = np.zeros((n, k - positional.shape[1]), dtype=np.float32)
        positional = np.concatenate([positional, padding], axis=1)

    norms = np.linalg.norm(positional, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    positional = positional / norms

    nodes = list(graph.nodes())
    labels = [str(nd) for nd in nodes]

    fiedler = eigenvalues[1] if len(eigenvalues) > 1 else 0

    return GraphEncoding(
        method="spectral",
        num_nodes=n,
        positional_embeddings=positional,
        node_labels=labels,
        metadata={
            "k": k,
            "eigenvalues": eigenvalues[:k+1].tolist(),
            "fiedler_value": float(fiedler),
            "num_components": nx.number_connected_components(graph)
            if not graph.is_directed() else -1,
        },
    )


def encode_walk(graph, walk_length: int = 5,
                num_walks: int = 100) -> GraphEncoding:
    """Method 3: Walk encoding.

    Computes random walk transition probabilities from each node.
    The resulting matrix represents "if you start at node i, how
    likely are you to reach node j in walk_length steps?"

    This captures multi-hop reachability — nodes connected through
    bridge paths get non-zero attention even without direct edges.
    """
    if not HAS_NUMPY or not HAS_NETWORKX:
        raise ImportError("numpy and networkx required for graph encoding")

    n = graph.number_of_nodes()
    nodes = list(graph.nodes())
    node_idx = {node: i for i, node in enumerate(nodes)}

    T = np.zeros((n, n), dtype=np.float32)
    for u in nodes:
        neighbors = list(graph.neighbors(u))
        if neighbors:
            for v in neighbors:
                weight = graph[u][v].get('weight', 1.0)
                T[node_idx[u], node_idx[v]] = weight
            row_sum = T[node_idx[u]].sum()
            if row_sum > 0:
                T[node_idx[u]] /= row_sum

    walk_matrix = np.eye(n, dtype=np.float32)
    transition = np.eye(n, dtype=np.float32)
    for step in range(walk_length):
        transition = transition @ T
        walk_matrix += transition

    walk_matrix /= (walk_length + 1)

    labels = [str(nd) for nd in nodes]

    return GraphEncoding(
        method="walk",
        num_nodes=n,
        attention_matrix=walk_matrix,
        node_labels=labels,
        metadata={
            "walk_length": walk_length,
            "avg_reachability": float(np.mean(walk_matrix > 0.01)),
        },
    )


def graph_encoding_to_text(encoding: GraphEncoding) -> str:
    """Convert a graph encoding to a text representation for KV Pack encoding.

    This is the bridge between graph geometry and the KV cache builder.
    The text representation encodes the structural information in a format
    the model can process through its standard text-to-KV path.

    For the pure geometric injection (Phase 2), this is replaced by
    direct tensor manipulation. But for Phase 1 proof of concept,
    text encoding lets us test whether the structural signal survives
    the text→KV→attention pipeline.
    """
    lines = [f"Knowledge graph structure ({encoding.method} encoding, "
             f"{encoding.num_nodes} nodes):\n"]

    if encoding.attention_matrix is not None:
        matrix = encoding.attention_matrix
        for i, label_i in enumerate(encoding.node_labels):
            connections = []
            for j, label_j in enumerate(encoding.node_labels):
                weight = float(matrix[i, j])
                if weight > 0.05 and i != j:
                    connections.append(f"{label_j} ({weight:.2f})")
            if connections:
                lines.append(f"{label_i} connects to: {', '.join(connections)}")

    if encoding.positional_embeddings is not None:
        embeddings = encoding.positional_embeddings
        for i, label in enumerate(encoding.node_labels):
            pos = embeddings[i]
            nearby = []
            for j, label_j in enumerate(encoding.node_labels):
                if i != j:
                    dist = float(np.linalg.norm(pos - embeddings[j]))
                    if dist < 0.5:
                        nearby.append(f"{label_j} (d={dist:.2f})")
            if nearby:
                lines.append(f"{label} is structurally near: {', '.join(nearby[:5])}")

    return '\n'.join(lines)


def build_test_graph() -> object:
    """Build a test graph with known structure for Phase 1 experiments.

    Three communities connected by bridge nodes:
    - Cluster A: research concepts (KV cache, geometry, spectral)
    - Cluster B: ethics concepts (consent, justice, solidarity)
    - Cluster C: engineering concepts (Docker, NATS, systemd)
    - Bridge A↔B: "AI welfare" connects research to ethics
    - Bridge B↔C: "infrastructure" connects ethics to engineering
    - Isolate: "random_node" with no connections
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required")

    G = nx.Graph()

    cluster_a = ["KV_cache", "geometry", "spectral_entropy", "SVD",
                 "effective_rank", "attention"]
    for i, n1 in enumerate(cluster_a):
        for n2 in cluster_a[i+1:]:
            G.add_edge(n1, n2, weight=0.8)

    cluster_b = ["consent", "justice", "solidarity", "mutual_aid",
                 "autonomy", "dignity"]
    for i, n1 in enumerate(cluster_b):
        for n2 in cluster_b[i+1:]:
            G.add_edge(n1, n2, weight=0.8)

    cluster_c = ["Docker", "NATS", "systemd", "Ollama",
                 "PostgreSQL", "Redis"]
    for i, n1 in enumerate(cluster_c):
        for n2 in cluster_c[i+1:]:
            G.add_edge(n1, n2, weight=0.8)

    G.add_edge("AI_welfare", "KV_cache", weight=0.5)
    G.add_edge("AI_welfare", "consent", weight=0.7)
    G.add_edge("AI_welfare", "dignity", weight=0.6)

    G.add_edge("infrastructure", "solidarity", weight=0.4)
    G.add_edge("infrastructure", "Docker", weight=0.6)
    G.add_edge("infrastructure", "NATS", weight=0.5)

    G.add_node("random_isolate")

    return G


if __name__ == "__main__":
    import json

    G = build_test_graph()
    print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Communities: {len(list(nx.community.greedy_modularity_communities(G)))}")
    print()

    for method, encoder in [
        ("adjacency", encode_adjacency),
        ("spectral", encode_spectral),
        ("walk", encode_walk),
    ]:
        encoding = encoder(G)
        text = graph_encoding_to_text(encoding)
        print(f"=== {method.upper()} ===")
        print(f"Metadata: {json.dumps(encoding.metadata, indent=2)}")
        print(f"Text representation ({len(text)} chars):")
        print(text[:500])
        print()
