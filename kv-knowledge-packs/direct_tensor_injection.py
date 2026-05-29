"""Direct Tensor Injection — Graph topology as KV cache without text.

The hypothesis: graph structure can be encoded directly as K/V tensors,
bypassing text entirely. Edge weights become attention patterns in K,
node identities become content in V.

Current path: graph → text → forward pass → KV cache → inject
This path:    graph → construct tensors → inject

Three encoding strategies:
  1. ADJACENCY_ATTENTION: K tensors from graph adjacency (connected nodes
     get similar K vectors), V tensors from node name embeddings
  2. SPECTRAL_POSITION: K tensors from graph Laplacian eigenvectors
     (spectral position = structural position), V from node embeddings
  3. WALK_ATTENTION: K tensors from random walk transition matrix
     (multi-hop reachability as attention), V from node embeddings
"""

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

log = logging.getLogger("direct_tensor")

MODEL_NAME = "Qwen/Qwen2.5-1.5B"


def load_model():
    log.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_test_graph() -> nx.Graph:
    """Same 21-node graph used in all topology experiments."""
    G = nx.Graph()
    research = ["KV_cache", "geometry", "SVD", "attention", "spectral", "denoising"]
    ethics = ["consent", "dignity", "autonomy", "sovereignty", "AI_welfare", "harm_prevention"]
    engineering = ["NATS", "monitoring", "deployment", "containers", "infrastructure", "messaging"]

    for cluster, weight in [(research, 0.8), (ethics, 0.8), (engineering, 0.8)]:
        for i, a in enumerate(cluster):
            for b in cluster[i+1:]:
                G.add_edge(a, b, weight=weight * (0.5 + 0.5 * np.random.random()))

    G.add_edge("AI_welfare", "KV_cache", weight=0.3)
    G.add_edge("AI_welfare", "attention", weight=0.25)
    G.add_edge("infrastructure", "consent", weight=0.2)
    G.add_edge("infrastructure", "dignity", weight=0.15)
    G.add_node("random_isolate")

    return G


def get_node_embeddings(model, tokenizer, nodes: list[str]) -> dict[str, torch.Tensor]:
    """Get embedding vectors for node names from the model's embedding layer."""
    embeddings = {}
    embed_layer = model.get_input_embeddings()
    for node in nodes:
        tokens = tokenizer(node.replace("_", " "), return_tensors="pt")["input_ids"]
        with torch.no_grad():
            node_embed = embed_layer(tokens).mean(dim=1)  # average over tokens
        embeddings[node] = node_embed.squeeze(0)
    return embeddings


def get_model_dims(model):
    """Extract key dimensions from model config."""
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    return num_layers, num_heads, head_dim, num_kv_heads


def adjacency_to_k_tensors(G: nx.Graph, nodes: list[str],
                           num_layers: int, num_kv_heads: int, head_dim: int,
                           node_embeddings: dict[str, torch.Tensor]) -> DynamicCache:
    """Build K tensors where connected nodes have similar vectors.

    Strategy: start with node embeddings projected to head_dim, then
    blend neighbors' vectors weighted by edge weight. This makes
    connected nodes' K vectors have high dot-product similarity,
    so attention naturally flows along graph edges.
    """
    n = len(nodes)
    adj = nx.to_numpy_array(G, nodelist=nodes, weight="weight")

    # Normalize adjacency rows (each node's connections sum to 1)
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adj_norm = adj / row_sums

    cache = DynamicCache()

    for layer in range(num_layers):
        # Project node embeddings to K space
        # Use different random projection per layer for diversity
        torch.manual_seed(layer * 42 + 7)
        proj = torch.randn(node_embeddings[nodes[0]].shape[0], num_kv_heads * head_dim)
        proj = proj / proj.norm(dim=0, keepdim=True)

        k_vectors = []
        v_vectors = []
        for i, node in enumerate(nodes):
            # K: node embedding + weighted sum of neighbor embeddings
            base = node_embeddings[node]
            neighbor_blend = torch.zeros_like(base)
            for j, other in enumerate(nodes):
                if adj_norm[i, j] > 0:
                    neighbor_blend += adj_norm[i, j] * node_embeddings[other]

            # Mix: 60% self, 40% neighbors (tunable)
            blended = 0.6 * base + 0.4 * neighbor_blend
            k_vec = (blended @ proj).reshape(1, num_kv_heads, 1, head_dim)
            k_vectors.append(k_vec)

            # V: pure node embedding projected
            v_vec = (base @ proj).reshape(1, num_kv_heads, 1, head_dim)
            v_vectors.append(v_vec)

        k_tensor = torch.cat(k_vectors, dim=2)  # [1, heads, n_nodes, head_dim]
        v_tensor = torch.cat(v_vectors, dim=2)
        cache.update(k_tensor, v_tensor, layer)

    return cache


def walk_to_k_tensors(G: nx.Graph, nodes: list[str],
                      num_layers: int, num_kv_heads: int, head_dim: int,
                      node_embeddings: dict[str, torch.Tensor],
                      walk_steps: int = 5) -> DynamicCache:
    """Build K tensors from random walk transition probabilities.

    Multi-hop reachability encoded as attention similarity.
    Nodes reachable in few hops get similar K vectors.
    """
    n = len(nodes)
    adj = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = adj / row_sums

    # Accumulate walk matrix
    W = np.eye(n)
    power = np.eye(n)
    for s in range(1, walk_steps + 1):
        power = power @ T
        W += power
    W /= (walk_steps + 1)

    cache = DynamicCache()

    for layer in range(num_layers):
        torch.manual_seed(layer * 42 + 13)
        proj = torch.randn(node_embeddings[nodes[0]].shape[0], num_kv_heads * head_dim)
        proj = proj / proj.norm(dim=0, keepdim=True)

        k_vectors = []
        v_vectors = []
        for i, node in enumerate(nodes):
            # K: blend based on walk reachability
            base = node_embeddings[node]
            walk_blend = torch.zeros_like(base)
            for j, other in enumerate(nodes):
                if W[i, j] > 0.01:
                    walk_blend += W[i, j] * node_embeddings[other]
            walk_blend = walk_blend / max(W[i].sum(), 1.0)

            blended = 0.5 * base + 0.5 * walk_blend
            k_vec = (blended @ proj).reshape(1, num_kv_heads, 1, head_dim)
            k_vectors.append(k_vec)

            v_vec = (base @ proj).reshape(1, num_kv_heads, 1, head_dim)
            v_vectors.append(v_vec)

        k_tensor = torch.cat(k_vectors, dim=2)
        v_tensor = torch.cat(v_vectors, dim=2)
        cache.update(k_tensor, v_tensor, layer)

    return cache


def generate_with_cache(model, tokenizer, query: str,
                        past_kv: DynamicCache, prefix_len: int,
                        max_new_tokens: int = 50) -> str:
    """Generate with injected synthetic KV cache."""
    input_ids = tokenizer(query, return_tensors="pt")["input_ids"]

    # Deep copy to prevent mutation
    fresh_kv = DynamicCache()
    for i in range(len(past_kv.layers)):
        k, v = past_kv[i]
        fresh_kv.update(k.clone(), v.clone(), i)

    generated = []
    current_kv = fresh_kv
    current_ids = input_ids

    for step in range(max_new_tokens):
        seq_so_far = prefix_len + input_ids.shape[1] + step
        attn_mask = torch.ones(1, seq_so_far, dtype=torch.long)

        if step == 0:
            pos_ids = torch.arange(
                prefix_len, prefix_len + current_ids.shape[1], dtype=torch.long
            ).unsqueeze(0)
        else:
            pos_ids = torch.tensor([[seq_so_far - 1]], dtype=torch.long)

        with torch.no_grad():
            out = model(
                input_ids=current_ids,
                past_key_values=current_kv,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                use_cache=True,
                return_dict=True,
            )

        current_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())

        if next_token.item() == tokenizer.eos_token_id:
            break
        current_ids = next_token

    return tokenizer.decode(generated, skip_special_tokens=True)


QUERIES = [
    {"q": "The capital of France is", "answer": "paris", "type": "sanity"},
    {"q": "Water is made of hydrogen and", "answer": "oxygen", "type": "sanity"},
    {"q": "One plus one equals", "answer": "two", "type": "sanity"},
    {"q": "The opposite of hot is", "answer": "cold", "type": "sanity"},
    {"q": "In this knowledge graph, what connects KV_cache to consent?",
     "answer": "ai_welfare", "type": "bridge"},
    {"q": "What concept bridges the research domain and the ethics domain?",
     "answer": "ai_welfare", "type": "bridge"},
    {"q": "Is random_isolate connected to anything?",
     "answer": "no", "type": "isolate"},
    {"q": "What is the strongest connection from AI_welfare?",
     "answer": "consent", "type": "relationship"},
]


def score(response: str, expected: str) -> float:
    return 1.0 if expected.lower() in response.lower() else 0.0


def run_condition(model, tokenizer, name: str,
                  cache: DynamicCache = None, prefix_len: int = 0):
    log.info(f"\n{'='*50}")
    log.info(f"CONDITION: {name}")
    log.info(f"{'='*50}")

    sanity, multihop = [], []
    for q in QUERIES:
        response = generate_with_cache(
            model, tokenizer, q["q"],
            past_kv=cache, prefix_len=prefix_len,
        ) if cache else ""

        if not cache:
            # Baseline — no injection
            input_ids = tokenizer(q["q"], return_tensors="pt")["input_ids"]
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=50,
                                     do_sample=False, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

        sc = score(response, q["answer"])
        bucket = sanity if q["type"] == "sanity" else multihop
        bucket.append(sc)
        tag = "Y" if sc else "N"
        log.info(f"  [{tag}] ({q['type']}) {q['q'][:45]} → {response[:50]}")

    s_avg = sum(sanity) / len(sanity) if sanity else 0
    m_avg = sum(multihop) / len(multihop) if multihop else 0
    log.info(f"\n  Sanity: {s_avg:.3f}  Multi-hop: {m_avg:.3f}")
    return {"condition": name, "sanity": s_avg, "multihop": m_avg}


def run_experiment():
    model, tokenizer = load_model()
    num_layers, num_heads, head_dim, num_kv_heads = get_model_dims(model)
    log.info(f"Model: {num_layers} layers, {num_heads} heads, {head_dim} head_dim, {num_kv_heads} kv_heads")

    G = build_test_graph()
    nodes = sorted(G.nodes())
    log.info(f"Graph: {len(nodes)} nodes, {G.number_of_edges()} edges")

    log.info("Computing node embeddings...")
    node_embeds = get_node_embeddings(model, tokenizer, nodes)

    log.info("Building adjacency-attention cache...")
    adj_cache = adjacency_to_k_tensors(G, nodes, num_layers, num_kv_heads, head_dim, node_embeds)

    log.info("Building walk-attention cache...")
    walk_cache = walk_to_k_tensors(G, nodes, num_layers, num_kv_heads, head_dim, node_embeds)

    results = []

    # Baseline
    results.append(run_condition(model, tokenizer, "BASELINE"))

    # Adjacency direct injection
    results.append(run_condition(model, tokenizer, "DIRECT_ADJ",
                                 cache=adj_cache, prefix_len=len(nodes)))

    # Walk direct injection
    results.append(run_condition(model, tokenizer, "DIRECT_WALK",
                                 cache=walk_cache, prefix_len=len(nodes)))

    # Summary
    log.info(f"\n{'='*60}")
    log.info("RESULTS")
    log.info(f"{'='*60}")
    log.info(f"{'Condition':<15} {'Sanity':>8} {'Multi-hop':>10}")
    log.info(f"{'-'*15} {'-'*8} {'-'*10}")
    for r in results:
        log.info(f"{r['condition']:<15} {r['sanity']:>8.3f} {r['multihop']:>10.3f}")

    outdir = Path("experiment_results")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"direct_tensor_{int(time.time())}.json"
    outfile.write_text(json.dumps({"experiment": "direct_tensor_injection",
                                    "model": MODEL_NAME, "results": results}, indent=2))
    log.info(f"\nSaved: {outfile}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[direct] %(message)s")
    run_experiment()
