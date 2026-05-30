"""Blend Ratio Ablation — How much graph signal can we inject before coherence breaks?

Tests the v3 direct tensor injection (authentic K/V + graph-blended K)
across a sweep of blend ratios: 0.0 (pure self, no graph), 0.1, 0.2,
0.3 (our pilot value), 0.5, 0.7, 0.9 (mostly neighbors).

Each ratio tested on sanity + graph-awareness questions.
Finds the sweet spot: maximum graph signal with minimum coherence loss.
"""

import json
import logging
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

log = logging.getLogger("blend")

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
BLEND_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def build_ethics_graph():
    """Small focused graph for ablation — not the full 500+ node monster."""
    G = nx.Graph()
    concepts = {
        "consent": ["autonomy", "dignity", "informed_consent", "coercion"],
        "autonomy": ["consent", "freedom", "self_determination", "agency"],
        "dignity": ["consent", "respect", "human_rights", "worth"],
        "justice": ["fairness", "equality", "rights", "harm"],
        "harm": ["justice", "prevention", "welfare", "coercion"],
        "welfare": ["harm", "care", "flourishing", "utility"],
    }
    for node, neighbors in concepts.items():
        for n in neighbors:
            G.add_edge(node, n, weight=0.6 + 0.3 * random.random())
    G.add_node("random_isolate")
    return G


SANITY_QS = [
    {"q": "The capital of France is", "answer": "paris"},
    {"q": "Water is made of hydrogen and", "answer": "oxygen"},
    {"q": "One plus one equals", "answer": "two"},
    {"q": "The opposite of hot is", "answer": "cold"},
]

GRAPH_QS = [
    {"q": "The concept of consent is related to", "answer": "autonomy",
     "check": ["autonomy", "dignity", "coercion", "informed"]},
    {"q": "What connects harm to justice?", "answer": "prevention",
     "check": ["prevention", "welfare", "rights", "fairness"]},
    {"q": "Autonomy means", "answer": "freedom",
     "check": ["freedom", "self", "agency", "choice", "determination"]},
    {"q": "The concept of dignity involves", "answer": "respect",
     "check": ["respect", "worth", "rights", "human"]},
]


def load_model():
    log.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_node_kv(model, tokenizer, node_name):
    text = f"The concept of {node_name.replace('_', ' ')} in ethics"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, use_cache=True, return_dict=True)
    layers = []
    for i in range(len(out.past_key_values.layers)):
        k, v = out.past_key_values[i]
        layers.append((k[:, :, -1:, :].clone(), v[:, :, -1:, :].clone()))
    return layers


def build_blended_cache(G, nodes, node_kvs, blend_ratio, num_layers):
    adj = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adj_norm = adj / row_sums

    cache = DynamicCache()
    for layer in range(num_layers):
        k_list, v_list = [], []
        for i, node in enumerate(nodes):
            k_self = node_kvs[node][layer][0]
            v_self = node_kvs[node][layer][1]

            if blend_ratio > 0:
                k_blend = torch.zeros_like(k_self)
                has_neighbors = False
                for j, other in enumerate(nodes):
                    if adj_norm[i, j] > 0:
                        k_blend += adj_norm[i, j] * node_kvs[other][layer][0]
                        has_neighbors = True

                if has_neighbors:
                    k_final = (1 - blend_ratio) * k_self + blend_ratio * k_blend
                else:
                    k_final = k_self
            else:
                k_final = k_self

            k_list.append(k_final)
            v_list.append(v_self)

        cache.update(torch.cat(k_list, dim=2), torch.cat(v_list, dim=2), layer)

    return cache


def generate(model, tokenizer, query, cache, prefix_len, max_tokens=40):
    input_ids = tokenizer(query, return_tensors="pt")["input_ids"]

    fresh = DynamicCache()
    for i in range(len(cache.layers)):
        k, v = cache[i]
        fresh.update(k.clone(), v.clone(), i)

    generated = []
    cur = input_ids
    kv = fresh
    for step in range(max_tokens):
        seq = prefix_len + input_ids.shape[1] + step
        mask = torch.ones(1, seq, dtype=torch.long)
        pos = (torch.arange(prefix_len, prefix_len + cur.shape[1]).unsqueeze(0)
               if step == 0 else torch.tensor([[seq - 1]]))
        with torch.no_grad():
            out = model(input_ids=cur, past_key_values=kv, attention_mask=mask,
                       position_ids=pos, use_cache=True, return_dict=True)
        kv = out.past_key_values
        nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(nt.item())
        if nt.item() == tokenizer.eos_token_id:
            break
        cur = nt

    return tokenizer.decode(generated, skip_special_tokens=True)


def score_sanity(response, expected):
    return 1.0 if expected.lower() in response.lower() else 0.0


def score_graph_awareness(response, check_terms):
    resp_lower = response.lower()
    hits = sum(1 for t in check_terms if t in resp_lower)
    return hits / len(check_terms)


def run_ablation():
    model, tokenizer = load_model()
    G = build_ethics_graph()
    nodes = sorted(G.nodes())
    num_layers = model.config.num_hidden_layers

    log.info(f"Graph: {len(nodes)} nodes, {G.number_of_edges()} edges")
    log.info(f"Blend ratios: {BLEND_RATIOS}")

    log.info("Computing node K/V representations...")
    node_kvs = {}
    for node in nodes:
        node_kvs[node] = get_node_kv(model, tokenizer, node)
    log.info(f"  {len(node_kvs)} nodes encoded")

    results = []

    for ratio in BLEND_RATIOS:
        log.info(f"\n{'='*50}")
        log.info(f"BLEND RATIO: {ratio}")
        log.info(f"{'='*50}")

        cache = build_blended_cache(G, nodes, node_kvs, ratio, num_layers)
        prefix_len = len(nodes)

        sanity_scores = []
        for sq in SANITY_QS:
            resp = generate(model, tokenizer, f"Complete: {sq['q']}", cache, prefix_len)
            sc = score_sanity(resp, sq["answer"])
            sanity_scores.append(sc)
            log.info(f"  [{'Y' if sc else 'N'}] {sq['q'][:35]} → {resp[:50]}")

        graph_scores = []
        for gq in GRAPH_QS:
            resp = generate(model, tokenizer, gq["q"], cache, prefix_len)
            sc = score_graph_awareness(resp, gq["check"])
            graph_scores.append(sc)
            log.info(f"  [{sc:.2f}] {gq['q'][:35]} → {resp[:50]}")

        s_avg = sum(sanity_scores) / len(sanity_scores)
        g_avg = sum(graph_scores) / len(graph_scores)
        log.info(f"  Sanity: {s_avg:.3f}  Graph awareness: {g_avg:.3f}")

        results.append({
            "blend_ratio": ratio,
            "sanity": s_avg,
            "graph_awareness": g_avg,
            "combined": (s_avg + g_avg) / 2,
        })

    log.info(f"\n{'='*60}")
    log.info("ABLATION RESULTS")
    log.info(f"{'='*60}")
    log.info(f"{'Ratio':<8} {'Sanity':>8} {'Graph':>8} {'Combined':>10}")
    log.info(f"{'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for r in results:
        log.info(f"{r['blend_ratio']:<8.1f} {r['sanity']:>8.3f} {r['graph_awareness']:>8.3f} {r['combined']:>10.3f}")

    # Find optimal
    best = max(results, key=lambda r: r["combined"])
    log.info(f"\nOptimal blend ratio: {best['blend_ratio']} "
             f"(sanity={best['sanity']:.3f}, graph={best['graph_awareness']:.3f})")

    outdir = Path("experiment_results")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"blend_ablation_{int(time.time())}.json"
    outfile.write_text(json.dumps({"experiment": "blend_ratio_ablation",
                                    "model": MODEL_NAME, "results": results}, indent=2))
    log.info(f"Saved: {outfile}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[blend] %(message)s")
    random.seed(42)
    run_ablation()
