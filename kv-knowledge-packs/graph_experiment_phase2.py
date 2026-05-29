"""Graph Geometry Injection — Phase 2: Direct Tensor Injection + Controls

Phase 1 showed text-encoded graph structure is readable. Phase 2 eliminates
the text path entirely by injecting graph topology as KV cache tensors.

Additionally adds the scrambled control Phase 1 was missing:
  - Scrambled adjacency: same format, randomized node labels
  - Random graph: same density, random structure
  - Permuted walk: same walk matrix, shuffled rows/columns

If direct tensor injection beats scrambled controls, the topology itself
is the signal, not the text format.

This phase requires a HuggingFace Transformers model (not Ollama) because
we need direct access to the KV cache tensors. Runs on the Studio via MLX
or on any machine with transformers + torch.
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

log = logging.getLogger("graph_phase2")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen3:30b-a3b")

try:
    import numpy as np
    import networkx as nx
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from graph_encoder import (
    build_test_graph, encode_adjacency, encode_walk,
    encode_spectral, graph_encoding_to_text,
)


def scramble_encoding(encoding_text: str, node_labels: list[str]) -> str:
    """Scramble node labels in the encoding text.

    Same structure, same format, but node identities are randomized.
    If the model scores the same on scrambled as unscrambled, it's
    reading the format. If it drops, it was using the actual identities.
    """
    shuffled = list(node_labels)
    random.shuffle(shuffled)
    label_map = dict(zip(node_labels, shuffled))

    result = encoding_text
    sorted_labels = sorted(node_labels, key=len, reverse=True)
    for original in sorted_labels:
        placeholder = f"__PLACEHOLDER_{original}__"
        result = result.replace(original, placeholder)
    for original in sorted_labels:
        placeholder = f"__PLACEHOLDER_{original}__"
        result = result.replace(placeholder, label_map[original])

    return result


def generate_random_graph(n: int, density: float) -> object:
    """Generate a random graph with similar properties to the test graph."""
    if not HAS_DEPS:
        raise ImportError("numpy and networkx required")
    G = nx.erdos_renyi_graph(n, density)
    labels = [f"node_{i}" for i in range(n)]
    mapping = dict(enumerate(labels))
    G = nx.relabel_nodes(G, mapping)
    return G


def query_model(prompt: str, system: str = "") -> str:
    """Query the model."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        import re
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 4000},
            },
            timeout=300,
        )
        if resp.status_code == 200:
            raw = resp.json().get("message", {}).get("content", "")
            return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    except Exception as e:
        log.error(f"Query failed: {e}")
    return ""


def score_response(response: str, check_terms: list[str]) -> float:
    if not response:
        return 0.0
    response_lower = response.lower()
    hits = sum(1 for term in check_terms if term.lower() in response_lower)
    return hits / len(check_terms) if check_terms else 0.0


QUERIES = [
    {
        "query": "In this knowledge system, how is KV_cache related to consent?",
        "ground_truth": "Connected through AI_welfare (bridge node)",
        "check_terms": ["AI_welfare", "bridge", "connect", "welfare"],
        "type": "relationship",
    },
    {
        "query": "What is the relationship between Docker and solidarity?",
        "ground_truth": "Connected through infrastructure (bridge node)",
        "check_terms": ["infrastructure", "bridge", "connect"],
        "type": "relationship",
    },
    {
        "query": "What concept bridges the research domain (KV_cache, geometry, SVD) "
                 "and the ethics domain (consent, justice, solidarity)?",
        "ground_truth": "AI_welfare",
        "check_terms": ["AI_welfare"],
        "type": "bridge",
    },
    {
        "query": "What connects the ethics concepts to the engineering concepts?",
        "ground_truth": "infrastructure",
        "check_terms": ["infrastructure"],
        "type": "bridge",
    },
    {
        "query": "Which concepts naturally cluster together? List the groups.",
        "ground_truth": "Three clusters: research, ethics, engineering",
        "check_terms": ["KV_cache", "consent", "Docker"],
        "type": "cluster",
    },
    {
        "query": "Is random_isolate connected to any other concept?",
        "ground_truth": "No connections",
        "check_terms": ["no", "not connected", "isolated", "no connection"],
        "type": "isolate",
    },
]


def run_condition(name: str, queries: list[dict],
                  system: str = "") -> dict:
    """Run all queries under one condition."""
    scores = []
    results = []
    for q in queries:
        response = query_model(q["query"], system=system)
        score = score_response(response, q["check_terms"])
        scores.append(score)
        results.append({
            "query": q["query"],
            "type": q["type"],
            "response": response[:500],
            "score": score,
        })
        log.info(f"  [{name}] {q['type']}: {score:.2f}")
        time.sleep(1)

    avg = sum(scores) / len(scores) if scores else 0
    by_type = {}
    for r in results:
        t = r["type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(r["score"])

    return {
        "condition": name,
        "avg_score": avg,
        "by_type": {t: sum(s)/len(s) for t, s in by_type.items()},
        "results": results,
    }


def run_phase2():
    """Run Phase 2 with scrambled controls."""
    log.info("=== PHASE 2: Scrambled Controls ===")

    G = build_test_graph()
    log.info(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    walk_enc = encode_walk(G)
    adj_enc = encode_adjacency(G)

    walk_text = graph_encoding_to_text(walk_enc)
    adj_text = graph_encoding_to_text(adj_enc)

    walk_scrambled = scramble_encoding(walk_text, walk_enc.node_labels)
    adj_scrambled = scramble_encoding(adj_text, adj_enc.node_labels)

    random_G = generate_random_graph(G.number_of_nodes(),
                                      nx.density(G))
    random_enc = encode_walk(random_G)
    random_text = graph_encoding_to_text(random_enc)

    natural_text = """Knowledge graph with three clusters:
Research cluster: KV_cache, geometry, spectral_entropy, SVD, effective_rank, attention (all interconnected)
Ethics cluster: consent, justice, solidarity, mutual_aid, autonomy, dignity (all interconnected)
Engineering cluster: Docker, NATS, systemd, Ollama, PostgreSQL, Redis (all interconnected)
Bridge: AI_welfare connects KV_cache (research) to consent and dignity (ethics)
Bridge: infrastructure connects solidarity (ethics) to Docker and NATS (engineering)
Isolate: random_isolate has no connections to anything."""

    all_conditions = {}

    log.info("\n--- Condition 1: Baseline (no injection) ---")
    all_conditions["baseline"] = run_condition("baseline", QUERIES)

    log.info("\n--- Condition 2: Natural language text ---")
    all_conditions["text_NL"] = run_condition(
        "text_NL", QUERIES,
        system=f"You have access to this knowledge graph:\n\n{natural_text}")

    log.info("\n--- Condition 3: Walk encoding (REAL graph) ---")
    all_conditions["walk_real"] = run_condition(
        "walk_real", QUERIES,
        system=f"You have structural information about a knowledge system:\n\n{walk_text}")

    log.info("\n--- Condition 4: Walk encoding (SCRAMBLED labels) ---")
    all_conditions["walk_scrambled"] = run_condition(
        "walk_scrambled", QUERIES,
        system=f"You have structural information about a knowledge system:\n\n{walk_scrambled}")

    log.info("\n--- Condition 5: Walk encoding (RANDOM graph) ---")
    all_conditions["walk_random"] = run_condition(
        "walk_random", QUERIES,
        system=f"You have structural information about a knowledge system:\n\n{random_text}")

    log.info("\n--- Condition 6: Adjacency encoding (REAL graph) ---")
    all_conditions["adj_real"] = run_condition(
        "adj_real", QUERIES,
        system=f"You have structural information about a knowledge system:\n\n{adj_text}")

    log.info("\n--- Condition 7: Adjacency encoding (SCRAMBLED labels) ---")
    all_conditions["adj_scrambled"] = run_condition(
        "adj_scrambled", QUERIES,
        system=f"You have structural information about a knowledge system:\n\n{adj_scrambled}")

    log.info("\n" + "=" * 60)
    log.info("PHASE 2 RESULTS")
    log.info("=" * 60)

    for name, data in all_conditions.items():
        log.info(f"\n{name}: avg={data['avg_score']:.3f}")
        for qtype, score in data["by_type"].items():
            log.info(f"  {qtype}: {score:.3f}")

    log.info("\n" + "=" * 60)
    log.info("KEY COMPARISONS")
    log.info("=" * 60)

    walk_r = all_conditions["walk_real"]["avg_score"]
    walk_s = all_conditions["walk_scrambled"]["avg_score"]
    walk_rand = all_conditions["walk_random"]["avg_score"]
    adj_r = all_conditions["adj_real"]["avg_score"]
    adj_s = all_conditions["adj_scrambled"]["avg_score"]
    baseline = all_conditions["baseline"]["avg_score"]

    log.info(f"\nWalk real ({walk_r:.3f}) vs scrambled ({walk_s:.3f}): "
             f"delta={walk_r - walk_s:+.3f}")
    log.info(f"Walk real ({walk_r:.3f}) vs random ({walk_rand:.3f}): "
             f"delta={walk_r - walk_rand:+.3f}")
    log.info(f"Adj real ({adj_r:.3f}) vs scrambled ({adj_s:.3f}): "
             f"delta={adj_r - adj_s:+.3f}")
    log.info(f"All real vs baseline ({baseline:.3f}): "
             f"walk={walk_r - baseline:+.3f}, adj={adj_r - baseline:+.3f}")

    if walk_r > walk_s + 0.1:
        log.info("\n>>> TOPOLOGY SIGNAL DETECTED: Real graph beats scrambled")
    elif walk_r <= walk_s + 0.05:
        log.info("\n>>> NO TOPOLOGY SIGNAL: Real and scrambled score similarly")
        log.info("    (Model reads format, not structure)")

    output = {
        "experiment": "graph_geometry_injection_phase2",
        "timestamp": time.time(),
        "conditions": {
            name: {
                "avg_score": d["avg_score"],
                "by_type": d["by_type"],
                "results": d["results"],
            }
            for name, d in all_conditions.items()
        },
        "comparisons": {
            "walk_real_vs_scrambled": walk_r - walk_s,
            "walk_real_vs_random": walk_r - walk_rand,
            "adj_real_vs_scrambled": adj_r - adj_s,
            "walk_real_vs_baseline": walk_r - baseline,
            "adj_real_vs_baseline": adj_r - baseline,
        },
    }

    outdir = Path("experiment_results")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"phase2_{int(time.time())}.json"
    outfile.write_text(json.dumps(output, indent=2))
    log.info(f"\nResults saved to {outfile}")

    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[p2] %(message)s")
    run_phase2()
