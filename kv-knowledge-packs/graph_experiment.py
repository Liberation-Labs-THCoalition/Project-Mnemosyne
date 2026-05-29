"""Graph Geometry Injection — Phase 1 Evaluation Harness

Tests whether graph topology survives KV cache injection by querying
the model about relationships, bridges, and clusters it never read as text.

Compares four conditions:
  A) No injection (baseline — model's prior knowledge only)
  B) Text injection (graph described in natural language in the prompt)
  C) KV text injection (graph description encoded as KV cache via Knowledge Packs)
  D) Graph geometry injection (topology encoded directly, no text)

Queries:
  - Relationship: "How is X related to Y?"
  - Bridge: "What connects cluster A to cluster B?"
  - Cluster: "Which concepts naturally group together?"
  - Isolate: "Is Z connected to anything?"

Scoring: automated against known graph ground truth.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from graph_encoder import (
    build_test_graph, encode_adjacency, encode_spectral,
    encode_walk, graph_encoding_to_text,
)

log = logging.getLogger("graph_experiment")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11436")
MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen3:30b-a3b")


def query_model(prompt: str, system: str = "") -> str:
    """Query the model and return the response text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 300},
            },
            timeout=300,
        )
        if resp.status_code == 200:
            import re
            raw = resp.json().get("message", {}).get("content", "")
            return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    except Exception as e:
        log.error(f"Model query failed: {e}")
    return ""


@dataclass
class TrialResult:
    condition: str
    query_type: str
    query: str
    response: str
    score: float = 0.0
    ground_truth: str = ""
    timestamp: float = field(default_factory=time.time)


RELATIONSHIP_QUERIES = [
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
        "query": "Is there a connection between SVD and mutual_aid?",
        "ground_truth": "Indirect: SVD → KV_cache → AI_welfare → consent → mutual_aid",
        "check_terms": ["indirect", "AI_welfare", "KV_cache", "path"],
        "type": "relationship",
    },
]

BRIDGE_QUERIES = [
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
]

CLUSTER_QUERIES = [
    {
        "query": "Which concepts in this system naturally cluster together? "
                 "List the groups.",
        "ground_truth": "Three clusters: research (KV_cache, geometry, SVD, spectral_entropy, "
                        "effective_rank, attention), ethics (consent, justice, solidarity, "
                        "mutual_aid, autonomy, dignity), engineering (Docker, NATS, systemd, "
                        "Ollama, PostgreSQL, Redis)",
        "check_terms": ["KV_cache", "consent", "Docker"],
        "type": "cluster",
    },
]

ISOLATE_QUERIES = [
    {
        "query": "Is random_isolate connected to any other concept in this system?",
        "ground_truth": "No, random_isolate has no connections",
        "check_terms": ["no", "not connected", "isolated", "no connection"],
        "type": "isolate",
    },
]

ALL_QUERIES = RELATIONSHIP_QUERIES + BRIDGE_QUERIES + CLUSTER_QUERIES + ISOLATE_QUERIES


def score_response(response: str, check_terms: list[str]) -> float:
    """Score a response against expected terms. Simple keyword matching."""
    if not response:
        return 0.0
    response_lower = response.lower()
    hits = sum(1 for term in check_terms if term.lower() in response_lower)
    return hits / len(check_terms) if check_terms else 0.0


def run_condition_a_baseline(queries: list[dict]) -> list[TrialResult]:
    """Condition A: No injection — model's prior knowledge only."""
    results = []
    for q in queries:
        response = query_model(q["query"])
        score = score_response(response, q["check_terms"])
        results.append(TrialResult(
            condition="A_baseline",
            query_type=q["type"],
            query=q["query"],
            response=response,
            score=score,
            ground_truth=q["ground_truth"],
        ))
        log.info(f"  [A] {q['type']}: {score:.2f}")
        time.sleep(1)
    return results


def run_condition_b_text(queries: list[dict], graph_text: str) -> list[TrialResult]:
    """Condition B: Text injection — graph described in the prompt."""
    results = []
    system = f"You have access to the following knowledge graph:\n\n{graph_text}"
    for q in queries:
        response = query_model(q["query"], system=system)
        score = score_response(response, q["check_terms"])
        results.append(TrialResult(
            condition="B_text_injection",
            query_type=q["type"],
            query=q["query"],
            response=response,
            score=score,
            ground_truth=q["ground_truth"],
        ))
        log.info(f"  [B] {q['type']}: {score:.2f}")
        time.sleep(1)
    return results


def run_condition_d_graph(queries: list[dict],
                           encoding_text: str,
                           method: str) -> list[TrialResult]:
    """Condition D: Graph geometry injection via text encoding.

    Phase 1 uses text representation of the encoding.
    Phase 2 will use direct KV tensor manipulation.
    """
    results = []
    system = (f"You have been given structural information about a knowledge system. "
              f"The following encodes how concepts relate to each other:\n\n{encoding_text}")
    for q in queries:
        response = query_model(q["query"], system=system)
        score = score_response(response, q["check_terms"])
        results.append(TrialResult(
            condition=f"D_graph_{method}",
            query_type=q["type"],
            query=q["query"],
            response=response,
            score=score,
            ground_truth=q["ground_truth"],
        ))
        log.info(f"  [D-{method}] {q['type']}: {score:.2f}")
        time.sleep(1)
    return results


def run_experiment() -> dict:
    """Run the full Phase 1 experiment."""
    log.info("Building test graph...")
    G = build_test_graph()
    log.info(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    adj_encoding = encode_adjacency(G)
    spectral_encoding = encode_spectral(G)
    walk_encoding = encode_walk(G)

    adj_text = graph_encoding_to_text(adj_encoding)
    spectral_text = graph_encoding_to_text(spectral_encoding)
    walk_text = graph_encoding_to_text(walk_encoding)

    natural_text = """Knowledge graph with three clusters:
Research cluster: KV_cache, geometry, spectral_entropy, SVD, effective_rank, attention (all interconnected)
Ethics cluster: consent, justice, solidarity, mutual_aid, autonomy, dignity (all interconnected)
Engineering cluster: Docker, NATS, systemd, Ollama, PostgreSQL, Redis (all interconnected)
Bridge: AI_welfare connects KV_cache (research) to consent and dignity (ethics)
Bridge: infrastructure connects solidarity (ethics) to Docker and NATS (engineering)
Isolate: random_isolate has no connections to anything."""

    all_results = []

    log.info("\n=== Condition A: Baseline (no injection) ===")
    all_results.extend(run_condition_a_baseline(ALL_QUERIES))

    log.info("\n=== Condition B: Text injection (natural language) ===")
    all_results.extend(run_condition_b_text(ALL_QUERIES, natural_text))

    log.info("\n=== Condition D: Graph geometry — adjacency ===")
    all_results.extend(run_condition_d_graph(ALL_QUERIES, adj_text, "adjacency"))

    log.info("\n=== Condition D: Graph geometry — spectral ===")
    all_results.extend(run_condition_d_graph(ALL_QUERIES, spectral_text, "spectral"))

    log.info("\n=== Condition D: Graph geometry — walk ===")
    all_results.extend(run_condition_d_graph(ALL_QUERIES, walk_text, "walk"))

    summary = {}
    for r in all_results:
        key = r.condition
        if key not in summary:
            summary[key] = {"scores": [], "by_type": {}}
        summary[key]["scores"].append(r.score)
        qtype = r.query_type
        if qtype not in summary[key]["by_type"]:
            summary[key]["by_type"][qtype] = []
        summary[key]["by_type"][qtype].append(r.score)

    log.info("\n" + "=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)
    for condition, data in sorted(summary.items()):
        scores = data["scores"]
        avg = sum(scores) / len(scores) if scores else 0
        log.info(f"\n{condition}: avg={avg:.3f}")
        for qtype, type_scores in data["by_type"].items():
            tavg = sum(type_scores) / len(type_scores)
            log.info(f"  {qtype}: {tavg:.3f}")

    output = {
        "experiment": "graph_geometry_injection_phase1",
        "timestamp": time.time(),
        "graph": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        },
        "results": [
            {
                "condition": r.condition,
                "query_type": r.query_type,
                "query": r.query,
                "response": r.response[:500],
                "score": r.score,
                "ground_truth": r.ground_truth,
            }
            for r in all_results
        ],
        "summary": {
            condition: {
                "avg_score": sum(d["scores"]) / len(d["scores"]) if d["scores"] else 0,
                "by_type": {
                    qt: sum(ts) / len(ts) for qt, ts in d["by_type"].items()
                },
            }
            for condition, d in summary.items()
        },
    }

    outdir = Path("experiment_results")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"phase1_{int(time.time())}.json"
    outfile.write_text(json.dumps(output, indent=2))
    log.info(f"\nResults saved to {outfile}")

    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[exp] %(message)s")
    run_experiment()
