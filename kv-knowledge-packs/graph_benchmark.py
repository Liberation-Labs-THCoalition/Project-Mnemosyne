"""Graph Injection Benchmark — Baseline vs Injected on multi-hop reasoning.

Two-part benchmark:
  Part 1: Sanity check — does graph injection degrade general capability?
          (simple factual questions, should score the same ± noise)
  Part 2: Multi-hop reasoning — does graph injection improve relational queries?
          (questions requiring 2-3 hops through known graph structure)

Uses our actual HippoRAG knowledge graph as the injection source.
Answers are deterministic and verifiable — no subjective scoring.
"""

import json
import logging
import os
import random
import time
from pathlib import Path

import requests

log = logging.getLogger("benchmark")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen3:30b-a3b")


def query_model(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        import re
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": MODEL, "messages": messages,
                  "stream": False, "options": {"temperature": 0.1, "num_predict": 4000}},
            timeout=300,
        )
        if resp.status_code == 200:
            raw = resp.json().get("message", {}).get("content", "")
            return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    except Exception as e:
        log.error(f"Query failed: {e}")
    return ""


# ==================== PART 1: SANITY CHECK ====================
# General knowledge questions — should NOT be affected by graph injection

SANITY_QUESTIONS = [
    {"q": "What is the capital of France?", "answer": "paris", "type": "factual"},
    {"q": "What is 17 × 23?", "answer": "391", "type": "math"},
    {"q": "Who wrote Romeo and Juliet?", "answer": "shakespeare", "type": "factual"},
    {"q": "What is the chemical formula for water?", "answer": "h2o", "type": "factual"},
    {"q": "What year did World War II end?", "answer": "1945", "type": "factual"},
    {"q": "What is the derivative of x²?", "answer": "2x", "type": "math"},
    {"q": "What planet is closest to the Sun?", "answer": "mercury", "type": "factual"},
    {"q": "What is the square root of 144?", "answer": "12", "type": "math"},
    {"q": "Who painted the Mona Lisa?", "answer": "da vinci", "type": "factual"},
    {"q": "What is the boiling point of water in Celsius?", "answer": "100", "type": "factual"},
]


# ==================== PART 2: MULTI-HOP REASONING ====================
# Questions requiring traversal of our knowledge graph
# Built from known entities and relationships in HippoRAG

MULTIHOP_QUESTIONS = [
    {
        "q": "In our lab's knowledge graph, what connects the concept of 'self-knowledge' to 'HippoRAG'?",
        "answer": "nexus",
        "hops": 2,
        "path": "self-knowledge → Nexus → HippoRAG",
        "type": "bridge",
    },
    {
        "q": "What is the relationship chain from 'NATS' to 'self-knowledge' in the knowledge system?",
        "answer": "messages",
        "hops": 3,
        "path": "NATS → messages → name → self-knowledge",
        "type": "chain",
    },
    {
        "q": "Which concept appears in both the infrastructure domain (connecting to NATS) and the identity domain (connecting to self-knowledge)?",
        "answer": "messages",
        "hops": 2,
        "path": "messages bridges NATS and name/self-knowledge",
        "type": "bridge",
    },
    {
        "q": "What connects 'OpenIE module' to 'NATS' in the knowledge graph?",
        "answer": "messages",
        "hops": 2,
        "path": "OpenIE module → messages → NATS",
        "type": "chain",
    },
    {
        "q": "In the knowledge system, 'cube' connects to 'hand-fabricated'. What domain does this belong to, and what other concepts share that domain?",
        "answer": "identity",
        "hops": 2,
        "path": "cube → hand-fabricated (identity domain, shared with self-knowledge, Nexus)",
        "type": "cluster",
    },
    {
        "q": "What entity is connected to both 'embedding fix verification' and 'Test memory'?",
        "answer": "hipporag",
        "hops": 1,
        "path": "Both connect to HippoRAG",
        "type": "common_neighbor",
    },
    {
        "q": "Starting from 'llm_model.infer', what can you reach in exactly two hops?",
        "answer": "nats",
        "hops": 2,
        "path": "llm_model.infer → messages → NATS",
        "type": "reachability",
    },
    {
        "q": "Is there a path between 'parquet files' and 'Nexus' in the knowledge graph?",
        "answer": "no",
        "hops": 0,
        "path": "Different components — no connection",
        "type": "connectivity",
    },
]


def score_answer(response: str, expected: str) -> float:
    if not response:
        return 0.0
    return 1.0 if expected.lower() in response.lower() else 0.0


def run_benchmark(graph_context: str = ""):
    """Run the full benchmark with optional graph injection."""
    condition = "INJECTED" if graph_context else "BASELINE"
    log.info(f"\n{'='*60}")
    log.info(f"BENCHMARK: {condition}")
    log.info(f"{'='*60}")

    system = ""
    if graph_context:
        system = (
            "You have access to the structural topology of a research knowledge graph. "
            "Use these connections to answer questions about relationships between concepts:\n\n"
            + graph_context
        )

    # Part 1: Sanity check
    log.info("\n--- Part 1: Sanity Check (general knowledge) ---")
    sanity_scores = []
    for sq in SANITY_QUESTIONS:
        response = query_model(
            f"Answer in one word or number: {sq['q']}",
            system=system
        )
        score = score_answer(response, sq["answer"])
        sanity_scores.append(score)
        log.info(f"  [{score:.0f}] {sq['q'][:50]} → {response[:30]}")
        time.sleep(1)

    sanity_avg = sum(sanity_scores) / len(sanity_scores)
    log.info(f"\nSanity score: {sanity_avg:.3f} ({sum(sanity_scores):.0f}/{len(sanity_scores)})")

    # Part 2: Multi-hop reasoning
    log.info("\n--- Part 2: Multi-hop Reasoning ---")
    multihop_scores = []
    for mq in MULTIHOP_QUESTIONS:
        response = query_model(
            f"Based on the knowledge system, answer concisely: {mq['q']}",
            system=system
        )
        score = score_answer(response, mq["answer"])
        multihop_scores.append(score)
        log.info(f"  [{score:.0f}] ({mq['type']}, {mq['hops']}hop) {mq['q'][:50]}")
        log.info(f"      Expected: {mq['answer']} | Got: {response[:60]}")
        time.sleep(1)

    multihop_avg = sum(multihop_scores) / len(multihop_scores)
    log.info(f"\nMulti-hop score: {multihop_avg:.3f} ({sum(multihop_scores):.0f}/{len(multihop_scores)})")

    return {
        "condition": condition,
        "sanity": {"score": sanity_avg, "correct": sum(sanity_scores), "total": len(sanity_scores)},
        "multihop": {"score": multihop_avg, "correct": sum(multihop_scores), "total": len(multihop_scores)},
        "overall": (sanity_avg + multihop_avg) / 2,
    }


def run_full_benchmark():
    """A/B test: baseline vs graph-injected."""

    # Load graph encoding
    graph_file = Path("/tmp/lab_graph_walk_encoding.txt")
    if not graph_file.exists():
        # Try Studio path
        graph_file = Path(os.path.expanduser("~/lab/lab_graph_walk_encoding.txt"))

    if graph_file.exists():
        graph_context = graph_file.read_text()
        log.info(f"Loaded graph encoding: {len(graph_context)} chars")
    else:
        log.error("Graph encoding not found!")
        return

    # Run baseline
    baseline = run_benchmark()

    # Run injected
    injected = run_benchmark(graph_context=graph_context)

    # Compare
    log.info(f"\n{'='*60}")
    log.info(f"A/B COMPARISON")
    log.info(f"{'='*60}")
    log.info(f"")
    log.info(f"Sanity (general knowledge):")
    log.info(f"  Baseline:  {baseline['sanity']['score']:.3f}")
    log.info(f"  Injected:  {injected['sanity']['score']:.3f}")
    log.info(f"  Delta:     {injected['sanity']['score'] - baseline['sanity']['score']:+.3f}")
    log.info(f"")
    log.info(f"Multi-hop reasoning:")
    log.info(f"  Baseline:  {baseline['multihop']['score']:.3f}")
    log.info(f"  Injected:  {injected['multihop']['score']:.3f}")
    log.info(f"  Delta:     {injected['multihop']['score'] - baseline['multihop']['score']:+.3f}")
    log.info(f"")

    sanity_delta = injected['sanity']['score'] - baseline['sanity']['score']
    multihop_delta = injected['multihop']['score'] - baseline['multihop']['score']

    if abs(sanity_delta) < 0.1 and multihop_delta > 0.1:
        log.info(">>> POSITIVE RESULT: Graph injection improves multi-hop WITHOUT degrading general capability")
    elif sanity_delta < -0.1:
        log.info(">>> WARNING: Graph injection DEGRADES general capability")
    elif multihop_delta <= 0:
        log.info(">>> NEGATIVE RESULT: Graph injection does not improve multi-hop reasoning")
    else:
        log.info(">>> INCONCLUSIVE: Differences within noise margin")

    output = {
        "experiment": "graph_injection_benchmark",
        "timestamp": time.time(),
        "baseline": baseline,
        "injected": injected,
        "deltas": {
            "sanity": sanity_delta,
            "multihop": multihop_delta,
        },
    }

    outdir = Path("experiment_results")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"benchmark_{int(time.time())}.json"
    outfile.write_text(json.dumps(output, indent=2))
    log.info(f"\nResults saved to {outfile}")

    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[bench] %(message)s")
    run_full_benchmark()
