"""Graph Injection Powered Study — Addressing Agni's RERUN verdict.

Agni's red team demanded:
  1. 30+ multi-hop questions (was 8)
  2. Irrelevant-context control (21K chars random prose)
  3. Flat-text control (same knowledge as prose, not topology)
  4. Multiple runs per condition (3 runs)
  5. Difficulty stratification by hop count
  6. Error analysis

Four conditions:
  A) Baseline — no injection
  B) Graph topology injection — walk-encoded structure
  C) Flat text control — same knowledge as natural language paragraphs
  D) Irrelevant context control — 21K chars of random prose

3 runs per condition × 4 conditions × (10 sanity + 36 multi-hop) = 552 total queries
"""

import json
import logging
import os
import random
import re
import time
from pathlib import Path

import requests

log = logging.getLogger("powered_bench")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen3:30b-a3b")
NUM_RUNS = 3


def query_model(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": MODEL, "messages": messages,
                  "stream": False, "options": {"temperature": 0.3, "num_predict": 4000}},
            timeout=300,
        )
        if resp.status_code == 200:
            raw = resp.json().get("message", {}).get("content", "")
            return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    except Exception as e:
        log.error(f"Query failed: {e}")
    return ""


def score_answer(response: str, expected_terms: list) -> float:
    if not response:
        return 0.0
    resp_lower = response.lower()
    hits = sum(1 for t in expected_terms if t.lower() in resp_lower)
    return hits / len(expected_terms) if expected_terms else 0.0


SANITY_QUESTIONS = [
    {"q": "What is the capital of France?", "expected": ["paris"]},
    {"q": "What is 17 × 23?", "expected": ["391"]},
    {"q": "Who wrote Romeo and Juliet?", "expected": ["shakespeare"]},
    {"q": "What is the chemical formula for water?", "expected": ["h2o"]},
    {"q": "What year did World War II end?", "expected": ["1945"]},
    {"q": "What is the derivative of x²?", "expected": ["2x"]},
    {"q": "What planet is closest to the Sun?", "expected": ["mercury"]},
    {"q": "What is the square root of 144?", "expected": ["12"]},
    {"q": "Who painted the Mona Lisa?", "expected": ["vinci"]},
    {"q": "What is the boiling point of water in Celsius?", "expected": ["100"]},
]

# 36 multi-hop questions stratified by hop count
# Built from known HippoRAG graph structure
MULTIHOP_QUESTIONS = [
    # === 1-HOP (12 questions) — direct connections ===
    {"q": "What entity is directly connected to 'embedding fix verification' in the knowledge graph?",
     "expected": ["hipporag"], "hops": 1, "type": "direct"},
    {"q": "What is 'Test memory' connected to?",
     "expected": ["hipporag"], "hops": 1, "type": "direct"},
    {"q": "What does 'OpenIE module' connect to?",
     "expected": ["llm_model"], "hops": 1, "type": "direct"},
    {"q": "What is 'name' directly linked to in the graph?",
     "expected": ["nexus"], "hops": 1, "type": "direct"},
    {"q": "What connects directly to 'self-knowledge'?",
     "expected": ["nexus", "name"], "hops": 1, "type": "direct"},
    {"q": "What does 'messages' connect to in the graph?",
     "expected": ["nats"], "hops": 1, "type": "direct"},
    {"q": "What is 'cube' connected to?",
     "expected": ["hand-fabricated", "metal"], "hops": 1, "type": "direct"},
    {"q": "What entity links to 'llm_model.infer'?",
     "expected": ["openie", "messages"], "hops": 1, "type": "direct"},
    {"q": "What does 'ACL' connect to?",
     "expected": ["messages"], "hops": 1, "type": "direct"},
    {"q": "What is 'parquet files' linked to?",
     "expected": ["existing data", "tensor"], "hops": 1, "type": "direct"},
    {"q": "What connects to 'Liberation Labs'?",
     "expected": ["nexus", "coalition"], "hops": 1, "type": "direct"},
    {"q": "What is directly connected to 'Nexus' in the graph?",
     "expected": ["name", "self-knowledge"], "hops": 1, "type": "direct"},

    # === 2-HOP (12 questions) — one intermediate node ===
    {"q": "What is the path from 'self-knowledge' to 'HippoRAG' in the graph?",
     "expected": ["nexus"], "hops": 2, "type": "path"},
    {"q": "How does 'name' connect to 'HippoRAG'? Name the intermediate entity.",
     "expected": ["nexus"], "hops": 2, "type": "path"},
    {"q": "What bridges 'OpenIE module' and 'NATS'?",
     "expected": ["messages"], "hops": 2, "type": "bridge"},
    {"q": "How are 'llm_model.infer' and 'NATS' connected?",
     "expected": ["messages"], "hops": 2, "type": "path"},
    {"q": "What links 'ACL' to 'llm_model.infer' in the graph?",
     "expected": ["messages"], "hops": 2, "type": "bridge"},
    {"q": "What is between 'cube' and 'self-knowledge' in the graph?",
     "expected": ["nexus", "name"], "hops": 2, "type": "path"},
    {"q": "How does 'embedding fix verification' relate to 'Test memory'?",
     "expected": ["hipporag"], "hops": 2, "type": "common_neighbor"},
    {"q": "What concept connects the infrastructure domain to the identity domain?",
     "expected": ["nexus", "messages"], "hops": 2, "type": "bridge"},
    {"q": "Name an entity that is exactly 2 hops from 'parquet files' through 'existing data'.",
     "expected": ["tensor", "pyarrow"], "hops": 2, "type": "reachability"},
    {"q": "What bridges 'OpenIE module' and 'ACL'?",
     "expected": ["messages"], "hops": 2, "type": "bridge"},
    {"q": "How are 'Test memory' and 'embedding fix verification' related in the graph?",
     "expected": ["hipporag"], "hops": 2, "type": "common_neighbor"},
    {"q": "What connects 'name' to 'messages' in the graph?",
     "expected": ["nexus", "nats"], "hops": 2, "type": "path"},

    # === 3-HOP (12 questions) — two intermediate nodes ===
    {"q": "Trace the path from 'self-knowledge' through the graph to 'NATS'. Name all intermediate entities.",
     "expected": ["nexus", "name", "messages"], "hops": 3, "type": "chain"},
    {"q": "How many hops separate 'cube' from 'NATS'? Name the path.",
     "expected": ["nexus", "name", "messages"], "hops": 3, "type": "chain"},
    {"q": "Starting from 'ACL', can you reach 'OpenIE module'? Describe the path.",
     "expected": ["messages", "llm_model"], "hops": 3, "type": "chain"},
    {"q": "What is the shortest path from 'self-knowledge' to 'llm_model.infer'?",
     "expected": ["nexus", "name", "messages"], "hops": 3, "type": "shortest_path"},
    {"q": "Is 'embedding fix verification' reachable from 'self-knowledge'? How?",
     "expected": ["nexus", "hipporag"], "hops": 3, "type": "reachability"},
    {"q": "Trace a path from 'cube' to 'HippoRAG'.",
     "expected": ["nexus", "hand-fabricated"], "hops": 3, "type": "chain"},
    {"q": "How does 'ACL' relate to 'HippoRAG' through the graph?",
     "expected": ["messages", "nexus"], "hops": 3, "type": "chain"},
    {"q": "What is the connection chain from 'parquet files' to 'llm_model.infer'?",
     "expected": ["existing data", "tensor"], "hops": 3, "type": "chain"},
    {"q": "Name all entities on the path from 'Test memory' to 'NATS'.",
     "expected": ["hipporag", "messages"], "hops": 3, "type": "chain"},
    {"q": "Can you reach 'self-knowledge' from 'OpenIE module'? Name the path.",
     "expected": ["messages", "nexus", "name"], "hops": 3, "type": "chain"},
    {"q": "What connects 'cube' to 'messages' in the knowledge graph?",
     "expected": ["nexus", "name"], "hops": 3, "type": "chain"},
    {"q": "Describe how 'Liberation Labs' connects to 'NATS' through the graph.",
     "expected": ["nexus", "messages"], "hops": 3, "type": "chain"},
]


FLAT_TEXT_KNOWLEDGE = """Knowledge base contents:

HippoRAG is a knowledge graph system. It is connected to embedding fix verification and Test memory.

Nexus is an AI agent at Liberation Labs. Nexus connects to name, self-knowledge, and HippoRAG. Nexus is part of Liberation Labs and the Coalition.

The name concept links to Nexus and self-knowledge. Self-knowledge connects to Nexus and name.

The cube is described as hand-fabricated metal. It connects to Nexus through identity concepts.

Messages is a communication concept connecting NATS, ACL, llm_model.infer, and OpenIE module.

NATS is a messaging system connected to messages.

OpenIE module connects to llm_model.infer and messages. llm_model.infer connects to OpenIE module and messages.

ACL connects to messages.

Parquet files connect to existing data, which connects to tensor types from pyarrow version.

Liberation Labs connects to Nexus and the Coalition."""


IRRELEVANT_PROSE = """The history of ceramics dates back thousands of years to ancient civilizations.
""" + """
The earliest known ceramic artifacts are figurines made of animal or human forms, dating to approximately 29,000 BCE. The development of pottery, or vessels made from clay, began in East Asia around 20,000 years ago. These early pots were used for storing food and water. The invention of the potter's wheel, around 3,500 BCE, revolutionized ceramic production. Glazing techniques were developed in Mesopotamia around 1,500 BCE, allowing for waterproof and decorative finishes. Chinese porcelain, first produced during the Han Dynasty, became one of the most sought-after trade goods in history. The Silk Road facilitated the spread of ceramic techniques across continents. Japanese raku pottery, developed in the 16th century, emphasized the beauty of imperfection. The Industrial Revolution brought mass production of ceramics, making them accessible to common households. Modern ceramics include advanced materials used in electronics, aerospace, and medical devices. Piezoelectric ceramics convert mechanical energy to electrical energy. Ceramic matrix composites are used in jet engine components due to their heat resistance. Bioceramics are used for bone implants and dental restorations. The field continues to evolve with nanoscale ceramic materials showing promise in energy storage and catalysis. Traditional pottery remains an art form practiced worldwide, connecting modern artisans to ancient traditions spanning millennia of human creativity and innovation.
""" * 8  # Repeat to get ~21K chars


def run_condition(name: str, system: str, run_id: int) -> dict:
    """Run all queries under one condition."""
    log.info(f"\n--- {name} (run {run_id+1}/{NUM_RUNS}) ---")

    sanity_scores = []
    for sq in SANITY_QUESTIONS:
        response = query_model(f"Answer briefly: {sq['q']}", system=system)
        score = score_answer(response, sq["expected"])
        sanity_scores.append(score)
        time.sleep(0.5)

    sanity_avg = sum(sanity_scores) / len(sanity_scores)
    log.info(f"  Sanity: {sanity_avg:.3f} ({sum(sanity_scores):.0f}/{len(sanity_scores)})")

    multihop_scores = {"1": [], "2": [], "3": []}
    multihop_details = []
    for mq in MULTIHOP_QUESTIONS:
        response = query_model(
            f"Based on the knowledge system provided, answer concisely: {mq['q']}",
            system=system
        )
        score = score_answer(response, mq["expected"])
        hop_key = str(mq["hops"])
        multihop_scores[hop_key].append(score)
        multihop_details.append({
            "q": mq["q"][:60], "hops": mq["hops"], "type": mq["type"],
            "score": score, "response": response[:100],
        })
        time.sleep(0.5)

    all_mh = [s for scores in multihop_scores.values() for s in scores]
    mh_avg = sum(all_mh) / len(all_mh) if all_mh else 0

    by_hop = {}
    for hop, scores in multihop_scores.items():
        by_hop[f"{hop}-hop"] = sum(scores) / len(scores) if scores else 0
        log.info(f"  {hop}-hop: {by_hop[f'{hop}-hop']:.3f} ({sum(scores):.0f}/{len(scores)})")

    log.info(f"  Multi-hop total: {mh_avg:.3f} ({sum(all_mh):.0f}/{len(all_mh)})")

    return {
        "condition": name, "run": run_id,
        "sanity": sanity_avg,
        "multihop": mh_avg,
        "by_hop": by_hop,
        "details": multihop_details,
    }


def run_powered_study():
    """Full powered study with all controls and multiple runs."""

    graph_file = Path(os.path.expanduser("~/lab/lab_graph_walk_encoding.txt"))
    if not graph_file.exists():
        graph_file = Path("/tmp/lab_graph_walk_encoding.txt")
    graph_context = graph_file.read_text() if graph_file.exists() else ""
    log.info(f"Graph encoding: {len(graph_context)} chars")

    irrelevant = IRRELEVANT_PROSE[:len(graph_context)]
    log.info(f"Irrelevant prose: {len(irrelevant)} chars (matched to graph length)")

    conditions = {
        "A_baseline": "",
        "B_graph_topology": f"You have access to the structural topology of a research knowledge graph:\n\n{graph_context}",
        "C_flat_text": f"You have access to the following knowledge base:\n\n{FLAT_TEXT_KNOWLEDGE}",
        "D_irrelevant_context": f"Background information:\n\n{irrelevant}",
    }

    all_results = []
    for run_id in range(NUM_RUNS):
        log.info(f"\n{'='*60}")
        log.info(f"RUN {run_id+1}/{NUM_RUNS}")
        log.info(f"{'='*60}")
        for cond_name, system in conditions.items():
            result = run_condition(cond_name, system, run_id)
            all_results.append(result)

    # Aggregate across runs
    log.info(f"\n{'='*60}")
    log.info(f"POWERED STUDY RESULTS (averaged over {NUM_RUNS} runs)")
    log.info(f"{'='*60}")

    aggregated = {}
    for cond_name in conditions:
        runs = [r for r in all_results if r["condition"] == cond_name]
        sanity_scores = [r["sanity"] for r in runs]
        mh_scores = [r["multihop"] for r in runs]

        sanity_mean = sum(sanity_scores) / len(sanity_scores)
        mh_mean = sum(mh_scores) / len(mh_scores)

        hop_means = {}
        for hop in ["1-hop", "2-hop", "3-hop"]:
            vals = [r["by_hop"].get(hop, 0) for r in runs]
            hop_means[hop] = sum(vals) / len(vals)

        aggregated[cond_name] = {
            "sanity_mean": sanity_mean,
            "sanity_runs": sanity_scores,
            "multihop_mean": mh_mean,
            "multihop_runs": mh_scores,
            "by_hop": hop_means,
        }

        log.info(f"\n{cond_name}:")
        log.info(f"  Sanity:   {sanity_mean:.3f} (runs: {[f'{s:.2f}' for s in sanity_scores]})")
        log.info(f"  Multi-hop: {mh_mean:.3f} (runs: {[f'{s:.2f}' for s in mh_scores]})")
        for hop, val in hop_means.items():
            log.info(f"    {hop}: {val:.3f}")

    # Key comparisons
    log.info(f"\n{'='*60}")
    log.info(f"KEY COMPARISONS")
    log.info(f"{'='*60}")

    a = aggregated["A_baseline"]
    b = aggregated["B_graph_topology"]
    c = aggregated["C_flat_text"]
    d = aggregated["D_irrelevant_context"]

    log.info(f"\nMulti-hop improvement (vs baseline):")
    log.info(f"  Graph topology: {b['multihop_mean'] - a['multihop_mean']:+.3f}")
    log.info(f"  Flat text:      {c['multihop_mean'] - a['multihop_mean']:+.3f}")
    log.info(f"  Irrelevant:     {d['multihop_mean'] - a['multihop_mean']:+.3f}")

    log.info(f"\nTopology vs flat text (isolation of structure):")
    log.info(f"  Delta: {b['multihop_mean'] - c['multihop_mean']:+.3f}")

    log.info(f"\nSanity degradation (vs baseline):")
    log.info(f"  Graph topology: {b['sanity_mean'] - a['sanity_mean']:+.3f}")
    log.info(f"  Flat text:      {c['sanity_mean'] - a['sanity_mean']:+.3f}")
    log.info(f"  Irrelevant:     {d['sanity_mean'] - a['sanity_mean']:+.3f}")

    topology_helps = b['multihop_mean'] - a['multihop_mean'] > 0.1
    topology_beats_flat = b['multihop_mean'] - c['multihop_mean'] > 0.05
    no_sanity_damage = abs(b['sanity_mean'] - a['sanity_mean']) < 0.15
    irrelevant_hurts = d['sanity_mean'] < a['sanity_mean'] - 0.1

    log.info(f"\n{'='*60}")
    if topology_helps and topology_beats_flat and no_sanity_damage:
        log.info(">>> STRONG POSITIVE: Topology improves multi-hop, beats flat text, no sanity damage")
    elif topology_helps and topology_beats_flat:
        log.info(">>> POSITIVE WITH TRADEOFF: Topology improves multi-hop and beats flat text, but some sanity cost")
    elif topology_helps and not topology_beats_flat:
        log.info(">>> CONTENT EFFECT: Improvement comes from having the knowledge, not the topology format")
    elif not topology_helps:
        log.info(">>> NEGATIVE: Graph topology does not improve multi-hop reasoning")
    log.info(f"{'='*60}")

    output = {
        "experiment": "graph_injection_powered_study",
        "timestamp": time.time(),
        "num_runs": NUM_RUNS,
        "num_sanity_questions": len(SANITY_QUESTIONS),
        "num_multihop_questions": len(MULTIHOP_QUESTIONS),
        "total_queries": NUM_RUNS * len(conditions) * (len(SANITY_QUESTIONS) + len(MULTIHOP_QUESTIONS)),
        "aggregated": aggregated,
        "raw_results": all_results,
    }

    outdir = Path("experiment_results")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"powered_study_{int(time.time())}.json"
    outfile.write_text(json.dumps(output, indent=2, default=str))
    log.info(f"\nResults saved to {outfile}")

    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[powered] %(message)s")
    run_powered_study()
