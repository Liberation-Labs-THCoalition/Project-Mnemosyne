"""Ethics KV Pack Builder — Converts ethics datasets into knowledge graphs
and walk-encoded KV packs for Oracle.

Pipeline:
  1. Load ethics data (Stanford Encyclopedia + other datasets)
  2. Extract entities and relationships via local LLM (OpenIE-style)
  3. Build knowledge graph from triples
  4. Walk-encode subgraphs by topic
  5. Output pack-ready encodings for KV injection

Dual purpose: packs serve as both powered study test data AND Oracle ethics library.
"""

import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import requests

log = logging.getLogger("ethics_packs")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("EXTRACT_MODEL", "deepseek-v2:16b")
OUTPUT_DIR = Path(os.environ.get("ETHICS_PACK_DIR",
    os.path.expanduser("~/Agent-Memory-Architectures/kv-knowledge-packs/ethics_packs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    source: str = ""
    confidence: float = 1.0


@dataclass
class EthicsPack:
    name: str
    category: str
    triples: list[Triple] = field(default_factory=list)
    graph: nx.Graph = field(default_factory=nx.Graph)
    walk_encoding: str = ""
    entry_count: int = 0


EXTRACT_PROMPT = """Extract entity-relationship triples from this philosophy text.
Return JSON array only. Each triple: {{"s": "subject", "p": "predicate", "o": "object"}}
Keep entities short (2-4 words). Use specific predicates (argues_for, defines, contrasts_with, requires, enables, undermines, extends, grounds).
Max 8 triples. Focus on philosophical relationships, not meta/bibliographic.
/no_think

Text: {text}

JSON:"""


def llm_extract(text: str) -> list[dict]:
    prompt = EXTRACT_PROMPT.format(text=text[:1500])
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.1, "num_predict": 800, "num_gpu": 5}},
            timeout=120,
        )
        if resp.status_code != 200:
            return []
        raw = resp.json().get("response", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Find JSON array
        start = raw.find("[")
        end = raw.rfind("]")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
    except (json.JSONDecodeError, Exception) as e:
        log.debug(f"Extract failed: {e}")
    return []


def normalize_entity(e: str) -> str:
    return re.sub(r'\s+', '_', e.strip().lower())[:50]


def build_graph(triples: list[Triple]) -> nx.Graph:
    G = nx.Graph()
    for t in triples:
        s = normalize_entity(t.subject)
        o = normalize_entity(t.object)
        if s and o and s != o:
            if G.has_edge(s, o):
                G[s][o]["weight"] += 0.1
                G[s][o]["predicates"].add(t.predicate)
            else:
                G.add_edge(s, o, weight=0.5, predicates={t.predicate})
    return G


def walk_encode(G: nx.Graph, steps: int = 5) -> str:
    if len(G.nodes) == 0:
        return ""

    # Compute random walk transition matrix
    adj = nx.to_numpy_array(G, weight="weight")
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = adj / row_sums

    import numpy as np
    nodes = list(G.nodes)
    W = np.eye(len(nodes))
    power = np.eye(len(nodes))
    for s in range(1, steps + 1):
        power = power @ T
        W += power
    W /= (steps + 1)

    lines = [f"Knowledge graph topology ({len(nodes)} concepts, walk encoding):"]
    for i, node in enumerate(nodes):
        connections = []
        for j, other in enumerate(nodes):
            if i != j and W[i, j] > 0.01:
                connections.append((other, W[i, j]))
        connections.sort(key=lambda x: -x[1])
        top = connections[:8]
        if top:
            conn_str = ", ".join(f"{n} ({w:.3f})" for n, w in top)
            lines.append(f"Node: {node} connects to: {conn_str}")
        else:
            lines.append(f"Node: {node} connects to: (none)")

    return "\n".join(lines)


def load_stanford_ethics(max_per_category: int = 20) -> dict[str, list[dict]]:
    from datasets import load_from_disk
    ds = load_from_disk("/mnt/data1/training-data/ethics/stanford_encyclopedia_philosophy")
    train = ds["train"]

    ethics_keywords = [
        "ethic", "moral", "virtue", "justice", "rights", "consent", "dignity",
        "autonomy", "welfare", "harm", "duty", "deontol", "utilitar", "care",
        "feminis", "animal", "bioethic", "environ", "responsib", "freedom",
    ]

    by_category = defaultdict(list)
    for i in range(len(train)):
        cat = train[i]["category"]
        if any(k in cat.lower() for k in ethics_keywords):
            if len(by_category[cat]) < max_per_category:
                by_category[cat].append({
                    "text": train[i]["text"],
                    "category": cat,
                    "url": train[i]["metadata"],
                })

    return dict(by_category)


def build_category_pack(category: str, entries: list[dict]) -> EthicsPack:
    pack = EthicsPack(name=category, category=category, entry_count=len(entries))
    all_triples = []

    for entry in entries:
        raw_triples = llm_extract(entry["text"])
        for rt in raw_triples:
            if isinstance(rt, dict) and "s" in rt and "o" in rt:
                all_triples.append(Triple(
                    subject=rt["s"],
                    predicate=rt.get("p", "related_to"),
                    object=rt["o"],
                    source=category,
                ))

    pack.triples = all_triples
    pack.graph = build_graph(all_triples)
    pack.walk_encoding = walk_encode(pack.graph)

    return pack


def save_pack(pack: EthicsPack):
    pack_dir = OUTPUT_DIR / pack.name
    pack_dir.mkdir(parents=True, exist_ok=True)

    # Save triples
    triples_data = [{"s": t.subject, "p": t.predicate, "o": t.object}
                    for t in pack.triples]
    (pack_dir / "triples.json").write_text(json.dumps(triples_data, indent=2))

    # Save walk encoding
    if pack.walk_encoding:
        (pack_dir / "walk_encoding.txt").write_text(pack.walk_encoding)

    # Save graph stats
    stats = {
        "category": pack.category,
        "entries_processed": pack.entry_count,
        "triples": len(pack.triples),
        "nodes": pack.graph.number_of_nodes(),
        "edges": pack.graph.number_of_edges(),
        "density": nx.density(pack.graph) if pack.graph.number_of_nodes() > 1 else 0,
        "encoding_tokens_approx": len(pack.walk_encoding.split()),
    }
    (pack_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    log.info(f"  Saved: {pack.name} — {stats['nodes']} nodes, {stats['edges']} edges, "
             f"~{stats['encoding_tokens_approx']} tokens")
    return stats


def run(categories: list[str] = None, max_per_category: int = 10,
        max_categories: int = 10):
    """Build ethics packs from Stanford Encyclopedia.

    Args:
        categories: specific categories to process (None = top N by size)
        max_per_category: entries to process per category
        max_categories: how many categories to process
    """
    log.info("Loading Stanford Encyclopedia ethics entries...")
    by_cat = load_stanford_ethics(max_per_category=max_per_category)
    log.info(f"Found {len(by_cat)} ethics categories, {sum(len(v) for v in by_cat.values())} entries")

    if categories:
        selected = {k: v for k, v in by_cat.items() if k in categories}
    else:
        selected = dict(sorted(by_cat.items(), key=lambda x: -len(x[1]))[:max_categories])

    log.info(f"Processing {len(selected)} categories...")
    all_stats = []

    for cat, entries in selected.items():
        log.info(f"\n{'='*40}")
        log.info(f"Category: {cat} ({len(entries)} entries)")
        log.info(f"{'='*40}")

        pack = build_category_pack(cat, entries)
        stats = save_pack(pack)
        all_stats.append(stats)

    # Summary
    total_nodes = sum(s["nodes"] for s in all_stats)
    total_edges = sum(s["edges"] for s in all_stats)
    total_triples = sum(s["triples"] for s in all_stats)
    total_tokens = sum(s["encoding_tokens_approx"] for s in all_stats)

    log.info(f"\n{'='*60}")
    log.info(f"SUMMARY: {len(all_stats)} packs built")
    log.info(f"  Total nodes: {total_nodes}")
    log.info(f"  Total edges: {total_edges}")
    log.info(f"  Total triples: {total_triples}")
    log.info(f"  Total encoding tokens: ~{total_tokens}")
    log.info(f"  Output: {OUTPUT_DIR}")

    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(all_stats, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[ethics] %(message)s")
    run(max_per_category=10, max_categories=5)
