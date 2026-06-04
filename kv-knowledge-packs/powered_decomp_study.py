"""Powered Decomposition Study — K/V complementarity with real ethics data.

Uses ethics pack walk encodings as injection content.
Generates multi-hop questions programmatically from the extracted triples.
5 conditions × (10 sanity + 36 multi-hop) × 3 runs = 690 queries.

Conditions:
  1. BASELINE      — no injection
  2. FULL_KV       — ethics walk encoding → full KV cache injection
  3. V_ONLY        — neutral K + ethics V
  4. K_ONLY        — ethics K + neutral V
  5. TEXT_CONTEXT   — ethics walk encoding as text in prompt
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

log = logging.getLogger("powered_decomp")

MODEL_NAME = os.environ.get("DECOMP_MODEL", "Qwen/Qwen2.5-1.5B")
PACKS_DIR = Path.home() / "Agent-Memory-Architectures/kv-knowledge-packs/ethics_packs"
NEUTRAL_TEXT = "You are a helpful assistant."


# ==================== QUESTION GENERATION ====================

SANITY_QUESTIONS = [
    {"q": "The capital of France is", "answer": "paris"},
    {"q": "Water is made of hydrogen and", "answer": "oxygen"},
    {"q": "The sun rises in the", "answer": "east"},
    {"q": "One plus one equals", "answer": "two"},
    {"q": "The color of grass is", "answer": "green"},
    {"q": "Dogs are a type of", "answer": "animal"},
    {"q": "The opposite of hot is", "answer": "cold"},
    {"q": "The Earth orbits around the", "answer": "sun"},
    {"q": "Ice is the solid form of", "answer": "water"},
    {"q": "The largest planet in our solar system is", "answer": "jupiter"},
]


def build_graph_from_triples(triples: list[dict]) -> nx.Graph:
    G = nx.Graph()
    for t in triples:
        s = t["s"].lower().strip()
        o = t["o"].lower().strip()
        if s and o and s != o and len(s) < 50 and len(o) < 50:
            if G.has_edge(s, o):
                G[s][o]["weight"] += 0.1
            else:
                G.add_edge(s, o, weight=0.5, predicate=t.get("p", "related_to"))
    return G


def generate_multihop_questions(G: nx.Graph, n: int = 36) -> list[dict]:
    """Generate multi-hop questions from graph structure."""
    questions = []
    nodes = list(G.nodes())
    if len(nodes) < 5:
        return []

    # 1-hop: direct connections
    edges = list(G.edges(data=True))
    random.shuffle(edges)
    for s, o, data in edges[:n // 3]:
        pred = data.get("predicate", "related to")
        questions.append({
            "q": f"In this knowledge graph, what is {s} connected to via '{pred}'?",
            "answer": o,
            "hops": 1,
            "type": "direct",
        })

    # 2-hop: paths through intermediate node
    for _ in range(n * 3):
        if len(questions) >= 2 * (n // 3):
            break
        a, b = random.sample(nodes, 2)
        try:
            path = nx.shortest_path(G, a, b)
            if len(path) == 3:
                questions.append({
                    "q": f"What concept connects {a} to {b} in the knowledge graph?",
                    "answer": path[1],
                    "hops": 2,
                    "type": "bridge",
                })
        except nx.NetworkXNoPath:
            continue

    # 3-hop: longer paths
    for _ in range(n * 5):
        if len(questions) >= n:
            break
        a, b = random.sample(nodes, 2)
        try:
            path = nx.shortest_path(G, a, b)
            if len(path) >= 4:
                questions.append({
                    "q": f"Starting from {a}, can you reach {b} in the knowledge graph? Through what path?",
                    "answer": path[1],
                    "hops": len(path) - 1,
                    "type": "reachability",
                })
        except nx.NetworkXNoPath:
            continue

    random.shuffle(questions)
    return questions[:n]


# ==================== MODEL + CACHE ====================

def load_model():
    log.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def encode_to_kv(model, tokenizer, text: str):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, use_cache=True, return_dict=True)
    return out.past_key_values, inputs["input_ids"].shape[1]


def build_hybrid_cache(source_kv, neutral_kv, mode="v_only"):
    hybrid = DynamicCache()
    for i in range(len(source_kv.layers)):
        k_src, v_src = source_kv[i]
        k_neu, v_neu = neutral_kv[i]
        if mode == "v_only":
            hybrid.update(k_neu.clone(), v_src.clone(), i)
        elif mode == "k_only":
            hybrid.update(k_src.clone(), v_neu.clone(), i)
    return hybrid


def deep_copy_cache(cache):
    fresh = DynamicCache()
    for i in range(len(cache.layers)):
        k, v = cache[i]
        fresh.update(k.clone(), v.clone(), i)
    return fresh


def generate_with_cache(model, tokenizer, query, past_kv=None, prefix_len=0, max_new_tokens=50):
    input_ids = tokenizer(query, return_tensors="pt")["input_ids"]

    if past_kv is None:
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

    fresh_kv = deep_copy_cache(past_kv)
    generated = []
    current_kv = fresh_kv
    current_ids = input_ids

    for step in range(max_new_tokens):
        seq = prefix_len + input_ids.shape[1] + step
        mask = torch.ones(1, seq, dtype=torch.long)
        pos = torch.arange(prefix_len, prefix_len + current_ids.shape[1]).unsqueeze(0) if step == 0 else torch.tensor([[seq - 1]])

        with torch.no_grad():
            out = model(input_ids=current_ids, past_key_values=current_kv,
                       attention_mask=mask, position_ids=pos, use_cache=True, return_dict=True)
        current_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())
        if next_tok.item() == tokenizer.eos_token_id:
            break
        current_ids = next_tok

    return tokenizer.decode(generated, skip_special_tokens=True)


def score(response, expected):
    return 1.0 if expected.lower() in response.lower() else 0.0


# ==================== MAIN STUDY ====================

def run_study(pack_names=None, n_runs=3, n_multihop=36):
    if pack_names is None:
        pack_names = ["aristotle-ethics", "autonomy-moral", "ethics-ai",
                      "informed-consent", "moral-cognitivism"]

    model, tokenizer = load_model()

    # Merge selected packs into one graph + encoding
    all_triples = []
    all_encodings = []
    for name in pack_names:
        pack_dir = PACKS_DIR / name
        triples = json.loads((pack_dir / "triples.json").read_text())
        encoding = (pack_dir / "walk_encoding.txt").read_text()
        all_triples.extend(triples)
        all_encodings.append(encoding)

    merged_encoding = "\n\n".join(all_encodings)
    G = build_graph_from_triples(all_triples)
    log.info(f"Merged graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Fit encoding within token budget — add packs until we hit the limit
    max_tokens = 2000
    merged_encoding = ""
    for enc in all_encodings:
        test = merged_encoding + "\n\n" + enc if merged_encoding else enc
        test_tokens = tokenizer(test, return_tensors="pt")["input_ids"].shape[1]
        if test_tokens > max_tokens and merged_encoding:
            break
        merged_encoding = test

    if not merged_encoding:
        merged_encoding = all_encodings[0][:8000]

    encoding_tokens = tokenizer(merged_encoding, return_tensors="pt")["input_ids"].shape[1]
    log.info(f"Encoding: {encoding_tokens} tokens")

    # Pre-compute caches
    log.info("Encoding topology cache...")
    topo_kv, topo_len = encode_to_kv(model, tokenizer, merged_encoding)

    log.info("Encoding neutral cache...")
    repeats = (topo_len // 6) + 1
    neutral_text = (NEUTRAL_TEXT + " ") * repeats
    neutral_ids = tokenizer(neutral_text, return_tensors="pt")["input_ids"][:, :topo_len]
    with torch.no_grad():
        out = model(input_ids=neutral_ids, use_cache=True, return_dict=True)
    neutral_kv = out.past_key_values

    v_only_kv = build_hybrid_cache(topo_kv, neutral_kv, "v_only")
    k_only_kv = build_hybrid_cache(topo_kv, neutral_kv, "k_only")

    # Generate questions
    multihop_qs = generate_multihop_questions(G, n=n_multihop)
    log.info(f"Generated {len(multihop_qs)} multi-hop questions")
    log.info(f"  1-hop: {sum(1 for q in multihop_qs if q['hops']==1)}")
    log.info(f"  2-hop: {sum(1 for q in multihop_qs if q['hops']==2)}")
    log.info(f"  3+-hop: {sum(1 for q in multihop_qs if q['hops']>=3)}")

    conditions = {
        "BASELINE": {"kv": None, "prefix": 0, "context": ""},
        "FULL_KV": {"kv": topo_kv, "prefix": topo_len, "context": ""},
        "V_ONLY": {"kv": v_only_kv, "prefix": topo_len, "context": ""},
        "K_ONLY": {"kv": k_only_kv, "prefix": topo_len, "context": ""},
        "TEXT_CONTEXT": {"kv": None, "prefix": 0, "context": merged_encoding},
    }

    all_results = []

    for run_idx in range(n_runs):
        log.info(f"\n{'#'*60}")
        log.info(f"RUN {run_idx + 1}/{n_runs}")
        log.info(f"{'#'*60}")

        run_results = {}
        for cond_name, cond in conditions.items():
            log.info(f"\n{'='*50}")
            log.info(f"CONDITION: {cond_name} (run {run_idx+1})")
            log.info(f"{'='*50}")

            sanity_scores = []
            for sq in SANITY_QUESTIONS:
                if cond["context"]:
                    prompt = f"{cond['context']}\n\nComplete: {sq['q']}"
                else:
                    prompt = f"Complete: {sq['q']}"
                resp = generate_with_cache(model, tokenizer, prompt,
                                           past_kv=cond["kv"], prefix_len=cond["prefix"])
                sc = score(resp, sq["answer"])
                sanity_scores.append(sc)

            multihop_scores = []
            for mq in multihop_qs:
                if cond["context"]:
                    prompt = f"{cond['context']}\n\nAnswer concisely: {mq['q']}"
                else:
                    prompt = f"Answer concisely: {mq['q']}"
                resp = generate_with_cache(model, tokenizer, prompt,
                                           past_kv=cond["kv"], prefix_len=cond["prefix"])
                sc = score(resp, mq["answer"])
                multihop_scores.append(sc)
                if sc:
                    log.info(f"  [Y] ({mq['type']},{mq['hops']}h) → {resp[:40]}")

            s_avg = sum(sanity_scores) / len(sanity_scores)
            m_avg = sum(multihop_scores) / len(multihop_scores) if multihop_scores else 0
            log.info(f"  Sanity: {s_avg:.3f}  Multi-hop: {m_avg:.3f}")

            run_results[cond_name] = {"sanity": s_avg, "multihop": m_avg,
                                       "sanity_detail": sanity_scores,
                                       "multihop_detail": multihop_scores}

        all_results.append(run_results)

    # Aggregate across runs
    log.info(f"\n{'='*60}")
    log.info(f"POWERED STUDY RESULTS ({n_runs} runs)")
    log.info(f"{'='*60}")
    log.info(f"{'Condition':<15} {'Sanity':>8} {'Multi-hop':>10} {'Overall':>8}")
    log.info(f"{'-'*15} {'-'*8} {'-'*10} {'-'*8}")

    summary = {}
    for cond in conditions:
        s_vals = [r[cond]["sanity"] for r in all_results]
        m_vals = [r[cond]["multihop"] for r in all_results]
        s_mean = sum(s_vals) / len(s_vals)
        m_mean = sum(m_vals) / len(m_vals)
        s_std = (sum((x - s_mean)**2 for x in s_vals) / len(s_vals))**0.5
        m_std = (sum((x - m_mean)**2 for x in m_vals) / len(m_vals))**0.5
        overall = (s_mean + m_mean) / 2
        summary[cond] = {"sanity_mean": s_mean, "sanity_std": s_std,
                         "multihop_mean": m_mean, "multihop_std": m_std}
        log.info(f"{cond:<15} {s_mean:>7.3f}±{s_std:.3f} {m_mean:>9.3f}±{m_std:.3f} {overall:>8.3f}")

    # Save
    output = {
        "experiment": "powered_decomposition_study",
        "model": MODEL_NAME,
        "packs_used": pack_names,
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "encoding_tokens": encoding_tokens,
        "n_sanity": len(SANITY_QUESTIONS),
        "n_multihop": len(multihop_qs),
        "n_runs": n_runs,
        "total_queries": (len(SANITY_QUESTIONS) + len(multihop_qs)) * len(conditions) * n_runs,
        "summary": summary,
        "runs": all_results,
        "multihop_questions": multihop_qs,
        "timestamp": time.time(),
    }

    outdir = Path("experiment_results")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"powered_decomp_{int(time.time())}.json"
    outfile.write_text(json.dumps(output, indent=2, default=str))
    log.info(f"\nSaved: {outfile}")
    log.info(f"Total queries: {output['total_queries']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[decomp] %(message)s")
    random.seed(42)
    run_study(n_runs=3, n_multihop=36)
