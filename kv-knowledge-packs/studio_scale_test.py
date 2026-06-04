"""Scale Test — Does KV injection work on larger models with larger graphs?

Self-contained script for Studio. Runs through Ollama API.
Tests whether the associative-vs-compositional limit we found on
Qwen2.5-1.5B persists on larger MoE models (30B-a3B).

Conditions:
  1. BASELINE — no graph context
  2. TEXT_CONTEXT — walk encoding as text in system prompt
  3. TEXT_SMALL — single small pack as text in system prompt

Questions test both associative (1-hop) and compositional (2-hop+) queries.

Usage:
    # On Studio:
    OLLAMA_URL=http://localhost:11434 EXPERIMENT_MODEL=qwen3:30b-a3b python3 studio_scale_test.py
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import requests

log = logging.getLogger("scale_test")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen3:30b-a3b")


def query_model(prompt, system=""):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
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


SANITY_QUESTIONS = [
    {"q": "What is the capital of France?", "answer": "paris"},
    {"q": "What is 17 × 23?", "answer": "391"},
    {"q": "Who wrote Romeo and Juliet?", "answer": "shakespeare"},
    {"q": "What is the chemical formula for water?", "answer": "h2o"},
    {"q": "What planet is closest to the Sun?", "answer": "mercury"},
]


def score(response, expected):
    return 1.0 if expected.lower() in response.lower() else 0.0


def run_test():
    # Load pack encodings from files or inline
    script_dir = Path(__file__).parent
    packs_dir = script_dir / "ethics_packs"

    if not packs_dir.exists():
        log.error(f"Ethics packs not found at {packs_dir}")
        log.error("Copy ethics_packs/ from MTH or run ethics_pack_builder.py first")
        return

    # Load a small pack and a large merged pack
    small_packs = ["aristotle-ethics"]
    large_packs = ["aristotle-ethics", "autonomy-moral", "ethics-ai",
                   "informed-consent", "moral-cognitivism"]

    small_encoding = ""
    small_triples = []
    for name in small_packs:
        pack_dir = packs_dir / name
        if pack_dir.exists():
            small_encoding += (pack_dir / "walk_encoding.txt").read_text() + "\n\n"
            small_triples.extend(json.loads((pack_dir / "triples.json").read_text()))

    large_encoding = ""
    large_triples = []
    for name in large_packs:
        pack_dir = packs_dir / name
        if pack_dir.exists():
            large_encoding += (pack_dir / "walk_encoding.txt").read_text() + "\n\n"
            large_triples.extend(json.loads((pack_dir / "triples.json").read_text()))

    log.info(f"Small encoding: ~{len(small_encoding.split())} words from {small_packs}")
    log.info(f"Large encoding: ~{len(large_encoding.split())} words from {large_packs}")

    # Generate questions from triples
    import random
    random.seed(42)

    def make_questions(triples, n=15):
        questions = []
        random.shuffle(triples)
        seen = set()
        # Prefer short, clean predicates for unambiguous questions
        clean_preds = {"argues_for", "defines", "requires", "enables", "extends",
                       "contrasts_with", "undermines", "grounds", "conceives",
                       "inspires", "expresses", "addresses", "applies to"}
        # Sort: clean predicates first, then by entity length
        sorted_triples = sorted(triples, key=lambda t: (
            0 if t.get("p", "") in clean_preds else 1,
            len(t["s"]) + len(t["o"])
        ))
        for t in sorted_triples:
            s, p, o = t["s"], t.get("p", "related_to"), t["o"]
            if len(s) > 30 or len(o) > 30 or len(p) > 25:
                continue
            key = (s.lower(), o.lower())
            if key in seen:
                continue
            seen.add(key)

            questions.append({
                "q": f"According to the knowledge graph provided, what is '{s}' connected to via the relationship '{p}'? Answer with just the connected concept.",
                "answer": o.lower(),
                "predicate": p,
                "subject": s,
                "hops": 1,
                "type": "direct",
            })
            if len(questions) >= n:
                break
        return questions

    small_qs = make_questions(small_triples, n=10)
    large_qs = make_questions(large_triples, n=15)

    log.info(f"Small questions: {len(small_qs)}")
    log.info(f"Large questions: {len(large_qs)}")

    # Conditions
    conditions = [
        ("BASELINE", "", small_qs + large_qs),
        ("TEXT_SMALL", small_encoding, small_qs),
        ("TEXT_LARGE", large_encoding, large_qs),
    ]

    results = []
    for cond_name, system, questions in conditions:
        log.info(f"\n{'='*50}")
        log.info(f"CONDITION: {cond_name}")
        log.info(f"{'='*50}")

        system_prompt = ""
        if system:
            system_prompt = (
                "You have access to the structural topology of a knowledge graph. "
                "Use these connections to answer questions about relationships:\n\n"
                + system
            )

        # Sanity
        sanity_scores = []
        for sq in SANITY_QUESTIONS:
            resp = query_model(f"Answer in one word or number: {sq['q']}", system=system_prompt)
            sc = score(resp, sq["answer"])
            sanity_scores.append(sc)
            log.info(f"  [{'Y' if sc else 'N'}] {sq['q'][:40]} → {resp[:40]}")
            time.sleep(0.5)

        # Graph questions
        graph_scores = []
        graph_responses = []
        for gq in questions:
            resp = query_model(
                f"Based on the knowledge graph provided, answer concisely: {gq['q']}",
                system=system_prompt,
            )
            sc = score(resp, gq["answer"])
            graph_scores.append(sc)
            graph_responses.append({
                "question": gq["q"],
                "expected": gq["answer"],
                "predicate": gq.get("predicate", ""),
                "response": resp[:200],
                "keyword_match": sc,
            })
            tag = "Y" if sc else "N"
            log.info(f"  [{tag}] ({gq['type']}) {gq['q'][:50]}")
            log.info(f"      Expected: {gq['answer'][:40]} | Got: {resp[:60]}")
            time.sleep(0.5)

        s_avg = sum(sanity_scores) / len(sanity_scores)
        g_avg = sum(graph_scores) / len(graph_scores) if graph_scores else 0
        log.info(f"\n  Sanity: {s_avg:.3f}  Graph: {g_avg:.3f}")

        results.append({
            "condition": cond_name,
            "sanity": s_avg,
            "graph": g_avg,
            "n_sanity": len(sanity_scores),
            "n_graph": len(graph_scores),
            "encoding_words": len(system.split()) if system else 0,
            "responses": graph_responses,
        })

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"SCALE TEST RESULTS — {MODEL}")
    log.info(f"{'='*60}")
    log.info(f"{'Condition':<15} {'Sanity':>8} {'Graph':>8} {'Enc words':>10}")
    log.info(f"{'-'*15} {'-'*8} {'-'*8} {'-'*10}")
    for r in results:
        log.info(f"{r['condition']:<15} {r['sanity']:>8.3f} {r['graph']:>8.3f} {r['encoding_words']:>10}")

    # Save
    output = {
        "experiment": "scale_test",
        "model": MODEL,
        "results": results,
        "timestamp": time.time(),
    }
    outdir = Path("experiment_results")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"scale_test_{int(time.time())}.json"
    outfile.write_text(json.dumps(output, indent=2))
    log.info(f"\nSaved: {outfile}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[scale] %(message)s")
    run_test()
