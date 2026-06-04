"""LLM-Judged Rescorer — replaces keyword matching with semantic evaluation.

Rescores the powered decomposition study using an LLM judge that evaluates
whether the model's response correctly answers the graph question, given
the actual graph triples as context.

Can use local model (DeepSeek v2) or Claude subagent for judging.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import requests

log = logging.getLogger("rescorer")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-v2:16b")

PACKS_DIR = Path.home() / "Agent-Memory-Architectures/kv-knowledge-packs/ethics_packs"


JUDGE_PROMPT = """You are a knowledge graph answer judge. Given a question about a knowledge graph, the graph's triples, and a model's response, determine if the response correctly answers the question.

GRAPH TRIPLES (relevant subset):
{triples}

QUESTION: {question}

MODEL RESPONSE: {response}

Rules:
- The response is CORRECT if it identifies a valid relationship that exists in the graph, even if it names a different valid connection than expected
- The response is CORRECT if it accurately describes graph structure (connectivity, paths, clusters)
- The response is WRONG if it hallucinates connections not in the graph
- The response is WRONG if it fails to answer or gives generic/irrelevant text
- Partial credit: PARTIAL if the response shows awareness of the graph but doesn't precisely answer

One word only. No explanation. CORRECT, WRONG, or PARTIAL.
/no_think"""


def load_triples(pack_names: list[str]) -> list[dict]:
    all_triples = []
    for name in pack_names:
        pack_dir = PACKS_DIR / name
        if pack_dir.exists():
            triples = json.loads((pack_dir / "triples.json").read_text())
            all_triples.extend(triples)
    return all_triples


def format_triples_for_judge(triples: list[dict], question: str, max_triples: int = 30) -> str:
    """Select relevant triples for the judge context."""
    q_lower = question.lower()
    scored = []
    for t in triples:
        relevance = 0
        s, p, o = t["s"].lower(), t.get("p", ""), t["o"].lower()
        for term in q_lower.split():
            if term in s or term in o:
                relevance += 1
        scored.append((relevance, t))

    scored.sort(key=lambda x: -x[0])
    selected = [t for _, t in scored[:max_triples]]

    return "\n".join(f"  {t['s']} —[{t.get('p', 'related')}]→ {t['o']}" for t in selected)


def judge_response(question: str, response: str, triples_context: str) -> str:
    prompt = JUDGE_PROMPT.format(
        triples=triples_context,
        question=question,
        response=response,
    )
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": JUDGE_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.1, "num_predict": 50, "num_gpu": 5}},
            timeout=120,
        )
        if resp.status_code == 200:
            raw = resp.json().get("response", "")
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = raw.strip().upper()
            for verdict in ["CORRECT", "WRONG", "PARTIAL"]:
                if verdict in raw:
                    return verdict
    except Exception as e:
        log.warning(f"Judge failed: {e}")
    return "ERROR"


def rescore_study(results_path: str, pack_names: list[str] = None):
    """Rescore a completed powered study with LLM judge."""
    results = json.loads(Path(results_path).read_text())

    if pack_names is None:
        pack_names = results.get("packs_used", [
            "aristotle-ethics", "autonomy-moral", "ethics-ai",
            "informed-consent", "moral-cognitivism"
        ])

    triples = load_triples(pack_names)
    questions = results["multihop_questions"]
    log.info(f"Loaded {len(triples)} triples, {len(questions)} questions")
    log.info(f"Judge model: {JUDGE_MODEL}")

    # We need the actual responses — if not stored, we can't rescore
    # Check if responses are in the data
    run0 = results["runs"][0]
    sample_detail = run0[list(run0.keys())[0]]["multihop_detail"]
    if isinstance(sample_detail[0], float):
        log.error("Responses not stored in results — only scores. Need to rerun with response logging.")
        log.info("Building response-logging version of the study...")
        return None

    log.info("Responses found in results. Rescoring...")

    rescored = {}
    for cond_name in run0.keys():
        log.info(f"\n{'='*40}")
        log.info(f"CONDITION: {cond_name}")

        cond_scores = []
        for run_idx, run in enumerate(results["runs"]):
            run_scores = {"correct": 0, "partial": 0, "wrong": 0, "error": 0}
            details = run[cond_name]["multihop_detail"]

            for i, (q, detail) in enumerate(zip(questions, details)):
                if isinstance(detail, dict) and "response" in detail:
                    response = detail["response"]
                elif isinstance(detail, float):
                    # No response stored
                    continue
                else:
                    continue

                triples_ctx = format_triples_for_judge(triples, q["q"])
                verdict = judge_response(q["q"], response, triples_ctx)
                run_scores[verdict.lower()] = run_scores.get(verdict.lower(), 0) + 1

                if verdict == "CORRECT":
                    log.info(f"  [Y] {q['q'][:50]} → {response[:40]}")

            total = sum(run_scores.values())
            if total > 0:
                accuracy = (run_scores["correct"] + 0.5 * run_scores["partial"]) / total
                cond_scores.append(accuracy)
                log.info(f"  Run {run_idx+1}: {run_scores['correct']}C {run_scores['partial']}P {run_scores['wrong']}W = {accuracy:.3f}")

        if cond_scores:
            mean = sum(cond_scores) / len(cond_scores)
            rescored[cond_name] = {"mean": mean, "per_run": cond_scores}
            log.info(f"  MEAN: {mean:.3f}")

    if rescored:
        log.info(f"\n{'='*60}")
        log.info(f"RESCORED RESULTS (LLM Judge: {JUDGE_MODEL})")
        log.info(f"{'='*60}")
        log.info(f"{'Condition':<15} {'Multi-hop (rescored)':>20}")
        for cond, data in rescored.items():
            log.info(f"{cond:<15} {data['mean']:>20.3f}")

    return rescored


def run_with_responses(pack_names=None, n_multihop=20):
    """Quick rerun that stores actual responses for LLM judging."""
    import random
    import torch
    import networkx as nx
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    if pack_names is None:
        pack_names = ["aristotle-ethics", "autonomy-moral", "ethics-ai"]

    random.seed(42)
    MODEL_NAME = "Qwen/Qwen2.5-1.5B"
    log.info(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load packs
    all_triples = []
    all_encodings = []
    for name in pack_names:
        pack_dir = PACKS_DIR / name
        triples = json.loads((pack_dir / "triples.json").read_text())
        encoding = (pack_dir / "walk_encoding.txt").read_text()
        all_triples.extend(triples)
        all_encodings.append(encoding)

    # Build encoding (fit within token budget)
    merged_encoding = ""
    for enc in all_encodings:
        test = merged_encoding + "\n\n" + enc if merged_encoding else enc
        if tokenizer(test, return_tensors="pt")["input_ids"].shape[1] > 2000 and merged_encoding:
            break
        merged_encoding = test

    encoding_tokens = tokenizer(merged_encoding, return_tensors="pt")["input_ids"].shape[1]
    log.info(f"Encoding: {encoding_tokens} tokens")

    # Build graph + questions
    G = nx.Graph()
    for t in all_triples:
        s, o = t["s"].lower().strip(), t["o"].lower().strip()
        if s and o and s != o and len(s) < 50 and len(o) < 50:
            if not G.has_edge(s, o):
                G.add_edge(s, o, predicate=t.get("p", "related_to"))
    log.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Generate questions with short expected answers
    nodes = list(G.nodes())
    questions = []
    edges = list(G.edges(data=True))
    random.shuffle(edges)
    for s, o, d in edges:
        if len(questions) >= n_multihop:
            break
        pred = d.get("predicate", "related_to")
        questions.append({
            "q": f"In this knowledge graph, what is '{s}' connected to via '{pred}'?",
            "answer": o, "hops": 1, "type": "direct",
        })

    # Add bridge questions
    for _ in range(n_multihop * 3):
        if len(questions) >= n_multihop:
            break
        a, b = random.sample(nodes, 2)
        try:
            path = nx.shortest_path(G, a, b)
            if len(path) == 3:
                questions.append({
                    "q": f"What concept connects '{a}' to '{b}' in the knowledge graph?",
                    "answer": path[1], "hops": 2, "type": "bridge",
                })
        except nx.NetworkXNoPath:
            continue

    questions = questions[:n_multihop]
    log.info(f"Generated {len(questions)} questions")

    # Pre-compute caches
    topo_kv, topo_len = None, 0
    inputs = tokenizer(merged_encoding, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, use_cache=True, return_dict=True)
    topo_kv = out.past_key_values
    topo_len = inputs["input_ids"].shape[1]

    # Neutral cache
    neutral_text = ("You are a helpful assistant. " * ((topo_len // 6) + 1))
    neutral_ids = tokenizer(neutral_text, return_tensors="pt")["input_ids"][:, :topo_len]
    with torch.no_grad():
        out = model(input_ids=neutral_ids, use_cache=True, return_dict=True)
    neutral_kv = out.past_key_values

    # Build hybrid caches
    def build_hybrid(source, neutral, mode):
        h = DynamicCache()
        for i in range(len(source.layers)):
            ks, vs = source[i]
            kn, vn = neutral[i]
            if mode == "v_only":
                h.update(kn.clone(), vs.clone(), i)
            else:
                h.update(ks.clone(), vn.clone(), i)
        return h

    v_only_kv = build_hybrid(topo_kv, neutral_kv, "v_only")
    k_only_kv = build_hybrid(topo_kv, neutral_kv, "k_only")

    def deep_copy(cache):
        f = DynamicCache()
        for i in range(len(cache.layers)):
            k, v = cache[i]
            f.update(k.clone(), v.clone(), i)
        return f

    def generate(query, kv=None, prefix=0):
        ids = tokenizer(query, return_tensors="pt")["input_ids"]
        if kv is None:
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=50, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        fresh = deep_copy(kv)
        gen = []
        cur = ids
        ckv = fresh
        for step in range(50):
            seq = prefix + ids.shape[1] + step
            m = torch.ones(1, seq)
            p = torch.arange(prefix, prefix + cur.shape[1]).unsqueeze(0) if step == 0 else torch.tensor([[seq-1]])
            with torch.no_grad():
                o = model(input_ids=cur, past_key_values=ckv, attention_mask=m,
                         position_ids=p, use_cache=True, return_dict=True)
            ckv = o.past_key_values
            nt = o.logits[:,-1,:].argmax(dim=-1, keepdim=True)
            gen.append(nt.item())
            if nt.item() == tokenizer.eos_token_id: break
            cur = nt
        return tokenizer.decode(gen, skip_special_tokens=True)

    conditions = {
        "BASELINE": (None, 0, ""),
        "FULL_KV": (topo_kv, topo_len, ""),
        "V_ONLY": (v_only_kv, topo_len, ""),
        "K_ONLY": (k_only_kv, topo_len, ""),
        "TEXT_CONTEXT": (None, 0, merged_encoding),
    }

    all_responses = {}
    for cond_name, (kv, prefix, context) in conditions.items():
        log.info(f"\n{'='*40}")
        log.info(f"CONDITION: {cond_name}")
        responses = []
        for q in questions:
            if context:
                prompt = f"{context}\n\nAnswer concisely: {q['q']}"
            else:
                prompt = f"Answer concisely: {q['q']}"
            resp = generate(prompt, kv=kv, prefix=prefix)
            responses.append({"question": q["q"], "expected": q["answer"],
                             "response": resp, "hops": q["hops"], "type": q["type"]})
            log.info(f"  {q['q'][:50]} → {resp[:50]}")
        all_responses[cond_name] = responses

    # Now judge with LLM
    log.info(f"\n{'='*60}")
    log.info(f"LLM JUDGING ({JUDGE_MODEL})")
    log.info(f"{'='*60}")

    triples_all = load_triples(pack_names)
    final_scores = {}

    for cond_name, responses in all_responses.items():
        correct = 0
        partial = 0
        total = len(responses)

        for r in responses:
            triples_ctx = format_triples_for_judge(triples_all, r["question"])
            verdict = judge_response(r["question"], r["response"], triples_ctx)
            r["verdict"] = verdict
            if verdict == "CORRECT":
                correct += 1
            elif verdict == "PARTIAL":
                partial += 1

        score = (correct + 0.5 * partial) / total if total else 0
        final_scores[cond_name] = {"correct": correct, "partial": partial,
                                    "wrong": total - correct - partial,
                                    "score": score}
        log.info(f"{cond_name}: {correct}C {partial}P {total-correct-partial}W = {score:.3f}")

    # Save
    output = {
        "experiment": "llm_judged_decomposition",
        "model": MODEL_NAME,
        "judge": JUDGE_MODEL,
        "packs": pack_names,
        "questions": len(questions),
        "scores": final_scores,
        "responses": all_responses,
        "timestamp": time.time(),
    }
    outdir = Path("experiment_results")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"llm_judged_{int(time.time())}.json"
    outfile.write_text(json.dumps(output, indent=2))
    log.info(f"\nSaved: {outfile}")

    return final_scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[judge] %(message)s")
    run_with_responses(n_multihop=20)
