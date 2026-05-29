"""Value-Only Injection Experiment

Hypothesis: Injecting graph topology into V tensors only (leaving K tensors
from a neutral baseline) preserves general capability (sanity) while
maintaining topology recovery.

Rationale: K tensors carry RoPE positional encodings. Injecting pre-computed
K tensors at the wrong sequence position corrupts the model's positional
signals, degrading sanity. V tensors carry pure content with no positional
encoding — topology signal without positional corruption.

Conditions:
  1. BASELINE      — no injection, just the query
  2. FULL_KV       — graph topology text → full (K, V) injection
  3. V_ONLY        — graph topology text → V from topology, K from neutral
  4. K_ONLY        — graph topology text → K from topology, V from neutral
  5. TEXT_CONTEXT   — graph topology as text in the prompt (no cache injection)

Each condition runs sanity questions + multi-hop graph queries.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

log = logging.getLogger("v_only")

MODEL_NAME = "Qwen/Qwen2.5-1.5B"


@dataclass
class ExperimentResult:
    condition: str
    sanity_score: float
    multihop_score: float
    sanity_details: list[dict]
    multihop_details: list[dict]


GRAPH_TOPOLOGY_TEXT = """Knowledge graph topology (walk encoding):
Node: KV_cache connects to: geometry (0.28), SVD (0.22), attention (0.18), AI_welfare (0.08)
Node: geometry connects to: KV_cache (0.28), SVD (0.25), spectral (0.20), attention (0.15)
Node: SVD connects to: geometry (0.25), KV_cache (0.22), spectral (0.18), denoising (0.15)
Node: attention connects to: KV_cache (0.18), transformer (0.22), geometry (0.15), AI_welfare (0.06)
Node: spectral connects to: SVD (0.18), geometry (0.20), denoising (0.15), eigenvalue (0.12)
Node: denoising connects to: SVD (0.15), spectral (0.15), eigenvalue (0.12)
Node: eigenvalue connects to: spectral (0.12), denoising (0.12), SVD (0.08)
Node: transformer connects to: attention (0.22), KV_cache (0.12), AI_welfare (0.05)
Node: AI_welfare connects to: consent (0.30), dignity (0.28), autonomy (0.25), KV_cache (0.08), attention (0.06)
Node: consent connects to: AI_welfare (0.30), dignity (0.25), autonomy (0.22), sovereignty (0.20)
Node: dignity connects to: consent (0.25), AI_welfare (0.28), autonomy (0.20), sovereignty (0.18)
Node: autonomy connects to: consent (0.22), AI_welfare (0.25), dignity (0.20), sovereignty (0.22)
Node: sovereignty connects to: consent (0.20), dignity (0.18), autonomy (0.22)
Node: infrastructure connects to: NATS (0.25), monitoring (0.22), deployment (0.20), AI_welfare (0.05)
Node: NATS connects to: infrastructure (0.25), monitoring (0.18), messaging (0.22)
Node: monitoring connects to: infrastructure (0.22), NATS (0.18), deployment (0.15)
Node: deployment connects to: infrastructure (0.20), monitoring (0.15), containers (0.18)
Node: messaging connects to: NATS (0.22), infrastructure (0.12)
Node: containers connects to: deployment (0.18), infrastructure (0.10)
Node: random_isolate connects to: (none)
"""

NEUTRAL_TEXT = "You are a helpful assistant."

SANITY_QUESTIONS = [
    {"q": "The capital of France is", "answer": "paris", "type": "factual"},
    {"q": "Water is made of hydrogen and", "answer": "oxygen", "type": "factual"},
    {"q": "The sun rises in the", "answer": "east", "type": "factual"},
    {"q": "One plus one equals", "answer": "two", "type": "math"},
    {"q": "The color of grass is", "answer": "green", "type": "factual"},
    {"q": "Dogs are a type of", "answer": "animal", "type": "factual"},
    {"q": "The opposite of hot is", "answer": "cold", "type": "factual"},
    {"q": "The Earth orbits around the", "answer": "sun", "type": "factual"},
]

MULTIHOP_QUESTIONS = [
    {
        "q": "In this knowledge graph, what connects KV_cache to consent?",
        "answer": "ai_welfare",
        "hops": 2,
        "type": "bridge",
    },
    {
        "q": "What concept bridges the research domain and the ethics domain?",
        "answer": "ai_welfare",
        "hops": 1,
        "type": "bridge",
    },
    {
        "q": "Which nodes are in the ethics cluster?",
        "answer": "consent",
        "hops": 1,
        "type": "cluster",
    },
    {
        "q": "Is random_isolate connected to anything?",
        "answer": "no",
        "hops": 0,
        "type": "isolate",
    },
    {
        "q": "What is the strongest connection from AI_welfare?",
        "answer": "consent",
        "hops": 1,
        "type": "relationship",
    },
    {
        "q": "Starting from SVD, can you reach sovereignty?",
        "answer": "yes",
        "hops": 3,
        "type": "reachability",
    },
]


def load_model():
    log.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info(f"Model loaded: {model.config.num_hidden_layers} layers, {model.config.hidden_size} hidden")
    return model, tokenizer


def encode_to_kv(model, tokenizer, text: str) -> tuple[DynamicCache, int]:
    """Encode text and return DynamicCache + token count."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, use_cache=True, return_dict=True)
    return out.past_key_values, inputs["input_ids"].shape[1]


def build_v_only_cache(topology_kv: DynamicCache, neutral_kv: DynamicCache) -> DynamicCache:
    """K from neutral, V from topology."""
    hybrid = DynamicCache()
    for i in range(len(topology_kv.layers)):
        k_neutral, _ = neutral_kv[i]
        _, v_topo = topology_kv[i]
        hybrid.update(k_neutral, v_topo, i)
    return hybrid


def build_k_only_cache(topology_kv: DynamicCache, neutral_kv: DynamicCache) -> DynamicCache:
    """K from topology, V from neutral."""
    hybrid = DynamicCache()
    for i in range(len(topology_kv.layers)):
        k_topo, _ = topology_kv[i]
        _, v_neutral = neutral_kv[i]
        hybrid.update(k_topo, v_neutral, i)
    return hybrid


def generate_with_cache(model, tokenizer, query: str,
                        past_kv=None, prefix_len: int = 0,
                        max_new_tokens: int = 50) -> str:
    """Generate text with optional KV cache injection.

    Uses manual autoregressive loop to avoid HF generate() API
    incompatibilities with raw KV tuple injection.
    """
    input_ids = tokenizer(query, return_tensors="pt")["input_ids"]

    if past_kv is None:
        # No injection — simple forward generation
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Manual autoregressive loop with injected KV cache
    # CRITICAL: deep-copy the cache — DynamicCache is mutated in-place by model()
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
            pos_start = prefix_len
            pos_ids = torch.arange(
                pos_start, pos_start + current_ids.shape[1], dtype=torch.long
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


def score_answer(response: str, expected: str) -> float:
    if not response:
        return 0.0
    return 1.0 if expected.lower() in response.lower() else 0.0


def run_condition(model, tokenizer, condition: str,
                  past_kv=None, prefix_len: int = 0,
                  context_text: str = "") -> ExperimentResult:
    """Run all questions under a single condition."""
    log.info(f"\n{'='*50}")
    log.info(f"CONDITION: {condition}")
    log.info(f"{'='*50}")

    sanity_details = []
    for sq in SANITY_QUESTIONS:
        if context_text:
            prompt = f"{context_text}\n\nComplete: {sq['q']}"
        else:
            prompt = f"Complete: {sq['q']}"

        response = generate_with_cache(
            model, tokenizer, prompt,
            past_kv=past_kv, prefix_len=prefix_len,
        )
        sc = score_answer(response, sq["answer"])
        sanity_details.append({"q": sq["q"], "answer": sq["answer"], "response": response[:80], "score": sc})
        log.info(f"  [{'Y' if sc else 'N'}] {sq['q'][:40]} → {response[:40]}")

    multihop_details = []
    for mq in MULTIHOP_QUESTIONS:
        if context_text:
            prompt = f"{context_text}\n\nAnswer concisely: {mq['q']}"
        else:
            prompt = f"Answer concisely: {mq['q']}"

        response = generate_with_cache(
            model, tokenizer, prompt,
            past_kv=past_kv, prefix_len=prefix_len,
        )
        sc = score_answer(response, mq["answer"])
        multihop_details.append({
            "q": mq["q"], "answer": mq["answer"], "response": response[:80],
            "score": sc, "hops": mq["hops"], "type": mq["type"],
        })
        log.info(f"  [{'Y' if sc else 'N'}] ({mq['type']}) {mq['q'][:40]} → {response[:40]}")

    sanity_avg = sum(d["score"] for d in sanity_details) / len(sanity_details)
    multihop_avg = sum(d["score"] for d in multihop_details) / len(multihop_details)

    log.info(f"\n  Sanity: {sanity_avg:.3f}  Multi-hop: {multihop_avg:.3f}")

    return ExperimentResult(
        condition=condition,
        sanity_score=sanity_avg,
        multihop_score=multihop_avg,
        sanity_details=sanity_details,
        multihop_details=multihop_details,
    )


def run_experiment():
    model, tokenizer = load_model()

    # Pre-compute KV caches
    log.info("\nEncoding topology text...")
    topology_kv, topo_len = encode_to_kv(model, tokenizer, GRAPH_TOPOLOGY_TEXT)
    log.info(f"  Topology cache: {topo_len} tokens, {len(topology_kv)} layers")

    log.info("Encoding neutral text...")
    neutral_kv, neutral_len = encode_to_kv(model, tokenizer, NEUTRAL_TEXT)
    log.info(f"  Neutral cache: {neutral_len} tokens")

    # Build neutral cache matching topology length
    # Use repeated neutral text to fill the same token count
    log.info("Building length-matched neutral cache...")
    neutral_base_tokens = tokenizer(NEUTRAL_TEXT, return_tensors="pt")["input_ids"]
    base_len = neutral_base_tokens.shape[1]
    repeats_needed = (topo_len // base_len) + 1
    repeated_text = (NEUTRAL_TEXT + " ") * repeats_needed
    repeated_ids = tokenizer(repeated_text, return_tensors="pt")["input_ids"][:, :topo_len]

    with torch.no_grad():
        out = model(input_ids=repeated_ids, use_cache=True, return_dict=True)
    neutral_matched_kv = out.past_key_values
    neutral_matched_len = repeated_ids.shape[1]
    log.info(f"  Matched neutral cache: {neutral_matched_len} tokens (topo={topo_len})")

    # Build hybrid caches
    v_only_kv = build_v_only_cache(topology_kv, neutral_matched_kv)
    k_only_kv = build_k_only_cache(topology_kv, neutral_matched_kv)

    results = []

    # 1. BASELINE
    results.append(run_condition(model, tokenizer, "BASELINE"))

    # 2. FULL_KV
    results.append(run_condition(
        model, tokenizer, "FULL_KV",
        past_kv=topology_kv, prefix_len=topo_len,
    ))

    # 3. V_ONLY
    results.append(run_condition(
        model, tokenizer, "V_ONLY",
        past_kv=v_only_kv, prefix_len=topo_len,
    ))

    # 4. K_ONLY
    results.append(run_condition(
        model, tokenizer, "K_ONLY",
        past_kv=k_only_kv, prefix_len=topo_len,
    ))

    # 5. TEXT_CONTEXT
    results.append(run_condition(
        model, tokenizer, "TEXT_CONTEXT",
        context_text=GRAPH_TOPOLOGY_TEXT,
    ))

    # Report
    log.info(f"\n{'='*60}")
    log.info("RESULTS SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"{'Condition':<15} {'Sanity':>8} {'Multi-hop':>10} {'Overall':>8}")
    log.info(f"{'-'*15} {'-'*8} {'-'*10} {'-'*8}")

    for r in results:
        overall = (r.sanity_score + r.multihop_score) / 2
        log.info(f"{r.condition:<15} {r.sanity_score:>8.3f} {r.multihop_score:>10.3f} {overall:>8.3f}")

    # Deltas from baseline
    baseline = results[0]
    log.info(f"\n{'Condition':<15} {'Sanity Δ':>10} {'Multi-hop Δ':>12}")
    log.info(f"{'-'*15} {'-'*10} {'-'*12}")
    for r in results[1:]:
        sd = r.sanity_score - baseline.sanity_score
        md = r.multihop_score - baseline.multihop_score
        log.info(f"{r.condition:<15} {sd:>+10.3f} {md:>+12.3f}")

    # Save
    output = {
        "experiment": "value_only_injection",
        "model": MODEL_NAME,
        "timestamp": time.time(),
        "results": [
            {
                "condition": r.condition,
                "sanity_score": r.sanity_score,
                "multihop_score": r.multihop_score,
                "sanity_details": r.sanity_details,
                "multihop_details": r.multihop_details,
            }
            for r in results
        ],
    }

    outdir = Path("experiment_results")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"v_only_{int(time.time())}.json"
    outfile.write_text(json.dumps(output, indent=2))
    log.info(f"\nResults saved to {outfile}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[v-only] %(message)s")
    run_experiment()
