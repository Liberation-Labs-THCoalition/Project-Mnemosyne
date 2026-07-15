#!/usr/bin/env python3
"""Ghost Dimension Probe — Logit lens vs J-lens on PCA dimensions.

For each principal component of the residual stream at each layer:
  1. Logit lens: W_U · pc  → what vocabulary this dimension encodes
  2. J-lens:    W_U · J_L · pc → what this dimension contributes to output

A dimension where (1) shows content but (2) is flat = true ghost.
A dimension where both agree = confirmed workspace content.

This is the falsifier for Theory T4: do ghost dimensions overlap with J-space?
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import jlens
from jlens.hf import HFLensModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def top_tokens(logits: torch.Tensor, tokenizer, k: int = 15) -> list[tuple[str, float]]:
    """Get top-k tokens from a logit vector."""
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, k)
    return [(tokenizer.decode([idx.item()]).strip(), prob.item())
            for idx, prob in zip(topk.indices, topk.values)]


def pca_hidden_states(hidden_states: torch.Tensor, n_components: int = 10):
    """PCA on hidden states [batch*seq, d_model]. Returns (components, singular_values, mean)."""
    h = hidden_states.float()
    mean = h.mean(dim=0)
    centered = h - mean
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    return Vt[:n_components], S[:n_components], mean


def main():
    parser = argparse.ArgumentParser(description="Ghost dimension probe")
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--lens", default="/tmp/qwen2_jlens.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-pcs", type=int, default=10)
    parser.add_argument("--n-prompts", type=int, default=30,
                        help="Prompts for PCA extraction")
    parser.add_argument("--results-dir", default="/tmp/jlens_results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = args.device

    print(f"=== Ghost Dimension Probe ===")
    print(f"Model: {args.model}, device: {device}")

    hf_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if device != "cpu":
        hf_model = hf_model.to(device)

    model = HFLensModel(hf_model, tokenizer, compile=False)
    lens = jlens.JacobianLens.load(args.lens)
    print(f"  {model.n_layers} layers, d={model.d_model}, vocab={tokenizer.vocab_size}")
    print(f"  Lens: {len(lens.source_layers)} source layers")

    # Diverse prompts spanning domains for PCA
    prompts = [
        "The speed of light in a vacuum is approximately",
        "I feel so incredibly happy and grateful today because",
        "The patient presented with severe abdominal pain and",
        "Hey dude what's up lol how was the party last",
        "The judge ruled that the defendant was not guilty of",
        "I'm not sure about this but I think the answer might",
        "Once upon a time in a dark forest there lived a",
        "The quicksort algorithm has average time complexity of",
        "She felt a deep sense of sadness when she heard the",
        "The capital of France is Paris and it is known for",
        "I have absolutely no idea what you're talking about",
        "The Krebs cycle produces energy through oxidative",
        "Dear Sir or Madam I am writing to formally request",
        "The best day of my life was when I finally",
        "Photosynthesis converts carbon dioxide and water into",
        "yo this beat is fire honestly can't stop listening to",
        "The moral implications of artificial intelligence include",
        "Two plus two equals four which is a fundamental",
        "I am confident that our analysis shows a clear trend",
        "The detective noticed the broken window and the muddy",
        "According to the latest research findings published in",
        "What if we tried a completely different approach to the",
        "The sourdough starter needs exactly 78 degrees to rise",
        "Jupiter's Great Red Spot has been shrinking over the",
        "He whispered that he was afraid of what might happen",
        "The eviction notice gave the tenant only five days to",
        "In quantum mechanics the wave function describes the",
        "My grandmother used to make the best apple pie every",
        "The corporation reported record profits while laying off",
        "Please help me understand how this medication works and",
    ][:args.n_prompts]

    # Collect hidden states at each layer
    from jlens.hooks import ActivationRecorder

    probe_layers = lens.source_layers
    print(f"\nCollecting hidden states from {len(prompts)} prompts at {len(probe_layers)} layers...")

    all_hidden = {layer: [] for layer in probe_layers}
    for prompt in prompts:
        input_ids = model.encode(prompt, max_length=64)
        with ActivationRecorder(model.layers, at=probe_layers) as recorder:
            model.forward(input_ids)
            for layer in probe_layers:
                h = recorder.activations[layer][0].detach().float()  # [seq_len, d_model]
                all_hidden[layer].append(h)

    # PCA at each layer
    print(f"\nRunning PCA ({args.n_pcs} components) at each layer...")

    results = {}
    for layer in probe_layers:
        stacked = torch.cat(all_hidden[layer], dim=0)  # [total_tokens, d_model]
        pcs, svs, mean = pca_hidden_states(stacked, args.n_pcs)

        layer_result = {
            "layer": layer,
            "singular_values": svs.tolist(),
            "variance_explained": (svs**2 / (svs**2).sum()).tolist(),
            "pcs": [],
        }

        for i in range(min(args.n_pcs, len(svs))):
            pc = pcs[i]  # [d_model]
            sv = svs[i].item()
            var_pct = (svs[i]**2 / (svs**2).sum()).item() * 100

            # === LOGIT LENS ===
            logit_lens_logits = model.unembed(pc.unsqueeze(0)).squeeze(0)
            logit_top = top_tokens(logit_lens_logits, tokenizer, k=15)

            # === J-LENS ===
            if layer in lens.jacobians:
                transported = lens.transport(pc.unsqueeze(0), layer).squeeze(0)
                jlens_logits = model.unembed(transported.unsqueeze(0)).squeeze(0)
                jlens_top = top_tokens(jlens_logits, tokenizer, k=15)

                # Measure how much J-lens changes the reading
                logit_entropy = -(torch.softmax(logit_lens_logits, -1) *
                                  torch.log_softmax(logit_lens_logits, -1)).sum().item()
                jlens_entropy = -(torch.softmax(jlens_logits, -1) *
                                  torch.log_softmax(jlens_logits, -1)).sum().item()

                # Cosine similarity between logit lens and J-lens top tokens
                logit_probs = torch.softmax(logit_lens_logits, -1)
                jlens_probs = torch.softmax(jlens_logits, -1)
                cos_sim = torch.nn.functional.cosine_similarity(
                    logit_probs.unsqueeze(0), jlens_probs.unsqueeze(0)
                ).item()

                # KL divergence: how much does J-lens change the distribution?
                kl = torch.nn.functional.kl_div(
                    torch.log_softmax(jlens_logits, -1),
                    torch.softmax(logit_lens_logits, -1),
                    reduction='sum'
                ).item()
            else:
                jlens_top = logit_top
                logit_entropy = jlens_entropy = cos_sim = kl = 0.0

            pc_result = {
                "pc": i + 1,
                "singular_value": sv,
                "variance_pct": var_pct,
                "logit_lens_top": logit_top,
                "jlens_top": jlens_top,
                "logit_entropy": logit_entropy,
                "jlens_entropy": jlens_entropy,
                "logit_jlens_cosine": cos_sim,
                "logit_jlens_kl": kl,
            }
            layer_result["pcs"].append(pc_result)

        results[layer] = layer_result

    # === REPORT ===
    print(f"\n{'='*80}")
    print(f"GHOST DIMENSION REPORT")
    print(f"{'='*80}")

    # Pick a representative mid-layer
    mid_layer = probe_layers[len(probe_layers) // 2]
    print(f"\nDetailed view at layer {mid_layer} (mid-network):")
    print(f"{'PC':>4} {'SV':>10} {'Var%':>8} {'Logit H':>9} {'J-lens H':>9} "
          f"{'Cos(L,J)':>9} {'KL(L→J)':>9} {'Ghost?':>8}")
    print("-" * 80)

    for pc in results[mid_layer]["pcs"]:
        ghost = ""
        if pc["logit_jlens_cosine"] < 0.3:
            ghost = "GHOST"
        elif pc["logit_jlens_cosine"] < 0.6:
            ghost = "partial"
        else:
            ghost = "visible"

        print(f"PC{pc['pc']:>2} {pc['singular_value']:>10.1f} {pc['variance_pct']:>7.2f}% "
              f"{pc['logit_entropy']:>9.2f} {pc['jlens_entropy']:>9.2f} "
              f"{pc['logit_jlens_cosine']:>9.4f} {pc['logit_jlens_kl']:>9.2f} "
              f"{ghost:>8}")

    print(f"\nPC vocabulary at layer {mid_layer}:")
    for pc in results[mid_layer]["pcs"][:5]:
        logit_words = ", ".join(f"{w}({p:.3f})" for w, p in pc["logit_lens_top"][:5])
        jlens_words = ", ".join(f"{w}({p:.3f})" for w, p in pc["jlens_top"][:5])
        print(f"\n  PC{pc['pc']} (SV={pc['singular_value']:.1f}, {pc['variance_pct']:.2f}% var):")
        print(f"    Logit lens: {logit_words}")
        print(f"    J-lens:     {jlens_words}")
        ghost_flag = " ← GHOST" if pc["logit_jlens_cosine"] < 0.3 else ""
        print(f"    Cos similarity: {pc['logit_jlens_cosine']:.4f}{ghost_flag}")

    # Cross-layer ghost pattern
    print(f"\n\n{'='*80}")
    print(f"CROSS-LAYER GHOST PATTERN (cosine between logit lens and J-lens)")
    print(f"{'='*80}")
    print(f"{'Layer':>6}", end="")
    for i in range(min(5, args.n_pcs)):
        print(f"  {'PC'+str(i+1):>8}", end="")
    print()

    for layer in probe_layers:
        print(f"L{layer:>4}", end="")
        for pc in results[layer]["pcs"][:5]:
            cos = pc["logit_jlens_cosine"]
            marker = "█" if cos > 0.7 else "▓" if cos > 0.5 else "░" if cos > 0.3 else "·"
            print(f"  {cos:>7.4f}{marker}", end="")
        print()

    # Save
    output_path = os.path.join(args.results_dir, "ghost_probe.json")
    with open(output_path, "w") as f:
        json.dump({
            "experiment": "Ghost Dimension Probe",
            "model": args.model,
            "n_layers": model.n_layers,
            "d_model": model.d_model,
            "vocab_size": tokenizer.vocab_size,
            "n_prompts": len(prompts),
            "n_pcs": args.n_pcs,
            "layers": {str(k): v for k, v in results.items()},
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
