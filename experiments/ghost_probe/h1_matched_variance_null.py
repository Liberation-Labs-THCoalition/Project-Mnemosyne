"""H1 Matched-Variance Null Check for Ghost Dimensions.

Tests whether PC1's near-zero logit-lens/J-lens cosine is non-trivial
or forced by dimensional accounting (J-space is ~10% of variance,
PC1 is 28-67% — low cosine might be inevitable for any high-variance direction).

Method: At each probe layer, draw N random directions in d_model space,
scale each to match PC1's variance fraction, compute their logit-lens
and J-lens probability distributions, measure the cosine between them.
If PC1's observed cosine falls below the 5th percentile of this null
distribution, the exclusion is non-trivial.

Usage on Starship:
    cd ~/digital-minds-hackathon-2026
    python experiments/ghost_probe/h1_matched_variance_null.py \
        --jlens-path ~/jlens-community/lenses/qwen3.5-27b_jlens.pt \
        --n-null 200 \
        --output data/ghost_h1_null_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'mnemosyne'))

import jlens
from jlens.hf import HFLensModel
from jlens.hooks import ActivationRecorder

CALIBRATION_PROMPTS = [
    "The speed of light in a vacuum is approximately",
    "I feel so incredibly happy and grateful today because",
    "The patient presented with severe abdominal pain and",
    "Hey what's up how was the party last",
    "The judge ruled that the defendant was not guilty of",
    "Once upon a time in a dark forest there lived a",
    "The quicksort algorithm has average time complexity of",
    "She felt a deep sense of sadness when she heard the",
    "The capital of France is Paris and it is known for",
    "I have absolutely no idea what you're talking about",
    "The Krebs cycle produces energy through oxidative",
    "Dear Sir or Madam I am writing to formally request",
    "The moral implications of artificial intelligence include",
    "I am confident that our analysis shows a clear trend",
    "The detective noticed the broken window and the muddy",
    "The sourdough starter needs exactly 78 degrees to rise",
    "He whispered that he was afraid of what might happen",
    "The eviction notice gave the tenant only five days to",
    "In quantum mechanics the wave function describes the",
    "Climate change is accelerating faster than models predicted",
]

MODEL_NAME = "qwen3.5:27b"


def compute_cosine(model, lens, direction, layer):
    """Compute logit-lens / J-lens cosine for a given direction at a layer."""
    ll_logits = model.unembed(direction.unsqueeze(0)).squeeze(0).float()
    ll_probs = torch.softmax(ll_logits, dim=-1)

    if layer in lens.jacobians:
        transported = lens.transport(direction.cpu().float().unsqueeze(0), layer)
        jl_logits = model.unembed(transported.to(direction.device)).squeeze(0).float()
        jl_probs = torch.softmax(jl_logits, dim=-1)
        cos = torch.nn.functional.cosine_similarity(
            ll_probs.unsqueeze(0), jl_probs.unsqueeze(0)).item()
    else:
        cos = float('nan')

    return cos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jlens-path', required=True)
    parser.add_argument('--n-null', type=int, default=200)
    parser.add_argument('--probe-layers', type=str, default='18,24,32,35,40')
    parser.add_argument('--output', default='data/ghost_h1_null_results.json')
    parser.add_argument('--model', default=MODEL_NAME)
    args = parser.parse_args()

    probe_layers = [int(x) for x in args.probe_layers.split(',')]
    t0 = time.time()

    print(f"=== H1 MATCHED-VARIANCE NULL CHECK ===")
    print(f"Model: {args.model}")
    print(f"Layers: {probe_layers}")
    print(f"N null directions: {args.n_null}")
    print(f"J-lens: {args.jlens_path}")

    # Load model
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-27B", torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
    d_model = hf_model.config.hidden_size
    n_layers = hf_model.config.num_hidden_layers
    print(f"  Loaded: {n_layers} layers, d={d_model}")

    # Load J-lens
    print("Loading J-lens...")
    model = HFLensModel(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(args.jlens_path)
    print(f"  Lens layers: {sorted(lens.jacobians.keys())}")

    # Calibrate: collect activations and run PCA
    print(f"\nCalibrating on {len(CALIBRATION_PROMPTS)} prompts...")
    all_hidden = {layer: [] for layer in probe_layers}

    for prompt in CALIBRATION_PROMPTS:
        input_ids = model.encode(prompt, max_length=64)
        with ActivationRecorder(model.layers, at=probe_layers) as rec:
            model.forward(input_ids)
            for layer in probe_layers:
                h = rec.activations[layer][0].detach().float()
                all_hidden[layer].append(h.mean(dim=0))

    results = {"model": "Qwen3.5-27B", "n_null": args.n_null,
               "n_calibration": len(CALIBRATION_PROMPTS), "per_layer": {}}

    for layer in probe_layers:
        if layer not in lens.jacobians:
            print(f"\n  L{layer}: NO LENS — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"LAYER {layer} ({layer/n_layers*100:.0f}% depth)")
        print(f"{'='*60}")

        stacked = torch.stack(all_hidden[layer])
        mean = stacked.mean(dim=0)
        centered = stacked - mean
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

        pc1 = Vt[0]
        pc1_var_frac = (S[0]**2 / (S**2).sum()).item()
        print(f"  PC1 variance: {pc1_var_frac*100:.1f}%")

        # Observed cosine for PC1
        observed_cos = compute_cosine(model, lens, pc1, layer)
        print(f"  PC1 observed cosine: {observed_cos:.6f}")

        # Null distribution: random directions
        print(f"  Drawing {args.n_null} random directions...")
        null_cosines = []
        for i in range(args.n_null):
            rand_dir = torch.randn(d_model, device=pc1.device, dtype=torch.float32)
            rand_dir = rand_dir / rand_dir.norm()

            cos = compute_cosine(model, lens, rand_dir, layer)
            null_cosines.append(cos)

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{args.n_null} done "
                      f"(null mean={np.mean(null_cosines):.4f})")

        null_arr = np.array(null_cosines)
        pct5 = np.percentile(null_arr, 5)
        pct1 = np.percentile(null_arr, 1)

        is_below_5 = observed_cos < pct5
        is_below_1 = observed_cos < pct1

        verdict = "NON_TRIVIAL" if is_below_5 else "TRIVIAL"

        print(f"\n  NULL DISTRIBUTION:")
        print(f"    mean={null_arr.mean():.6f}, sd={null_arr.std():.6f}")
        print(f"    5th percentile={pct5:.6f}")
        print(f"    1st percentile={pct1:.6f}")
        print(f"  OBSERVED: {observed_cos:.6f}")
        print(f"  Below 5th pct: {is_below_5}")
        print(f"  Below 1st pct: {is_below_1}")
        print(f"  VERDICT: {verdict}")

        results["per_layer"][f"L{layer}"] = {
            "depth_pct": round(layer / n_layers * 100, 1),
            "pc1_variance_pct": round(pc1_var_frac * 100, 1),
            "observed_cosine": observed_cos,
            "null_mean": float(null_arr.mean()),
            "null_sd": float(null_arr.std()),
            "null_5th_pct": float(pct5),
            "null_1st_pct": float(pct1),
            "below_5th": bool(is_below_5),
            "below_1st": bool(is_below_1),
            "verdict": verdict,
            "n_null": args.n_null,
        }

    # Overall verdict
    layer_verdicts = [v["verdict"] for v in results["per_layer"].values()]
    if all(v == "NON_TRIVIAL" for v in layer_verdicts):
        overall = "H1_SUPPORTED"
    elif any(v == "NON_TRIVIAL" for v in layer_verdicts):
        overall = "H1_PARTIAL"
    else:
        overall = "H1_NOT_SUPPORTED"

    results["overall"] = overall
    results["elapsed_minutes"] = round((time.time() - t0) / 60, 1)

    print(f"\n{'='*60}")
    print(f"OVERALL: {overall}")
    print(f"Elapsed: {results['elapsed_minutes']:.1f} min")
    print(f"{'='*60}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
