"""Full depth profile of J-space coupling: PC1 vs random directions at every lens layer.

Tests whether the "ghost regime" (low J-space coupling) is depth-specific.
At each layer where a J-lens Jacobian exists, measures:
  - PC1's logit-lens/J-lens cosine
  - 50 random directions' cosines (null distribution)

Usage on Starship:
    python experiments/ghost_probe/depth_coupling_profile.py \
        --jlens-path ~/jlens-community/lenses/qwen3.5-27b_jlens.pt \
        --output data/ghost_depth_coupling_profile.json
"""

import argparse, json, sys, time
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

N_RANDOM = 50


def compute_cosine(model, lens, direction, layer):
    ll_logits = model.unembed(direction.unsqueeze(0)).squeeze(0).float()
    ll_probs = torch.softmax(ll_logits, dim=-1)
    transported = lens.transport(direction.cpu().float().unsqueeze(0), layer)
    jl_logits = model.unembed(transported.to(direction.device)).squeeze(0).float()
    jl_probs = torch.softmax(jl_logits, dim=-1)
    return torch.nn.functional.cosine_similarity(
        ll_probs.unsqueeze(0), jl_probs.unsqueeze(0)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jlens-path', required=True)
    parser.add_argument('--output', default='data/ghost_depth_coupling_profile.json')
    args = parser.parse_args()

    t0 = time.time()
    print("=== DEPTH COUPLING PROFILE ===")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-27B", torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
    d_model = hf_model.config.hidden_size
    n_layers = hf_model.config.num_hidden_layers

    model = HFLensModel(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(args.jlens_path)

    available_layers = sorted(lens.jacobians.keys())
    print(f"Lens layers: {available_layers} ({len(available_layers)} of {n_layers})")

    # Calibrate PCA at every available layer
    print(f"Calibrating on {len(CALIBRATION_PROMPTS)} prompts...")
    all_hidden = {layer: [] for layer in available_layers}

    for prompt in CALIBRATION_PROMPTS:
        input_ids = model.encode(prompt, max_length=64)
        with ActivationRecorder(model.layers, at=available_layers) as rec:
            model.forward(input_ids)
            for layer in available_layers:
                h = rec.activations[layer][0].detach().float()
                all_hidden[layer].append(h.mean(dim=0))

    results = {"model": "Qwen3.5-27B", "n_layers": n_layers,
               "n_random": N_RANDOM, "per_layer": []}

    for layer in available_layers:
        stacked = torch.stack(all_hidden[layer])
        mean = stacked.mean(dim=0)
        centered = stacked - mean
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        pc1 = Vt[0]
        pc1_var = (S[0]**2 / (S**2).sum()).item()

        pc1_cos = compute_cosine(model, lens, pc1, layer)

        rand_cosines = []
        for _ in range(N_RANDOM):
            rd = torch.randn(d_model, device=pc1.device, dtype=torch.float32)
            rd = rd / rd.norm()
            rand_cosines.append(compute_cosine(model, lens, rd, layer))

        ra = np.array(rand_cosines)
        depth_pct = round(layer / n_layers * 100, 1)

        entry = {
            "layer": layer, "depth_pct": depth_pct,
            "pc1_cosine": round(pc1_cos, 6),
            "pc1_variance_pct": round(pc1_var * 100, 1),
            "null_mean": round(float(ra.mean()), 6),
            "null_sd": round(float(ra.std()), 6),
            "null_5th": round(float(np.percentile(ra, 5)), 6),
            "null_95th": round(float(np.percentile(ra, 95)), 6),
            "pc1_percentile": round(float((ra < pc1_cos).mean() * 100), 1),
        }
        results["per_layer"].append(entry)

        print(f"L{layer:2d} ({depth_pct:4.1f}%): PC1={pc1_cos:.4f} null={ra.mean():.4f}±{ra.std():.4f} "
              f"[5th={np.percentile(ra,5):.4f} 95th={np.percentile(ra,95):.4f}] "
              f"PC1@{(ra < pc1_cos).mean()*100:.0f}th pct")

    results["elapsed_minutes"] = round((time.time() - t0) / 60, 1)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output} ({results['elapsed_minutes']:.1f} min)")


if __name__ == '__main__':
    main()
