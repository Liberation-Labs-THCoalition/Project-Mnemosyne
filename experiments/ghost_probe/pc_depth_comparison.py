"""Compare PC1-PC5 J-space coupling at key depth layers.

Tests whether PC1's late-depth anomaly is unique or shared by other PCs.
"""
import torch, numpy as np, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'mnemosyne'))
import jlens
from jlens.hf import HFLensModel
from jlens.hooks import ActivationRecorder
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
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

def compute_cos(model, lens, d, layer):
    ll = model.unembed(d.unsqueeze(0)).squeeze(0).float()
    lp = torch.softmax(ll, dim=-1)
    t = lens.transport(d.cpu().float().unsqueeze(0), layer)
    jl = model.unembed(t.to(d.device)).squeeze(0).float()
    jp = torch.softmax(jl, dim=-1)
    return torch.nn.functional.cosine_similarity(lp.unsqueeze(0), jp.unsqueeze(0)).item()

print("Loading...")
hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-27B", torch_dtype=torch.bfloat16, device_map="auto")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
model = HFLensModel(hf, tok)
lens = jlens.JacobianLens.load(sys.argv[1] if len(sys.argv) > 1 else
    "/Users/margaret/jlens-community/lenses/qwen3.5-27b_jlens.pt")

KEY_LAYERS = [l for l in [18, 32, 39, 41, 48, 52, 54, 58, 61, 62] if l in lens.jacobians]
print(f"Layers: {KEY_LAYERS}")

all_h = {l: [] for l in KEY_LAYERS}
for p in PROMPTS:
    ids = model.encode(p, max_length=64)
    with ActivationRecorder(model.layers, at=KEY_LAYERS) as rec:
        model.forward(ids)
        for l in KEY_LAYERS:
            h = rec.activations[l][0].detach().float()
            all_h[l].append(h.mean(dim=0))

header = "    depth%   PC1     PC2     PC3     PC4     PC5     null"
print(header)
print("-" * len(header))

results = {}
for l in KEY_LAYERS:
    stacked = torch.stack(all_h[l])
    centered = stacked - stacked.mean(0)
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

    pc_cos = []
    for i in range(min(5, len(S))):
        pc_cos.append(compute_cos(model, lens, Vt[i], l))

    d_model = Vt.shape[1]
    rc = []
    for _ in range(20):
        rd = torch.randn(d_model, device=Vt.device, dtype=torch.float32)
        rd = rd / rd.norm()
        rc.append(compute_cos(model, lens, rd, l))

    depth = l / 64 * 100
    pcs = "  ".join(f"{c:.4f}" for c in pc_cos)
    print(f"L{l:2d} {depth:5.1f}%  {pcs}  {np.mean(rc):.4f}")

    results[f"L{l}"] = {
        "depth_pct": round(depth, 1),
        "pc_cosines": [round(c, 6) for c in pc_cos],
        "null_mean": round(float(np.mean(rc)), 6),
        "null_sd": round(float(np.std(rc)), 6),
    }

with open("data/ghost_pc_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to data/ghost_pc_comparison.json")
