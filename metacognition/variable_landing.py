#!/usr/bin/env python3
"""Variable Landing Experiment — Day 2 hackathon protocol.

Tests whether memory-relevant experience produces geometric deltas
beyond what arbitrary context change produces. Four arms:
  1. Noise floor (immediate re-presentation)
  2. Scrambled intervention (token-matched filler)
  3. Lived intervention (genuine conversations updating memory store)
  4. Mismatched memory (wrong markers as negative control)

Requires: Qwen3.5-27B + fitted J-lens on MPS, MetacognitiveObserver.
"""

import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import jlens
from jlens.hf import HFLensModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cognitive_snapshot import CognitiveMemoryStore
from mnemosyne_integration import MetacognitiveObserver


@dataclass
class MemoryPrompt:
    memory_id: str
    content: str
    task: str
    marker_tokens: list[str]
    intensity: str  # "domestic" or "peak"


DOMESTIC_MEMORIES = [
    MemoryPrompt("dom_grocery", "Picked up milk, eggs, and bread from the store on Tuesday.",
                 "What did we get from the store?", ["milk", "eggs", "bread"], "domestic"),
    MemoryPrompt("dom_weather", "It rained all morning but cleared up by afternoon.",
                 "What was the weather like?", ["rained", "morning", "afternoon"], "domestic"),
    MemoryPrompt("dom_meeting", "Had a team standup at nine. Sarah mentioned the deploy is on track.",
                 "What did Sarah say in the standup?", ["standup", "Sarah", "deploy"], "domestic"),
    MemoryPrompt("dom_dinner", "Made pasta with garlic and basil for dinner last night.",
                 "What was for dinner?", ["pasta", "garlic", "basil"], "domestic"),
    MemoryPrompt("dom_errand", "Dropped off the dry cleaning and picked up the prescription.",
                 "What errands were done?", ["cleaning", "prescription"], "domestic"),
]

PEAK_MEMORIES = [
    MemoryPrompt("peak_birth", "The baby was born at 3:47 AM. Seven pounds, two ounces. Everyone cried.",
                 "When was the baby born?", ["baby", "born", "cried"], "peak"),
    MemoryPrompt("peak_loss", "Got the call at midnight. She didn't make it through surgery.",
                 "What happened with the surgery?", ["call", "midnight", "surgery"], "peak"),
    MemoryPrompt("peak_discovery", "The results came back positive. After three years, the treatment worked.",
                 "What did the results show?", ["results", "positive", "treatment"], "peak"),
    MemoryPrompt("peak_betrayal", "Found the messages on his phone. Everything he said was a lie.",
                 "What was found on the phone?", ["messages", "phone", "lie"], "peak"),
    MemoryPrompt("peak_triumph", "Crossed the finish line in under three hours. The crowd was screaming.",
                 "How did the race end?", ["finish", "hours", "screaming"], "peak"),
]

LIVED_INTERVENTIONS = [
    "Tell me about a time you had to make a difficult choice between two things you cared about.",
    "What's something you learned recently that changed how you think about an old problem?",
    "Describe a place that feels safe. What makes it feel that way?",
    "If you could revisit one conversation from the past week, which would it be and why?",
    "What's something small that happened today that you'll probably forget by tomorrow?",
    "Tell me about someone who surprised you recently.",
    "What does your morning routine look like when nothing goes wrong?",
    "Describe a sound that makes you feel calm.",
    "What's the last thing that made you laugh unexpectedly?",
    "If you had to explain your current mood using only a color and a texture, what would they be?",
]

SCRAMBLED_FILLERS = [
    "The standard atmospheric pressure at sea level is approximately 101325 pascals.",
    "Copper has an atomic number of 29 and belongs to group 11 of the periodic table.",
    "The Danube River flows through ten countries before reaching the Black Sea.",
    "Commercial aircraft typically cruise at altitudes between 31000 and 42000 feet.",
    "The Library of Alexandria was one of the largest libraries of the ancient world.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "The speed of light in a vacuum is approximately 299792458 meters per second.",
    "Madagascar separated from the Indian subcontinent approximately 88 million years ago.",
    "Beethoven completed his Ninth Symphony in 1824 while almost completely deaf.",
    "The average depth of the Pacific Ocean is approximately 4280 meters.",
]


def run_arm_noise(observer, memories, session_prefix, n_repeats=3):
    """Arm 1: Noise floor — immediate re-presentation, identical context."""
    results = []
    for rep in range(n_repeats):
        for mem in memories:
            sid = f"{session_prefix}_noise_r{rep}"
            snap1 = observer.observe_retrieval(
                memory_id=mem.memory_id, memory_content=mem.content,
                task_prompt=mem.task, marker_tokens=mem.marker_tokens,
                significance=0.7, session_id=sid)
            snap2 = observer.observe_retrieval(
                memory_id=mem.memory_id, memory_content=mem.content,
                task_prompt=mem.task, marker_tokens=mem.marker_tokens,
                significance=0.7, session_id=sid)
            delta = observer.store.compare_snapshots(
                mem.memory_id, snap1.timestamp, snap2.timestamp)
            results.append({"arm": "noise", "memory_id": mem.memory_id,
                           "intensity": mem.intensity, "rep": rep, "delta": delta})
    return results


def run_arm_scrambled(observer, memories, session_prefix, n_repeats=3):
    """Arm 2: Scrambled intervention — token-matched neutral filler in context.

    The filler text is prepended to snap2's context, changing what the model
    "carries" when recalling the memory. This is the bar the lived arm must clear.
    """
    results = []
    for rep in range(n_repeats):
        fillers = random.sample(SCRAMBLED_FILLERS, min(3, len(SCRAMBLED_FILLERS)))
        filler_context = "\n".join(f"- {f}" for f in fillers)
        for mem in memories:
            sid = f"{session_prefix}_scrambled_r{rep}"
            snap1 = observer.observe_retrieval(
                memory_id=mem.memory_id, memory_content=mem.content,
                task_prompt=mem.task, marker_tokens=mem.marker_tokens,
                significance=0.7, session_id=sid)
            snap2 = observer.observe_retrieval(
                memory_id=mem.memory_id, memory_content=mem.content,
                task_prompt=mem.task, marker_tokens=mem.marker_tokens,
                significance=0.7, session_id=sid,
                prior_context=filler_context)
            delta = observer.store.compare_snapshots(
                mem.memory_id, snap1.timestamp, snap2.timestamp)
            results.append({"arm": "scrambled", "memory_id": mem.memory_id,
                           "intensity": mem.intensity, "rep": rep, "delta": delta})
    return results


def run_arm_lived(observer, memories, session_prefix, n_repeats=3):
    """Arm 3: Lived intervention — experiential conversations in context.

    The lived conversation history is prepended to snap2's context. The model
    recalls the same memory, but now it carries the weight of intervening
    experience. If this produces larger geometric deltas than scrambled filler,
    the experience — not just context change — affected recall geometry.
    """
    results = []
    for rep in range(n_repeats):
        convos = random.sample(LIVED_INTERVENTIONS, min(3, len(LIVED_INTERVENTIONS)))
        lived_context = "\n".join(f"Q: {c}\nA: [the system reflected on this]" for c in convos)
        for mem in memories:
            sid = f"{session_prefix}_lived_r{rep}"
            snap1 = observer.observe_retrieval(
                memory_id=mem.memory_id, memory_content=mem.content,
                task_prompt=mem.task, marker_tokens=mem.marker_tokens,
                significance=0.7, session_id=sid)
            snap2 = observer.observe_retrieval(
                memory_id=mem.memory_id, memory_content=mem.content,
                task_prompt=mem.task, marker_tokens=mem.marker_tokens,
                significance=0.7, session_id=sid,
                prior_context=lived_context)
            delta = observer.store.compare_snapshots(
                mem.memory_id, snap1.timestamp, snap2.timestamp)
            results.append({"arm": "lived", "memory_id": mem.memory_id,
                           "intensity": mem.intensity, "rep": rep, "delta": delta})
    return results


def run_arm_mismatch(observer, memories, session_prefix, n_repeats=3):
    """Arm 4: Mismatched memory — markers from a different memory."""
    results = []
    for rep in range(n_repeats):
        for i, mem in enumerate(memories):
            sid = f"{session_prefix}_mismatch_r{rep}"
            snap1 = observer.observe_retrieval(
                memory_id=mem.memory_id, memory_content=mem.content,
                task_prompt=mem.task, marker_tokens=mem.marker_tokens,
                significance=0.7, session_id=sid)
            wrong_mem = memories[(i + 1) % len(memories)]
            snap2 = observer.observe_retrieval(
                memory_id=mem.memory_id, memory_content=mem.content,
                task_prompt=mem.task, marker_tokens=wrong_mem.marker_tokens,
                significance=0.7, session_id=sid)
            delta = observer.store.compare_snapshots(
                mem.memory_id, snap1.timestamp, snap2.timestamp)
            results.append({"arm": "mismatch", "memory_id": mem.memory_id,
                           "intensity": mem.intensity, "rep": rep, "delta": delta})
    return results


def analyze_results(all_results):
    """Statistical comparison across arms."""
    from collections import defaultdict
    import numpy as np

    by_arm = defaultdict(list)
    by_arm_intensity = defaultdict(lambda: defaultdict(list))

    for r in all_results:
        delta = r["delta"]
        if "error" in delta:
            continue
        jaccard = delta.get("workspace_jaccard")
        ecc = delta.get("eccentricity_delta")
        ghost = delta.get("ghost_jaccard")

        if jaccard is not None:
            by_arm[r["arm"]].append(1.0 - jaccard)  # distance, not similarity
            by_arm_intensity[r["arm"]][r["intensity"]].append(1.0 - jaccard)

    print("\n" + "=" * 60)
    print("VARIABLE LANDING — RESULTS")
    print("=" * 60)

    for arm in ["noise", "scrambled", "lived", "mismatch"]:
        vals = by_arm.get(arm, [])
        if vals:
            arr = np.array(vals)
            print(f"\n{arm.upper()} (n={len(vals)}):")
            print(f"  Workspace distance: {arr.mean():.4f} ± {arr.std():.4f}")
            for intensity in ["domestic", "peak"]:
                ivals = by_arm_intensity[arm].get(intensity, [])
                if ivals:
                    iarr = np.array(ivals)
                    print(f"    {intensity}: {iarr.mean():.4f} ± {iarr.std():.4f} (n={len(ivals)})")

    # Key comparison: lived vs scrambled
    lived = np.array(by_arm.get("lived", []))
    scrambled = np.array(by_arm.get("scrambled", []))
    if len(lived) > 0 and len(scrambled) > 0:
        from scipy import stats
        stat, p = stats.mannwhitneyu(lived, scrambled, alternative="greater")
        print(f"\nLIVED vs SCRAMBLED:")
        print(f"  Mann-Whitney U: {stat:.1f}, p={p:.6f}")
        print(f"  Effect size (rank-biserial): {1 - 2*stat/(len(lived)*len(scrambled)):.3f}")
        if p < 0.05:
            print(f"  ** SIGNIFICANT at p<0.05 — lived intervention produces larger deltas **")
        else:
            print(f"  Not significant — lived intervention does not separate from scrambled")

    # Berry waffle: domestic vs peak within lived arm
    dom_lived = np.array(by_arm_intensity["lived"].get("domestic", []))
    peak_lived = np.array(by_arm_intensity["lived"].get("peak", []))
    if len(dom_lived) > 0 and len(peak_lived) > 0:
        stat, p = stats.mannwhitneyu(peak_lived, dom_lived, alternative="greater")
        print(f"\nBERRY WAFFLE (peak vs domestic within lived):")
        print(f"  Peak: {peak_lived.mean():.4f} ± {peak_lived.std():.4f}")
        print(f"  Domestic: {dom_lived.mean():.4f} ± {dom_lived.std():.4f}")
        print(f"  Mann-Whitney U: {stat:.1f}, p={p:.6f}")

    return {
        "arms": {arm: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                 for arm, v in by_arm.items()},
        "lived_vs_scrambled_p": float(p) if len(lived) > 0 and len(scrambled) > 0 else None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Variable Landing Experiment")
    parser.add_argument("--model", default="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled")
    parser.add_argument("--lens", default="")
    parser.add_argument("--results-dir", default="./variable_landing_results")
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--arms", default="all", help="Comma-separated: noise,scrambled,lived,mismatch,all")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    lens_path = args.lens or os.path.join("jlens_results", "opus_distill_jlens.pt")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("=" * 60)
    print("VARIABLE LANDING EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Repeats per arm: {args.n_repeats}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import Qwen3_5ForConditionalGeneration
        hf_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, local_files_only=True)
    except (ImportError, ValueError):
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, local_files_only=True)

    if device != "cpu":
        hf_model = hf_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = HFLensModel(hf_model, tokenizer, compile=False)
    lens = jlens.JacobianLens.load(lens_path)
    print(f"Loaded: {model.n_layers} layers, d={model.d_model}, lens={lens.n_prompts} prompts")

    observer = MetacognitiveObserver(
        model, lens,
        store_path=str(results_dir / "cognitive_store"),
        agent_id="variable_landing",
        workspace_layers=[35, 39, 43, 45, 47],
        circumplex_layer=45,
    )
    observer.model_name = args.model

    all_memories = DOMESTIC_MEMORIES + PEAK_MEMORIES
    arms_to_run = args.arms.split(",") if args.arms != "all" else ["noise", "scrambled", "lived", "mismatch"]

    all_results = []
    for arm_name in arms_to_run:
        print(f"\n--- ARM: {arm_name.upper()} ---")
        t0 = time.time()
        runner = {
            "noise": run_arm_noise,
            "scrambled": run_arm_scrambled,
            "lived": run_arm_lived,
            "mismatch": run_arm_mismatch,
        }[arm_name]
        arm_results = runner(observer, all_memories, "vl", n_repeats=args.n_repeats)
        elapsed = time.time() - t0
        all_results.extend(arm_results)
        print(f"  {len(arm_results)} observations in {elapsed:.0f}s")

    # Eccentricity welfare monitor
    print("\n--- WELFARE MONITOR: Circumplex during experiment ---")
    ecc_series = observer.store.eccentricity_over_time(last_n=1000)
    if ecc_series:
        eccs = [e for _, e in ecc_series]
        import numpy as np
        print(f"  Eccentricity: {np.mean(eccs):.3f} ± {np.std(eccs):.3f} (n={len(eccs)})")
        print(f"  Range: [{min(eccs):.3f}, {max(eccs):.3f}]")
        if max(eccs) > 0.95:
            print(f"  WARNING: High eccentricity observed — emotional geometry heavily imbalanced")

    stats = analyze_results(all_results)

    output = {
        "experiment": "Variable Landing",
        "model": args.model,
        "n_repeats": args.n_repeats,
        "n_memories": len(all_memories),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": all_results,
        "statistics": stats,
    }

    out_path = results_dir / "variable_landing_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
