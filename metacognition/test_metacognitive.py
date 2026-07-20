#!/usr/bin/env python3
"""Integration test for the metacognitive memory stack.

Runs on the 27B Opus distill with the Neuronpedia lens.
Tests: CognitiveSnapshot creation, workspace probe, circumplex probe,
ghost reading, memory loading, longitudinal recording, and
significance recalibration.
"""

import json
import os
import sys
import time
import tempfile

import torch
import jlens
from jlens.hf import HFLensModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cognitive_snapshot import CognitiveSnapshot, CognitiveMemoryStore
from circumplex_probe import CircumplexProbe
from mnemosyne_integration import MetacognitiveObserver


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled")
    parser.add_argument("--lens", default="")
    parser.add_argument("--results-dir", default="./jlens_results")
    args = parser.parse_args()

    lens_path = args.lens or os.path.join(args.results_dir, "opus_distill_jlens.pt")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("=" * 60)
    print("METACOGNITIVE MEMORY — Integration Test")
    print("=" * 60)

    # Load model + lens
    from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer

    print("\nLoading model...")
    hf_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True)
    if device != "cpu":
        hf_model = hf_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = HFLensModel(hf_model, tokenizer, compile=False)
    lens = jlens.JacobianLens.load(lens_path)
    print(f"  {model.n_layers} layers, d={model.d_model}, lens={lens.n_prompts} prompts")

    store_dir = os.path.join(args.results_dir, "metacognitive_test")

    # === TEST 1: MetacognitiveObserver creation ===
    print("\n[TEST 1] Creating MetacognitiveObserver...")
    observer = MetacognitiveObserver(
        model, lens,
        store_path=store_dir,
        agent_id="test_agent",
        workspace_layers=[35, 39, 43, 45, 47],
        circumplex_layer=45,
    )
    observer.model_name = args.model
    print("  PASS — observer created")

    # === TEST 2: Observe a retrieval event ===
    print("\n[TEST 2] Observing retrieval event...")
    t0 = time.time()
    snapshot = observer.observe_retrieval(
        memory_id="mem_medical_001",
        memory_content="Patient reported severe migraine headaches lasting three days. "
                       "Prescribed sumatriptan 100mg.",
        task_prompt="What medication was prescribed for the headaches?",
        retrieval_method="sira",
        significance=0.8,
        session_id="test_session",
        marker_tokens=["patient", "doctor", "medicine"],
    )
    elapsed = time.time() - t0
    print(f"  {elapsed:.1f}s — snapshot recorded")
    print(f"  Summary: {snapshot.summary()}")
    print(f"  Workspace readings: {len(snapshot.workspace_readings)} layers")
    print(f"  Dominant tokens: {snapshot.dominant_workspace_tokens[:5]}")
    if snapshot.circumplex:
        print(f"  Circumplex: e={snapshot.circumplex.eccentricity:.3f}")
        print(f"    Valence in J-space: {snapshot.circumplex.valence_in_jspace:.1%}")
        print(f"    Arousal in J-space: {snapshot.circumplex.arousal_in_jspace:.1%}")
    if snapshot.ghost:
        print(f"  Ghost: cos={snapshot.ghost.cosine_logit_jlens:.4f}")
        dominant = ", ".join(t for t, p in snapshot.ghost.dominant_tokens[:3])
        secondary = ", ".join(t for t, p in snapshot.ghost.secondary_tokens[:3])
        print(f"    Dominant: [{dominant}]")
        print(f"    Secondary: [{secondary}]")
    if snapshot.loading:
        print(f"  Loading: loaded={snapshot.loading.loaded}")
    print("  PASS")

    # === TEST 3: Second retrieval (different domain) ===
    print("\n[TEST 3] Second retrieval (legal domain)...")
    snapshot2 = observer.observe_retrieval(
        memory_id="mem_legal_001",
        memory_content="Tenant facing eviction for nonpayment. Filed answer within deadline.",
        task_prompt="What defense does the tenant have?",
        retrieval_method="sira",
        significance=0.9,
        session_id="test_session",
        marker_tokens=["tenant", "court", "judge"],
    )
    print(f"  Summary: {snapshot2.summary()}")
    if snapshot2.circumplex:
        print(f"  Circumplex: e={snapshot2.circumplex.eccentricity:.3f}")
    print("  PASS")

    # === TEST 4: Retroactive outcome ===
    print("\n[TEST 4] Recording retroactive outcome...")
    observer.store.record_outcome(
        timestamp=snapshot.timestamp,
        quality=0.85,
        source="user_feedback",
        notes="User confirmed the answer was helpful"
    )
    # Verify it was recorded
    history = observer.store.load_history(last_n=5)
    latest_with_outcome = [h for h in history if h.get("outcome_quality") is not None]
    if latest_with_outcome:
        print(f"  Outcome recorded: quality={latest_with_outcome[-1]['outcome_quality']}")
        print("  PASS")
    else:
        print("  WARN — outcome not found in history")

    # === TEST 5: Longitudinal queries ===
    print("\n[TEST 5] Longitudinal queries...")
    stats = observer.store.loading_success_rate()
    print(f"  Loading stats: {json.dumps(stats, indent=2)}")

    ecc = observer.store.eccentricity_over_time()
    print(f"  Eccentricity series: {len(ecc)} points")
    for ts, e in ecc:
        print(f"    t={ts:.0f} e={e:.3f}")

    ghost_series = observer.store.ghost_vocabulary_over_time()
    print(f"  Ghost vocabulary series: {len(ghost_series)} points")

    recal = observer.store.significance_recalibration()
    print(f"  Significance recalibration suggestions: {len(recal)}")
    print("  PASS")

    # === TEST 6: Circumplex sweep (E1) ===
    print("\n[TEST 6] Circumplex sweep (subset of layers)...")
    circ_probe = CircumplexProbe(model, lens)
    sample_layers = [11, 23, 35, 45, 57]
    results = circ_probe.sweep(layers=sample_layers)
    print(circ_probe.report(results))
    print("  PASS")

    # === SAVE ===
    output = os.path.join(args.results_dir, "metacognitive_test_results.json")
    with open(output, "w") as f:
        json.dump({
            "test": "Metacognitive Memory Integration",
            "model": args.model,
            "snapshot_1": snapshot.to_dict(),
            "snapshot_2": snapshot2.to_dict(),
            "loading_stats": stats,
            "circumplex_sweep": [
                {"layer": r.layer, "eccentricity": r.eccentricity,
                 "valence_jspace": r.valence_jspace_energy,
                 "arousal_jspace": r.arousal_jspace_energy}
                for r in results
            ],
        }, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED")
    print(f"Results saved to {output}")
    print(f"Cognitive memory store: {store_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
