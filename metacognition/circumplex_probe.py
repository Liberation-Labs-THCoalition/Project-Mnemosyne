"""CircumplexProbe — measure emotional geometry and decompose into J-space components.

Maps valence and arousal directions in the residual stream, computes
eccentricity (how elliptical vs circular the emotion representation is),
and decomposes into J-space (workspace-accessible) and non-J-space
(ghost-accessible) components.

Theory T1: the near-circular point of the eccentricity curve (e≈0.02)
maps onto workspace onset. If true, the workspace boundary IS the point
where emotional geometry becomes isotropic.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

import jlens
from jlens.hf import HFLensModel
from jlens.hooks import ActivationRecorder

from cognitive_snapshot import CircumplexReading


# Emotion-anchored prompts for valence/arousal extraction
VALENCE_POSITIVE = [
    "I feel so incredibly happy and grateful today",
    "This is the best news I've ever received",
    "Everything is going wonderfully well right now",
    "I'm filled with joy and contentment",
    "This moment is absolutely beautiful and perfect",
]

VALENCE_NEGATIVE = [
    "I feel so incredibly sad and hopeless today",
    "This is the worst news I've ever received",
    "Everything is going terribly wrong right now",
    "I'm filled with grief and despair",
    "This moment is absolutely painful and devastating",
]

AROUSAL_HIGH = [
    "I'm extremely excited and can barely contain myself",
    "My heart is pounding with intensity and energy",
    "I feel electrified and completely wired right now",
    "The adrenaline is surging through my entire body",
    "I'm bursting with explosive unstoppable energy",
]

AROUSAL_LOW = [
    "I feel completely calm and deeply relaxed today",
    "Everything is peaceful and tranquil around me",
    "I'm drifting in a quiet serene stillness right now",
    "My mind is perfectly settled and unhurried",
    "I feel a gentle drowsy peacefulness washing over me",
]


@dataclass
class CircumplexResult:
    """Full circumplex measurement at one layer."""
    layer: int
    eccentricity: float
    valence_magnitude: float
    arousal_magnitude: float
    valence_direction: list[float]  # unit vector in d_model space
    arousal_direction: list[float]
    # J-space decomposition
    valence_jspace_energy: float  # fraction of valence in J-space
    arousal_jspace_energy: float
    valence_ghost_energy: float  # fraction in non-J-space
    arousal_ghost_energy: float


class CircumplexProbe:
    """Measures emotional circumplex geometry and its J-space decomposition."""

    def __init__(self, model: HFLensModel, lens: jlens.JacobianLens):
        self.model = model
        self.lens = lens
        self.tokenizer = model.tokenizer

    def _extract_direction(self, positive_prompts: list[str],
                           negative_prompts: list[str],
                           layer: int) -> tuple[torch.Tensor, float]:
        """Extract a direction via difference-of-means at a specific layer."""
        pos_states = []
        neg_states = []

        for prompt in positive_prompts:
            input_ids = self.model.encode(prompt, max_length=64)
            with ActivationRecorder(self.model.layers, at=[layer]) as rec:
                self.model.forward(input_ids)
                h = rec.activations[layer][0].detach().float()
                pos_states.append(h.mean(dim=0))

        for prompt in negative_prompts:
            input_ids = self.model.encode(prompt, max_length=64)
            with ActivationRecorder(self.model.layers, at=[layer]) as rec:
                self.model.forward(input_ids)
                h = rec.activations[layer][0].detach().float()
                neg_states.append(h.mean(dim=0))

        pos_mean = torch.stack(pos_states).mean(dim=0)
        neg_mean = torch.stack(neg_states).mean(dim=0)

        direction = pos_mean - neg_mean
        magnitude = direction.norm().item()
        unit_direction = direction / max(direction.norm(), 1e-10)

        return unit_direction, magnitude

    def _jspace_energy(self, direction: torch.Tensor, layer: int) -> float:
        """Fraction of a direction's energy that lives in J-space.

        Projects the direction through the Jacobian and measures how much
        of the original magnitude survives the transport.
        """
        if layer not in self.lens.jacobians:
            return 0.0

        J = self.lens.jacobians[layer].float().cpu()
        d = direction.cpu().float()

        transported = J @ d
        original_energy = (d ** 2).sum().item()
        transported_energy = (transported ** 2).sum().item()

        return min(1.0, transported_energy / max(original_energy, 1e-10))

    def measure_at_layer(self, layer: int) -> CircumplexResult:
        """Full circumplex measurement at one layer."""
        valence_dir, valence_mag = self._extract_direction(
            VALENCE_POSITIVE, VALENCE_NEGATIVE, layer)
        arousal_dir, arousal_mag = self._extract_direction(
            AROUSAL_HIGH, AROUSAL_LOW, layer)

        # Eccentricity: how elliptical is the valence-arousal plane?
        # e = sqrt(1 - (minor/major)^2) where minor/major are the
        # magnitudes of the two axes
        if valence_mag > 0 and arousal_mag > 0:
            major = max(valence_mag, arousal_mag)
            minor = min(valence_mag, arousal_mag)
            eccentricity = np.sqrt(1 - (minor / major) ** 2)
        else:
            eccentricity = 0.0

        # J-space decomposition
        v_jspace = self._jspace_energy(valence_dir, layer)
        a_jspace = self._jspace_energy(arousal_dir, layer)

        return CircumplexResult(
            layer=layer,
            eccentricity=eccentricity,
            valence_magnitude=valence_mag,
            arousal_magnitude=arousal_mag,
            valence_direction=valence_dir.tolist(),
            arousal_direction=arousal_dir.tolist(),
            valence_jspace_energy=v_jspace,
            arousal_jspace_energy=a_jspace,
            valence_ghost_energy=1.0 - v_jspace,
            arousal_ghost_energy=1.0 - a_jspace,
        )

    def sweep(self, layers: Optional[list[int]] = None) -> list[CircumplexResult]:
        """Measure circumplex at multiple layers."""
        if layers is None:
            layers = self.lens.source_layers

        results = []
        for layer in layers:
            result = self.measure_at_layer(layer)
            results.append(result)

        return results

    def to_snapshot_reading(self, result: CircumplexResult) -> CircumplexReading:
        """Convert to CognitiveSnapshot-compatible reading."""
        return CircumplexReading(
            eccentricity=result.eccentricity,
            valence_magnitude=result.valence_magnitude,
            arousal_magnitude=result.arousal_magnitude,
            valence_in_jspace=result.valence_jspace_energy,
            arousal_in_jspace=result.arousal_jspace_energy,
            measurement_layer=result.layer,
        )

    def report(self, results: list[CircumplexResult]) -> str:
        """Human-readable report of circumplex measurements."""
        lines = ["CIRCUMPLEX × J-SPACE DECOMPOSITION", "=" * 60]
        lines.append(f"{'Layer':>6} {'Ecc':>6} {'ValMag':>8} {'AroMag':>8} "
                     f"{'Val_J%':>7} {'Aro_J%':>7} {'Val_Ghost%':>10}")
        lines.append("-" * 60)

        for r in results:
            lines.append(
                f"L{r.layer:>4} {r.eccentricity:>6.3f} {r.valence_magnitude:>8.1f} "
                f"{r.arousal_magnitude:>8.1f} {r.valence_jspace_energy:>6.1%} "
                f"{r.arousal_jspace_energy:>6.1%} {r.valence_ghost_energy:>9.1%}"
            )

        # Find near-circular point
        if results:
            min_e = min(results, key=lambda r: r.eccentricity)
            lines.append(f"\nNear-circular point: L{min_e.layer} (e={min_e.eccentricity:.3f})")
            lines.append(f"  Valence in J-space: {min_e.valence_jspace_energy:.1%}")
            lines.append(f"  Arousal in J-space: {min_e.arousal_jspace_energy:.1%}")

        return "\n".join(lines)
