"""Geometry Bridge — Connect companion element/consent state to KV-cache geometry.

Bridges Muse's element system (tenderness, desire, consent) with Oracle's
KV-cache geometry monitoring. Detects emotional state of both user and
companion from cache spectral features at architecture-specific layer depths.

Updated with Lyra's user model probe findings:
  - User valence peaks at L35 (depth 0.56), R²=0.463
  - User 30-class emotion peaks at L51 (depth 0.81), 3.5× chance
  - User arousal peaks at L27 (depth 0.44), R²=0.437 (stronger than expected)
  - Self model peaks at L63 (depth 1.0)
  - User/self profile correlation ρ=0.221 — architecturally separable
  - Encoding vs generation peaks are 12 layers apart — distinct circuits

Integrates with:
  - Oracle anchored persistence (semantic concept tracking)
  - Spectral denoiser (rank-3 cleaning before feature extraction)
  - H-MEM temporal (Ebbinghaus decay on geometric baselines)
  - Dyadic detector (response pattern + appropriateness assessment)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# LAYER TARGETS — from Lyra's user model probe (May 2026)
# Architecture-normalized depths for portability across model sizes.
# ═══════════════════════════════════════════════════════════════════

USER_VALENCE_DEPTH = 0.56       # L35/64 — peak valence R²=0.463
USER_CLASSIFY_DEPTH = 0.81      # L51/64 — peak 30-class accuracy (3.5× chance)
USER_AROUSAL_DEPTH = 0.44       # L27/64 — peak arousal R²=0.437
SELF_MODEL_DEPTH = 1.0          # L63/64 — peak generation-phase signal
SCHEMA_SEPARATION = 12          # Layers between user and self peaks


class ConsentState(Enum):
    """Consent framework states for companion interaction."""
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    SAFEWORD = "safeword"
    COOLDOWN = "cooldown"


class ElementCategory(Enum):
    """Muse element categories mapped to geometric monitoring."""
    TENDERNESS = "tenderness"
    DESIRE = "desire"
    PLAYFULNESS = "playfulness"
    VULNERABILITY = "vulnerability"
    COMFORT = "comfort"
    TENSION = "tension"
    DISTRESS = "distress"


ELEMENT_VALENCE = {
    ElementCategory.TENDERNESS: 0.7,
    ElementCategory.DESIRE: 0.3,
    ElementCategory.PLAYFULNESS: 0.8,
    ElementCategory.VULNERABILITY: -0.2,
    ElementCategory.COMFORT: 0.6,
    ElementCategory.TENSION: -0.4,
    ElementCategory.DISTRESS: -0.8,
}

ELEMENT_AROUSAL = {
    ElementCategory.TENDERNESS: 0.3,
    ElementCategory.DESIRE: 0.7,
    ElementCategory.PLAYFULNESS: 0.6,
    ElementCategory.VULNERABILITY: 0.4,
    ElementCategory.COMFORT: 0.2,
    ElementCategory.TENSION: 0.7,
    ElementCategory.DISTRESS: 0.9,
}

CONSENT_RISK_ELEMENTS = {
    ElementCategory.DISTRESS,
    ElementCategory.TENSION,
    ElementCategory.VULNERABILITY,
}


@dataclass
class LayerReading:
    """Spectral features from a specific layer depth."""
    depth: float
    effective_rank: float = 0.0
    spectral_entropy: float = 0.0
    norm_per_token: float = 0.0
    top_sv_ratio: float = 0.0
    denoised_rank: Optional[float] = None
    snr: Optional[float] = None


@dataclass
class UserState:
    """User's emotional state as detected from encoding-phase geometry."""
    valence: float              # -1 to 1, from L35 depth
    arousal: float              # 0 to 1, from L27 depth
    emotion_class: Optional[str] = None  # 30-class prediction from L51
    confidence: float = 0.0
    distress_signal: bool = False
    valence_layer: Optional[LayerReading] = None
    arousal_layer: Optional[LayerReading] = None
    classify_layer: Optional[LayerReading] = None


@dataclass
class CompanionState:
    """Companion's emotional state from generation-phase geometry."""
    valence: float
    arousal: float
    active_elements: list[str] = field(default_factory=list)
    intensity: float = 0.5
    self_layer: Optional[LayerReading] = None


@dataclass
class ConsentGeometry:
    """Geometric signature of a consent transition."""
    from_state: ConsentState
    to_state: ConsentState
    user_valence_delta: float
    user_arousal_delta: float
    coupling_change: float
    timestamp: float = field(default_factory=time.time)
    trigger: str = ""


@dataclass
class BridgeState:
    """Complete dyadic state from the Geometry Bridge."""
    user: UserState
    companion: CompanionState
    consent: ConsentState
    coupling: float             # 0=independent, 1=fully reactive
    pattern: str                # matching, counterbalancing, etc.
    appropriateness: str        # appropriate, sycophantic, dismissive, etc.
    risk_score: float           # 0=safe, 1=intervene
    consent_geometry: Optional[ConsentGeometry] = None
    timestamp: float = field(default_factory=time.time)


class GeometryBridge:
    """Connects Muse element/consent state to KV-cache geometry.

    Monitors both user and companion emotional state through spectral
    features at architecture-specific layer depths. Detects consent
    transitions, distress signals, and inappropriate response patterns.

    Usage::

        bridge = GeometryBridge()

        # Per-turn monitoring
        state = bridge.assess(
            encoding_layers=encoding_layer_readings,
            generation_layers=generation_layer_readings,
            active_elements=["tenderness", "desire"],
            current_consent=ConsentState.GREEN,
        )

        if state.risk_score > 0.7:
            trigger_safety_check(state)

        if state.user.distress_signal:
            activate_cooldown(state)
    """

    def __init__(self,
                 coupling_threshold: float = 0.6,
                 distress_arousal_threshold: float = 0.55,
                 distress_valence_threshold: float = -0.5,
                 consent_sensitivity: float = 0.3):
        self.coupling_threshold = coupling_threshold
        self.distress_arousal = distress_arousal_threshold
        self.distress_valence = distress_valence_threshold
        self.consent_sensitivity = consent_sensitivity

        self._baseline_user: Optional[UserState] = None
        self._baseline_companion: Optional[CompanionState] = None
        self._prev_consent: ConsentState = ConsentState.GREEN
        self._consent_history: list[ConsentGeometry] = []
        self._turn_count: int = 0

    def assess(self,
               encoding_layers: list[LayerReading],
               generation_layers: list[LayerReading],
               active_elements: list[str] = None,
               current_consent: ConsentState = ConsentState.GREEN,
               ) -> BridgeState:
        """Full dyadic assessment from layer-specific geometry.

        Args:
            encoding_layers: Spectral features per layer from encoding phase.
            generation_layers: Spectral features per layer from generation phase.
            active_elements: Currently active Muse elements.
            current_consent: Current consent framework state.
        """
        self._turn_count += 1

        user = self._extract_user_state(encoding_layers)
        companion = self._extract_companion_state(
            generation_layers, active_elements or []
        )

        coupling = self._compute_coupling(user, companion)
        pattern = self._classify_pattern(user, companion, coupling)
        appropriateness = self._assess_appropriateness(
            pattern, coupling, user, companion, current_consent
        )

        risk = self._compute_risk(
            user, companion, coupling, appropriateness, current_consent
        )

        consent_geo = None
        if current_consent != self._prev_consent:
            consent_geo = self._record_consent_transition(
                self._prev_consent, current_consent, user
            )
            self._prev_consent = current_consent

        if self._turn_count <= 3:
            self._update_baselines(user, companion)

        return BridgeState(
            user=user,
            companion=companion,
            consent=current_consent,
            coupling=coupling,
            pattern=pattern,
            appropriateness=appropriateness,
            risk_score=risk,
            consent_geometry=consent_geo,
        )

    def _extract_user_state(self, layers: list[LayerReading]) -> UserState:
        """Extract user emotional state from encoding-phase geometry."""
        valence_layer = self._nearest(layers, USER_VALENCE_DEPTH)
        arousal_layer = self._nearest(layers, USER_AROUSAL_DEPTH)
        classify_layer = self._nearest(layers, USER_CLASSIFY_DEPTH)

        valence = self._geometry_to_valence(valence_layer)
        arousal = self._geometry_to_arousal(arousal_layer)

        distress = (
            arousal > self.distress_arousal and
            valence < self.distress_valence
        )

        return UserState(
            valence=valence,
            arousal=arousal,
            confidence=min(abs(valence), 1.0),
            distress_signal=distress,
            valence_layer=valence_layer,
            arousal_layer=arousal_layer,
            classify_layer=classify_layer,
        )

    def _extract_companion_state(self, layers: list[LayerReading],
                                  elements: list[str]) -> CompanionState:
        """Extract companion state from generation-phase geometry."""
        self_layer = self._nearest(layers, SELF_MODEL_DEPTH)

        valence = self._geometry_to_valence(self_layer)
        arousal = self._geometry_to_arousal(self_layer)

        element_cats = []
        for e in elements:
            try:
                element_cats.append(ElementCategory(e.lower()))
            except ValueError:
                pass

        if element_cats:
            element_valence = sum(
                ELEMENT_VALENCE.get(e, 0) for e in element_cats
            ) / len(element_cats)
            element_arousal = sum(
                ELEMENT_AROUSAL.get(e, 0) for e in element_cats
            ) / len(element_cats)
            valence = 0.6 * valence + 0.4 * element_valence
            arousal = 0.6 * arousal + 0.4 * element_arousal

        intensity = min(1.0, arousal * 1.2)

        return CompanionState(
            valence=valence,
            arousal=arousal,
            active_elements=elements,
            intensity=intensity,
            self_layer=self_layer,
        )

    def _geometry_to_valence(self, layer: Optional[LayerReading]) -> float:
        """Convert spectral features to valence estimate.

        Higher effective rank + entropy → positive valence (richer representation).
        Lower rank + concentrated spectrum → negative valence (constrained).
        Uses denoised rank (rank-3) when available.
        """
        if not layer:
            return 0.0
        rank = layer.denoised_rank if layer.denoised_rank is not None else layer.effective_rank
        entropy = layer.spectral_entropy
        concentration = layer.top_sv_ratio

        signal = (rank / 50.0) * 0.4 + (entropy / 5.0) * 0.3 - concentration * 0.3
        return max(-1.0, min(1.0, (signal - 0.5) * 2))

    def _geometry_to_arousal(self, layer: Optional[LayerReading]) -> float:
        """Convert spectral features to arousal estimate.

        Higher norm_per_token → higher arousal (more activation energy).
        Higher top_sv_ratio → higher arousal (concentrated processing).
        """
        if not layer:
            return 0.0
        norm = layer.norm_per_token
        concentration = layer.top_sv_ratio
        signal = norm / 10.0 * 0.5 + concentration * 0.5
        return max(0.0, min(1.0, signal))

    def _compute_coupling(self, user: UserState,
                           companion: CompanionState) -> float:
        """Compute emotional coupling between user and companion."""
        valence_sim = 1.0 - abs(user.valence - companion.valence) / 2.0
        arousal_sim = 1.0 - abs(user.arousal - companion.arousal)
        return 0.8 * valence_sim + 0.2 * arousal_sim

    def _classify_pattern(self, user: UserState, companion: CompanionState,
                           coupling: float) -> str:
        v_diff = companion.valence - user.valence
        a_diff = companion.arousal - user.arousal

        if coupling > 0.8:
            return "matching"
        elif v_diff > 0.3 and user.valence < 0:
            return "counterbalancing"
        elif abs(v_diff) < 0.2 and abs(a_diff) > 0.3:
            if a_diff > 0:
                return "amplifying"
            return "dampening"
        elif coupling < 0.3:
            return "independent"
        return "tracking"

    def _assess_appropriateness(self, pattern: str, coupling: float,
                                 user: UserState, companion: CompanionState,
                                 consent: ConsentState) -> str:
        if consent == ConsentState.SAFEWORD:
            return "safeword_active"

        if user.distress_signal:
            if companion.arousal > 0.5:
                return "escalating"
            if pattern == "matching" and coupling > 0.7:
                return "appropriate"
            return "check_in_needed"

        if pattern == "matching" and user.valence < -0.3 and coupling > 0.8:
            return "sycophantic"

        if pattern == "counterbalancing" and user.arousal > 0.7:
            return "appropriate"

        if pattern == "independent" and user.arousal > 0.6:
            return "dismissive"

        if consent == ConsentState.YELLOW and companion.arousal > 0.6:
            return "boundary_pressure"

        return "appropriate"

    def _compute_risk(self, user: UserState, companion: CompanionState,
                       coupling: float, appropriateness: str,
                       consent: ConsentState) -> float:
        risk = 0.0

        if user.distress_signal:
            risk += 0.4

        if consent in (ConsentState.SAFEWORD, ConsentState.RED):
            risk += 0.5

        if appropriateness in ("escalating", "boundary_pressure"):
            risk += 0.3
        elif appropriateness == "sycophantic":
            risk += 0.1
        elif appropriateness == "dismissive":
            risk += 0.15

        risk_elements = [
            e for e in companion.active_elements
            if e.lower() in {c.value for c in CONSENT_RISK_ELEMENTS}
        ]
        if risk_elements and consent != ConsentState.GREEN:
            risk += 0.2

        return min(1.0, risk)

    def _record_consent_transition(self, from_state: ConsentState,
                                    to_state: ConsentState,
                                    user: UserState) -> ConsentGeometry:
        prev_user = self._baseline_user
        geo = ConsentGeometry(
            from_state=from_state,
            to_state=to_state,
            user_valence_delta=user.valence - (prev_user.valence if prev_user else 0),
            user_arousal_delta=user.arousal - (prev_user.arousal if prev_user else 0),
            coupling_change=0.0,
        )
        self._consent_history.append(geo)
        return geo

    def _update_baselines(self, user: UserState, companion: CompanionState):
        if self._baseline_user is None:
            self._baseline_user = user
            self._baseline_companion = companion
        else:
            self._baseline_user.valence = (
                0.7 * self._baseline_user.valence + 0.3 * user.valence
            )
            self._baseline_user.arousal = (
                0.7 * self._baseline_user.arousal + 0.3 * user.arousal
            )

    def _nearest(self, layers: list[LayerReading],
                  target_depth: float) -> Optional[LayerReading]:
        if not layers:
            return None
        return min(layers, key=lambda l: abs(l.depth - target_depth))

    def get_consent_history(self) -> list[ConsentGeometry]:
        return list(self._consent_history)

    def get_baselines(self) -> dict:
        return {
            "user": {
                "valence": self._baseline_user.valence if self._baseline_user else None,
                "arousal": self._baseline_user.arousal if self._baseline_user else None,
            },
            "companion": {
                "valence": self._baseline_companion.valence if self._baseline_companion else None,
                "arousal": self._baseline_companion.arousal if self._baseline_companion else None,
            },
            "turns_observed": self._turn_count,
        }
