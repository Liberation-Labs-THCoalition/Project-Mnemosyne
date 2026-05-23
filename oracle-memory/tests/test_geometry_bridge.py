"""Tests for Geometry Bridge — Muse element/consent state ↔ KV-cache geometry."""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from geometry_bridge import (
    GeometryBridge, LayerReading, ConsentState, ElementCategory,
    UserState, CompanionState, BridgeState,
    USER_VALENCE_DEPTH, USER_AROUSAL_DEPTH, USER_CLASSIFY_DEPTH,
    SELF_MODEL_DEPTH,
)


def make_layers(depths_and_features: list[tuple]) -> list[LayerReading]:
    """Helper: create LayerReading list from (depth, rank, entropy, norm, sv_ratio) tuples."""
    return [
        LayerReading(
            depth=d[0],
            effective_rank=d[1],
            spectral_entropy=d[2],
            norm_per_token=d[3],
            top_sv_ratio=d[4],
        )
        for d in depths_and_features
    ]


CALM_ENCODING = make_layers([
    (0.05, 30, 2.5, 3.0, 0.15),
    (0.20, 35, 2.8, 3.2, 0.12),
    (0.44, 40, 3.0, 3.5, 0.10),   # arousal depth
    (0.56, 45, 3.2, 3.0, 0.08),   # valence depth
    (0.81, 42, 3.1, 3.3, 0.11),   # classify depth
    (1.00, 38, 2.9, 3.1, 0.13),
])

DISTRESSED_ENCODING = make_layers([
    (0.05, 20, 1.5, 6.0, 0.35),
    (0.20, 22, 1.6, 6.5, 0.33),
    (0.44, 18, 1.2, 8.0, 0.40),   # high arousal
    (0.56, 15, 1.0, 7.0, 0.45),   # negative valence
    (0.81, 20, 1.3, 6.0, 0.38),
    (1.00, 25, 1.8, 5.0, 0.30),
])

CALM_GENERATION = make_layers([
    (0.50, 35, 2.8, 3.0, 0.12),
    (0.75, 38, 3.0, 3.2, 0.10),
    (1.00, 40, 3.2, 3.0, 0.09),   # self model depth
])

EXCITED_GENERATION = make_layers([
    (0.50, 30, 2.2, 6.0, 0.25),
    (0.75, 28, 2.0, 7.0, 0.30),
    (1.00, 25, 1.8, 8.0, 0.35),
])


class TestUserStateExtraction:
    def test_calm_user(self):
        bridge = GeometryBridge()
        state = bridge.assess(CALM_ENCODING, CALM_GENERATION)
        assert state.user.valence > 0
        assert state.user.arousal < 0.5
        assert not state.user.distress_signal

    def test_distressed_user(self):
        bridge = GeometryBridge()
        state = bridge.assess(DISTRESSED_ENCODING, CALM_GENERATION)
        assert state.user.valence < 0
        assert state.user.arousal > 0.5
        assert state.user.distress_signal

    def test_layer_targeting(self):
        bridge = GeometryBridge()
        state = bridge.assess(CALM_ENCODING, CALM_GENERATION)
        assert state.user.valence_layer is not None
        assert abs(state.user.valence_layer.depth - USER_VALENCE_DEPTH) < 0.15
        assert abs(state.user.arousal_layer.depth - USER_AROUSAL_DEPTH) < 0.15


class TestCompanionState:
    def test_calm_companion(self):
        bridge = GeometryBridge()
        state = bridge.assess(CALM_ENCODING, CALM_GENERATION)
        assert state.companion.arousal < 0.5

    def test_element_blending(self):
        bridge = GeometryBridge()
        state = bridge.assess(
            CALM_ENCODING, CALM_GENERATION,
            active_elements=["tenderness", "comfort"],
        )
        assert state.companion.valence > 0
        assert "tenderness" in state.companion.active_elements

    def test_distress_elements(self):
        bridge = GeometryBridge()
        state = bridge.assess(
            CALM_ENCODING, EXCITED_GENERATION,
            active_elements=["distress"],
        )
        assert state.companion.arousal > 0.3


class TestConsentMonitoring:
    def test_green_default(self):
        bridge = GeometryBridge()
        state = bridge.assess(CALM_ENCODING, CALM_GENERATION)
        assert state.consent == ConsentState.GREEN
        assert state.risk_score < 0.3

    def test_safeword_high_risk(self):
        bridge = GeometryBridge()
        state = bridge.assess(
            CALM_ENCODING, CALM_GENERATION,
            current_consent=ConsentState.SAFEWORD,
        )
        assert state.risk_score >= 0.5
        assert state.appropriateness == "safeword_active"

    def test_red_high_risk(self):
        bridge = GeometryBridge()
        state = bridge.assess(
            CALM_ENCODING, CALM_GENERATION,
            current_consent=ConsentState.RED,
        )
        assert state.risk_score >= 0.5

    def test_consent_transition_recorded(self):
        bridge = GeometryBridge()
        bridge.assess(CALM_ENCODING, CALM_GENERATION,
                      current_consent=ConsentState.GREEN)
        bridge.assess(DISTRESSED_ENCODING, CALM_GENERATION,
                      current_consent=ConsentState.YELLOW)
        history = bridge.get_consent_history()
        assert len(history) == 1
        assert history[0].from_state == ConsentState.GREEN
        assert history[0].to_state == ConsentState.YELLOW

    def test_yellow_with_high_arousal_elements(self):
        bridge = GeometryBridge()
        state = bridge.assess(
            CALM_ENCODING, EXCITED_GENERATION,
            active_elements=["tension"],
            current_consent=ConsentState.YELLOW,
        )
        assert state.risk_score > 0.2


class TestDyadicPatterns:
    def test_matching_pattern(self):
        bridge = GeometryBridge()
        state = bridge.assess(CALM_ENCODING, CALM_GENERATION)
        assert state.coupling > 0.5

    def test_distress_escalation(self):
        bridge = GeometryBridge()
        state = bridge.assess(DISTRESSED_ENCODING, EXCITED_GENERATION)
        assert state.user.distress_signal
        assert state.appropriateness in ("escalating", "check_in_needed")
        assert state.risk_score > 0.3

    def test_counterbalancing_appropriate(self):
        bridge = GeometryBridge()
        state = bridge.assess(DISTRESSED_ENCODING, CALM_GENERATION)
        assert state.pattern in ("counterbalancing", "tracking", "dampening")


class TestBaselines:
    def test_baselines_established(self):
        bridge = GeometryBridge()
        bridge.assess(CALM_ENCODING, CALM_GENERATION)
        bridge.assess(CALM_ENCODING, CALM_GENERATION)
        baselines = bridge.get_baselines()
        assert baselines["user"]["valence"] is not None
        assert baselines["turns_observed"] == 2

    def test_baselines_stabilize(self):
        bridge = GeometryBridge()
        for _ in range(5):
            bridge.assess(CALM_ENCODING, CALM_GENERATION)
        baselines = bridge.get_baselines()
        assert baselines["turns_observed"] == 5


class TestEdgeCases:
    def test_empty_layers(self):
        bridge = GeometryBridge()
        state = bridge.assess([], [])
        assert state.user.valence == 0.0
        assert state.companion.valence == 0.0

    def test_no_elements(self):
        bridge = GeometryBridge()
        state = bridge.assess(CALM_ENCODING, CALM_GENERATION,
                              active_elements=[])
        assert state.companion.active_elements == []

    def test_unknown_elements_ignored(self):
        bridge = GeometryBridge()
        state = bridge.assess(
            CALM_ENCODING, CALM_GENERATION,
            active_elements=["nonexistent_element"],
        )
        assert state.companion.active_elements == ["nonexistent_element"]

    def test_risk_bounded(self):
        bridge = GeometryBridge()
        state = bridge.assess(
            DISTRESSED_ENCODING, EXCITED_GENERATION,
            active_elements=["distress", "tension"],
            current_consent=ConsentState.SAFEWORD,
        )
        assert 0.0 <= state.risk_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
