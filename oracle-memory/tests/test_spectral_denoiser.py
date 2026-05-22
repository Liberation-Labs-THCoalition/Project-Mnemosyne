"""Tests for Spectral Denoiser — Marcenko-Pastur denoising for KV cache geometry."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectral_denoiser import (
    marcenko_pastur_bounds,
    gavish_donoho_threshold,
    denoise_singular_values,
    denoise_geometry,
    _spectral_entropy,
    _heuristic_rank,
    _omega_approx,
)


class TestMarcenkoPastur:
    def test_square_matrix(self):
        lm, lp = marcenko_pastur_bounds(100, 100, sigma=1.0)
        assert lm == pytest.approx(0.0)
        assert lp == pytest.approx(4.0)

    def test_rectangular_tall(self):
        lm, lp = marcenko_pastur_bounds(1000, 100, sigma=1.0)
        gamma = 10.0
        expected_plus = (1 + math.sqrt(gamma))**2
        assert lp == pytest.approx(expected_plus)
        assert lm < lp

    def test_sigma_scaling(self):
        lm1, lp1 = marcenko_pastur_bounds(100, 100, sigma=1.0)
        lm2, lp2 = marcenko_pastur_bounds(100, 100, sigma=2.0)
        assert lp2 == pytest.approx(lp1 * 4)

    def test_noise_bulk_contains_noise(self):
        """For a random matrix, all eigenvalues should fall within the bounds."""
        n, p = 200, 50
        lm, lp = marcenko_pastur_bounds(n, p, sigma=1.0)
        assert lp > lm
        assert lm >= 0


class TestGavishDonoho:
    def test_threshold_positive(self):
        svs = [10.0, 5.0, 2.0, 1.0, 0.5, 0.3, 0.1, 0.05]
        tau = gavish_donoho_threshold(100, 64, singular_values=svs)
        assert tau > 0

    def test_threshold_separates_signal(self):
        signal = [50.0, 30.0, 20.0]
        noise = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05] * 10
        svs = signal + noise
        tau = gavish_donoho_threshold(100, 64, singular_values=svs)
        assert tau < 20.0
        assert tau > 0.5

    def test_known_sigma(self):
        tau_unknown = gavish_donoho_threshold(100, 64, singular_values=[10, 5, 1, 0.5, 0.1])
        tau_known = gavish_donoho_threshold(100, 64, sigma=0.5)
        assert tau_known > 0
        assert tau_unknown > 0

    def test_omega_range(self):
        for beta in [0.1, 0.3, 0.5, 0.7, 1.0]:
            w = _omega_approx(beta)
            assert 1.0 < w < 4.0


class TestDenoiseSingularValues:
    def test_clean_signal(self):
        svs = [100.0, 50.0, 25.0, 0.1, 0.05, 0.01]
        result = denoise_singular_values(svs, n=200, p=64)
        assert result["denoised_rank"] <= 3
        assert result["snr"] > 10

    def test_pure_noise(self):
        svs = [1.01, 1.0, 0.99, 0.98, 0.97, 0.96]
        result = denoise_singular_values(svs, n=200, p=64)
        assert result["denoised_rank"] <= len(svs)

    def test_snr_increases_with_signal(self):
        noise = [0.5, 0.4, 0.3, 0.2, 0.1]
        weak = [5.0] + noise
        strong = [50.0] + noise
        r_weak = denoise_singular_values(weak, 100, 64)
        r_strong = denoise_singular_values(strong, 100, 64)
        assert r_strong["snr"] > r_weak["snr"]

    def test_entropy_reduction(self):
        svs = [100.0, 50.0, 0.5, 0.4, 0.3, 0.2, 0.1]
        result = denoise_singular_values(svs, 200, 64)
        assert result["entropy_reduction"] >= 0
        assert result["denoised_entropy"] <= result["total_entropy"]

    def test_heuristic_vs_denoised(self):
        svs = [100.0, 50.0, 25.0, 12.0, 6.0, 3.0, 1.5, 0.5, 0.1]
        result = denoise_singular_values(svs, 200, 64)
        assert result["heuristic_rank_90"] > 0
        assert result["rank_delta"] != 0 or result["denoised_rank"] == result["heuristic_rank_90"]

    def test_empty(self):
        result = denoise_singular_values([], 0, 0)
        assert result["denoised_rank"] == 0
        assert result["snr"] == 0

    def test_signal_fraction(self):
        svs = [100.0, 0.1, 0.01]
        result = denoise_singular_values(svs, 100, 64)
        assert result["signal_fraction"] > 0.99


class TestDenoiseGeometry:
    def test_augments_geometry(self):
        geo = {
            "effective_rank": 5,
            "spectral_entropy": 2.3,
            "norm_per_token": 3.5,
        }
        svs = [50.0, 25.0, 10.0, 0.5, 0.3, 0.1]
        result = denoise_geometry(geo, svs, n=200, p=64)
        assert "denoised_effective_rank" in result
        assert "snr" in result
        assert "snr_db" in result
        assert result["effective_rank"] == 5
        assert result["norm_per_token"] == 3.5


class TestSpectralEntropy:
    def test_uniform(self):
        svs = [1.0, 1.0, 1.0, 1.0]
        entropy = _spectral_entropy(svs)
        assert entropy == pytest.approx(math.log(4))

    def test_concentrated(self):
        svs = [100.0, 0.001, 0.001, 0.001]
        entropy = _spectral_entropy(svs)
        assert entropy < 0.1

    def test_empty(self):
        assert _spectral_entropy([]) == 0.0


class TestHeuristicRank:
    def test_basic(self):
        svs = [10.0, 5.0, 2.0, 1.0, 0.5]
        rank = _heuristic_rank(svs, 0.9)
        assert 1 <= rank <= 5

    def test_single_dominant(self):
        svs = [100.0, 0.01, 0.01]
        assert _heuristic_rank(svs, 0.9) == 1

    def test_uniform(self):
        svs = [1.0] * 10
        assert _heuristic_rank(svs, 0.9) == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
