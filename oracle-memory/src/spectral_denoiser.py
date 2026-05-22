"""Spectral Denoiser — Marcenko-Pastur denoising for KV cache geometry.

Replaces the heuristic "90% cumulative variance" effective rank with a
principled signal/noise boundary derived from random matrix theory.

The Marcenko-Pastur law describes the eigenvalue distribution of large
random matrices. For a matrix of shape (n, p) with aspect ratio
gamma = n/p, the noise eigenvalues fall within:

    lambda_- = sigma^2 * (1 - sqrt(gamma))^2
    lambda_+ = sigma^2 * (1 + sqrt(gamma))^2

Eigenvalues above lambda_+ are signal. Below are noise.

For KV cache matrices: n = seq_len (tokens), p = head_dim.
The aspect ratio tells us where to draw the line.

Gavish & Donoho (2014) give the optimal singular value threshold
for the asymptotic regime: tau* = lambda_med * omega(beta),
where beta = min(n,p) / max(n,p) and omega is a known function.

This module computes:
  1. Denoised effective rank (signal eigenvalues only)
  2. Signal-to-noise ratio of the KV cache matrix
  3. Optimal singular value shrinkage (Gavish-Donoho)
  4. Cleaned spectral entropy (entropy of signal eigenvalues only)

No external dependencies beyond numpy (or pure Python fallback).
"""

import math
from typing import Optional


def marcenko_pastur_bounds(n: int, p: int,
                           sigma: float = 1.0) -> tuple[float, float]:
    """Compute the Marcenko-Pastur noise bulk boundaries.

    Args:
        n: Number of rows (tokens / seq_len).
        p: Number of columns (head_dim).
        sigma: Noise standard deviation (estimated or assumed).

    Returns:
        (lambda_minus, lambda_plus): eigenvalue boundaries of the noise bulk.
    """
    gamma = n / p if p > 0 else 1.0
    sqrt_gamma = math.sqrt(gamma)
    lambda_minus = sigma**2 * (1 - sqrt_gamma)**2
    lambda_plus = sigma**2 * (1 + sqrt_gamma)**2
    return lambda_minus, lambda_plus


def gavish_donoho_threshold(n: int, p: int,
                             sigma: Optional[float] = None,
                             singular_values: list[float] = None) -> float:
    """Compute the optimal singular value hard threshold.

    Gavish & Donoho (2014): for an n x p matrix with noise level sigma,
    the optimal threshold is tau* = omega(beta) * sqrt(median_eigenvalue)
    where beta = min(n,p) / max(n,p).

    If sigma is unknown, estimates it from the median singular value
    using the Marcenko-Pastur median.

    Args:
        n: Rows (tokens).
        p: Columns (head_dim).
        sigma: Known noise level (None = estimate from data).
        singular_values: Sorted descending singular values (needed if sigma is None).

    Returns:
        Optimal singular value threshold.
    """
    beta = min(n, p) / max(n, p) if max(n, p) > 0 else 1.0
    omega = _omega_approx(beta)

    if sigma is not None:
        mu_mp_median = _mp_median(beta)
        return omega * sigma * math.sqrt(max(n, p)) * math.sqrt(mu_mp_median)

    if singular_values:
        median_sv = _median(singular_values)
        mu_mp_median = _mp_median(beta)
        sigma_est = median_sv / math.sqrt(max(n, p) * mu_mp_median)
        return omega * sigma_est * math.sqrt(max(n, p)) * math.sqrt(mu_mp_median)

    return 0.0


def _omega_approx(beta: float) -> float:
    """Approximate the omega function from Gavish & Donoho (2014).

    omega(beta) for optimal hard thresholding. Uses the numerical
    approximation valid for beta in (0, 1].
    """
    if beta <= 0:
        return 0.0
    if beta >= 1:
        beta = 1.0
    return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43


def _mp_median(beta: float) -> float:
    """Approximate the median of the Marcenko-Pastur distribution.

    For beta in (0, 1], the median sits slightly above the center of
    the support. Uses a polynomial fit to numerical values.
    """
    if beta <= 0:
        return 1.0
    return (1 + math.sqrt(beta))**2 * (1 - 0.1 * (1 - beta))


def denoise_singular_values(singular_values: list[float],
                             n: int, p: int,
                             sigma: Optional[float] = None) -> dict:
    """Apply optimal singular value thresholding.

    Separates signal from noise using the Gavish-Donoho threshold.
    Returns denoised metrics.

    Args:
        singular_values: Singular values in descending order.
        n: Rows (tokens / seq_len).
        p: Columns (head_dim).
        sigma: Known noise level (None = estimate from data).

    Returns:
        Dict with denoised metrics:
          - signal_svs: singular values above threshold (signal)
          - noise_svs: singular values below threshold (noise)
          - threshold: the Gavish-Donoho threshold used
          - denoised_rank: count of signal singular values
          - heuristic_rank_90: traditional 90% cumulative variance rank
          - snr: signal-to-noise ratio (sum of signal^2 / sum of noise^2)
          - denoised_entropy: spectral entropy of signal SVs only
          - total_entropy: spectral entropy of all SVs
    """
    if not singular_values:
        return _empty_result()

    threshold = gavish_donoho_threshold(n, p, sigma, singular_values)

    signal_svs = [sv for sv in singular_values if sv > threshold]
    noise_svs = [sv for sv in singular_values if sv <= threshold]

    signal_energy = sum(sv**2 for sv in signal_svs)
    noise_energy = sum(sv**2 for sv in noise_svs)
    total_energy = signal_energy + noise_energy

    snr = signal_energy / noise_energy if noise_energy > 0 else float('inf')

    denoised_entropy = _spectral_entropy(signal_svs) if signal_svs else 0.0
    total_entropy = _spectral_entropy(singular_values)

    heuristic_rank = _heuristic_rank(singular_values, 0.9)

    return {
        "signal_svs": signal_svs,
        "noise_svs": noise_svs,
        "threshold": threshold,
        "denoised_rank": len(signal_svs),
        "heuristic_rank_90": heuristic_rank,
        "rank_delta": len(signal_svs) - heuristic_rank,
        "snr": snr,
        "snr_db": 10 * math.log10(snr) if snr > 0 and snr != float('inf') else None,
        "signal_fraction": signal_energy / total_energy if total_energy > 0 else 0,
        "denoised_entropy": denoised_entropy,
        "total_entropy": total_entropy,
        "entropy_reduction": total_entropy - denoised_entropy,
    }


def denoise_geometry(geometry: dict, singular_values: list[float],
                      n: int, p: int) -> dict:
    """Augment a geometry reading with denoised metrics.

    Takes an existing geometry dict (from oracle-memory) and adds
    denoised versions of the spectral features.
    """
    result = denoise_singular_values(singular_values, n, p)

    return {
        **geometry,
        "denoised_effective_rank": result["denoised_rank"],
        "denoised_spectral_entropy": result["denoised_entropy"],
        "snr": result["snr"],
        "snr_db": result["snr_db"],
        "signal_fraction": result["signal_fraction"],
        "sv_threshold": result["threshold"],
        "rank_delta": result["rank_delta"],
        "entropy_reduction": result["entropy_reduction"],
    }


def _spectral_entropy(singular_values: list[float]) -> float:
    """Shannon entropy of the squared singular value distribution."""
    sq = [sv**2 for sv in singular_values]
    total = sum(sq)
    if total == 0:
        return 0.0
    probs = [s / total for s in sq]
    return -sum(p * math.log(p) for p in probs if p > 0)


def _heuristic_rank(singular_values: list[float], threshold: float = 0.9) -> int:
    """Traditional effective rank: dimensions for threshold% cumulative variance."""
    sq = [sv**2 for sv in singular_values]
    total = sum(sq)
    if total == 0:
        return 0
    cumulative = 0
    for i, s in enumerate(sq):
        cumulative += s
        if cumulative / total >= threshold:
            return i + 1
    return len(sq)


def _median(values: list[float]) -> float:
    """Median of a sorted (descending) list."""
    n = len(values)
    if n == 0:
        return 0.0
    s = sorted(values)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def _empty_result() -> dict:
    return {
        "signal_svs": [], "noise_svs": [], "threshold": 0,
        "denoised_rank": 0, "heuristic_rank_90": 0, "rank_delta": 0,
        "snr": 0, "snr_db": None, "signal_fraction": 0,
        "denoised_entropy": 0, "total_entropy": 0, "entropy_reduction": 0,
    }
