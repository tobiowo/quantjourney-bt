# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""ERC solver must converge, equalise risk contributions and stay stable."""

from __future__ import annotations

import numpy as np
import pytest

from backtester.risk import RiskParityModel


def _risk_contributions(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    return weights * (covariance @ weights)


def _random_covariance(n: int = 6, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(250, n))
    covariance = np.cov(factors.T)
    return covariance + np.eye(n) * 1e-8


def test_diagonal_covariance_matches_inverse_volatility() -> None:
    covariance = np.diag([0.04, 0.01])  # vols 0.20 and 0.10
    weights = RiskParityModel()._solve_erc(covariance, 2)  # noqa: SLF001

    assert weights[0] == pytest.approx(1.0 / 3.0, rel=1e-8)
    assert weights[1] == pytest.approx(2.0 / 3.0, rel=1e-8)


def test_risk_contributions_are_equal() -> None:
    covariance = _random_covariance()
    weights = RiskParityModel()._solve_erc(covariance, 6)  # noqa: SLF001

    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert (weights > 0).all()
    contributions = _risk_contributions(weights, covariance)
    dispersion = (contributions.max() - contributions.min()) / contributions.mean()
    assert dispersion < 1e-6


def test_solution_is_stable_under_tiny_covariance_perturbation() -> None:
    covariance = _random_covariance()
    base = RiskParityModel()._solve_erc(covariance, 6)  # noqa: SLF001

    rng = np.random.default_rng(11)
    noise = rng.normal(size=covariance.shape)
    noise = (noise + noise.T) / 2.0
    perturbed = covariance + noise * 1e-8
    shifted = RiskParityModel()._solve_erc(perturbed, 6)  # noqa: SLF001

    assert np.max(np.abs(base - shifted)) < 1e-4


def test_near_singular_covariance_returns_usable_weights() -> None:
    covariance = np.ones((4, 4)) + np.eye(4) * 1e-12
    weights = RiskParityModel()._solve_erc(covariance, 4)  # noqa: SLF001

    assert np.isfinite(weights).all()
    assert (weights > 0).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_non_square_covariance_is_rejected() -> None:
    with pytest.raises(ValueError, match="covariance matrix"):
        RiskParityModel()._solve_erc(np.eye(3), 4)  # noqa: SLF001
