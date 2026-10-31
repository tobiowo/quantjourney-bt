"""
Risk-Parity (Equal Risk Contribution) Model
============================================

Adjusts weights so that each instrument contributes equally to
total portfolio risk.  Uses a fast iterative approximation
(no optimiser dependency) suitable for backtesting loops.

For *n* instruments with covariance matrix Σ, the risk-parity
solution satisfies::

    w_i * (Σ w)_i  =  constant  ∀ i

We solve the equivalent strictly convex problem

    min_w  0.5 * wᵀΣw - (1/n) * Σ log(w_i)

by cyclical coordinate descent: each weight is set to the exact positive
root of a quadratic, so the iterate converges instead of oscillating.
Inverse-volatility weights are the starting point. Convergence typically
takes fewer than 20 iterations for 5–20 asset portfolios, and the solution
is stable under the small covariance perturbations data providers introduce
between otherwise identical requests.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.risk.base import RiskModel


@dataclass
class RiskParityModel(RiskModel):
    """
    Equal Risk Contribution weighting.

    Parameters
    ----------
    lookback : int
        Rolling window for covariance estimation.
    max_iter : int
        Maximum iterations for the ERC solver.
    tol : float
        Convergence tolerance on weight change.
    rebalance_freq : str
        How often to recompute (default: monthly).
    """

    lookback: int = 63
    max_iter: int = 50
    tol: float = 1e-8
    rebalance_freq: str = "BMS"

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")

    def adjust(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        *,
        metadata: dict | None = None,
    ) -> pd.DataFrame:
        n = len(weights)
        if n == 0:
            return weights

        out = weights.copy()

        # Detect rescoring dates
        if self.rebalance_freq == "D":
            is_rescore = pd.Series(True, index=weights.index)
        else:
            periods = pd.Series(
                weights.index.to_period(
                    {"BMS": "M", "W-MON": "W", "MS": "M", "QS": "Q"}.get(self.rebalance_freq, "M")
                ),
                index=weights.index,
            )
            is_rescore = periods != periods.shift(1)

        prev_rp_weights = None
        prev_signature = None

        weight_values = weights.to_numpy(dtype=float, copy=True)
        output_values = weight_values.copy()
        rescore_values = is_rescore.to_numpy(dtype=bool)

        for i in range(self.lookback, n):
            row_w = weight_values[i]
            active_mask = np.abs(row_w) > 1e-10
            n_active = int(active_mask.sum())
            signature = tuple(np.sign(row_w))

            if n_active < 2:
                # Single or no asset — pass through
                output_values[i] = row_w
                prev_rp_weights = None
                prev_signature = signature
                continue

            if rescore_values[i] or prev_rp_weights is None or signature != prev_signature:
                # Compute new risk-parity weights for active instruments
                window = returns.iloc[max(0, i - self.lookback) : i]
                active_cols = weights.columns[active_mask]
                sub_returns = window[active_cols]

                # ERC solves for positive magnitudes. Transform each return
                # series by the strategy's sign, then restore those signs to
                # the solution. This preserves long/short intent while making
                # risk contributions consistent with the signed portfolio.
                signs = np.sign(row_w[active_mask])
                cov = sub_returns.mul(signs, axis=1).cov().values
                rp_w = self._solve_erc(cov, n_active)

                # Build full weight vector
                full_rp = np.zeros(len(weights.columns), dtype=float)
                full_rp[active_mask] = signs * rp_w

                # Preserve original total exposure
                original_exposure = float(np.nansum(np.abs(row_w)))
                full_rp = full_rp * original_exposure

                prev_rp_weights = full_rp
                prev_signature = signature

            output_values[i] = prev_rp_weights

        out.iloc[:] = output_values

        return out

    def _solve_erc(self, cov: np.ndarray, n: int) -> np.ndarray:
        """Solve Equal Risk Contribution weights by coordinate descent.

        Minimises the strictly convex problem

            f(w) = 0.5 * wᵀΣw - (1/n) * Σ log(w_i)

        whose stationary point satisfies ``w_i * (Σw)_i = 1/n`` for every i
        (the ERC condition) and ``Σ w_i = 1``. Each coordinate is updated to
        the exact positive root of the resulting quadratic, which makes the
        solution stable under the tiny covariance perturbations that data
        providers introduce between otherwise identical requests. The previous
        multiplicative heuristic could oscillate and return an unconverged
        iterate for correlated multi-asset covariances.
        """
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (n, n):
            raise ValueError(f"covariance matrix must be {n}x{n}, got {cov.shape}")

        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        diagonal: np.ndarray = np.diag(cov).copy()
        floor = 1e-12
        diagonal = np.maximum(diagonal, floor)
        np.fill_diagonal(cov, diagonal)

        # Inverse-volatility start, matching the risk-parity literature.
        w: np.ndarray = 1.0 / np.sqrt(diagonal)
        w = w / w.sum()

        off_diagonal: np.ndarray = cov.copy()
        np.fill_diagonal(off_diagonal, 0.0)

        for _ in range(self.max_iter):
            previous = w.copy()
            for i in range(n):
                a = diagonal[i]
                b = float(off_diagonal[i] @ w)
                w[i] = (-b + np.sqrt(b * b + 4.0 * a / n)) / (2.0 * a)
            total = float(w.sum())
            if not np.isfinite(total) or total <= 0.0:
                return np.ones(n) / n
            if np.max(np.abs(w - previous)) < self.tol:
                break

        total = float(w.sum())
        if not np.isfinite(total) or total <= 0.0:
            return np.ones(n) / n
        return w / total

    def __repr__(self) -> str:
        return f"RiskParityModel(lookback={self.lookback}, freq={self.rebalance_freq})"
