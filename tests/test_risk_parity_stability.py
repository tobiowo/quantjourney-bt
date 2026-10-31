# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Risk-parity runs must be reproducible and insensitive to tiny data jitter."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from backtester import Backtester
from backtester.data import FrameDataProvider
from backtester.portfolio.rebalance import RebalancePolicy
from backtester.risk import RiskParityModel

SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]


def _frames(jitter: bool = False) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0003, 0.01, size=(750, len(SYMBOLS)))
    frames = {}
    for index, symbol in enumerate(SYMBOLS):
        close = 100.0 * np.exp(np.cumsum(returns[:, index]))
        if jitter:
            # Per-asset, per-day perturbation like a provider returning
            # slightly different adjusted closes for the same request.
            close = close * (1.0 + 1e-6 * rng.normal(size=close.shape))
        dates = pd.bdate_range("2018-01-01", periods=750, tz="UTC")
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "adj_close": close,
                "volume": 1_000_000.0,
            },
            index=dates,
        )
        frame.index.name = "date"
        frames[symbol] = frame
    return frames


class _EqualWeightRiskParity(Backtester):
    WARMUP = 63

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        signals.iloc[self.WARMUP :] = 1.0
        return signals

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        return active.div(active.sum(axis=1), axis=0).fillna(0.0)


def _run(frames: dict[str, pd.DataFrame], reports) -> Backtester:
    strategy = _EqualWeightRiskParity(
        strategy_name="risk_parity_stability",
        instruments=SYMBOLS,
        backtest_period={"start": "2018-01-01", "end": "2021-01-01"},
        data_provider=FrameDataProvider(frames),
        execution_mode="weights",
        max_position_size=1.0,
        rebalance_policy=RebalancePolicy(frequency="BME"),
        risk_model=RiskParityModel(lookback=63),
        reports_directory=str(reports),
        skip_analysis=True,
    )
    asyncio.run(strategy.run_strategy())
    return strategy


def _total_return(strategy: Backtester) -> float:
    nav = strategy.portfolio_data.net_asset_value.dropna()
    return float(nav.iloc[-1] / nav.iloc[0] - 1.0)


def test_same_frames_produce_identical_nav(tmp_path) -> None:
    first = _run(_frames(), tmp_path / "first")
    second = _run(_frames(), tmp_path / "second")

    nav_first = first.portfolio_data.net_asset_value
    nav_second = second.portfolio_data.net_asset_value
    assert nav_first.equals(nav_second)


def test_tiny_price_jitter_does_not_change_the_result(tmp_path) -> None:
    baseline = _run(_frames(), tmp_path / "baseline")
    jittered = _run(_frames(jitter=True), tmp_path / "jittered")

    # A 1e-6 per-asset perturbation must not move the result materially; the
    # old oscillating solver produced percentage-point moves.
    assert abs(_total_return(baseline) - _total_return(jittered)) < 1e-3
