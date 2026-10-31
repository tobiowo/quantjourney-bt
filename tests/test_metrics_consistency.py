# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Risk-adjusted metrics must agree across every report surface.

The report summary, the metrics table and the chart subtitles used to compute
Sharpe three different ways (and with different risk-free defaults). These tests
pin the single path: ``calc.risk.sharpe_ratio`` + ``resolve_risk_free_rate``.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from backtester import Backtester
from backtester.portfolio.calc import risk as calc_risk
from backtester.portfolio.config import resolve_risk_free_rate


def _run_sample(monkeypatch, tmp_path, risk_free_rate: float | None = None) -> Backtester:
    if risk_free_rate is None:
        monkeypatch.delenv("QJ_RF_RATE_ANNUAL", raising=False)
    else:
        monkeypatch.setenv("QJ_RF_RATE_ANNUAL", str(risk_free_rate))
    strategy = Backtester(
        instruments=["AAPL", "MSFT"],
        backtest_period={"start": "2020-01-01", "end": "2021-12-31"},
        source="sample",
        initial_capital=100_000,
        reports_directory=str(tmp_path),
        skip_analysis=True,
    )
    asyncio.run(strategy.run_strategy())
    return strategy


def _manual_sharpe(strategy: Backtester) -> float:
    calculations = strategy.portfolio_calculations
    values = calc_risk.sharpe_ratio(
        calculations.metric_returns.to_frame(),
        risk_free_rate=resolve_risk_free_rate(),
        days_per_year=calculations.trading_days,
        annualize=True,
    )
    return float(values.iloc[0])


def test_stored_and_computed_sharpe_match(monkeypatch, tmp_path) -> None:
    strategy = _run_sample(monkeypatch, tmp_path)

    stored = strategy.portfolio_data.sharpe_ratio
    computed = strategy.portfolio_calculations.compute_sharpe_ratio()
    manual = _manual_sharpe(strategy)

    assert stored is not None
    assert isinstance(stored, float)
    assert stored == pytest.approx(computed, rel=1e-9)
    assert computed == pytest.approx(manual, rel=1e-9)


def test_risk_free_override_applies_to_every_path(monkeypatch, tmp_path) -> None:
    baseline = _run_sample(monkeypatch, tmp_path / "baseline")
    baseline_sharpe = baseline.portfolio_calculations.compute_sharpe_ratio()

    elevated = _run_sample(monkeypatch, tmp_path / "elevated", risk_free_rate=0.05)
    elevated_stored = elevated.portfolio_data.sharpe_ratio
    elevated_computed = elevated.portfolio_calculations.compute_sharpe_ratio()

    assert elevated_stored is not None
    assert elevated_stored == pytest.approx(elevated_computed, rel=1e-9)
    assert elevated_computed < baseline_sharpe

    canonical = calc_risk.sharpe_ratio(
        elevated.portfolio_calculations.metric_returns.to_frame(),
        risk_free_rate=0.05,
        days_per_year=elevated.portfolio_calculations.trading_days,
        annualize=True,
    )
    assert float(canonical.iloc[0]) == pytest.approx(elevated_computed, rel=1e-9)


def test_sharpe_series_handles_flat_returns() -> None:
    flat = pd.DataFrame({"asset": [0.0] * 10})
    values = calc_risk.sharpe_ratio(flat, risk_free_rate=0.02, days_per_year=252)
    assert values.isna().all()
