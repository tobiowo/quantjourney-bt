# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""The weight-mode cash reserve must be explicit and configurable."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from backtester import Backtester


class _Trend(Backtester):
    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("close")
        fast = close.rolling(20).mean()
        slow = close.rolling(60).mean()
        return ((fast > slow) & fast.notna() & slow.notna()).astype(float)

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        return active.div(active.sum(axis=1).replace(0, 1), axis=0).fillna(0.0)


def _run(cash_buffer: float | None, reports: Path) -> Backtester:
    strategy = _Trend(
        strategy_name="cash_buffer_test",
        instruments=["AAPL", "MSFT"],
        backtest_period={"start": "2020-01-01", "end": "2021-12-31"},
        source="sample",
        cash_buffer=cash_buffer,
        reports_directory=str(reports),
        skip_analysis=True,
    )
    asyncio.run(strategy.run_strategy())
    return strategy


def test_default_cash_buffer_holds_five_percent(tmp_path) -> None:
    strategy = _run(None, tmp_path / "default")
    realized = strategy.portfolio_data.weights
    assert realized is not None
    assert realized.sum(axis=1).max() == pytest.approx(0.95, abs=1e-6)


def test_zero_cash_buffer_invests_fully(tmp_path) -> None:
    strategy = _run(0.0, tmp_path / "full")
    realized = strategy.portfolio_data.weights
    assert realized is not None
    assert realized.sum(axis=1).max() == pytest.approx(1.0, abs=1e-6)


def test_invalid_cash_buffer_is_rejected() -> None:
    with pytest.raises(ValueError, match="cash_buffer"):
        Backtester(
            instruments=["AAPL"],
            backtest_period={"start": "2020-01-01", "end": "2021-01-01"},
            source="sample",
            cash_buffer=1.5,
        )
