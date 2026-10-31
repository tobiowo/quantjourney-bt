# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Pre-trade limits are opt-in; the default must not silently cap leverage.

The engine used to create ``PreTradeRisk(max_margin_utilization=1.0)`` for the
weight-to-orders path, which rejected whole orders for any target above 1x and
could leave a leveraged strategy with no position at all. The default is now
passthrough, limits are recorded in run metadata, and rejections are warned.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pandas as pd

from backtester import Backtester
from backtester.risk.pre_trade import PreTradeRisk

TARGET_WEIGHT = 1.5


class _Levered(Backtester):
    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("close")
        return pd.DataFrame(1.0, index=close.index, columns=close.columns)

    def _compute_weights(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("close")
        return pd.DataFrame(TARGET_WEIGHT, index=close.index, columns=close.columns)


def _run(tmp_path, pre_trade_risk=None) -> Backtester:
    strategy = _Levered(
        strategy_name="pretrade_test",
        instruments=["AAPL"],
        backtest_period={"start": "2020-01-01", "end": "2021-01-01"},
        source="sample",
        execution_mode="weights",
        weight_execution="orders",
        max_position_size=None,
        pre_trade_risk=pre_trade_risk,
        reports_directory=str(tmp_path),
        skip_analysis=True,
    )
    asyncio.run(strategy.run_strategy())
    return strategy


def _metadata(tmp_path) -> dict:
    return json.loads((tmp_path / "pretrade_test" / "run_metadata.json").read_text())


def test_default_pre_trade_risk_is_passthrough(tmp_path) -> None:
    strategy = _run(tmp_path)

    final_weight = float(strategy.portfolio_data.weights.iloc[-1].sum())
    assert final_weight > 1.0
    assert strategy.pre_trade_risk.is_passthrough

    execution = _metadata(tmp_path)["execution"]
    risk = execution["pre_trade_risk"]
    assert risk["is_passthrough"] is True
    assert risk["max_margin_utilization"] is None
    assert risk["max_gross_leverage"] is None
    assert execution["rejected_orders"] == 0


def test_explicit_cap_rejects_and_is_recorded(tmp_path) -> None:
    strategy = _run(tmp_path, PreTradeRisk(max_margin_utilization=1.0))

    final_weight = float(strategy.portfolio_data.weights.iloc[-1].sum())
    assert final_weight < 0.05
    rejected = [order for order in strategy.fill_engine.order_history if order.rejection_reason]
    assert rejected

    execution = _metadata(tmp_path)["execution"]
    risk = execution["pre_trade_risk"]
    assert risk["is_passthrough"] is False
    assert risk["max_margin_utilization"] == 1.0
    assert execution["rejected_orders"] == len(rejected)


def test_leverage_and_rejection_warnings(tmp_path, caplog) -> None:
    # The package logger sets propagate=False, so caplog must attach to it.
    package_logger = logging.getLogger("backtester")
    package_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="backtester")
    try:
        _run(tmp_path / "default")
        messages = [record.getMessage() for record in caplog.records]
        assert any("target gross leverage" in message for message in messages)
        assert not any("order rejected" in message for message in messages)

        caplog.clear()
        _run(tmp_path / "capped", PreTradeRisk(max_margin_utilization=1.0))
        messages = [record.getMessage() for record in caplog.records]
        assert any("order rejected" in message for message in messages)
    finally:
        package_logger.removeHandler(caplog.handler)
