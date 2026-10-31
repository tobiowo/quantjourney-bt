# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Order quantities and trade notionals must respect contract specifications."""

from __future__ import annotations

import pandas as pd
import pytest

from backtester import Backtester
from backtester.execution import ContractSpec


def _backtester(symbol: str, spec: ContractSpec) -> Backtester:
    return Backtester(
        instruments=[symbol],
        backtest_period={"start": "2024-01-01", "end": "2024-12-31"},
        execution_mode="orders",
        contract_specs={symbol: spec},
        show_text_reports=False,
        skip_analysis=True,
        reports_directory="reports/_unused",
    )


def _set_context(strategy: Backtester, symbol: str, price: float = 100.0) -> None:
    strategy._order_context = {
        "date": pd.Timestamp("2024-01-02", tz="UTC"),
        "bars": {symbol: None},
        "positions": {symbol: 0.0},
        "nav": 100_000.0,
    }


def test_futures_order_quantity_rounds_to_whole_contracts() -> None:
    strategy = _backtester("ES", ContractSpec.future("ES", multiplier=50.0))
    _set_context(strategy, "ES")

    assert strategy.order_market("ES", 1.7) is not None
    assert strategy.fill_engine.pending_orders[0].quantity == pytest.approx(1.0)

    # A nonzero delta below one contract must not create an order.
    assert strategy.order_market("ES", 0.4) is None
    assert len(strategy.fill_engine.pending_orders) == 1


def test_equity_orders_default_to_whole_shares() -> None:
    strategy = _backtester("AAPL", ContractSpec.equity("AAPL"))
    _set_context(strategy, "AAPL")

    assert strategy.order_market("AAPL", 2.6) is not None
    assert strategy.fill_engine.pending_orders[0].quantity == pytest.approx(2.0)


def test_fractional_quantities_allowed_when_step_disabled() -> None:
    strategy = _backtester("AAPL", ContractSpec.equity("AAPL", quantity_step=None))
    _set_context(strategy, "AAPL")

    assert strategy.order_market("AAPL", 2.6) is not None
    assert strategy.fill_engine.pending_orders[0].quantity == pytest.approx(2.6)


def test_trade_value_applies_contract_multiplier() -> None:
    strategy = Backtester(
        instruments=["ES"],
        backtest_period={"start": "2024-01-01", "end": "2024-12-31"},
        contract_specs={"ES": ContractSpec.future("ES", multiplier=50.0)},
        show_text_reports=False,
        skip_analysis=True,
        reports_directory="reports/_unused",
    )
    dates = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC")]
    )
    position_changes = pd.DataFrame({"ES": [2.0, -2.0]}, index=dates)
    prices = pd.DataFrame({"ES": [100.0, 101.0]}, index=dates)

    trades = strategy._record_trades(position_changes, prices, dates)

    assert list(trades["TradeValue"]) == pytest.approx([2 * 100 * 50, 2 * 101 * 50])


def test_fx_trade_value_uses_lot_size() -> None:
    strategy = Backtester(
        instruments=["EURUSD"],
        backtest_period={"start": "2024-01-01", "end": "2024-12-31"},
        contract_specs={"EURUSD": ContractSpec.fx("EURUSD", lot_size=100_000.0)},
        show_text_reports=False,
        skip_analysis=True,
        reports_directory="reports/_unused",
    )
    dates = pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz="UTC")])
    position_changes = pd.DataFrame({"EURUSD": [1.0]}, index=dates)
    prices = pd.DataFrame({"EURUSD": [1.10]}, index=dates)

    trades = strategy._record_trades(position_changes, prices, dates)

    assert list(trades["TradeValue"]) == pytest.approx([1.10 * 100_000.0])
