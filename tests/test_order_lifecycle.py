# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Lifecycle tests for limit, stop, trailing, bracket and OCO orders.

The example catalogue teaches these order types but the suite only imported the
modules. These tests pin the matching rules on single bars, including gap and
no-fill cases, so execution semantics cannot drift silently.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtester.execution import FillEngine
from backtester.execution.order_types import (
    BarData,
    BracketSpec,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)

STAMP = pd.Timestamp("2024-01-02", tz="UTC")


def _bar(open_: float, high: float, low: float, close: float, volume: float = 1_000.0) -> BarData:
    return BarData(
        timestamp=STAMP,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _engine() -> FillEngine:
    return FillEngine(fill_at="open")


def test_limit_buy_fills_at_limit_or_better() -> None:
    engine = _engine()
    order = Order(
        instrument="AAA",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=97.0,
    )
    engine.submit(order)

    # Open above the limit: fill at the limit, never worse.
    fills = engine.process_bar("AAA", _bar(100.0, 101.0, 95.0, 99.0))
    assert len(fills) == 1
    assert fills[0].fill_price == pytest.approx(97.0)
    assert order.status is OrderStatus.FILLED


def test_limit_buy_gets_price_improvement_on_gap() -> None:
    engine = _engine()
    order = Order(
        instrument="AAA",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=97.0,
    )
    engine.submit(order)

    fills = engine.process_bar("AAA", _bar(95.0, 96.0, 94.0, 95.5))
    assert len(fills) == 1
    assert fills[0].fill_price == pytest.approx(95.0)
    assert order.status is OrderStatus.FILLED


def test_limit_buy_does_not_fill_above_the_limit() -> None:
    engine = _engine()
    order = Order(
        instrument="AAA",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=97.0,
    )
    engine.submit(order)

    fills = engine.process_bar("AAA", _bar(99.0, 100.0, 98.0, 99.5))
    assert fills == []
    assert order.status is OrderStatus.PENDING


def test_stop_buy_gaps_through_the_trigger() -> None:
    engine = _engine()
    order = Order(
        instrument="AAA",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.STOP,
        stop_price=105.0,
    )
    engine.submit(order)

    fills = engine.process_bar("AAA", _bar(108.0, 109.0, 107.0, 108.5))
    assert len(fills) == 1
    # A gap above the stop fills at the open, not at the stale stop price.
    assert fills[0].fill_price == pytest.approx(108.0)
    assert order.status is OrderStatus.FILLED


def test_stop_limit_waits_for_a_reachable_limit() -> None:
    engine = _engine()
    order = Order(
        instrument="AAA",
        side=OrderSide.SELL,
        quantity=10,
        order_type=OrderType.STOP_LIMIT,
        stop_price=95.0,
        limit_price=94.0,
    )
    engine.submit(order)

    # Stop triggers, but the bar never trades back up to the 94 limit: no fill.
    fills = engine.process_bar("AAA", _bar(90.0, 92.0, 88.0, 89.0))
    assert fills == []
    assert order.status is OrderStatus.PENDING

    # A later bar trades through the limit, so the order fills at no worse.
    fills = engine.process_bar("AAA", _bar(90.0, 96.0, 89.0, 95.0))
    assert len(fills) == 1
    assert fills[0].fill_price == pytest.approx(94.0)
    assert order.status is OrderStatus.FILLED


def test_trailing_stop_ratchets_with_highs() -> None:
    engine = _engine()
    order = Order(
        instrument="AAA",
        side=OrderSide.SELL,
        quantity=10,
        order_type=OrderType.STOP_TRAIL,
        trail_amount=5.0,
    )
    engine.submit(order)

    # High 110 sets the anchor; the stop sits at 105 and is not touched.
    assert engine.process_bar("AAA", _bar(100.0, 110.0, 106.0, 108.0)) == []

    # High 120 ratchets the anchor up; low 112 breaches 115 and fills at the open.
    fills = engine.process_bar("AAA", _bar(111.0, 120.0, 112.0, 118.0))
    assert len(fills) == 1
    assert fills[0].fill_price == pytest.approx(111.0)
    assert order.status is OrderStatus.FILLED


def test_bracket_entry_then_take_profit_cancels_stop_loss() -> None:
    engine = _engine()
    entry = Order(
        instrument="AAA",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.BRACKET,
        bracket=BracketSpec(take_profit_price=110.0, stop_loss_price=95.0),
    )
    engine.submit(entry)

    # Entry fills at the open; children are armed but not matched on this bar.
    entry_fills = engine.process_bar("AAA", _bar(100.0, 101.0, 99.0, 100.5))
    assert len(entry_fills) == 1
    assert entry_fills[0].fill_price == pytest.approx(100.0)

    children = [o for o in engine.order_history if o.tag.startswith("bracket_")]
    take_profit = next(o for o in children if o.tag.startswith("bracket_tp"))
    stop_loss = next(o for o in children if o.tag.startswith("bracket_sl"))
    assert take_profit.is_active and stop_loss.is_active

    # Take profit is reached: the OCO sibling is cancelled.
    exit_fills = engine.process_bar("AAA", _bar(105.0, 112.0, 104.0, 111.0))
    assert len(exit_fills) == 1
    assert exit_fills[0].fill_price == pytest.approx(110.0)
    assert take_profit.status is OrderStatus.FILLED
    assert stop_loss.status is OrderStatus.CANCELLED


def test_oco_pair_cancels_the_sibling_leg() -> None:
    engine = _engine()
    pair = "dip-or-breakout"
    dip = Order(
        instrument="AAA",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=95.0,
        oco_pair_id=pair,
    )
    breakout = Order(
        instrument="AAA",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.STOP,
        stop_price=105.0,
        oco_pair_id=pair,
    )
    engine.submit(dip)
    engine.submit(breakout)

    # The dip limit is touched; the breakout stop must be cancelled, not filled.
    fills = engine.process_bar("AAA", _bar(100.0, 101.0, 94.0, 96.0))
    assert len(fills) == 1
    assert fills[0].fill_price == pytest.approx(95.0)
    assert dip.status is OrderStatus.FILLED
    assert breakout.status is OrderStatus.CANCELLED
