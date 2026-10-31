# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Deterministic sample market-data payloads for credential-free demos.

The synthetic bars are illustrative, not historical market data. They exist so
the install path, the provider schema and the report pack can be exercised
without an account.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backtester.data.payload import build_payload


def _sample_dates(start: str, end: str) -> pd.DatetimeIndex:
    dates = pd.bdate_range(start=start, end=end, tz="UTC")
    if len(dates) < 260:
        dates = pd.bdate_range(end=pd.Timestamp(end, tz="UTC"), periods=260)
    return dates


def _synthetic_frame(symbol_index: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    t = np.arange(len(dates), dtype=float)
    drift = 0.00038 + symbol_index * 0.00004
    cycle = 0.00065 * np.sin(t / (23.0 + symbol_index) + symbol_index * 0.6)
    shorter_cycle = 0.00035 * np.cos(t / (9.0 + symbol_index * 0.5))
    stress = -0.0030 * np.exp(-0.5 * ((t - (1150.0 + symbol_index * 5.0)) / 16.0) ** 2)
    rebound = 0.0020 * np.exp(-0.5 * ((t - (1210.0 + symbol_index * 5.0)) / 24.0) ** 2)
    daily_return = drift + cycle + shorter_cycle + stress + rebound
    close = (80.0 + symbol_index * 12.0) * np.exp(np.cumsum(daily_return))
    open_ = close * (1.0 + 0.0015 * np.sin(t / 5.0 + symbol_index))
    high = np.maximum(open_, close) * (1.0 + 0.004 + 0.001 * np.sin(t / 11.0))
    low = np.minimum(open_, close) * (1.0 - 0.004 - 0.001 * np.cos(t / 13.0))
    volume = 1_000_000 + (symbol_index + 1) * 75_000 + (50_000 * (1.0 + np.sin(t / 9.0)))

    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=dates,
    )
    frame.index.name = "date"
    return frame


def build_sample_bt_payload(
    *,
    instruments: list[str],
    start: str,
    end: str,
    initial_nav: float = 100.0,
) -> dict[str, Any]:
    """Build a /bt/prepare-compatible payload from deterministic OHLCV data."""
    symbols = [str(symbol).strip().upper() for symbol in instruments if str(symbol).strip()]
    if not symbols:
        symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

    dates = _sample_dates(start, end)
    price_map = {symbol: _synthetic_frame(index, dates) for index, symbol in enumerate(symbols)}
    return build_payload(
        price_map,
        source="sample",
        dataset_id="sample-dataset",
        session_id="sample-session",
        initial_nav=initial_nav,
    )
