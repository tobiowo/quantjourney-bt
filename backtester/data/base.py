# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Shared contracts for local market-data providers.

A provider turns user-owned price history into the payload shape that
``Backtester._process_market_data`` already understands, so the engine does not
care whether the bars came from the hosted API or a CSV file.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

REQUIRED_PRICE_FIELDS = ("open", "high", "low", "close", "volume")


@runtime_checkable
class DataProvider(Protocol):
    """Anything that can supply one market-data payload for a backtest."""

    def load(
        self,
        *,
        instruments: list[str],
        start: str,
        end: str,
        granularity: str,
    ) -> dict[str, Any]:
        """Return a /bt/prepare-shaped payload for the requested universe."""
        ...


def _coerce_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a frame indexed by a sorted, unique, UTC DatetimeIndex."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "date" not in frame.columns:
            raise ValueError("price frame needs a DatetimeIndex or a 'date' column")
        frame = frame.set_index(pd.to_datetime(frame.pop("date")))
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    frame.index = index
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.index.name = "date"
    return frame


def normalize_price_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Validate one symbol's OHLCV frame and repair the fields that are optional.

    ``close`` is required. ``adj_close`` falls back to ``close``; ``open``,
    ``high`` and ``low`` fall back to ``close`` because the engine's schema
    requires the columns, so callers should supply true OHLC when order-level
    execution matters. ``volume`` falls back to ``NaN`` so a volume
    participation cap fails closed instead of silently filling orders.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{symbol}: expected a pandas DataFrame, got {type(frame).__name__}")
    frame = frame.rename(columns={column: str(column).strip().lower() for column in frame.columns})
    frame = _coerce_index(frame)

    if "close" not in frame.columns:
        raise ValueError(f"{symbol}: price frame must contain a 'close' column")

    missing = [field for field in ("open", "high", "low") if field not in frame.columns]
    if missing:
        for field in missing:
            frame[field] = frame["close"]
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]
    if "volume" not in frame.columns:
        frame["volume"] = float("nan")

    frame = frame[["open", "high", "low", "close", "adj_close", "volume"]]
    frame = frame[~frame["close"].isna()]
    if frame.empty:
        raise ValueError(f"{symbol}: price frame has no usable close observations")
    return frame
