# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Local market-data providers for offline, user-owned price history.

Pass one of these to ``Backtester(..., data_provider=...)`` to run the normal
strategy pipeline without the hosted ``/bt/prepare`` endpoint::

    from backtester.data import CSVDataProvider

    strategy = MyStrategy(
        instruments=["SPY", "QQQ"],
        backtest_period={"start": "2015-01-01", "end": "2025-01-01"},
        data_provider=CSVDataProvider("data/daily"),
    )

Every provider returns the same frozen payload contract the hosted API uses, so
signals, execution, accounting, metrics and reports are identical.
"""

from backtester.data.base import DataProvider, normalize_price_frame
from backtester.data.payload import build_payload, frame_payload, series_payload
from backtester.data.sources import CSVDataProvider, FrameDataProvider, ParquetDataProvider
from backtester.data.yahoo_source import YahooDataProvider

__all__ = [
    "CSVDataProvider",
    "DataProvider",
    "FrameDataProvider",
    "ParquetDataProvider",
    "YahooDataProvider",
    "build_payload",
    "frame_payload",
    "normalize_price_frame",
    "series_payload",
]
