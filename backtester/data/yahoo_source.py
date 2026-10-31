# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Yahoo Finance public-chart data provider.

Yahoo's chart endpoint is unauthenticated and needs no QuantJourney account. It
is a convenience source for local research, not a licensed market-data feed:
users are responsible for Yahoo's terms and for the quality of the bars.

This module deliberately depends on ``urllib`` rather than ``yfinance`` so the
engine keeps its dependency surface small.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd

from backtester.data.payload import build_payload

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Backtester granularity -> Yahoo interval.
INTERVAL_BY_GRANULARITY = {
    "1d": "1d",
    "1w": "1wk",
    "1h": "60m",
    "30m": "30m",
    "15m": "15m",
    "5m": "5m",
    "1m": "1m",
}


def chart_payload_to_frame(payload: dict[str, Any], symbol: str) -> pd.DataFrame:
    """Parse one Yahoo chart JSON response into an OHLCV frame."""
    chart = payload.get("chart") or {}
    error = chart.get("error")
    if error:
        description = error.get("description") if isinstance(error, dict) else str(error)
        raise ValueError(f"Yahoo chart request for {symbol} failed: {description}")
    results = chart.get("result") or []
    if not results:
        raise ValueError(f"Yahoo chart request for {symbol} returned no result")

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quotes = indicators.get("quote") or [{}]
    quote = quotes[0]
    adjusted = (indicators.get("adjclose") or [{}])[0].get("adjclose")

    frame = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "adj_close": adjusted if adjusted is not None else quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True),
    )
    frame.index.name = "date"
    return frame.dropna(subset=["close"])


class YahooDataProvider:
    """Fetch daily (or intraday) bars from Yahoo's public chart endpoint."""

    def __init__(
        self,
        *,
        interval: str | None = None,
        timeout: float = 30.0,
        dataset_id: str = "yahoo-public-chart",
        benchmark_returns: pd.Series | None = None,
        instrument_specs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._interval = interval
        self._timeout = float(timeout)
        self._dataset_id = dataset_id
        self._benchmark_returns = benchmark_returns
        self._instrument_specs = dict(instrument_specs) if instrument_specs else None

    def _interval_for(self, granularity: str) -> str:
        if self._interval is not None:
            return self._interval
        try:
            return INTERVAL_BY_GRANULARITY[str(granularity).strip().lower()]
        except KeyError as exc:
            raise ValueError(
                f"Yahoo provider does not support granularity {granularity!r}; "
                f"choose one of {sorted(INTERVAL_BY_GRANULARITY)} or pass interval="
            ) from exc

    def _fetch(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
        period2 = int(pd.Timestamp(end, tz="UTC").timestamp())
        query = urllib.parse.urlencode(
            {"period1": period1, "period2": period2, "interval": interval, "events": "div,split"}
        )
        request = urllib.request.Request(
            f"{YAHOO_CHART_URL.format(symbol=urllib.parse.quote(symbol))}?{query}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            payload = json.loads(response.read().decode())
        return chart_payload_to_frame(payload, symbol)

    def load_frames(
        self,
        *,
        instruments: list[str],
        start: str,
        end: str,
        granularity: str,
    ) -> dict[str, pd.DataFrame]:
        """Fetch raw OHLCV frames, before payload construction."""
        interval = self._interval_for(granularity)
        return {
            str(symbol).strip().upper(): self._fetch(str(symbol), start, end, interval)
            for symbol in instruments
        }

    def load(
        self,
        *,
        instruments: list[str],
        start: str,
        end: str,
        granularity: str,
    ) -> dict[str, Any]:
        frames = self.load_frames(
            instruments=instruments,
            start=start,
            end=end,
            granularity=granularity,
        )
        return build_payload(
            frames,
            source="yahoo",
            dataset_id=self._dataset_id,
            benchmark_returns=self._benchmark_returns,
            instrument_specs=self._instrument_specs,
        )
