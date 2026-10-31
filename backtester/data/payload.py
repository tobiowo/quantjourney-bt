# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Build a /bt/prepare-shaped payload from local price frames.

The engine consumes one frozen payload contract regardless of data origin. This
module is the single place that turns user-owned OHLCV frames into that
contract, so every local provider (CSV, Parquet, Yahoo, in-memory) produces
identical, validated output.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pandas as pd

from backtester.data.base import normalize_price_frame

METRIC_FIELDS = (
    "returns",
    "volatility",
    "daily_pnl",
    "transaction_costs",
    "net_asset_value",
    "gross_asset_value",
    "daily_net_return",
    "drawdown",
)

PARAMETER_FIELDS = (
    "exchange",
    "units",
    "eligibility",
    "active",
    "forecasts",
    "is_trading_day",
    "day_type",
)


def frame_payload(frame: pd.DataFrame) -> dict[str, Any]:
    """Encode an (instrument, field) DataFrame in the API's JSON shape."""
    columns: list[dict[str, str]] = [
        {"instrument": str(instrument), "field": str(field)}
        for instrument, field in cast(list[tuple[str, str]], list(frame.columns))
    ]
    null = cast(Any, None)
    safe = frame.astype(object).where(pd.notna(frame), null)
    return {
        "columns": columns,
        "index": [stamp.isoformat() for stamp in frame.index],
        "data": safe.values.tolist(),
    }


def series_payload(series: pd.Series) -> dict[str, Any]:
    """Encode a Series in the API's JSON shape."""
    null = cast(Any, None)
    safe = series.astype(object).where(pd.notna(series), null)
    return {"index": [stamp.isoformat() for stamp in series.index], "data": safe.tolist()}


def _normalize_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(f"{name}: expected a DatetimeIndex")
    index = pd.DatetimeIndex(series.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    series = series.copy()
    series.index = index
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series.astype("float64")


def build_payload(
    price_map: Mapping[str, pd.DataFrame],
    *,
    source: str = "local",
    dataset_id: str = "local-dataset",
    session_id: str = "local-session",
    initial_nav: float = 100.0,
    benchmark_returns: pd.Series | None = None,
    instrument_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the payload ``Backtester._process_market_data`` expects.

    ``price_map`` maps symbol to an OHLCV frame. Each frame is validated by
    :func:`normalize_price_frame`; missing optional fields are repaired there.
    The engine's metric and parameter tables are synthesised from the prices,
    because local providers do not ship the warehouse's pre-computed columns.
    """
    if not price_map:
        raise ValueError("price_map must contain at least one symbol")

    normalized: dict[str, pd.DataFrame] = {}
    for symbol, frame in price_map.items():
        normalized[str(symbol).strip().upper()] = normalize_price_frame(str(symbol), frame)

    prices = pd.concat(normalized, axis=1)
    prices.columns.names = ["instrument", "field"]
    dates = prices.index

    returns = pd.DataFrame(
        {
            symbol: frame["adj_close"].pct_change(fill_method=None)
            for symbol, frame in normalized.items()
        },
        index=dates,
    ).fillna(0.0)

    metric_parts: list[pd.DataFrame] = []
    parameter_parts: list[pd.DataFrame] = []
    for symbol in normalized:
        symbol_returns = returns[symbol]
        nav = (1.0 + symbol_returns).cumprod()
        metric_parts.append(
            pd.DataFrame(
                {
                    (symbol, "returns"): symbol_returns,
                    (symbol, "volatility"): (
                        symbol_returns.rolling(20, min_periods=2).std().fillna(0.0)
                    ),
                    (symbol, "daily_pnl"): symbol_returns,
                    (symbol, "transaction_costs"): 0.0,
                    (symbol, "net_asset_value"): nav,
                    (symbol, "gross_asset_value"): nav,
                    (symbol, "daily_net_return"): symbol_returns,
                    (symbol, "drawdown"): nav / nav.cummax() - 1.0,
                },
                index=dates,
            )
        )
        parameter_parts.append(
            pd.DataFrame(
                {
                    (symbol, "exchange"): 0.0,
                    (symbol, "units"): 0.0,
                    (symbol, "eligibility"): 1.0,
                    (symbol, "active"): 1.0,
                    (symbol, "forecasts"): 0.0,
                    (symbol, "is_trading_day"): 1.0,
                    (symbol, "day_type"): 1.0,
                },
                index=dates,
            )
        )

    metrics = pd.concat(metric_parts, axis=1)
    parameters = pd.concat(parameter_parts, axis=1)
    for frame in (metrics, parameters):
        frame.columns.names = ["instrument", "field"]

    portfolio_nav = float(initial_nav) * (1.0 + returns.mean(axis=1)).cumprod()
    payload: dict[str, Any] = {
        "session_id": session_id,
        "dataset_id": dataset_id,
        "instrument_names": list(normalized),
        "prices": frame_payload(prices),
        "metrics": frame_payload(metrics),
        "parameters": frame_payload(parameters),
        "nav": series_payload(portfolio_nav),
        "summary": {
            "source": source,
            "instruments": len(normalized),
            "dates": int(len(dates)),
            "start": dates[0].date().isoformat(),
            "end": dates[-1].date().isoformat(),
            "partial": False,
            "missing_instruments": [],
        },
    }
    if instrument_specs:
        payload["instrument_specs"] = {
            str(symbol).strip().upper(): dict(spec) for symbol, spec in instrument_specs.items()
        }
    if benchmark_returns is not None:
        payload["benchmark_returns"] = series_payload(
            _normalize_series(benchmark_returns, "benchmark_returns")
        )
    return payload
