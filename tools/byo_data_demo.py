# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Bring-your-own-data proof of concept for the QuantJourney Backtester.

The public engine only ships two data paths: the hosted ``/bt/prepare`` endpoint
and the bundled deterministic sample data. This script demonstrates a third,
local path: fetch daily OHLCV from Yahoo Finance's public chart endpoint, build
the same payload shape ``/bt/prepare`` returns, and inject it by overriding
``Backtester._fetch_market_data``.

It is intentionally a proof of concept, not a supported API. It relies on a
private method and on the shape of the market-data payload, both of which may
change between releases. The intended follow-up is a first-class
``data_provider=`` hook; this script is the temporary seam until that lands.

Usage:
    .venv\\Scripts\\python.exe tools\\byo_data_demo.py
    .venv\\Scripts\\python.exe tools\\byo_data_demo.py --symbols SPY,QQQ --years 10y
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester.portfolio.rebalance import RebalancePolicy  # noqa: E402
from strategies.example_weights_01_sma_daily import DailySMATrend  # noqa: E402

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def fetch_yahoo(symbol: str, years: str = "15y") -> pd.DataFrame:
    """Return daily OHLCV for one symbol from Yahoo's public chart endpoint."""
    request = urllib.request.Request(
        YAHOO_CHART.format(symbol=symbol) + f"?range={years}&interval=1d",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode())
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    index = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    frame = pd.DataFrame(
        {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "adj_close": adjusted,
            "volume": quote["volume"],
        },
        index=index,
    )
    return frame.dropna(subset=["close", "adj_close"])


def frame_payload(frame: pd.DataFrame) -> dict:
    """Encode a (instrument, field) DataFrame in the API's JSON shape."""
    columns = [
        {"instrument": str(instrument), "field": str(field)}
        for instrument, field in frame.columns.to_list()
    ]
    safe = frame.astype(object).where(pd.notna(frame), None)
    return {
        "columns": columns,
        "index": [stamp.isoformat() for stamp in frame.index],
        "data": safe.values.tolist(),
    }


def series_payload(series: pd.Series) -> dict:
    """Encode a Series in the API's JSON shape."""
    safe = series.astype(object).where(pd.notna(series), None)
    return {"index": [stamp.isoformat() for stamp in series.index], "data": safe.tolist()}


def build_payload(price_map: dict[str, pd.DataFrame]) -> dict:
    """Assemble the /bt/prepare-shaped payload the engine expects."""
    prices = pd.concat(price_map, axis=1)
    prices.columns.names = ["instrument", "field"]
    dates = prices.index

    returns = pd.DataFrame(
        {
            symbol: frame["adj_close"].pct_change(fill_method=None)
            for symbol, frame in price_map.items()
        },
        index=dates,
    ).fillna(0.0)

    metric_parts = []
    parameter_parts = []
    for symbol in price_map:
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

    portfolio_nav = 100.0 * (1.0 + returns.mean(axis=1)).cumprod()
    return {
        "session_id": "byo-session",
        "dataset_id": "yahoo-public-chart",
        "instrument_names": list(price_map),
        "prices": frame_payload(prices),
        "metrics": frame_payload(metrics),
        "parameters": frame_payload(parameters),
        "nav": series_payload(portfolio_nav),
        "summary": {"source": "yahoo", "instruments": len(price_map)},
    }


class BYODataStrategy(DailySMATrend):
    """W01 SMA trend that injects locally sourced data instead of calling the API."""

    payload: dict

    async def _fetch_market_data(self) -> None:
        self._api_response = self.payload
        self.session_id = self.payload["session_id"]
        self.dataset_id = self.payload["dataset_id"]
        self._validate_data_completeness_response()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="SPY,QQQ,IWM", help="comma-separated Yahoo symbols")
    parser.add_argument("--years", default="15y", help="Yahoo range, e.g. 5y, 10y, 15y")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    price_map = {symbol: fetch_yahoo(symbol, args.years) for symbol in symbols}
    first = list(price_map.values())[0].index[0]
    last = list(price_map.values())[0].index[-1]
    print(f"Loaded {len(price_map)} symbols, {first.date()} -> {last.date()}")

    strategy = BYODataStrategy(
        strategy_name="BYO_Yahoo_RealData",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=symbols,
        backtest_period={"start": str(first.date()), "end": str(last.date())},
        benchmark_symbol="^GSPC",
        benchmark_name="S&P 500",
        source="byo",  # unused: the fetch is overridden below
        execution_mode="weights",
        max_position_size=0.5,
        rebalance_policy=RebalancePolicy(frequency="W"),
        indicators_config=[
            {"function": "SMA", "price_cols": ["close"], "params": {"periods": [50, 200]}}
        ],
        reports_directory=str(ROOT / "reports"),
    )
    strategy.payload = build_payload(price_map)
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
