# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Run an example strategy on real Yahoo Finance prices, no QuantJourney account.

This is the supported local-data path: ``YahooDataProvider`` turns Yahoo's
public chart responses into the same payload the hosted ``/bt/prepare`` endpoint
returns, and ``Backtester(data_provider=...)`` consumes it. The strategy code is
unchanged from ``strategies/example_weights_01_sma_daily.py``.

Yahoo is a convenience source for local research, not a licensed feed; check its
terms before relying on it.

Usage:
    .venv\\Scripts\\python.exe tools\\byo_data_demo.py
    .venv\\Scripts\\python.exe tools\\byo_data_demo.py --symbols SPY,QQQ --years 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester.data import YahooDataProvider  # noqa: E402
from backtester.portfolio.rebalance import RebalancePolicy  # noqa: E402
from strategies.example_weights_01_sma_daily import DailySMATrend  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="SPY,QQQ,IWM", help="comma-separated Yahoo symbols")
    parser.add_argument("--years", type=int, default=15, help="lookback in years (default 15)")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    end = date.today()
    start = end - timedelta(days=365 * args.years)
    print(f"Fetching {len(symbols)} symbols, {start} -> {end}")

    strategy = DailySMATrend(
        strategy_name="BYO_Yahoo_RealData",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=symbols,
        backtest_period={"start": str(start), "end": str(end)},
        benchmark_symbol="^GSPC",
        benchmark_name="S&P 500",
        data_provider=YahooDataProvider(),
        execution_mode="weights",
        max_position_size=0.5,
        rebalance_policy=RebalancePolicy(frequency="W"),
        indicators_config=[
            {"function": "SMA", "price_cols": ["close"], "params": {"periods": [50, 200]}}
        ],
        reports_directory=str(ROOT / "reports"),
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
