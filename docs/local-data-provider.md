# Local Data Providers

Run the full strategy pipeline on your own price history instead of the hosted
`/bt/prepare` endpoint. Providers turn local OHLCV frames into the same payload
contract the API returns, so signals, execution, accounting, metrics and reports
behave identically.

This is for research on data you already have. It does not add point-in-time
fundamentals, survivorship-free universes, corporate-action ledgers or licensed
market data — those remain platform concerns.

## Quick start

```python
import asyncio

from backtester import Backtester
from backtester.data import CSVDataProvider
from backtester.portfolio.rebalance import RebalancePolicy


class Trend(Backtester):
    def _compute_signals(self):
        close = self.instruments_data.get_feature("close")
        fast, slow = close.rolling(50).mean(), close.rolling(200).mean()
        return ((fast > slow) & fast.notna() & slow.notna()).astype(float)

    def _compute_weights(self):
        active = self.signals == 1.0
        return active.div(active.sum(axis=1).replace(0, 1), axis=0).fillna(0.0)


async def main() -> None:
    strategy = Trend(
        strategy_name="trend_local",
        instruments=["SPY", "QQQ"],
        backtest_period={"start": "2015-01-01", "end": "2025-01-01"},
        data_provider=CSVDataProvider("data/daily"),
        rebalance_policy=RebalancePolicy(frequency="W"),
        reports_directory="./reports",
    )
    await strategy.run_strategy()
    strategy.print_summary()


asyncio.run(main())
```

`CSVDataProvider("data/daily")` expects `data/daily/SPY.csv`,
`data/daily/QQQ.csv`, and so on. Each file is a table with a `date` column (or a
date index) plus OHLCV columns.

## Available providers

| Provider | Input | Notes |
|---|---|---|
| `CSVDataProvider(directory)` | one `<SYMBOL>.csv` per instrument | simplest path; `date_column=` for a non-standard date header |
| `ParquetDataProvider(directory)` | one `<SYMBOL>.parquet` per instrument | needs `pyarrow` (or another pandas parquet engine) |
| `YahooDataProvider()` | fetched at run time | public chart endpoint; unauthenticated; convenience only |
| `FrameDataProvider({symbol: frame})` | in-memory DataFrames | notebooks, databases, tests |

All four accept `benchmark_returns=` and `instrument_specs=`.

## Frame contract

Each frame must contain `close`; `date` must be a column or the index. Anything
missing is repaired explicitly rather than silently:

- `adj_close` falls back to `close`
- `open`, `high`, `low` fall back to `close` (supply true OHLC when order-level
  execution matters — stop and limit triggers read intrabar highs and lows)
- `volume` falls back to `NaN`, so a volume-participation cap fails closed
  instead of filling orders against fabricated volume

Indexes are normalised to a sorted, unique, UTC `DatetimeIndex`. The provider
fails closed if any requested instrument is missing.

## Benchmark

Pass a benchmark return series to get benchmark-relative report metrics:

```python
from backtester.data import CSVDataProvider

provider = CSVDataProvider("data/daily", benchmark_returns=spy_returns)
```

`spy_returns` must be a `pd.Series` indexed by date.

## Assumptions and limits

- The engine treats the data as authoritative: look-ahead safety is your
  responsibility for adjusted prices, symbol changes and delistings.
- Order-mode modelling requires real intrabar OHLC and volume; a close-only
  dataset degrades stop, limit and participation behaviour.
- FX and futures need `instrument_specs=` (multiplier, lot size, settlement
  currency); otherwise positions are treated as cash equities.
- Yahoo is a convenience source, not a licensed feed; check its terms before
  relying on it for anything beyond local experiments.

## Do not commit data

Market data is usually licensed and often large. Keep it under `data/` (already
ignored) or outside the repository, and run `python
tools/check_private_artifacts.py --staged` before committing. The bundled
`tools/install_pre_commit_hook.py` wires that guard into Git on machines where
hooks can run.
