# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Tests for local market-data providers (offline runs without /bt/prepare)."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from backtester import Backtester
from backtester.data import (
    CSVDataProvider,
    FrameDataProvider,
    build_payload,
    normalize_price_frame,
)
from backtester.data.yahoo_source import chart_payload_to_frame
from backtester.portfolio.rebalance import RebalancePolicy


def _prices(periods: int = 300, seed: int = 7, start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0004, 0.01, periods)
    close = 100.0 * np.exp(np.cumsum(returns))
    index = pd.bdate_range(start, periods=periods, tz="UTC")
    frame = pd.DataFrame(
        {
            "open": close * (1.0 + rng.normal(0.0, 0.001, periods)),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000_000, 2_000_000, periods).astype(float),
        },
        index=index,
    )
    frame.index.name = "date"
    return frame


class _Trend(Backtester):
    """Minimal two-method strategy so the pipeline can run offline."""

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("close")
        fast = close.rolling(20).mean()
        slow = close.rolling(60).mean()
        return ((fast > slow) & fast.notna() & slow.notna()).astype(float)

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        count = active.sum(axis=1).replace(0, 1)
        return active.div(count, axis=0).fillna(0.0)


def test_payload_has_required_schema() -> None:
    payload = build_payload({"AAA": _prices(), "BBB": _prices(seed=11)}, source="test")

    assert payload["instrument_names"] == ["AAA", "BBB"]
    price_fields = {column["field"] for column in payload["prices"]["columns"]}
    assert {"open", "high", "low", "close", "adj_close", "volume"} <= price_fields
    metric_fields = {column["field"] for column in payload["metrics"]["columns"]}
    assert {
        "returns",
        "volatility",
        "daily_pnl",
        "transaction_costs",
        "net_asset_value",
        "gross_asset_value",
        "daily_net_return",
        "drawdown",
    } <= metric_fields
    parameter_fields = {column["field"] for column in payload["parameters"]["columns"]}
    assert {
        "exchange",
        "units",
        "eligibility",
        "active",
        "forecasts",
        "is_trading_day",
        "day_type",
    } <= parameter_fields
    assert payload["summary"]["instruments"] == 2


def test_normalize_price_frame_repairs_optional_fields() -> None:
    frame = _prices()[["close"]]
    normalized = normalize_price_frame("AAA", frame)

    assert list(normalized.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert (normalized["open"] == normalized["close"]).all()
    assert (normalized["adj_close"] == normalized["close"]).all()
    assert normalized["volume"].isna().all()
    assert str(normalized.index.tz) == "UTC"


def test_normalize_price_frame_requires_close() -> None:
    frame = _prices().drop(columns=["close"])
    with pytest.raises(ValueError, match="close"):
        normalize_price_frame("AAA", frame)


def test_csv_provider_reads_symbol_files_and_clips_window(tmp_path) -> None:
    _prices().to_csv(tmp_path / "AAA.csv", index_label="date")
    _prices(seed=11).to_csv(tmp_path / "BBB.csv", index_label="date")

    provider = CSVDataProvider(tmp_path)
    payload = provider.load(
        instruments=["AAA", "BBB"],
        start="2020-03-01",
        end="2020-12-31",
        granularity="1d",
    )

    assert payload["instrument_names"] == ["AAA", "BBB"]
    stamps = pd.to_datetime([row for row in payload["prices"]["index"]])
    assert stamps.min() >= pd.Timestamp("2020-03-01", tz="UTC")
    assert stamps.max() <= pd.Timestamp("2020-12-31", tz="UTC")


def test_frame_provider_fails_closed_on_missing_instrument() -> None:
    provider = FrameDataProvider({"AAA": _prices()})
    with pytest.raises(ValueError, match="missing requested instruments: BBB"):
        provider.load(
            instruments=["AAA", "BBB"], start="2020-01-01", end="2021-01-01", granularity="1d"
        )


def test_chart_payload_to_frame_parses_quotes_and_errors() -> None:
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "timestamp": [1577923200, 1578009600],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 101.0],
                                "high": [102.0, 103.0],
                                "low": [99.0, 100.0],
                                "close": [101.0, 102.0],
                                "volume": [1_000_000, 1_100_000],
                            }
                        ],
                        "adjclose": [{"adjclose": [100.5, 101.5]}],
                    },
                }
            ],
        }
    }
    frame = chart_payload_to_frame(payload, "AAA")
    assert len(frame) == 2
    assert frame["adj_close"].iloc[0] == pytest.approx(100.5)

    failed = {"chart": {"error": {"description": "Not Found"}, "result": None}}
    with pytest.raises(ValueError, match="Not Found"):
        chart_payload_to_frame(failed, "AAA")


def test_offline_backtest_runs_with_frame_provider(tmp_path) -> None:
    provider = FrameDataProvider({"AAA": _prices(), "BBB": _prices(seed=11)})
    strategy = _Trend(
        strategy_name="local_provider_test",
        instruments=["AAA", "BBB"],
        backtest_period={"start": "2020-02-01", "end": "2021-01-01"},
        data_provider=provider,
        initial_capital=100_000,
        rebalance_policy=RebalancePolicy(frequency="W"),
        reports_directory=str(tmp_path),
        skip_analysis=True,
    )

    asyncio.run(strategy.run_strategy())

    nav = strategy.portfolio_data.net_asset_value
    assert len(nav) > 0
    assert np.isfinite(nav.iloc[-1])
    assert (nav > 0).all()

    weights = strategy.weights
    assert weights is not None
    assert weights.sum(axis=1).max() <= 1.0 + 1e-9


def test_provider_benchmark_returns_are_attached(tmp_path) -> None:
    prices = _prices()
    benchmark = pd.Series(np.linspace(0.0, 0.2, len(prices)), index=prices.index)
    provider = FrameDataProvider({"AAA": prices}, benchmark_returns=benchmark)
    strategy = _Trend(
        strategy_name="local_benchmark_test",
        instruments=["AAA"],
        backtest_period={"start": "2020-02-01", "end": "2021-01-01"},
        data_provider=provider,
        reports_directory=str(tmp_path),
        skip_analysis=True,
    )

    asyncio.run(strategy.run_strategy())

    attached = strategy._benchmark_returns
    assert attached is not None
    assert len(attached) == len(benchmark)
    assert attached.name == "benchmark"
