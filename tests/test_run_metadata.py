# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Run metadata must identify the code, inputs and cost assumptions used."""

from __future__ import annotations

import asyncio
import json

import pandas as pd

from backtester import Backtester


class _Trend(Backtester):
    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("close")
        fast = close.rolling(20).mean()
        slow = close.rolling(60).mean()
        return ((fast > slow) & fast.notna() & slow.notna()).astype(float)

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        return active.div(active.sum(axis=1).replace(0, 1), axis=0).fillna(0.0)


def test_run_metadata_records_reproducibility_fields(tmp_path) -> None:
    strategy = _Trend(
        strategy_name="metadata_test",
        instruments=["AAPL", "MSFT"],
        backtest_period={"start": "2020-01-01", "end": "2021-12-31"},
        source="sample",
        reports_directory=str(tmp_path),
        skip_analysis=True,
    )
    asyncio.run(strategy.run_strategy())

    metadata = json.loads((tmp_path / "metadata_test" / "run_metadata.json").read_text())

    fingerprint = metadata["data_fingerprint"]
    assert fingerprint.startswith("sha256:")
    assert len(fingerprint) == len("sha256:") + 64

    revision = metadata["git_revision"]
    assert revision is None or (isinstance(revision, str) and len(revision) >= 40)

    execution = metadata["execution"]
    assert execution["mode"] == "weights"
    assert execution["fill_at"] is None  # fast weight path has no fill engine
    assert execution["rebalance_policy"]

    costs = metadata["costs"]
    assert costs["transaction_cost_fraction"] == 0.0001
    assert costs["weight_cost_model"] == "FixedBpsWeightCostModel"
    assert costs["slippage_model"] is None  # fast weight path has no fill engine
