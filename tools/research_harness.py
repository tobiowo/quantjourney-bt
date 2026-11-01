# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Single-name / sector research harness.

Runs preset strategies (SMA trend, cross-sectional momentum, mean reversion,
buy & hold) over cached Yahoo frames or local CSVs, optionally sweeping a
parameter grid, and writes a comparison table plus a full report for the best
run.

Data is loaded once and shared across runs through the pickle cache, so
parameter sweeps are bit-identical on the same data.

Usage:
    python tools/research_harness.py --tickers NVDA --start 2010-01-01
    python tools/research_harness.py --tickers XLK,XLF,XLE,XLV,XLY,XLP,XLU,XLI,XLB,XLRE,XLC \
        --start 2018-07-02 --strategy momentum --grid "lookback=63,126;top_n=2,3"
    python tools/research_harness.py --tickers AAPL,MSFT --source csv --data-dir data/daily
    python tools/research_harness.py --tickers NVDA --grid "fast=20,50;slow=100,200" \
        --walk-forward --wf-train-months 24 --wf-test-months 6
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import logging
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester import Backtester  # noqa: E402
from backtester.data import FrameDataProvider  # noqa: E402
from backtester.data.base import normalize_price_frame  # noqa: E402
from backtester.execution.commission import FixedBpsCommission  # noqa: E402
from backtester.execution.slippage import FixedBpsSlippage  # noqa: E402
from backtester.portfolio.calc import risk as calc_risk  # noqa: E402
from backtester.portfolio.config import resolve_risk_free_rate  # noqa: E402
from backtester.portfolio.rebalance import RebalancePolicy  # noqa: E402
from backtester.portfolio.weight_cost import FixedBpsWeightCostModel  # noqa: E402
from backtester.risk.vol_target import VolTargetModel  # noqa: E402
from tools.data_cache import CachedYahooFrames  # noqa: E402

DEFAULT_CACHE = ROOT / "data" / "research_cache"
DEFAULT_CSV = ROOT / "data" / "daily"
DEFAULT_OUTPUT = ROOT / "reports" / "_research"


class _PresetStrategy(Backtester):
    """Base class carrying the sweep parameters."""

    def __init__(
        self,
        *,
        params: dict[str, Any],
        eval_start: pd.Timestamp | None = None,
        **kwargs: Any,
    ) -> None:
        self.params = dict(params)
        self._eval_start = eval_start
        super().__init__(**kwargs)

    def _equal_weight(self) -> pd.DataFrame:
        active = (self.signals == 1.0).astype(float)
        if self._eval_start is not None:
            # Warmup bars may be supplied for indicator history; hold cash
            # until the evaluation window starts.
            active.loc[active.index < self._eval_start] = 0.0
        sums = active.sum(axis=1).replace(0.0, 1.0)
        return active.div(sums, axis=0).fillna(0.0)


class SMATrend(_PresetStrategy):
    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        fast = close.rolling(int(self.params.get("fast", 50))).mean()
        slow = close.rolling(int(self.params.get("slow", 200))).mean()
        return ((fast > slow) & fast.notna() & slow.notna()).astype(float)

    def _compute_weights(self) -> pd.DataFrame:
        return self._equal_weight()


class Momentum(_PresetStrategy):
    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        lookback = int(self.params.get("lookback", 126))
        top_n = int(self.params.get("top_n", 3))
        momentum = close.pct_change(lookback)
        ranks = momentum.rank(axis=1, ascending=False)
        active = (ranks <= top_n) & momentum.notna() & (momentum > 0)
        return active.astype(float)

    def _compute_weights(self) -> pd.DataFrame:
        return self._equal_weight()


class MeanReversion(_PresetStrategy):
    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        lookback = int(self.params.get("lookback", 20))
        entry_z = float(self.params.get("entry_z", 1.0))
        mean = close.rolling(lookback).mean()
        std = close.rolling(lookback).std()
        zscore = ((close - mean) / std).to_numpy(dtype=float)

        signals = np.zeros_like(zscore)
        holding = np.zeros(zscore.shape[1], dtype=bool)
        for row_index in range(len(zscore)):
            row = zscore[row_index]
            valid = ~np.isnan(row)
            holding = np.where(valid, np.where(row <= -entry_z, True, row >= 0.0), holding)
            signals[row_index] = holding.astype(float)
        return pd.DataFrame(signals, index=close.index, columns=close.columns)

    def _compute_weights(self) -> pd.DataFrame:
        return self._equal_weight()


class BuyHold(_PresetStrategy):
    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("adj_close")
        return close.notna().astype(float)

    def _compute_weights(self) -> pd.DataFrame:
        return self._equal_weight()


PRESETS: dict[str, type[_PresetStrategy]] = {
    "sma_trend": SMATrend,
    "momentum": Momentum,
    "mean_reversion": MeanReversion,
    "buy_hold": BuyHold,
}
DEFAULT_PARAMS = {
    "sma_trend": {"fast": 50, "slow": 200},
    "momentum": {"lookback": 126, "top_n": 3},
    "mean_reversion": {"lookback": 20, "entry_z": 1.0},
    "buy_hold": {},
}
DEFAULT_REBALANCE = {
    "sma_trend": "D",
    "momentum": "W",
    "mean_reversion": "D",
    "buy_hold": "BME",
}


def _parse_value(raw: str) -> Any:
    text = raw.strip().strip("\"'")
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _parse_kv(spec: str, defaults: dict[str, Any]) -> dict[str, Any]:
    params = dict(defaults)
    for part in spec.split(";"):
        if not part.strip():
            continue
        name, _, value = part.partition("=")
        if not value:
            raise SystemExit(f"invalid parameter {part!r}; expected name=value")
        params[name.strip()] = _parse_value(value)
    return params


def _expand_grid(base: dict[str, Any], grid: str) -> list[dict[str, Any]]:
    if not grid.strip():
        return [dict(base)]
    axes: list[tuple[str, list[Any]]] = []
    for part in grid.split(";"):
        if not part.strip():
            continue
        name, _, values = part.partition("=")
        if not values:
            raise SystemExit(f"invalid grid axis {part!r}; expected name=v1,v2")
        axes.append((name.strip(), [_parse_value(value) for value in values.split(",")]))
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*[axis_values for _, axis_values in axes]):
        params = dict(base)
        params.update({name: value for (name, _), value in zip(axes, values, strict=True)})
        combos.append(params)
    return combos


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _load_frames(args: argparse.Namespace, tickers: list[str]) -> dict[str, pd.DataFrame]:
    if args.source == "csv":
        frames: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            path = Path(args.data_dir) / f"{ticker}.csv"
            if not path.is_file():
                raise SystemExit(f"missing CSV for {ticker}: {path}")
            raw = pd.read_csv(path, parse_dates=["date"], index_col="date")
            frame = normalize_price_frame(ticker, raw)
            start = pd.Timestamp(args.start, tz="UTC")
            end = pd.Timestamp(args.end, tz="UTC")
            frame = frame.loc[(frame.index >= start) & (frame.index < end)]
            if frame.empty:
                raise SystemExit(f"CSV for {ticker} has no rows in {args.start}..{args.end}")
            frames[ticker] = frame
        return frames

    cache = CachedYahooFrames(args.cache_dir, refresh=args.refresh)
    return cache.load_frames(
        instruments=tickers,
        start=args.start,
        end=args.end,
        granularity="1d",
    )


def _frames_digest(frames: dict[str, pd.DataFrame]) -> str:
    hasher = hashlib.sha256()
    for symbol in sorted(frames):
        hasher.update(str(symbol).encode())
        hasher.update(np.nan_to_num(frames[symbol].to_numpy(dtype=float)).tobytes())
    return hasher.hexdigest()[:16]


def _strategy_kwargs(
    args: argparse.Namespace,
    tickers: list[str],
    provider: FrameDataProvider,
    strategy_name: str,
    *,
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "strategy_name": strategy_name,
        "instruments": tickers,
        "backtest_period": {"start": args.start, "end": args.end},
        "data_provider": provider,
        "execution_mode": "weights",
        "cash_buffer": args.cash_buffer,
        "rebalance_policy": RebalancePolicy(frequency=args.rebalance),
        "benchmark_symbol": args.benchmark,
        "benchmark_name": args.benchmark,
        "reports_directory": str(runs_dir or (args.output / "runs")),
        "skip_analysis": True,
    }
    if args.execution == "orders":
        kwargs.update(
            weight_execution="orders",
            slippage_model=FixedBpsSlippage(bps=args.slippage_bps),
            commission_scheme=FixedBpsCommission(bps=args.commission_bps),
        )
    else:
        kwargs["weight_cost_model"] = FixedBpsWeightCostModel(total_bps=args.cost_bps)
    if args.target_vol is not None:
        kwargs["risk_model"] = VolTargetModel(
            target_vol=float(args.target_vol),
            lookback=int(args.vol_lookback),
        )
    return kwargs


_METRIC_KEYS = ("total_return", "cagr", "sharpe", "volatility", "max_drawdown", "turnover", "bars")


def _metrics(strategy: Backtester, eval_start: pd.Timestamp | None = None) -> dict[str, Any]:
    data = strategy.portfolio_data
    nav = data.net_asset_value.dropna()
    weights = getattr(data, "weights", None)
    if eval_start is not None:
        nav = nav.loc[nav.index >= eval_start]
        if weights is not None:
            weights = weights.loc[weights.index >= eval_start]
    if len(nav) < 3:
        raise ValueError("run produced fewer than three NAV observations")
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    returns = nav.pct_change().dropna()
    annualization = max(int(getattr(data, "periods_per_year", 252) or 252), 1)
    sharpe_values = calc_risk.sharpe_ratio(
        returns.to_frame(),
        risk_free_rate=resolve_risk_free_rate(),
        days_per_year=annualization,
        annualize=True,
    )
    sharpe = float(sharpe_values.iloc[0]) if len(sharpe_values) else float("nan")
    drawdown = nav / nav.cummax() - 1.0
    turnover = (
        float(weights.diff().abs().sum(axis=1).mean()) if weights is not None else float("nan")
    )
    return {
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0),
        "sharpe": None if np.isnan(sharpe) else sharpe,
        "volatility": float(returns.std() * np.sqrt(annualization)),
        "max_drawdown": float(drawdown.min()),
        "turnover": None if np.isnan(turnover) else turnover,
        "bars": int(len(nav)),
        "final_nav": float(nav.iloc[-1]),
    }


def _execute_one(
    args: argparse.Namespace,
    tickers: list[str],
    frames: dict[str, pd.DataFrame],
    preset: str,
    params: dict[str, Any],
    label: str,
    *,
    eval_start: pd.Timestamp | None = None,
    runs_dir: Path | None = None,
) -> tuple[dict[str, Any], pd.Series]:
    provider = FrameDataProvider(frames)
    kwargs = _strategy_kwargs(
        args,
        tickers,
        provider,
        _sanitize(label),
        runs_dir=runs_dir,
    )
    strategy = PRESETS[preset](params=params, eval_start=eval_start, **kwargs)
    started = time.perf_counter()
    record: dict[str, Any] = {
        "label": label,
        "preset": preset,
        "params": json.dumps(params, sort_keys=True),
        "status": "ok",
    }
    returns = pd.Series(dtype=float)
    try:
        asyncio.run(strategy.run_strategy())
        record.update(_metrics(strategy, eval_start=eval_start))
        nav = strategy.portfolio_data.net_asset_value.dropna()
        if eval_start is not None:
            nav = nav.loc[nav.index >= eval_start]
        returns = nav.pct_change().dropna()
    except Exception as exc:  # noqa: BLE001 - research harness records failures
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    record["runtime_seconds"] = round(time.perf_counter() - started, 2)
    return record, returns


def _run_one(
    args: argparse.Namespace,
    tickers: list[str],
    frames: dict[str, pd.DataFrame],
    preset: str,
    params: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    record, _ = _execute_one(args, tickers, frames, preset, params, label)
    return record


def _objective_value(record: dict[str, Any], objective: str) -> float:
    value = record.get(objective)
    return float(value) if value is not None and np.isfinite(value) else float("-inf")


def _warmup_bars(preset: str, params: dict[str, Any]) -> int:
    if preset == "sma_trend":
        return max(int(params.get("fast", 0) or 0), int(params.get("slow", 0) or 0))
    if preset in {"momentum", "mean_reversion"}:
        return int(params.get("lookback", 0) or 0)
    return 0


def _slice_frames(
    frames: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    sliced: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        part = frame.loc[(frame.index >= start) & (frame.index < end)]
        if len(part) < 2:
            return {}
        sliced[symbol] = part
    return sliced


def _test_frames(
    frames: dict[str, pd.DataFrame],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    warmup: int,
) -> dict[str, pd.DataFrame]:
    sliced: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        start_pos = max(int(frame.index.searchsorted(test_start)) - warmup, 0)
        end_pos = int(frame.index.searchsorted(test_end))
        part = frame.iloc[start_pos:end_pos]
        if len(part) < 3:
            return {}
        sliced[symbol] = part
    return sliced


def _build_folds(args: argparse.Namespace) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    train_span = pd.DateOffset(months=int(args.wf_train_months))
    test_span = pd.DateOffset(months=int(args.wf_test_months))
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    index = 1
    while index <= 240:
        if args.wf_anchored:
            train_start = start
            train_end = start + pd.DateOffset(
                months=int(args.wf_train_months) + (index - 1) * int(args.wf_step_months)
            )
        else:
            train_start = start + pd.DateOffset(months=int(args.wf_step_months) * (index - 1))
            train_end = train_start + train_span
        test_end = train_end + test_span
        if test_end > end:
            break
        folds.append((train_start, train_end, test_end))
        index += 1
    return folds


def _aggregate_returns(series_list: list[pd.Series]) -> dict[str, Any]:
    usable = [series for series in series_list if len(series) >= 2]
    if not usable:
        return {}
    combined = pd.concat(usable).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    nav = (1.0 + combined).cumprod()
    years = max((combined.index[-1] - combined.index[0]).days / 365.25, 1e-9)
    sharpe_values = calc_risk.sharpe_ratio(
        combined.to_frame(),
        risk_free_rate=resolve_risk_free_rate(),
        days_per_year=252,
        annualize=True,
    )
    sharpe = float(sharpe_values.iloc[0]) if len(sharpe_values) else float("nan")
    drawdown = nav / nav.cummax() - 1.0
    return {
        "total_return": float(nav.iloc[-1] - 1.0),
        "cagr": float(nav.iloc[-1] ** (1.0 / years) - 1.0),
        "sharpe": None if np.isnan(sharpe) else sharpe,
        "volatility": float(combined.std() * np.sqrt(252)),
        "max_drawdown": float(drawdown.min()),
        "bars": int(len(combined)),
    }


def _write_walkforward(output: Path, payload: dict[str, Any]) -> None:
    (output / "walkforward.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    lines = [
        "# Walk-forward validation",
        "",
        f"- Strategy: {payload['strategy']}",
        f"- Window: {payload['start']} → {payload['end']} (end exclusive)",
        f"- Train {payload['train_months']}m / test {payload['test_months']}m / "
        f"step {payload['step_months']}m, anchored={payload['anchored']}",
        f"- Objective: {payload['objective']}",
        f"- Folds: {len(payload['folds'])}",
        "",
        "| Scheme | Total | CAGR | Sharpe | Vol | Max DD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("walk_forward", "WF re-optimised"), ("fixed", "Fixed full-window best")):
        aggregate = payload["aggregates"].get(name)
        if not aggregate:
            continue
        lines.append(
            "| {label} | {total:+.2%} | {cagr:+.2%} | {sharpe:.2f} | {vol:.2%} | {dd:.2%} |".format(
                label=label,
                total=aggregate["total_return"],
                cagr=aggregate["cagr"],
                sharpe=aggregate["sharpe"] or 0.0,
                vol=aggregate["volatility"],
                dd=aggregate["max_drawdown"],
            )
        )
    lines.extend(
        [
            "",
            "| Fold | Train | Test | Chosen | Train Sharpe | Test Total | Test Sharpe | Test DD |",
            "|---:|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for fold in payload["folds"]:
        lines.append(
            "| {fold} | {train_start}→{train_end} | {test_start}→{test_end} | {chosen} | "
            "{train_sharpe:.2f} | {test_total:+.2%} | {test_sharpe:.2f} | {test_dd:.2%} |".format(
                fold=fold["fold"],
                train_start=fold["train_start"],
                train_end=fold["train_end"],
                test_start=fold["test_start"],
                test_end=fold["test_end"],
                chosen=fold["chosen"],
                train_sharpe=fold["train"]["sharpe"] or 0.0,
                test_total=fold["test"]["total_return"],
                test_sharpe=fold["test"]["sharpe"] or 0.0,
                test_dd=fold["test"]["max_drawdown"],
            )
        )
    (output / "walkforward.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _walk_forward(
    args: argparse.Namespace,
    tickers: list[str],
    frames: dict[str, pd.DataFrame],
    combos: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    folds = _build_folds(args)
    if not folds:
        print("Walk-forward: no complete folds for this window", flush=True)
        return None

    strategy_records = [
        record
        for record in records
        if record["status"] == "ok" and record["preset"] == args.strategy
    ]
    if not strategy_records:
        print("Walk-forward: no successful in-sample runs to anchor on", flush=True)
        return None
    fixed_record = max(
        strategy_records,
        key=lambda record: _objective_value(record, args.wf_objective),
    )
    fixed_params = json.loads(fixed_record["params"])
    warmup = max((_warmup_bars(args.strategy, params) for params in combos), default=0) + 5
    runs_dir = args.output / "wf_runs"

    fold_rows: list[dict[str, Any]] = []
    wf_returns: list[pd.Series] = []
    fixed_returns: list[pd.Series] = []
    for fold_index, (train_start, train_end, test_end) in enumerate(folds, 1):
        train_frames = _slice_frames(frames, train_start, train_end)
        if not train_frames:
            continue
        train_best: dict[str, Any] | None = None
        for params in combos:
            label = f"wf{fold_index}_train_" + "_".join(
                f"{key}{value}" for key, value in params.items()
            )
            record, _ = _execute_one(
                args,
                tickers,
                train_frames,
                args.strategy,
                params,
                label,
                runs_dir=runs_dir,
            )
            if record["status"] != "ok":
                continue
            if train_best is None or _objective_value(record, args.wf_objective) > _objective_value(
                train_best, args.wf_objective
            ):
                train_best = record
        if train_best is None:
            continue
        chosen = json.loads(train_best["params"])

        test_frames = _test_frames(frames, train_end, test_end, warmup)
        if not test_frames:
            continue
        test_label = f"wf{fold_index}_test_" + "_".join(
            f"{key}{value}" for key, value in chosen.items()
        )
        test_record, test_series = _execute_one(
            args,
            tickers,
            test_frames,
            args.strategy,
            chosen,
            test_label,
            eval_start=train_end,
            runs_dir=runs_dir,
        )
        if test_record["status"] != "ok":
            continue
        wf_returns.append(test_series)

        fixed_fold: dict[str, Any] | None = None
        fixed_label = f"wf{fold_index}_fixed"
        fixed_record_fold, fixed_series = _execute_one(
            args,
            tickers,
            test_frames,
            args.strategy,
            fixed_params,
            fixed_label,
            eval_start=train_end,
            runs_dir=runs_dir,
        )
        if fixed_record_fold["status"] == "ok":
            fixed_returns.append(fixed_series)
            fixed_fold = {key: fixed_record_fold[key] for key in _METRIC_KEYS}

        fold_rows.append(
            {
                "fold": fold_index,
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": train_end.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "chosen": " ".join(f"{key}={value}" for key, value in chosen.items()),
                "train": {key: train_best[key] for key in _METRIC_KEYS},
                "test": {key: test_record[key] for key in _METRIC_KEYS},
                "fixed": fixed_fold,
            }
        )
        print(
            f"[wf {fold_index}/{len(folds)}] train {train_start.date()}→{train_end.date()} "
            f"test →{test_end.date()}: {fold_rows[-1]['chosen']} | "
            f"test sharpe={test_record['sharpe'] or float('nan'):.2f}",
            flush=True,
        )

    payload = {
        "strategy": args.strategy,
        "start": args.start,
        "end": args.end,
        "train_months": args.wf_train_months,
        "test_months": args.wf_test_months,
        "step_months": args.wf_step_months,
        "anchored": args.wf_anchored,
        "objective": args.wf_objective,
        "fixed_params": fixed_params,
        "folds": fold_rows,
        "aggregates": {
            "walk_forward": _aggregate_returns(wf_returns),
            "fixed": _aggregate_returns(fixed_returns),
        },
    }
    if not fold_rows:
        print("Walk-forward: every fold failed", flush=True)
        return None
    _write_walkforward(args.output, payload)
    wf_aggregate = payload["aggregates"]["walk_forward"]
    fixed_aggregate = payload["aggregates"]["fixed"]
    print(
        f"\nWalk-forward ({len(fold_rows)} folds): "
        f"WF sharpe={wf_aggregate['sharpe'] or float('nan'):.2f} "
        f"total={wf_aggregate['total_return']:+.2%} | "
        f"fixed-best OOS sharpe={fixed_aggregate.get('sharpe') or float('nan'):.2f} "
        f"total={fixed_aggregate.get('total_return', 0.0):+.2%}",
        flush=True,
    )
    print(f"Walk-forward report: {args.output / 'walkforward.md'}")
    return payload


def _write_outputs(
    output: Path,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    tickers: list[str],
    frames: dict[str, pd.DataFrame],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    frame = pd.DataFrame(records)
    frame.to_csv(output / "results.csv", index=False)
    (output / "results.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Research harness run",
        "",
        f"- Tickers: {', '.join(tickers)}",
        f"- Window: {config['start']} → {config['end']} (end exclusive)",
        f"- Strategy: {config['strategy']}; rebalance {config['rebalance']}",
        f"- Execution: {config['execution']}; cash buffer {config['cash_buffer']}",
        f"- Data: {config['source']}, digest `{config['data_digest']}`",
        "",
        "| # | Run | Status | Total | CAGR | Sharpe | Vol | Max DD | Turnover | Secs |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        records,
        key=lambda item: (item["status"] != "ok", -(item.get("sharpe") or -999.0)),
    )
    for index, record in enumerate(ordered, 1):
        if record["status"] != "ok":
            lines.append(
                f"| {index} | {record['label']} | failed | — | — | — | — | — | — | "
                f"{record['runtime_seconds']:.1f} |"
            )
            continue
        lines.append(
            "| {index} | {label} | ok | {total:+.2%} | {cagr:+.2%} | {sharpe:.2f} | "
            "{vol:.2%} | {dd:.2%} | {turnover:.2%} | {secs:.1f} |".format(
                index=index,
                label=record["label"],
                total=record.get("total_return") or 0.0,
                cagr=record.get("cagr") or 0.0,
                sharpe=record.get("sharpe") or 0.0,
                vol=record.get("volatility") or 0.0,
                dd=record.get("max_drawdown") or 0.0,
                turnover=record.get("turnover") or 0.0,
                secs=record.get("runtime_seconds") or 0.0,
            )
        )

    lines.extend(
        ["", "## Data coverage", "", "| Symbol | First | Last | Rows |", "|---|---|---|---:|"]
    )
    for symbol in sorted(frames):
        item = frames[symbol]
        lines.append(
            f"| {symbol} | {item.index[0].date()} | {item.index[-1].date()} | {len(item)} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", required=True, help="comma-separated symbols")
    parser.add_argument("--start", default="2018-07-02", help="start date (inclusive)")
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="end date, exclusive at 00:00 UTC (default: today)",
    )
    parser.add_argument("--strategy", choices=sorted(PRESETS), default="sma_trend")
    parser.add_argument("--params", default="", help="base params, e.g. 'fast=50;slow=200'")
    parser.add_argument("--grid", default="", help="grid, e.g. 'fast=20,50;slow=100,200'")
    parser.add_argument("--rebalance", default=None, help="D, W, BME (preset default if unset)")
    parser.add_argument("--cash-buffer", type=float, default=0.0)
    parser.add_argument("--cost-bps", type=float, default=5.0, help="weight-mode round-trip cost")
    parser.add_argument(
        "--execution",
        choices=("weights", "orders"),
        default="weights",
        help="fast weights or target-weight orders",
    )
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--target-vol", type=float, default=None)
    parser.add_argument("--vol-lookback", type=int, default=63)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--source", choices=("yahoo", "csv"), default="yahoo")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", help="show engine INFO logs")
    parser.add_argument("--walk-forward", action="store_true", help="run walk-forward validation")
    parser.add_argument("--wf-train-months", type=int, default=24)
    parser.add_argument("--wf-test-months", type=int, default=6)
    parser.add_argument("--wf-step-months", type=int, default=6)
    parser.add_argument(
        "--wf-anchored", action="store_true", help="expanding instead of rolling train"
    )
    parser.add_argument(
        "--wf-objective",
        choices=("sharpe", "cagr", "total_return"),
        default="sharpe",
        help="metric used to pick parameters on each train window",
    )
    parser.add_argument("--no-best-report", action="store_true")
    parser.add_argument("--no-benchmark-row", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.getLogger("backtester").setLevel(logging.INFO if args.verbose else logging.WARNING)
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    if not tickers:
        raise SystemExit("no tickers requested")
    if args.start >= args.end:
        raise SystemExit(f"start {args.start} must be before end {args.end}")

    args.rebalance = args.rebalance or DEFAULT_REBALANCE[args.strategy]
    if args.output is None:
        stamp = f"{args.strategy}_{'_'.join(tickers)}"
        args.output = DEFAULT_OUTPUT / _sanitize(stamp)[:80]
    args.output = Path(args.output).resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(tickers)} tickers {args.start} → {args.end} ({args.source})", flush=True)
    frames = _load_frames(args, tickers)
    digest = _frames_digest(frames)
    print(f"Loaded {len(frames)} symbols; digest {digest}", flush=True)

    base_params = _parse_kv(args.params, DEFAULT_PARAMS[args.strategy])
    combos = _expand_grid(base_params, args.grid)
    total = len(combos) + (0 if args.no_benchmark_row else 1)
    print(f"Running {total} backtests", flush=True)

    records: list[dict[str, Any]] = []
    for index, params in enumerate(combos, 1):
        label = args.strategy + " " + " ".join(f"{key}={value}" for key, value in params.items())
        record = _run_one(args, tickers, frames, args.strategy, params, label)
        records.append(record)
        if record["status"] == "ok":
            print(
                f"[{index}/{total}] {label}: total={record['total_return']:+.2%} "
                f"sharpe={record['sharpe']:.2f}",
                flush=True,
            )
        else:
            print(f"[{index}/{total}] {label}: FAILED {record['error']}", flush=True)

    if not args.no_benchmark_row:
        record = _run_one(args, tickers, frames, "buy_hold", {}, "buy_hold equal-weight")
        records.append(record)
        if record["status"] == "ok":
            print(
                f"[{total}/{total}] buy_hold: total={record['total_return']:+.2%} "
                f"sharpe={record['sharpe']:.2f}",
                flush=True,
            )

    config = {
        "tickers": tickers,
        "start": args.start,
        "end": args.end,
        "strategy": args.strategy,
        "rebalance": args.rebalance,
        "execution": args.execution,
        "cash_buffer": args.cash_buffer,
        "cost_bps": args.cost_bps,
        "slippage_bps": args.slippage_bps,
        "commission_bps": args.commission_bps,
        "target_vol": args.target_vol,
        "source": args.source,
        "data_digest": digest,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _write_outputs(args.output, config, records, tickers, frames)

    ok_records = [record for record in records if record["status"] == "ok"]
    ordered = sorted(ok_records, key=lambda item: -(item.get("sharpe") or -999.0))
    print("\nTop runs by Sharpe:")
    for record in ordered[: args.top]:
        print(
            f"  {record['sharpe']:.2f}  {record['label']}  "
            f"(total {record['total_return']:+.2%}, DD {record['max_drawdown']:.2%})"
        )
    print(f"\nResults: {args.output / 'summary.md'}")

    if args.walk_forward and ok_records:
        _walk_forward(args, tickers, frames, combos, ok_records)

    if ok_records and not args.no_best_report:
        best = ordered[0]
        params = json.loads(best["params"])
        print(f"\nBuilding full report for {best['label']} …", flush=True)
        provider = FrameDataProvider(frames)
        kwargs = _strategy_kwargs(
            args,
            tickers,
            provider,
            strategy_name=f"{args.strategy}_best",
        )
        kwargs.update(
            reports_directory=str(args.output / "best_report"),
            skip_analysis=False,
            show_text_reports=True,
            save_text_reports=True,
            save_portfolio_plots=True,
            show_portfolio_plots=False,
        )
        strategy = PRESETS[args.strategy](params=params, **kwargs)
        try:
            asyncio.run(strategy.run_strategy())
            report_path = args.output / "best_report" / f"{args.strategy}_best" / "dashboard.html"
            print(f"Report: {report_path}")
        except Exception as exc:  # noqa: BLE001 - best report is best-effort
            print(f"Best-report run failed: {type(exc).__name__}: {exc}")

    return 0 if ok_records else 1


if __name__ == "__main__":
    raise SystemExit(main())
