# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Run the example catalogue against local Yahoo Finance data.

Each strategy file is imported and its ``main()`` executed with the data-fetch
step redirected to ``YahooDataProvider``. Reports are disabled for speed; the
runner records headline metrics, runtime, skips and failures into JSON/CSV.

This is a research tool, not a test: it needs network access, results depend on
Yahoo's bars, and order-mode/FX/futures examples lack provider contract specs.

Usage:
    python tools/run_catalog_local.py
    python tools/run_catalog_local.py --filter weights --limit 5
    python tools/run_catalog_local.py --include-intraday --include-optimizers
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import importlib.util
import inspect
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester import Backtester  # noqa: E402
from backtester.data import YahooDataProvider  # noqa: E402
from backtester.portfolio.calc import risk as calc_risk  # noqa: E402
from backtester.portfolio.config import resolve_risk_free_rate  # noqa: E402

STRATEGIES = ROOT / "strategies"
DEFAULT_OUTPUT = ROOT / "reports" / "_catalog"

_ORIGINAL_FETCH = Backtester._fetch_market_data
_ORIGINAL_RUN = Backtester.run_strategy
_CAPTURED: list[Backtester] = []


class SkipRun(Exception):
    """Raised when the local provider cannot serve this example."""


async def _local_fetch(self: Backtester) -> None:
    granularity = str(getattr(self, "_granularity", "1d")).strip().lower()
    if granularity not in {"1d", "1w"}:
        raise SkipRun(f"granularity {granularity!r} is not served by the Yahoo provider")
    self._data_provider = YahooDataProvider()
    await _ORIGINAL_FETCH(self)


async def _local_run(self: Backtester) -> None:
    await _ORIGINAL_RUN(self)
    _CAPTURED.append(self)


def _install_patch() -> None:
    Backtester._fetch_market_data = _local_fetch  # type: ignore[method-assign]
    Backtester.run_strategy = _local_run  # type: ignore[method-assign]


def _strategy_code(path: Path) -> str:
    match = re.match(r"example_(weights|orders|wf)_(\d+)_", path.stem)
    if not match:
        return path.stem
    prefix = {"weights": "W", "orders": "O", "wf": "WF"}[match.group(1)]
    return f"{prefix}{int(match.group(2)):02d}"


def _title(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return path.stem
    docstring = ast.get_docstring(tree) or ""
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return path.stem


def _metrics(strategy: Backtester) -> dict[str, Any]:
    data = getattr(strategy, "portfolio_data", None)
    nav = None if data is None else getattr(data, "net_asset_value", None)
    payload: dict[str, Any] = {
        "final_nav": None,
        "total_return": None,
        "cagr": None,
        "sharpe": None,
        "max_drawdown": None,
        "observations": 0,
    }
    if nav is None or len(nav.dropna()) < 2:
        return payload

    nav = nav.dropna()
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    payload["final_nav"] = float(nav.iloc[-1])
    payload["total_return"] = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    payload["cagr"] = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)
    payload["observations"] = int(len(nav))

    returns = getattr(data, "returns_for_metrics", None)
    if returns is None or len(returns) < 2:
        returns = nav.pct_change().dropna()
    if len(returns) > 1:
        annualization = max(int(getattr(data, "periods_per_year", 252)), 1)
        values = calc_risk.sharpe_ratio(
            returns.to_frame(),
            risk_free_rate=resolve_risk_free_rate(),
            days_per_year=annualization,
            annualize=True,
        )
        sharpe = float(values.iloc[0]) if len(values) else float("nan")
        payload["sharpe"] = None if np.isnan(sharpe) else sharpe
        drawdown = nav / nav.cummax() - 1.0
        payload["max_drawdown"] = float(drawdown.min())
    return payload


def _load_main(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"catalog_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = getattr(module, "main", None)
    if main is None:
        raise RuntimeError(f"{path.name} has no main()")
    return main


def _run_one(path: Path, output: Path) -> dict[str, Any]:
    code = _strategy_code(path)
    record: dict[str, Any] = {
        "code": code,
        "file": path.name,
        "title": _title(path),
        "status": "ok",
        "runtime_seconds": None,
        "notes": "",
    }
    _CAPTURED.clear()
    log = io.StringIO()
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            main = _load_main(path)
            if inspect.iscoroutinefunction(main):
                asyncio.run(main())
            else:
                main()
    except SkipRun as exc:
        record["status"] = "skipped"
        record["notes"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - batch runner records every failure
        record["status"] = "failed"
        record["notes"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    finally:
        record["runtime_seconds"] = round(time.perf_counter() - started, 2)
        (output / f"{code}.log").write_text(log.getvalue(), encoding="utf-8")

    if record["status"] == "ok":
        if not _CAPTURED:
            record["status"] = "failed"
            record["notes"] = "strategy did not complete a run"
        else:
            strategy = _CAPTURED[-1]
            record["mode"] = getattr(strategy, "execution_mode", "")
            record["instruments"] = list(getattr(strategy, "instruments", []))
            record.update(_metrics(strategy))
            if code.startswith("WF"):
                record["notes"] = "; ".join(
                    filter(
                        None,
                        [
                            record["notes"],
                            "walk-forward workflow; metrics are the full-period backtest",
                        ],
                    )
                )
            if not getattr(strategy, "instrument_specs", None):
                classes = {
                    str(spec).upper()
                    for spec in getattr(strategy, "instruments", [])
                    if any(token in str(spec).upper() for token in ("=X", "=F"))
                }
                if classes:
                    record["notes"] = "provider supplied no contract specs; equity defaults used"
    return record


def _is_intraday(path: Path) -> bool:
    return any(token in path.stem for token in ("intraday", "_1m_", "_15m_", "_30m_", "_1h"))


def _skip_record(path: Path, note: str) -> dict[str, Any]:
    return {
        "code": _strategy_code(path),
        "file": path.name,
        "title": _title(path),
        "status": "skipped",
        "runtime_seconds": 0.0,
        "notes": note,
    }


def _write_reports(records: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    frame = pd.DataFrame(records)
    frame.to_csv(output / "results.csv", index=False)
    lines = ["# Local catalogue run", ""]
    for record in records:
        if record["status"] != "ok":
            lines.append(f"- {record['code']}: {record['status']} — {record['notes']}")
    lines.append("")
    lines.append("| Code | Mode | Total return | CAGR | Sharpe | Max DD | Seconds |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for record in records:
        if record["status"] != "ok":
            continue
        lines.append(
            (
                "| {code} | {mode} | {total:+.2%} | {cagr:+.2%} | "
                "{sharpe:.2f} | {dd:.2%} | {secs:.1f} |"
            ).format(
                code=record["code"],
                mode=record.get("mode", ""),
                total=record.get("total_return") or 0.0,
                cagr=record.get("cagr") or 0.0,
                sharpe=record.get("sharpe") or 0.0,
                dd=record.get("max_drawdown") or 0.0,
                secs=record.get("runtime_seconds") or 0.0,
            )
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", default="", help="substring match on the strategy file name")
    parser.add_argument("--limit", type=int, default=0, help="stop after N strategies (0 = all)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-intraday", action="store_true", help="keep intraday examples")
    parser.add_argument(
        "--include-optimizers",
        action="store_true",
        help="keep WF04/WF05 parameter-search workflows",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["QJ_NO_REPORTS"] = "1"
    os.environ["QJ_QUIET"] = "1"
    _install_patch()

    files = sorted(STRATEGIES.glob("example_*.py"))
    if args.filter:
        files = [path for path in files if args.filter in path.stem]
    if args.limit:
        files = files[: args.limit]

    plan: list[tuple[Path, str | None]] = []
    for path in files:
        if not args.include_intraday and _is_intraday(path):
            plan.append((path, "intraday example excluded (Yahoo provider serves daily only)"))
        elif not args.include_optimizers and path.stem.startswith(
            ("example_wf_04", "example_wf_05")
        ):
            plan.append((path, "optimizer workflow excluded (pass --include-optimizers)"))
        else:
            plan.append((path, None))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["QJ_OUTPUT_DIR"] = str(output / "_runs")
    records: list[dict[str, Any]] = []
    to_run = sum(1 for _, note in plan if note is None)
    print(f"Running {to_run} of {len(plan)} strategies -> {output}")
    for index, (path, note) in enumerate(plan, 1):
        record = _skip_record(path, note) if note else _run_one(path, output)
        records.append(record)
        state = {"ok": "PASS", "skipped": "SKIP", "failed": "FAIL"}.get(record["status"], "?")
        detail = ""
        if record["status"] == "ok":
            sharpe = record.get("sharpe")
            sharpe_text = "n/a" if sharpe is None else f"{sharpe:.2f}"
            detail = f"ret={record['total_return']:+.1%} sharpe={sharpe_text}"
        elif record["notes"]:
            detail = record["notes"][:90]
        print(f"[{index:02d}/{len(plan)}] {state} {record['code']:<4} {detail}")

    _write_reports(records, output)
    ok = sum(record["status"] == "ok" for record in records)
    skipped = sum(record["status"] == "skipped" for record in records)
    failed = sum(record["status"] == "failed" for record in records)
    print(f"\nDone: {ok} ok, {skipped} skipped, {failed} failed")
    print(f"Summary: {output / 'summary.md'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
