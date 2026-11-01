# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Fetch and cache daily market data for local research.

Frames are stored in the shared pickle cache used by the research tools, and
optionally exported as ``<SYMBOL>.csv`` for ``CSVDataProvider``. The manifest
records requested ranges, actual coverage, row counts and content hashes.

Defaults to the 11 US sector SPDR ETFs.

Usage:
    python tools/cache_market_data.py
    python tools/cache_market_data.py --start 2018-07-02
    python tools/cache_market_data.py --symbols XLK,XLF,XLE --no-csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_cache import CachedYahooFrames  # noqa: E402

DEFAULT_SYMBOLS = [
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "XLRE",
    "XLC",
]
DEFAULT_START = "2000-01-03"
DEFAULT_CACHE = ROOT / "data" / "research_cache"
DEFAULT_CSV_DIR = ROOT / "data" / "daily"


def _digest(frame: pd.DataFrame) -> str:
    values = np.nan_to_num(frame.to_numpy(dtype=float))
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def _manifest_entries(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = payload.get("entries")
    return entries if isinstance(entries, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="comma-separated symbols (default: 11 sector SPDR ETFs)",
    )
    parser.add_argument("--start", default=DEFAULT_START, help="requested start date")
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help=(
            "requested end date, exclusive at 00:00 UTC (default: today, i.e. through "
            "the last completed session)"
        ),
    )
    parser.add_argument("--granularity", default="1d", help="1d or 1w")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help="export <SYMBOL>.csv here; pass --no-csv to skip",
    )
    parser.add_argument("--no-csv", action="store_true", help="skip CSV export")
    parser.add_argument("--refresh", action="store_true", help="refetch even when cached")
    parser.add_argument(
        "--manifest", type=Path, help="manifest path (default: <cache>/manifest.json)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    if not symbols:
        print("no symbols requested")
        return 1
    if args.start > args.end:
        print(f"start {args.start} is after end {args.end}")
        return 1

    cache = CachedYahooFrames(args.cache_dir, refresh=args.refresh)
    frames = cache.load_frames(
        instruments=symbols,
        start=args.start,
        end=args.end,
        granularity=args.granularity,
    )

    csv_dir = None if args.no_csv else args.csv_dir
    if csv_dir is not None:
        csv_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest or (args.cache_dir / "manifest.json")
    entries = _manifest_entries(manifest_path)
    generated_at = datetime.now(UTC).isoformat()

    print(f"{'symbol':8} {'first':12} {'last':12} {'rows':>6} {'sha256':>16}  notes")
    failed = 0
    total_bytes = 0
    for symbol in symbols:
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            print(f"{symbol:8} {'-':12} {'-':12} {0:6d} {'-':>16}  NO DATA")
            failed += 1
            continue
        first = frame.index[0].date()
        last = frame.index[-1].date()
        digest = _digest(frame)
        notes = []
        if first.isoformat() > args.start:
            notes.append(f"starts {first.isoformat()} (later than requested)")
        if last.isoformat() < args.end and (pd.Timestamp(args.end) - pd.Timestamp(last)).days > 4:
            notes.append(f"ends {last.isoformat()} (before requested)")
        print(
            f"{symbol:8} {first.isoformat():12} {last.isoformat():12} {len(frame):6d} "
            f"{digest:>16}  {'; '.join(notes)}"
        )
        entries[f"{symbol}|{args.start}|{args.end}|{args.granularity}"] = {
            "symbol": symbol,
            "requested_start": args.start,
            "requested_end": args.end,
            "granularity": args.granularity,
            "first": first.isoformat(),
            "last": last.isoformat(),
            "rows": int(len(frame)),
            "sha256_16": digest,
            "fetched_at": generated_at,
        }
        if csv_dir is not None:
            destination = csv_dir / f"{symbol}.csv"
            frame.to_csv(destination, index_label="date")
            total_bytes += destination.stat().st_size

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"generated_at": generated_at, "entries": entries}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"symbols: {len(symbols) - failed}/{len(symbols)}")
    print(f"cache:   {args.cache_dir}")
    if csv_dir is not None:
        print(f"csv:     {csv_dir} ({total_bytes / 1024:.1f} KB)")
    print(f"manifest:{manifest_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
