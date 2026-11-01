# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Build ticker-day news features from FNSPID-derived Parquet extracts.

Input is any Parquet extract with ``date``, ``title`` and ``stock`` columns
(the shape of HuggingFace ``2084Collective/FNSPID_IMPROVED``). The builder
validates coverage, converts publication timestamps to US/Eastern, dedupes
ticker-day headlines, scores unique titles with VADER, and writes:

    data/events/fnspid_daily.parquet   # one row per ticker and publication day
    data/events/fnspid_daily_summary.json

Timestamps are kept in US/Eastern. ``after_close_rows`` counts headlines
published at or after 16:00 ET, so a strategy can defer those to the next
session instead of using them same-day.

Usage:
    python tools/build_news_features.py --download
    python tools/build_news_features.py --input-dir data/raw/fnspid_improved
    python tools/build_news_features.py --tickers NVDA,AMD --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "2084Collective/FNSPID_IMPROVED"
DEFAULT_INPUT = ROOT / "data" / "raw" / "fnspid_improved"
DEFAULT_OUT = ROOT / "data" / "events" / "fnspid_daily.parquet"
DEFAULT_SUMMARY = ROOT / "data" / "events" / "fnspid_daily_summary.json"
SENTIMENT_POSITIVE = 0.05
SENTIMENT_NEGATIVE = -0.05

_ANALYZER: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HuggingFace dataset repo")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--download", action="store_true", help="download shards via huggingface_hub"
    )
    parser.add_argument("--refresh", action="store_true", help="force re-download")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--tickers",
        default="",
        help="optional comma-separated ticker filter (default: all)",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=0, help="limit raw rows (smoke tests only)")
    parser.add_argument(
        "--min-date",
        default="1980-01-01",
        help="drop rows dated before this day (FNSPID contains a few bogus 1914/1969 rows)",
    )
    return parser.parse_args()


def _ensure_inputs(args: argparse.Namespace) -> list[Path]:
    files = sorted(Path(args.input_dir).rglob("*.parquet"))
    if files:
        return files
    if not args.download:
        raise SystemExit(
            f"no parquet files under {args.input_dir}; pass --download or copy the shards there"
        )
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("pip install huggingface_hub to use --download") from exc

    repo_files = sorted(
        name
        for name in HfApi().list_repo_files(args.repo, repo_type="dataset")
        if name.endswith(".parquet")
    )
    if not repo_files:
        raise SystemExit(f"no Parquet files found in {args.repo}")
    for name in repo_files:
        print(f"downloading {name}", flush=True)
        hf_hub_download(
            args.repo,
            name,
            repo_type="dataset",
            local_dir=str(args.input_dir),
            force_download=args.refresh,
        )
    files = sorted(Path(args.input_dir).rglob("*.parquet"))
    if not files:
        raise SystemExit("download finished but no parquet files were found")
    return files


def _quote_list(values: list[str]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


def _find_column(columns: dict[str, str], candidates: set[str]) -> str | None:
    for lower, original in columns.items():
        if lower in candidates:
            return original
    return None


def _init_vader() -> None:
    global _ANALYZER
    if _ANALYZER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _ANALYZER = SentimentIntensityAnalyzer()


def _score_batch(batch: list[str]) -> list[float]:
    _init_vader()
    return [float(_ANALYZER.polarity_scores(text)["compound"]) for text in batch]


def _score_titles(titles: list[str], workers: int) -> list[float]:
    if not titles:
        return []
    batch_size = 10_000
    batches = [titles[index : index + batch_size] for index in range(0, len(titles), batch_size)]
    if workers <= 1:
        scores: list[float] = []
        for index, batch in enumerate(batches, 1):
            scores.extend(_score_batch(batch))
            if index % 20 == 0:
                print(f"  sentiment {index}/{len(batches)} batches", flush=True)
        return scores

    context = multiprocessing.get_context("spawn")
    try:
        pool = context.Pool(processes=workers, initializer=_init_vader)
    except (OSError, PermissionError) as exc:
        # Some Windows sandboxes deny named pipes; fall back to sequential.
        print(f"  multiprocessing unavailable ({exc}); scoring sequentially", flush=True)
        return _score_titles(titles, 1)
    scores = []
    with pool:
        for index, batch_scores in enumerate(pool.imap(_score_batch, batches, chunksize=1), 1):
            scores.extend(batch_scores)
            if index % 20 == 0:
                print(f"  sentiment {index}/{len(batches)} batches", flush=True)
    return scores


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    files = _ensure_inputs(args)
    print(f"input: {len(files)} parquet files under {args.input_dir}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temp_titles = args.out.parent / "_fnspid_titles_sentiment.parquet"

    con = duckdb.connect()
    con.execute("SET threads = 4")
    source = f"read_parquet([{_quote_list([str(path.as_posix()) for path in files])}])"
    columns = {
        row["column_name"].lower(): row["column_name"]
        for _, row in con.execute(f"DESCRIBE SELECT * FROM {source}").df().iterrows()
    }
    date_column = _find_column(columns, {"date", "datetime", "published_at", "timestamp"})
    title_column = _find_column(columns, {"title", "headline", "headlines"})
    ticker_column = _find_column(columns, {"stock", "ticker", "symbol"})
    if not date_column or not title_column or not ticker_column:
        raise SystemExit(
            "input must contain date/headline/stock columns; found " + ", ".join(columns.values())
        )

    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    ticker_filter = (
        f"AND upper(trim(CAST({_quote(ticker_column)} AS VARCHAR))) IN ({_quote_list(tickers)})"
        if tickers
        else ""
    )
    limit_clause = f"LIMIT {args.limit}" if args.limit > 0 else ""

    input_rows = con.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0]
    unmapped_rows = con.execute(
        f"SELECT COUNT(*) FROM {source} WHERE {_quote(ticker_column)} IS NULL"
    ).fetchone()[0]
    min_date = date.fromisoformat(args.min_date)

    con.execute(
        f"""
        CREATE TEMP TABLE raw AS
        SELECT
            CAST(
                replace(CAST({_quote(date_column)} AS VARCHAR), ' UTC', '')
                AS TIMESTAMP
            ) AS published_utc,
            upper(trim(CAST({_quote(ticker_column)} AS VARCHAR))) AS ticker,
            trim(CAST({_quote(title_column)} AS VARCHAR)) AS title
        FROM {source}
        WHERE {_quote(ticker_column)} IS NOT NULL
          AND {_quote(title_column)} IS NOT NULL
          AND length(trim(CAST({_quote(title_column)} AS VARCHAR))) > 0
          AND length(trim(CAST({_quote(ticker_column)} AS VARCHAR))) > 0
          {ticker_filter}
        {limit_clause}
        """
    )
    con.execute(
        """
        ALTER TABLE raw ADD COLUMN date_only BOOLEAN;
        UPDATE raw SET date_only = (CAST(published_utc AS TIME) = TIME '00:00:00');
        ALTER TABLE raw ADD COLUMN published_et TIMESTAMP;
        UPDATE raw SET published_et =
            CASE WHEN date_only THEN published_utc
                 ELSE timezone('America/New_York', timezone('UTC', published_utc))
            END;
        ALTER TABLE raw ADD COLUMN trade_date DATE;
        UPDATE raw SET trade_date = CAST(published_et AS DATE);
        """
    )
    dropped_old_rows = con.execute(
        f"SELECT COUNT(*) FROM raw WHERE published_utc < TIMESTAMP '{min_date.isoformat()}'"
    ).fetchone()[0]
    if dropped_old_rows:
        con.execute(f"DELETE FROM raw WHERE published_utc < TIMESTAMP '{min_date.isoformat()}'")

    raw_rows = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
    if raw_rows == 0:
        raise SystemExit("no rows matched the filters")
    con.execute(
        """
        CREATE TEMP TABLE deduped AS
        SELECT DISTINCT ticker, trade_date, title FROM raw;
        CREATE TEMP TABLE raw_daily AS
        SELECT
            ticker,
            trade_date,
            COUNT(*) AS raw_rows,
            CAST(
                SUM(CASE WHEN date_only THEN 1 ELSE 0 END) AS BIGINT
            ) AS date_only_rows,
            CAST(
                SUM(CASE WHEN NOT date_only THEN 1 ELSE 0 END) AS BIGINT
            ) AS timed_rows,
            CAST(
                SUM(
                    CASE WHEN NOT date_only
                          AND CAST(published_et AS TIME) >= TIME '16:00:00'
                         THEN 1 ELSE 0 END
                )
                AS BIGINT
            ) AS after_close_rows,
            MIN(published_et) AS first_et,
            MAX(published_et) AS last_et
        FROM raw
        GROUP BY ticker, trade_date;
        CREATE TEMP TABLE titles AS SELECT DISTINCT title FROM deduped;
        """
    )
    deduped_rows = con.execute("SELECT COUNT(*) FROM deduped").fetchone()[0]
    title_count = con.execute("SELECT COUNT(*) FROM titles").fetchone()[0]
    print(
        f"raw rows: {raw_rows:,}; deduped ticker-days: {deduped_rows:,}; "
        f"unique titles: {title_count:,}",
        flush=True,
    )

    titles = con.execute("SELECT title FROM titles").df()["title"].tolist()
    print(f"scoring {len(titles):,} unique titles with VADER ({args.workers} workers)", flush=True)
    scores = _score_titles(titles, max(int(args.workers), 1))
    if len(scores) != len(titles):
        raise RuntimeError(f"scored {len(scores)} titles for {len(titles)} rows")
    pq.write_table(
        pa.table(
            {
                "title": pa.array(titles),
                "sentiment": pa.array(np.asarray(scores, dtype=np.float32)),
            }
        ),
        temp_titles,
    )
    con.execute(
        f"""
        CREATE TEMP TABLE title_sentiment AS
        SELECT title, sentiment FROM read_parquet('{temp_titles.as_posix()}');
        CREATE TEMP TABLE daily AS
        SELECT
            d.ticker,
            d.trade_date,
            COUNT(*) AS unique_titles,
            AVG(s.sentiment) AS sentiment_mean,
            STDDEV_POP(s.sentiment) AS sentiment_std,
            CAST(
                SUM(CASE WHEN s.sentiment >= {SENTIMENT_POSITIVE} THEN 1 ELSE 0 END)
                AS BIGINT
            ) AS positive_titles,
            CAST(
                SUM(CASE WHEN s.sentiment <= {SENTIMENT_NEGATIVE} THEN 1 ELSE 0 END)
                AS BIGINT
            ) AS negative_titles,
            CAST(
                SUM(
                    CASE WHEN s.sentiment > {SENTIMENT_NEGATIVE}
                           AND s.sentiment < {SENTIMENT_POSITIVE} THEN 1 ELSE 0 END
                )
                AS BIGINT
            ) AS neutral_titles
        FROM deduped d
        JOIN title_sentiment s USING(title)
        GROUP BY d.ticker, d.trade_date;
        CREATE TEMP TABLE features AS
        SELECT
            r.trade_date AS date,
            r.ticker,
            r.raw_rows,
            r.date_only_rows,
            r.timed_rows,
            r.after_close_rows,
            r.first_et,
            r.last_et,
            d.unique_titles,
            d.sentiment_mean,
            d.sentiment_std,
            d.positive_titles,
            d.negative_titles,
            d.neutral_titles,
            d.positive_titles * 1.0 / d.unique_titles AS positive_share,
            d.negative_titles * 1.0 / d.unique_titles AS negative_share,
            d.neutral_titles * 1.0 / d.unique_titles AS neutral_share
        FROM raw_daily r
        JOIN daily d USING(ticker, trade_date);
        """
    )
    con.execute(
        f"COPY (SELECT * FROM features ORDER BY date, ticker) "
        f"TO '{args.out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    aggregate = (
        con.execute(
            """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT ticker) AS tickers,
            MIN(date) AS first_date,
            MAX(date) AS last_date,
            AVG(sentiment_mean) AS sentiment_mean,
            AVG(positive_share) AS positive_share,
            AVG(negative_share) AS negative_share,
            SUM(date_only_rows) AS date_only_rows,
            SUM(timed_rows) AS timed_rows,
            SUM(after_close_rows) AS after_close_rows,
            SUM(after_close_rows) * 1.0 / NULLIF(SUM(timed_rows), 0) AS after_close_share
        FROM features
        """
        )
        .df()
        .iloc[0]
        .to_dict()
    )
    yearly = (
        con.execute(
            "SELECT EXTRACT(year FROM date) AS year, COUNT(*) AS rows "
            "FROM features GROUP BY 1 ORDER BY 1"
        )
        .df()
        .to_dict(orient="records")
    )
    top_tickers = (
        con.execute(
            "SELECT ticker, COUNT(*) AS rows FROM features GROUP BY 1 ORDER BY rows DESC LIMIT 10"
        )
        .df()
        .to_dict(orient="records")
    )
    duplicate_ratio = (raw_rows - deduped_rows) / raw_rows if raw_rows else 0.0
    mapped_rows = raw_rows + dropped_old_rows
    mapped_share = mapped_rows / input_rows if input_rows else 0.0
    date_only_share = aggregate["date_only_rows"] / raw_rows if raw_rows else 0.0

    warnings: list[str] = []
    if input_rows < 25_000_000 and args.limit == 0:
        warnings.append(f"input has {input_rows:,} rows; the source card reports ~28.6M")
    if mapped_share < 0.4 and not tickers:
        warnings.append(
            f"only {mapped_share:.1%} of input rows carry a ticker; "
            "the remainder are general news without a symbol"
        )
    if date_only_share > 0.8:
        warnings.append(
            f"{date_only_share:.1%} of mapped headlines are date-only (00:00 UTC); "
            "a strategy must apply a next-session lag"
        )
    if dropped_old_rows:
        warnings.append(f"dropped {dropped_old_rows:,} rows before {min_date.isoformat()}")
    after_close = aggregate.get("after_close_share") or 0.0
    if after_close > 0.6:
        warnings.append("more than 60% of timed headlines are timestamped after the US close")

    summary = {
        "source": {
            "repo": args.repo,
            "input_dir": str(args.input_dir),
            "files": [{"name": path.name, "bytes": path.stat().st_size} for path in files],
        },
        "parameters": {
            "tickers": tickers,
            "limit": args.limit,
            "workers": args.workers,
            "min_date": min_date.isoformat(),
            "sentiment_positive_threshold": SENTIMENT_POSITIVE,
            "sentiment_negative_threshold": SENTIMENT_NEGATIVE,
        },
        "validation": {
            "input_rows": int(input_rows),
            "unmapped_rows": int(unmapped_rows),
            "mapped_rows": int(mapped_rows),
            "mapped_share": float(mapped_share),
            "dropped_old_rows": int(dropped_old_rows),
            "raw_rows": int(raw_rows),
            "deduped_rows": int(deduped_rows),
            "duplicate_ratio": float(duplicate_ratio),
            "unique_titles": int(title_count),
            "tickers": int(aggregate["tickers"]),
            "first_date_et": str(aggregate["first_date"]),
            "last_date_et": str(aggregate["last_date"]),
            "date_only_rows": int(aggregate["date_only_rows"]),
            "timed_rows": int(aggregate["timed_rows"]),
            "date_only_share": float(date_only_share),
            "after_close_rows": int(aggregate["after_close_rows"]),
            "after_close_share": float(aggregate["after_close_share"] or 0.0),
            "sentiment_mean": float(aggregate["sentiment_mean"]),
            "positive_share": float(aggregate["positive_share"]),
            "negative_share": float(aggregate["negative_share"]),
            "yearly_rows": yearly,
            "top_tickers": top_tickers,
            "warnings": warnings,
        },
        "output": {
            "path": str(args.out),
            "bytes": args.out.stat().st_size,
            "sha256": _digest(args.out),
        },
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    args.summary.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    temp_titles.unlink(missing_ok=True)
    con.close()

    print(json.dumps(summary["validation"], indent=2, default=str)[:2000])
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"summary: {args.summary}")
    return 0


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


if __name__ == "__main__":
    raise SystemExit(main())
