# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Local market-data providers.

Each provider answers the same question — "give me validated OHLCV for these
symbols over this window" — and delegates payload construction to
:func:`backtester.data.payload.build_payload`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from backtester.data.payload import build_payload


def _select_frames(
    frames: Mapping[str, pd.DataFrame],
    instruments: list[str],
) -> dict[str, pd.DataFrame]:
    """Return requested instruments in the requested order, failing closed."""
    by_symbol = {str(symbol).strip().upper(): frame for symbol, frame in frames.items()}
    missing = [symbol for symbol in instruments if str(symbol).strip().upper() not in by_symbol]
    if missing:
        raise ValueError(
            "Local data provider is missing requested instruments: " + ", ".join(missing)
        )
    return {
        str(symbol).strip().upper(): by_symbol[str(symbol).strip().upper()]
        for symbol in instruments
    }


class FrameDataProvider:
    """Serve already-loaded frames, e.g. from a database or a notebook."""

    def __init__(
        self,
        frames: Mapping[str, pd.DataFrame],
        *,
        source: str = "local-frames",
        dataset_id: str = "local-frames",
        benchmark_returns: pd.Series | None = None,
        instrument_specs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._frames = dict(frames)
        self._source = source
        self._dataset_id = dataset_id
        self._benchmark_returns = benchmark_returns
        self._instrument_specs = dict(instrument_specs) if instrument_specs else None

    def load(
        self,
        *,
        instruments: list[str],
        start: str,
        end: str,
        granularity: str,
    ) -> dict[str, Any]:
        del start, end, granularity  # frames are already loaded for their own window
        return build_payload(
            _select_frames(self._frames, instruments),
            source=self._source,
            dataset_id=self._dataset_id,
            benchmark_returns=self._benchmark_returns,
            instrument_specs=self._instrument_specs,
        )


class _DirectoryProvider:
    """Base class for providers backed by one file per symbol."""

    suffix = ""

    def __init__(
        self,
        directory: str | Path,
        *,
        date_column: str = "date",
        source: str | None = None,
        dataset_id: str | None = None,
        benchmark_returns: pd.Series | None = None,
        instrument_specs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._directory = Path(directory)
        if not self._directory.is_dir():
            raise ValueError(f"Local data directory does not exist: {self._directory}")
        self._date_column = date_column
        self._source = source or f"local-{self.suffix}"
        self._dataset_id = dataset_id or f"local-{self.suffix}"
        self._benchmark_returns = benchmark_returns
        self._instrument_specs = dict(instrument_specs) if instrument_specs else None

    def _read(self, path: Path) -> pd.DataFrame:
        raise NotImplementedError

    def _load_frame(self, symbol: str) -> pd.DataFrame:
        candidates = [
            self._directory / f"{symbol}{self.suffix}",
            self._directory / f"{symbol.upper()}{self.suffix}",
            self._directory / f"{symbol.lower()}{self.suffix}",
        ]
        for candidate in candidates:
            if candidate.is_file():
                frame = self._read(candidate)
                if self._date_column != "date" and self._date_column in frame.columns:
                    frame = frame.rename(columns={self._date_column: "date"})
                return frame
        raise ValueError(f"No {self.suffix} file for {symbol} in {self._directory}")

    def load(
        self,
        *,
        instruments: list[str],
        start: str,
        end: str,
        granularity: str,
    ) -> dict[str, Any]:
        del granularity
        frames = {str(symbol).strip().upper(): self._load_frame(symbol) for symbol in instruments}
        window_start = pd.Timestamp(start, tz="UTC")
        window_end = pd.Timestamp(end, tz="UTC")
        clipped = {
            symbol: _clip(frame, window_start, window_end) for symbol, frame in frames.items()
        }
        return build_payload(
            clipped,
            source=self._source,
            dataset_id=self._dataset_id,
            benchmark_returns=self._benchmark_returns,
            instrument_specs=self._instrument_specs,
        )


def _clip(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Trim a frame to the requested window without inventing bars."""
    working = frame.copy()
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], utc=True)
        working = working.set_index("date")
    if not isinstance(working.index, pd.DatetimeIndex):
        return frame
    index = working.index
    if index.tz is None:
        index = index.tz_localize("UTC")
        working.index = index
    else:
        working.index = index.tz_convert("UTC")
    return working.loc[(working.index >= start) & (working.index <= end)]


class CSVDataProvider(_DirectoryProvider):
    """Read one ``<SYMBOL>.csv`` file per instrument from a directory."""

    suffix = ".csv"

    def _read(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)


class ParquetDataProvider(_DirectoryProvider):
    """Read one ``<SYMBOL>.parquet`` file per instrument from a directory."""

    suffix = ".parquet"

    def _read(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:  # pragma: no cover - depends on optional engine
            raise ImportError(
                "Parquet support requires an engine such as pyarrow: pip install pyarrow"
            ) from exc
