# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Deterministic Yahoo frame cache shared by the research tools.

Yahoo returns adjusted closes that differ by ~1e-5 between identical requests;
path-dependent strategies can bifurcate on that jitter. Caching the first fetch
as pickle (exact float round-trip) makes repeated runs bit-identical.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from backtester.data import YahooDataProvider, build_payload


class CachedYahooFrames:
    """Fetch Yahoo frames once and reuse them from a pickle cache."""

    def __init__(self, cache_dir: str | Path, *, refresh: bool = False) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._refresh = bool(refresh)
        self._inner = YahooDataProvider()

    def load_frames(
        self,
        *,
        instruments: list[str],
        start: str,
        end: str,
        granularity: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for symbol in instruments:
            normalized = str(symbol).strip().upper()
            key = hashlib.sha256(f"{normalized}|{start}|{end}|{granularity}".encode()).hexdigest()[
                :16
            ]
            path = self._cache_dir / f"{key}.pkl"
            if path.is_file() and not self._refresh:
                frame = pd.read_pickle(path)
            else:
                fetched = self._inner.load_frames(
                    instruments=[symbol], start=start, end=end, granularity=granularity
                )
                frame = fetched[normalized]
                frame.to_pickle(path)
            frames[normalized] = frame
        return frames


class CachedYahooDataProvider:
    """Payload provider backed by :class:`CachedYahooFrames`."""

    def __init__(self, cache_dir: str | Path, *, refresh: bool = False) -> None:
        self._frames = CachedYahooFrames(cache_dir, refresh=refresh)

    def load_frames(
        self,
        *,
        instruments: list[str],
        start: str,
        end: str,
        granularity: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        return self._frames.load_frames(
            instruments=instruments,
            start=start,
            end=end,
            granularity=granularity,
        )

    def load(
        self,
        *,
        instruments: list[str],
        start: str,
        end: str,
        granularity: str,
    ) -> dict[str, Any]:
        frames = self.load_frames(
            instruments=instruments,
            start=start,
            end=end,
            granularity=granularity,
        )
        return build_payload(frames, source="yahoo-cache", dataset_id="yahoo-cache")
