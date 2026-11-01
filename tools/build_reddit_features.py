# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Build point-in-time Reddit thread features for the backtester.

Pipeline (corrected after adversarial review):
  1. Fetch r/thewallstreet discussion threads and their full comment tree.
  2. Apply a trading-calendar point-in-time window:
     daily -> 16:00 ET on the session; nightly/weekend -> 09:30 ET next session.
  3. Drop post-cutoff comments, bot/AutoModerator posts, and edited bodies
     (latest-snapshot leak guard).
  4. Regex pre-extract candidate ticker/macro symbols from a point-in-time
     universe + alias maps; the LLM only disambiguates, scores and tags.
  5. One DeepSeek call per thread (JSON output, thinking disabled).

Usage:
    python tools/build_reddit_features.py --pilot 10 --dry-run
    $env:DEEPSEEK_API_KEY = "..."
    python tools/build_reddit_features.py --pilot 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime
from datetime import time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARCTIC = "https://arctic-shift.photon-reddit.com"
SUBREDDIT = "thewallstreet"
ET = ZoneInfo("America/New_York")
UTC_ZONE = ZoneInfo("UTC")
CAL = xcals.get_calendar("XNYS")
UA = "quantjourney-bt-research/0.1 (local research)"

REF_DIR = ROOT / "data" / "reference"
RAW_DIR = ROOT / "data" / "raw" / "reddit_thewallstreet"
OUT_DIR = ROOT / "data" / "events"

SYSTEM_PROMPT = """You extract structured trading information from Reddit daily discussion threads
for a systematic backtest.

Return valid JSON only, with this exact shape:
{
  "thread": {
    "date": "YYYY-MM-DD",
    "title": "string",
    "stance": "bullish|bearish|neutral|mixed",
    "overall_sentiment": -1.0,
    "risk_regime": "risk_on|risk_off|mixed|unknown",
    "summary": "one or two sentences"
  },
  "tickers": [
    {"symbol": "NVDA", "mentions": 3, "sentiment": 0.4, "confidence": 0.8,
     "aspects": ["earnings", "options-flow"],
     "position": "long|short|options|neutral",
     "price_targets": [200.0],
     "evidence_ids": ["abc123"]}
  ],
  "macro": [
    {"symbol": "SPX", "kind": "index", "mentions": 8, "sentiment": -0.4,
     "confidence": 0.8, "evidence_ids": ["abc123"]}
  ],
  "events": [{"topic": "FOMC", "sentiment": 0.1, "mentions": 2}],
  "notable_comments": [
    {"id": "abc123", "tags": ["ticker:SPY", "macro:VIX", "bearish"], "reason": "short reason"}
  ]
}

The comments are a reply tree. Indentation means a reply to the line above; use parent context.

Each comment line ends with "candidates: ..." — these are pre-extracted candidate symbols.
- Keep only candidates that really refer to a security; drop false positives.
- You may also map an unambiguous company name in the body to a symbol, even if it was
  not listed as a candidate.

Rules:
- tickers = tradable US-listed equities/ETFs only (NVDA, SPY, QQQ, TLT).
- macro = indices/futures/volatility/rates/FX/commodities. Never drop these; never put
  them in tickers.
- Canonicalize macro aliases:
  SPX/ES/spooz -> SPX; NDX/NQ -> NDX; VIX/vol -> VIX; TNX/10y/yields -> TNX;
  DXY/dollar -> DXY; WTI/oil/CL/USOIL -> CL; gold/GC -> GC.
- Include EVERY ticker with at least one explicit mention; no cap.
- mentions = number of distinct comments mentioning the symbol.
- evidence_ids must be ids that actually appear in the comments; at most 5 per symbol.
- sentiment is toward the symbol: bullish VIX = expect VIX up; bearish SPX = expect SPX down.
- sentiment is -1..+1; confidence is 0..1; price_targets is a list of floats.
- aspects are free but short: earnings, guidance, options-flow, technicals, macro,
  product, meme, other.
- position is the dominant stated position for that ticker in the thread.
- events = macro events discussed (FOMC, CPI, earnings season, opex, Fed, jobs, etc.);
  at most 10.
- notable_comments: at most 15; ids must exist; tags use ticker:SYMBOL / macro:SYMBOL /
  bullish / bearish / neutral.
- Do not use outside knowledge, do not fetch links, do not invent ids or symbols.
- Keep the JSON compact.
"""

MACRO_ALIASES = {
    "SPX": "SPX",
    "ES": "SPX",
    "/ES": "SPX",
    "SPOOS": "SPX",
    "S&P": "SPX",
    "NDX": "NDX",
    "NQ": "NDX",
    "/NQ": "NDX",
    "VIX": "VIX",
    "VOL": "VIX",
    "TNX": "TNX",
    "10Y": "TNX",
    "YIELDS": "TNX",
    "UST10Y": "TNX",
    "DXY": "DXY",
    "DOLLAR": "DXY",
    "USD": "DXY",
    "WTI": "CL",
    "OIL": "CL",
    "CL": "CL",
    "/CL": "CL",
    "USOIL": "CL",
    "GOLD": "GC",
    "GC": "GC",
    "/GC": "GC",
}
MACRO_KIND = {
    "SPX": "index",
    "NDX": "index",
    "VIX": "volatility",
    "TNX": "rates",
    "DXY": "fx",
    "CL": "commodity",
    "GC": "commodity",
}
COMPANY_ALIASES = {
    "nvidia": "NVDA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "baba": "BABA",
    "alibaba": "BABA",
    "intel": "INTC",
    "amd": "AMD",
    "roku": "ROKU",
    "paypal": "PYPL",
    "twitter": "TWTR",
    "exxon": "XOM",
    "berkshire": "BRK.B",
    "starbucks": "SBUX",
    "mcdonald": "MCD",
    "disney": "DIS",
    "walmart": "WMT",
    "target": "TGT",
    "costco": "COST",
    "uber": "UBER",
    "lucid": "LCID",
    "sofi": "SOFI",
}
DELISTED = {
    "TWTR": "TWTR",
    "FRC": "FRC",
    "SIVB": "SIVB",
    "SBNY": "SBNY",
    "TLRD": "TLRD",
    "BMBL": "BMBL",
    "VIAC": "VIAC",
    "FB": "META",
}
ETF_ADDENDUM = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "TLT",
    "GLD",
    "SLV",
    "GDX",
    "USO",
    "EEM",
    "XLE",
    "XLF",
    "XLK",
    "XLV",
    "XLY",
    "XLP",
    "XLU",
    "XLI",
    "XLB",
    "XLRE",
    "XLC",
    "SOXL",
    "SOXX",
    "SMH",
    "KWEB",
    "FXI",
    "ASHR",
    "TNA",
    "UVXY",
    "VXX",
    "BOIL",
    "UNG",
    "UGAZ",
    "GBTC",
    "ARKK",
    "LABU",
    "TQQQ",
    "SQQQ",
    "UPRO",
]
STOP_WORDS = {
    "A",
    "AM",
    "AN",
    "AND",
    "ARE",
    "AS",
    "AT",
    "BE",
    "BY",
    "GO",
    "IF",
    "IN",
    "IS",
    "IT",
    "ME",
    "MY",
    "NO",
    "OF",
    "ON",
    "OR",
    "SO",
    "TO",
    "UP",
    "WE",
    "ER",
    "PM",
    "CEO",
    "CPI",
    "FOMC",
    "IPO",
    "ETF",
    "GDP",
    "YOLO",
    "OPEX",
    "BTD",
    "FOMO",
    "ITM",
    "OTM",
    "RSI",
    "GTC",
    "EOD",
    "ATH",
    "DD",
    "IMO",
}


def _get_json(path: str, params: dict[str, object], timeout: int = 120) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"{ARCTIC}{path}?{query}"
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if attempt == 3 or exc.code not in (429, 500, 502, 503, 504, 522):
                raise
            time.sleep(5 * (attempt + 1))
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def _load_universe() -> set[str]:
    path = REF_DIR / "tickers.json"
    universe = set(ETF_ADDENDUM)
    if path.is_file():
        universe.update(json.loads(path.read_text(encoding="utf-8")).get("tickers", []))
    universe.update(DELISTED.values())
    universe = {symbol.upper() for symbol in universe}
    universe.update(ETF_ADDENDUM)
    return universe


def _et_date_naive(et_date: date) -> pd.Timestamp:
    utc_midnight = datetime.combine(et_date, dt_time(0, 0), tzinfo=ET).astimezone(UTC)
    return pd.Timestamp(utc_midnight.date())


def _session_on_or_after(et_date: date) -> pd.Timestamp:
    naive = _et_date_naive(et_date)
    return naive if CAL.is_session(naive) else _next_session_after(et_date)


def _next_session_after(et_date: date) -> pd.Timestamp:
    start = _et_date_naive(et_date)
    sessions = CAL.sessions_in_range(start, start + pd.Timedelta("14D"))
    after = [session for session in sessions if pd.Timestamp(session) > start]
    if not after:
        raise RuntimeError(f"no NYSE session found after {et_date}")
    return after[0]


def _window_end(thread_type: str, created_utc: float) -> tuple[datetime, str]:
    created_et = datetime.fromtimestamp(created_utc, ET)
    if thread_type == "daily":
        session = _session_on_or_after(created_et.date())
        session_date = pd.Timestamp(session).date()
        cutoff_et = datetime.combine(session_date, dt_time(16, 0), tzinfo=ET)
        signal = "T+1 open (available at close)"
    else:
        session = _next_session_after(created_et.date())
        session_date = pd.Timestamp(session).date()
        cutoff_et = datetime.combine(session_date, dt_time(9, 30), tzinfo=ET)
        signal = "same-session open"
    return cutoff_et.astimezone(UTC), signal


def _classify(title: str) -> str | None:
    if title.startswith("Daily Discussion"):
        return "daily"
    if title.startswith("Nightly Discussion"):
        return "nightly"
    if title.startswith("Weekend Market Discussion"):
        return "weekend"
    return None


def _fetch_threads(counts: dict[str, int], max_pages: int = 12) -> list[dict]:
    picked: dict[str, list[dict]] = {kind: [] for kind in counts}
    before: int | None = None
    for _ in range(max_pages):
        params: dict[str, object] = {
            "subreddit": SUBREDDIT,
            "limit": 100,
            "sort": "desc",
            "fields": "id,created_utc,title,num_comments",
        }
        if before is not None:
            params["before"] = before
        data = _get_json("/api/posts/search", params).get("data", [])
        if not data:
            break
        for item in data:
            kind = _classify(str(item.get("title", "")))
            if kind and len(picked[kind]) < counts[kind]:
                picked[kind].append(item)
        if all(len(picked[kind]) >= counts[kind] for kind in counts):
            break
        before = int(data[-1]["created_utc"]) - 1
    return [item for kind in counts for item in picked[kind]]


def _walk_tree(nodes: list[dict], depth: int, out: list[dict]) -> None:
    for node in nodes:
        if node.get("kind") != "t1":
            continue
        data = node["data"]
        item = {
            "id": data.get("id"),
            "parent_id": data.get("parent_id"),
            "author": data.get("author"),
            "body": " ".join(str(data.get("body") or "").split()),
            "created_utc": int(data.get("created_utc") or 0),
            "edited": data.get("edited"),
            "distinguished": data.get("distinguished"),
            "is_submitter": bool(data.get("is_submitter")),
            "score": data.get("score"),
            "depth": depth,
        }
        out.append(item)
        replies = (data.get("replies") or {}).get("data", {}).get("children", [])
        if replies:
            _walk_tree(replies, depth + 1, out)


def _fetch_comments(thread_id: str) -> list[dict]:
    data = _get_json(
        "/api/comments/tree", {"link_id": f"t3_{thread_id}", "limit": 9999}, timeout=180
    )
    comments: list[dict] = []
    _walk_tree(data.get("data", []), 0, comments)
    return comments


def _candidates(body: str, universe: set[str]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for symbol in re.findall(r"\$([A-Z][A-Z0-9.\-]{0,5})", body):
        key = ("ticker", symbol.upper())
        if key[1] not in seen:
            found.append(key)
            seen.add(key[1])
    for token in re.findall(r"\b[A-Z]{1,6}\b", body):
        symbol = token.upper()
        if len(symbol) < 2 or symbol in STOP_WORDS:
            continue
        if symbol in MACRO_ALIASES:
            canonical = MACRO_ALIASES[symbol]
            key = ("macro", canonical)
            if canonical not in seen:
                found.append(key)
                seen.add(canonical)
        elif symbol in universe and symbol not in STOP_WORDS:
            key = ("ticker", symbol)
            if symbol not in seen:
                found.append(key)
                seen.add(symbol)
    lowered = body.lower()
    for name, symbol in COMPANY_ALIASES.items():
        if name in lowered:
            key = ("ticker", symbol)
            if symbol not in seen:
                found.append(key)
                seen.add(symbol)
    return found


def _build_user_prompt(
    thread: dict, comments: list[dict], cutoff: datetime, universe: set[str]
) -> str:
    posted = datetime.fromtimestamp(int(thread["created_utc"]), ET).date().isoformat()
    lines = [
        f"Thread: {thread['title']}",
        f"Posted (ET): {posted}",
        f"Information cutoff (UTC): {cutoff.isoformat()}",
        "Only comments up to this cutoff are included; later comments were removed.",
        f"URL: https://www.reddit.com/r/thewallstreet/comments/{thread['id']}/",
        "",
        f"Comments ({len(comments)}), one per line; indentation means a reply:",
    ]
    for item in comments:
        ts = datetime.fromtimestamp(item["created_utc"], UTC).isoformat()
        body = item["body"] if len(item["body"]) <= 1200 else item["body"][:1200] + " …"
        cands = _candidates(body, universe)
        cand_text = ", ".join(symbol for _, symbol in cands) or "-"
        prefix = "  " * min(item["depth"], 8)
        parent = item["parent_id"].split("_", 1)[-1] if item["parent_id"] else ""
        arrow = f" ↳{parent}" if parent else " root"
        line = (
            f"{prefix}[{ts}] id={item['id']}{arrow} "
            f"author={item['author'] or '[deleted]'}: {body}  candidates: {cand_text}"
        )
        lines.append(line)
    return "\n".join(lines)


def _call_llm(messages: list[dict], model: str, api_key: str, base_url: str) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 16000,
        "stream": False,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": UA,
        },
        method="POST",
    )
    for _ in range(3):
        with urllib.request.urlopen(request, timeout=900) as response:
            data = json.loads(response.read())
        content = (data["choices"][0]["message"].get("content") or "").strip()
        if content and not content.isspace():
            try:
                json.loads(content)
                return data
            except json.JSONDecodeError:
                pass
        time.sleep(3)
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, default=10)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument(
        "--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    universe = _load_universe()
    daily = max(args.pilot // 2, 1)
    nightly = args.pilot // 4
    weekend = args.pilot - daily - nightly
    counts = {"daily": daily, "nightly": nightly, "weekend": weekend}
    threads = _fetch_threads(counts)
    print(f"universe={len(universe)} symbols; threads={len(threads)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, thread in enumerate(threads, 1):
        kind = _classify(thread["title"])
        cutoff, signal = _window_end(kind, int(thread["created_utc"]))
        comments = _fetch_comments(thread["id"])
        in_window = [c for c in comments if c["created_utc"] <= cutoff.timestamp()]
        dropped_late = len(comments) - len(in_window)
        before_edit = [c for c in in_window if c["edited"] is False]
        dropped_edited = len(in_window) - len(before_edit)
        keep = [
            c
            for c in before_edit
            if c["author"] != "AutoModerator"
            and c["distinguished"] not in ("moderator", "admin")
            and c["body"]
            and c["body"] not in ("[deleted]", "[removed]")
        ]
        user = _build_user_prompt(thread, keep, cutoff, universe)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        prompt_chars = len(SYSTEM_PROMPT) + len(user)
        approx_tokens = int(prompt_chars / 4)
        cand_total = sum(len(_candidates(c["body"], universe)) for c in keep)
        row = {
            "id": thread["id"],
            "type": kind,
            "date": datetime.fromtimestamp(int(thread["created_utc"]), ET).date().isoformat(),
            "cutoff_utc": cutoff.isoformat(),
            "signal": signal,
            "comments_total": len(comments),
            "in_window": len(in_window),
            "dropped_late": dropped_late,
            "dropped_edited": dropped_edited,
            "kept": len(keep),
            "candidate_hits": cand_total,
            "prompt_chars": prompt_chars,
            "approx_input_tokens": approx_tokens,
        }
        (RAW_DIR / f"{thread['id']}_prompt.json").write_text(
            json.dumps({"messages": messages}, indent=2) + "\n", encoding="utf-8"
        )
        if args.dry_run:
            row["est_cost_usd"] = round(approx_tokens * 0.15 / 1e6 + 4000 * 0.60 / 1e6, 4)
            print(
                f"[{index}/{len(threads)}] {kind:8} {row['date']} total={len(comments)} "
                f"in_window={len(in_window)} dropped_late={dropped_late} "
                f"edited={dropped_edited} kept={len(keep)} cand={cand_total} "
                f"tok={approx_tokens:,} est=${row['est_cost_usd']}"
            )
            rows.append(row)
            continue

        data = _call_llm(messages, args.model, api_key, args.base_url)
        content = (data["choices"][0]["message"].get("content") or "").strip()
        try:
            extraction = json.loads(content)
        except json.JSONDecodeError:
            extraction = {"_raw": content}
        row.update({"usage": data.get("usage"), "extraction": extraction})
        (RAW_DIR / f"{thread['id']}_result.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
        rows.append(row)
        print(f"[{index}/{len(threads)}] {kind} {thread['id']} tokens={data.get('usage')}")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dry_run": args.dry_run,
        "threads": rows,
        "totals": {
            "threads": len(rows),
            "comments_total": sum(r["comments_total"] for r in rows),
            "kept": sum(r["kept"] for r in rows),
            "approx_input_tokens": sum(r["approx_input_tokens"] for r in rows),
        },
    }
    (RAW_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"\nmanifest: {RAW_DIR / 'manifest.json'}")
    print(f"totals: {manifest['totals']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
