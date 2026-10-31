# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Compare a local catalogue run with the published quants.md numbers.

The published cards come from QuantJourney-managed data and their own execution
assumptions; the local run uses Yahoo bars (and no contract specs for FX/futures
examples). Treat the deltas as a data-source sanity check, not a like-for-like
benchmark.

Usage:
    python tools/compare_catalog_local.py
    python tools/compare_catalog_local.py --results reports/_catalog/results.json
    python tools/compare_catalog_local.py --from-html reports/_catalog/site.html
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

SITE_URL = "https://quants.md/"
CARD_RE = re.compile(
    r'<article class="card[^"]*" data-id="([A-Z0-9]+)">(.*?)</article>',
    re.DOTALL,
)
METRIC_RE = re.compile(r"<dt>([^<]+)</dt>\s*<dd[^>]*>([^<]+)</dd>")
TITLE_RE = re.compile(r'<h3 class="card-title">.*?<button[^>]*>([^<]+)</button>', re.DOTALL)
YEARS_RE = re.compile(r'<span class="span">([^<]+)</span>')


def _percent(raw: str) -> float | None:
    value = raw.strip().replace("−", "-").replace("%", "").replace("+", "")
    try:
        return float(value) / 100.0
    except ValueError:
        return None


def _number(raw: str) -> float | None:
    value = raw.strip().replace("−", "-").replace("+", "")
    try:
        return float(value)
    except ValueError:
        return None


def parse_site(html: str) -> dict[str, dict[str, Any]]:
    published: dict[str, dict[str, Any]] = {}
    for code, block in CARD_RE.findall(html):
        title_match = TITLE_RE.search(block)
        metrics = dict(METRIC_RE.findall(block))
        years_match = YEARS_RE.search(block)
        published[code] = {
            "title": title_match.group(1).strip() if title_match else "",
            "return": _percent(metrics.get("Return", "")),
            "cagr": _percent(metrics.get("CAGR", "")),
            "sharpe": _number(metrics.get("Sharpe", "")),
            "max_dd": _percent(metrics.get("Max DD", "")),
            "years": years_match.group(1).strip() if years_match else "",
        }
    return published


def fetch_site(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", "replace")


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _delta(local: float | None, published: float | None) -> str:
    if local is None or published is None:
        return "n/a"
    return f"{local - published:+.2f}"


def _summary_lines(
    results: list[dict[str, Any]],
    published: dict[str, dict[str, Any]],
) -> list[str]:
    sharpes: list[tuple[float, float]] = []
    returns: list[tuple[float, float]] = []
    for record in results:
        if record["status"] != "ok":
            continue
        card = published.get(record["code"])
        if not card:
            continue
        if record.get("sharpe") is not None and card.get("sharpe") is not None:
            sharpes.append((record["sharpe"], card["sharpe"]))
        if record.get("total_return") is not None and card.get("return") is not None:
            returns.append((record["total_return"], card["return"]))

    lines = ["## Summary", ""]
    if len(sharpes) >= 3 and len(returns) >= 3:
        sharpe_corr = float(np.corrcoef([a for a, _ in sharpes], [b for _, b in sharpes])[0, 1])
        return_corr = float(np.corrcoef([a for a, _ in returns], [b for _, b in returns])[0, 1])
        deltas = np.abs(np.array([a - b for a, b in sharpes]))
        lines.append(
            f"- Paired strategies: {len(sharpes)}. Sharpe correlation **{sharpe_corr:.2f}**, "
            f"return correlation **{return_corr:.2f}**."
        )
        lines.append(
            f"- Median absolute Sharpe delta **{np.median(deltas):.3f}**; "
            f"{np.mean(deltas < 0.10):.0%} within 0.10 and "
            f"{np.mean(deltas < 0.25):.0%} within 0.25."
        )
    else:
        lines.append("- Not enough paired strategies to compute agreement statistics.")
    lines.append("")
    return lines


def build_comparison(
    results: list[dict[str, Any]],
    published: dict[str, dict[str, Any]],
) -> tuple[list[str], list[tuple[str, float]]]:
    lines = [
        "# Local vs published catalogue",
        "",
        "Local: Yahoo bars via `tools/run_catalog_local.py`, 2% risk-free rate. "
        "Published: quants.md cards backed by QuantJourney data. Deltas are a "
        "data-source sanity check, not a like-for-like benchmark.",
        "",
    ]
    lines.extend(_summary_lines(results, published))
    lines.extend(
        [
            "| Code | Local return | Published | Δ Sharpe | Local Sharpe | "
            "Published | Δ DD | Note |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    deltas: list[tuple[str, float]] = []
    for record in results:
        code = record["code"]
        card = published.get(code)
        if record["status"] != "ok":
            lines.append(
                f"| {code} | {record['status']} | — | — | — | — | — | {record['notes'][:60]} |"
            )
            continue
        local_sharpe = record.get("sharpe")
        published_sharpe = card.get("sharpe") if card else None
        if local_sharpe is not None and published_sharpe is not None:
            deltas.append((code, local_sharpe - published_sharpe))
        local_dd = record.get("max_drawdown")
        published_dd = card.get("max_dd") if card else None
        dd_delta = (
            f"{local_dd - published_dd:+.2%}"
            if local_dd is not None and published_dd is not None
            else "n/a"
        )
        lines.append(
            "| {code} | {local_ret} | {pub_ret} | {dsharpe} | {local_sharpe} | "
            "{pub_sharpe} | {ddd} | {note} |".format(
                code=code,
                local_ret=_pct(record.get("total_return")),
                pub_ret=_pct(card.get("return")) if card else "n/a",
                dsharpe=_delta(local_sharpe, published_sharpe),
                local_sharpe=_num(local_sharpe),
                pub_sharpe=_num(published_sharpe),
                ddd=dd_delta,
                note=record.get("notes", "")[:40],
            )
        )
    return lines, deltas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("reports/_catalog/results.json"),
    )
    parser.add_argument("--from-html", type=Path, help="reuse a saved quants.md page")
    parser.add_argument("--site-url", default=SITE_URL)
    parser.add_argument("--output", type=Path, help="write the markdown comparison here")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.results.is_file():
        print(f"missing local results: {args.results}")
        return 1
    results = json.loads(args.results.read_text(encoding="utf-8"))
    html = (
        args.from_html.read_text(encoding="utf-8") if args.from_html else fetch_site(args.site_url)
    )
    published = parse_site(html)
    if not published:
        print("could not parse any strategy cards from the site")
        return 1

    lines, deltas = build_comparison(results, published)
    output = args.output or args.results.with_name("comparison.md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    compared = [item for item in results if item["status"] == "ok"]
    print(f"Compared {len(compared)} local runs against {len(published)} published cards")
    print(f"Wrote {output}")
    if deltas:
        deltas.sort(key=lambda item: item[1])
        selected = deltas if len(deltas) <= 6 else deltas[:3] + deltas[-3:]
        print("Largest Sharpe gaps (local - published):")
        for code, gap in selected:
            print(f"  {code}: {gap:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
