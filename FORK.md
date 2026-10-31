# Fork notes

This repository is a personal fork of
[`QuantJourneyOrg/quantjourney-bt`](https://github.com/QuantJourneyOrg/quantjourney-bt).
It is not affiliated with QuantJourney, and upstream has not reviewed these
changes. The licence remains Apache-2.0.

## Why the fork exists

Upstream treats market-data ingestion as part of its hosted platform and gates
distributable files through `release/public_artifacts.txt`. That makes it a poor
fit for the work below, so this fork is developed as its own product. Upstream
is kept as a read-only source of engine fixes.

## Divergence from upstream

| Area | Change |
|---|---|
| Local data | `backtester/data/` providers (CSV, Parquet, Yahoo, in-memory), shared payload builder, `Backtester(data_provider=...)` |
| Metrics | One risk-free-rate resolver; Sharpe/Sortino agree across summary, metrics table and charts; flat series report an undefined Sharpe |
| Sample data | `sample_data.py` rebuilt on the shared payload builder (output unchanged) |
| Tests | Provider schema/end-to-end, metrics consistency, order-lifecycle coverage (limit, stop, stop-limit, trailing, bracket, OCO) |
| Tooling | `tools/byo_data_demo.py`, `tools/check_private_artifacts.py`, `tools/install_pre_commit_hook.py` |

`main` also carries a reviewed `quality/mypy-baseline.json` refresh (788 → 782
diagnostics) caused by removing the duplicated sample-data helpers.

## Working here

```bash
uv venv .venv --python 3.12          # or python 3.11+
uv pip install -e ".[data]"          # plus the optional extras you need
python -m pytest -q                  # full suite
python tools/verify_public_release.py
python tools/byo_data_demo.py        # real Yahoo data, no account
python tools/run_catalog_local.py    # all examples on Yahoo bars
python tools/compare_catalog_local.py  # vs published quants.md cards
python tools/check_private_artifacts.py --staged
```

CI runs on every branch push and on pull requests into `main`: source/lint/type
gate plus the Python 3.11–3.14 test matrix. Standard runners on a public
repository are free; larger runners are not.

## Validating against the published catalogue

`run_catalog_local.py` executes each example's `main()` with the data fetch
redirected to Yahoo bars; `compare_catalog_local.py` joins the results with the
published quants.md cards. Last full run: 43 executed, 7 skipped (5 intraday,
2 optimizer workflows), 0 failures. Against the 40 strategies with published
metrics the Sharpe correlation was 0.97 (return 0.99), median absolute Sharpe
delta 0.024, and 98% of strategies were within 0.25.

Runs are data-dependent: Yahoo returns adjusted closes that differ by ~1e-5
between identical requests, and path-dependent strategies (partial rebalancing,
drift or threshold triggers) can bifurcate on that jitter. W12, for example,
alternates between roughly +237% and +269% total return when each run fetches
fresh bars. Pass `--cache-dir` to persist the first fetch; cached reruns are
bit-identical (43/43 metric rows matched in the last paired run).

Treat the comparison as a data-source sanity check, not parity: the published
cards use QuantJourney-managed data and execution assumptions, order-mode OHLC
comes from Yahoo, and FX/futures examples run without provider contract specs.

## Staying current with upstream

```bash
git fetch upstream
git switch main
git merge upstream/main
```

Expect conflicts only in files this fork edits: `backtester/core.py`,
`backtester/mixins/sdk_client.py`, the portfolio calculation modules,
`backtester/engines/performance.py`, `sample_data.py`, and
`release/public_artifacts.txt`. `backtester/data/` is additive.

Merge upstream releases when they tag rather than continuously; this fork pins
no upstream version and does not publish packages.

## Private work

Keep strategy alpha and licensed data out of this repository. Use a separate
private repository that imports this engine, and keep local scratch under
`private/`, `scratch/` or `notes/` (ignored). The commit guard refuses staged
data dumps, oversized files and token-shaped strings:

```bash
python tools/install_pre_commit_hook.py   # on machines where Git hooks can run
```
