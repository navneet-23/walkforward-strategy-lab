# Independent Verification Package
## Walk-Forward Strategy Study - NIFTY 100 + Midcap 100, Point-in-Time

**Claim under verification** (see `claims.json` for exact values):
Rs 1 crore -> Rs 23.19 crore over 2015-01-01 .. 2026-06-30
(CAGR 31.46%, annualised Sharpe 1.156, max drawdown -37.1%),
net of 0.1% per-leg transaction costs, on a point-in-time universe,
vs 15.9% CAGR for the equal-weight basket of the same universe.

Every deployed 6-month window is out-of-sample for the strategy chosen
in it: strategies are selected by a tournament over ~580 candidates on
the trailing 2 years, then traded unchanged (see the method note).

## What's in this package

| File | Role |
|---|---|
| `claims.json` | the headline numbers, machine-readable |
| `verify.py` | the verifier (levels 1 and 2) |
| `study_equity_pit.csv` | daily equity curve produced by the study |
| `study_log_pit.csv` | winner + trained/deployed return per window |
| `data_pit_wf/close.parquet`, `volume.parquet` | daily adjusted prices and volumes, 323 tickers, 2013-2026 |
| `data_pit_wf/membership.parquet` | point-in-time index membership (event-based) |
| `pit_events/events.csv` | the 769 inclusion/exclusion events parsed from official press releases |
| `pit_events/parse_report.txt` | parser balance checks per index per year |
| `pit_events/reconcile.txt` | diff vs the older snapshot-based membership + the 38 unrecoverable names |
| `walkforward_study.py` | the full study: signals, engine, tournament, walk-forward |
| `pit_reconstruct.py` | rebuilds membership from the index provider's press-release archive |
| `make_report.py` | regenerates the HTML report from the study outputs |
| `walkforward_report.html` | the report these claims appear in |

## Verification levels

**Level 1 - recompute the claims (seconds, no trust in our code):**
```
pip install pandas numpy pyarrow
python verify.py
```
Recomputes CAGR, Sharpe, drawdown, window count and the benchmark
directly from the raw CSV/parquet files with plain pandas.

**Level 2 - replay the strategies (minutes):**
```
python verify.py --replay
```
Rebuilds the portfolio engine from prices + membership, re-executes the
23 logged winning strategies over their deployment windows with the
0.1%/leg cost model, and checks the final value to 0.2%. This proves
the equity curve follows mechanically from (prices, membership, winner
specs, costs).

**Level 3 - rebuild everything from primary sources (hours):**
1. `python pit_reconstruct.py --all` - downloads the 204 index
   press releases from the provider's own archive, re-parses the
   inclusion/exclusion events, re-derives membership from today's
   official constituent lists walked backward. Diff against
   `data_pit_wf/membership.parquet`.
2. Re-download prices (script in the main repo builds the panel from
   Yahoo Finance; adjusted prices are restated over time, so expect
   approximate, not bit-exact, reproduction).
3. `python walkforward_study.py --workers 8 --mode pit` - re-runs the
   full tournament + walk-forward (~30-120 min). The search is
   deterministic given identical data.
4. `python make_report.py` - regenerates the report.

## Known limitations (also stated in the report)

- 38 historical members (listed in `pit_events/reconcile.txt`,
  concentrated pre-2019, skewed toward delisted failures) have no
  recoverable price history and are absent from the tradable universe:
  a modest upward bias.
- Prices are Yahoo Finance adjusted closes (dividends via adjustment);
  slippage/market impact and taxes are not modelled; fills at close of
  rebalance day. The 0.1%/leg cost approximates the statutory
  delivery-equity stack with zero discount brokerage.
- Level-3 reproduction is approximate because Yahoo restates adjusted
  prices; Levels 1-2 are exact against the shipped data.
