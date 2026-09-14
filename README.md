# Walk-Forward Strategy Lab — Indian Equities

A research program that asks one question honestly: **does automated
strategy search generalize out-of-sample, and how much of a backtest's
performance is manufactured by hindsight?**

Universe: NIFTY 100 + Midcap 100 (NSE). Daily adjusted prices via Yahoo
Finance. 2015–2026, transaction cost 0.1% per leg.

## The system

Every 6 months, a tree search evaluates ~500 candidate strategies
(45 signals across 9 style families x 3 universe variants x portfolio
sizes {4, 8} x condition/gate tree of depth 2) on the trailing 2 years,
and deploys the winner for the next 6 months, rebalancing fortnightly.
Every deployment window is fully out-of-sample for the strategy chosen
in it. Meta-parameters (lookback, cadence, costs, eligibility) were
fixed a priori and never tuned.

## The results ladder — each row removes one layer of hindsight

| Configuration                                  | CAGR   | Sharpe | MDD    |
|------------------------------------------------|--------|--------|--------|
| In-sample champion (2021–25 seen at selection)  | ~142%  | 3.24   | -24.5% |
| Walk-forward, today's index constituents        | 46.7%  | 1.33   | -34.1% |
| Walk-forward, point-in-time constituents        | 25.2%  | 0.96   | -65.0% |
| Eq-wt basket of the PIT universe (benchmark)    | ~15%   | —      | —      |

The headline finding: knowing today's index membership eleven years
early — and nothing else — was worth roughly 21 percentage points of
CAGR per year. After removing selection bias AND survivorship bias,
roughly +10pp/yr of genuine alpha over the honest universe remains,
delivered with violent drawdowns and no stable strategy identity.

## Point-in-time reconstruction

Historical index membership is rebuilt from Wayback Machine captures of
the NSE constituent files (union of each index's most recent archived
list; symbol renames mapped; demoted members' prices re-downloaded).
Known limitations: sparse snapshot coverage (nothing before Nov 2017;
a 2019–2023 gap in the Midcap list), ~14 unrecoverable delisted or
renamed names. The honest error bar on the PIT figure is a few
percentage points in either direction.

## Files

- `scripts/walkforward_study.py` — the final, self-contained study:
  runs both configurations end-to-end (`--mode both|current|pit`)
- `scripts/walkforward_wide.py` — wide-pool walk-forward with a
  `--criterion return|sharpe|calmar` selection experiment
- `scripts/walkforward_pit.py` — the point-in-time engine (Wayback
  membership, alias map, former-member downloads)
- `scripts/pit_audit.py` — reconstruction quality audit + a 2x2
  strategy-x-universe cross-deployment test
- `scripts/` (others) — earlier iterations kept for the record
- `docs/` — search-space specification, parameter sheet, results
  overview (Word)

## Usage

```
pip install -r requirements.txt
python scripts/walkforward_study.py --workers 8        # both runs, ~2h
```

First run downloads ~280 tickers of price history and the archived
constituent snapshots; everything caches locally.

## Disclaimers

Research code, not investment advice. Yahoo Finance data is restated
over time; results reproduce approximately, not to the decimal. The
survivorship-loaded configuration is included deliberately, as the
measured counterfactual — not as a performance claim.
