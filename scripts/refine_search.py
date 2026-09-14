"""
Stage-2 refinement: fine grid around the stage-1 winner (mom12_52wh weekly).

Every variant is still an explainable one-liner:
  score = z( momentum over L months, skipping most recent S ) * 1.0
        + z( proximity to 52-week high )                      * w52
        + z( -60d volatility )                                * wlv
  buy top N (8 or 10), weekly, equal or rank weights, +- market cash filter.

Grid: L in {3,6,9,12}m x S in {0,1}m x w52 in {0,0.5,1,1.5} x wlv in {0,0.5}
      x N in {8,10} x weighting in {equal,rank} x cash in {none,market}
      = 512 configs, all weekly. ~10-15 min.

Usage:
    python refine_search.py             # full grid, real data
    python refine_search.py --quick     # every 8th config (sanity pass)
    python refine_search.py --synthetic

Output: results/refine_results.csv (+ run folders), top-20 printed.
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

from backtest import rebalance_dates
from features import compute_features, zscore_row
from run_search import RESULTS_DIR, START, evaluate, load_data
from strategies import build_market_regime, eligible, to_weights

LOOKBACKS_M = [3, 6, 9, 12]     # months
SKIPS_M = [0, 1]
W52 = [0.0, 0.5, 1.0, 1.5]
WLV = [0.0, 0.5]
TOPN = [8, 10]
WEIGHTINGS = ["equal", "rank"]
CASH = ["none", "market"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(RESULTS_DIR, "runs"), exist_ok=True)
    print("loading data ...")
    close, volume = load_data(args.synthetic)
    feats = compute_features(close, volume)
    regime = build_market_regime(close)
    weekly = pd.DatetimeIndex([d for d in rebalance_dates(close.index, "W")
                               if d >= pd.Timestamp(START)])

    # precompute momentum variants (trading days: 21/month)
    print("precomputing momentum variants ...")
    mom = {}
    for L, S in itertools.product(LOOKBACKS_M, SKIPS_M):
        if S >= L:
            continue
        mom[(L, S)] = close.shift(S * 21).pct_change((L - S) * 21)

    # precompute per-date z-scores of the three ingredients at rebalance dates
    print("precomputing per-date z-scores ...")
    zmom = {k: {} for k in mom}
    z52, zlv, elig = {}, {}, {}
    for d in weekly:
        ok = eligible(feats, d)
        elig[d] = ok
        z52[d] = zscore_row(feats["dist_52wh"].loc[d].where(ok))
        zlv[d] = zscore_row((-feats["vol_60"].loc[d]).where(ok))
        for k, df in mom.items():
            zmom[k][d] = zscore_row(df.loc[d].where(ok))

    grid = [g for g in itertools.product(mom.keys(), W52, WLV, TOPN, WEIGHTINGS, CASH)]
    if args.quick:
        grid = grid[::8]
    print(f"{len(grid)} configs to run\n")

    rows = []
    for (L, S), w52, wlv, n, weighting, cm in grid:
        t0 = time.time()
        run_id = f"ref_L{L}S{S}_w52-{w52}_wlv-{wlv}_n{n}_{weighting}_{cm}"
        wbd = {}
        for d in weekly:
            if cm == "market" and not bool(regime.loc[d]):
                wbd[d] = pd.Series(dtype=float)
                continue
            score = zmom[(L, S)][d] + w52 * z52[d] + wlv * zlv[d]
            score = score.dropna()
            if len(score) >= n:
                wbd[d] = to_weights(score, n, weighting)
        evaluate(run_id, {"family": "refine", "lookback_m": L, "skip_m": S,
                          "w_52wh": w52, "w_lowvol": wlv, "top_n": n,
                          "weighting": weighting, "cash_mode": cm, "freq": "W"},
                 close, wbd, rows, t0)

    df = pd.DataFrame(rows).sort_values("pnl", ascending=False)
    out = os.path.join(RESULTS_DIR, "refine_results.csv")
    df.to_csv(out, index=False)
    print(f"\n{len(rows)} configs -> {out}")
    print("\nTOP 20 BY PNL:")
    cols = ["run_id", "pnl", "cagr_pct", "max_drawdown_pct", "sharpe", "n_trades"]
    with pd.option_context("display.width", 220):
        print(df.head(20)[cols].to_string(index=False))
    print("\nstage-1 champion for reference: rule_mom12_52wh_W_rank_none  "
          "PnL Rs10.59cr  CAGR 56.2%  MDD -32.0%")


if __name__ == "__main__":
    main()
