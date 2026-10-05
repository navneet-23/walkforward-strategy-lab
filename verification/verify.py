"""
verify.py - independent verification of the walk-forward study's claims.

Level 1 (seconds):   python verify.py
    Recomputes every headline number in claims.json directly from the raw
    equity curve (study_equity_pit.csv) and window log (study_log_pit.csv)
    using only pandas/numpy - none of the project's own code.

Level 2 (minutes):   python verify.py --replay
    Rebuilds the portfolio engine from the shipped price panel and
    point-in-time membership, re-executes every logged winning strategy
    over its deployment window, and checks that the resulting equity
    curve lands on the claimed final value. This proves the equity curve
    follows from (prices, membership, winner specs, cost model) - not
    from trust in the saved CSV.

Level 3 (hours, see README_VERIFICATION.md):
    Rebuild membership from the index provider's own press releases
    (pit_reconstruct.py --all), re-download prices, re-run the full
    strategy search (walkforward_study.py --mode pit), and diff.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

TOL_PCT = 0.15        # tolerance on percentage-point metrics
TOL_VAL = 0.002       # relative tolerance on monetary values


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: computed {got:,.3f} "
          f"vs claimed {want:,.3f}")
    return ok


def level1():
    claims = json.load(open("claims.json"))
    eq = pd.read_csv("study_equity_pit.csv", index_col=0,
                     parse_dates=True).iloc[:, 0]
    L = pd.read_csv("study_log_pit.csv")
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    ok = True
    ok &= check("final value (Rs)", eq.iloc[-1], claims["final_value_inr"],
                TOL_VAL * claims["final_value_inr"])
    ok &= check("CAGR %", 100 * ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1),
                claims["cagr_pct"], TOL_PCT)
    ok &= check("Sharpe (ann.)", r.mean() / r.std() * np.sqrt(252),
                claims["sharpe_annualised"], 0.02)
    ok &= check("max drawdown %", 100 * (eq / eq.cummax() - 1).min(),
                claims["max_drawdown_pct"], TOL_PCT)
    ok &= check("positive windows", (L["deployed_6m_%"] > 0).sum(),
                claims["windows_positive"], 0.5)
    # benchmark from the same membership + prices
    close = pd.read_parquet("data_pit_wf/close.parquet")
    mem = pd.read_parquet("data_pit_wf/membership.parquet")
    m = mem.shift(1).fillna(False).astype(float)
    br = ((close.pct_change() * m).sum(axis=1)
          / m.sum(axis=1).replace(0, np.nan)).fillna(0)
    b = (1 + br).cumprod()
    b = b[(b.index >= eq.index[0]) & (b.index <= eq.index[-1])]
    ok &= check("benchmark CAGR %",
                100 * ((b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1),
                claims["benchmark_eqwt_pit_cagr_pct"], TOL_PCT)
    return ok


def level2():
    import walkforward_study as ws
    claims = json.load(open("claims.json"))
    L = pd.read_csv("study_log_pit.csv", parse_dates=["deploy_from"])
    close = pd.read_parquet("data_pit_wf/close.parquet")
    volume = pd.read_parquet("data_pit_wf/volume.parquet")
    member = pd.read_parquet("data_pit_wf/membership.parquet")
    eng = ws.Engine(close, volume, member)
    start = pd.Timestamp(claims["period"][0])
    end = pd.Timestamp(claims["period"][1])
    ws.COST = claims["transaction_cost_per_leg"]

    def parse_spec(name):
        parts = name.split("|")
        conds, gates = [], []
        for extra in parts[1:]:
            for tok in extra.split("+"):
                (gates if tok in ws.GATES else conds).append(tok)
        core, nstr = parts[0].rsplit("_n", 1)
        rest, universe = core.rsplit("_", 1)
        return (tuple(rest.split("&")), universe, int(nstr),
                tuple(sorted(conds)), tuple(sorted(gates)))

    wbd = {}
    for _, row in L.iterrows():
        dep0 = pd.Timestamp(row["deploy_from"])
        dep1 = min(dep0 + pd.DateOffset(months=6), end + pd.Timedelta(days=1))
        dd = eng.fortnights[(eng.fortnights >= dep0) & (eng.fortnights < dep1)]
        wbd.update(eng.weights(parse_spec(row["winner"]), dd))
        print(f"  replayed {dep0.date()} {row['winner']}")
    eq = eng.backtest(wbd, start, end)
    print(f"\n  replayed final: Rs {eq.iloc[-1]:,.0f}")
    return check("REPLAY final value", eq.iloc[-1], claims["final_value_inr"],
                 TOL_VAL * claims["final_value_inr"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true",
                    help="level 2: re-execute logged strategies from prices")
    a = ap.parse_args()
    print("=== Level 1: recompute claims from raw outputs ===")
    ok = level1()
    if a.replay:
        print("\n=== Level 2: replay strategies from prices + membership ===")
        ok &= level2()
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if ok else 1)
