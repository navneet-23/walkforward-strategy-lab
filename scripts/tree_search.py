"""
BEAM TREE SEARCH over a LARGE strategy space, 2026H1 > 0 enforced at every node.

ROOT SPACE (evaluated competitively - the root is discovered, not chosen):
  base signals (~20): mom_5d, mom_1m, mom_2m, mom_3m, mom_6m, mom_12_1,
    sharpe_6m, rsi_14, cmo_14, tsi_25_13, stoch_14, updays_1m, tstat_1m,
    macd_8_17, macd_12_26, macd_5_35, trix_9, trix_12, trix_15, trix_21,
    kst, dmi_x_adx, adx_gate_mom, low_vol
  x universes {all, highvol, lowvol}   x frequencies {W, 2W}
  => ~140 roots, each backtested. Roots failing 2026H1>0 are pruned.

TREE: the best --beam roots are each expanded with every condition from the
library (AND-filters + portfolio gates); children failing 2026H1>0 are
pruned; the best --beam nodes across ALL parents survive to the next depth.
Duplicate condition-sets are deduped. Repeat to --steps.

Usage:
    python tree_search.py                          # beam 10, depth 3, ~25-40 min
    python tree_search.py --beam 20 --steps 4      # wider/deeper, ~1.5 hr
    python tree_search.py --synthetic --beam 3 --steps 1
Output: results/tree_search.csv (every node evaluated), top-30 printed,
best node -> results/greedy_spec.json  (package via: python final_submission.py greedy)
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from backtest import run_backtest
from features import compute_features
from final_submission import _rebal
from run_search import RESULTS_DIR, load_data
from strategies import eligible, to_weights

START, END = "2021-01-01", "2026-06-30"
OFF_END = "2025-12-31"
N = 8
CHUNKS = [("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
          ("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
          ("2025", "2025-01-01", "2025-12-31"), ("2026H1", "2026-01-01", "2026-06-30")]


# ---------------------------------------------------------------- signal frames
def build_frames(close, volume):
    ema = lambda s, n: s.ewm(span=n, adjust=False).mean()
    wema = lambda s: s.ewm(alpha=1 / 14, adjust=False).mean()
    r = close.pct_change()
    d = close.diff()

    ru, rd = d.clip(lower=0).pipe(wema), (-d.clip(upper=0)).pipe(wema)
    rsi = 100 - 100 / (1 + ru / rd.replace(0, np.nan))
    su, sd_ = d.clip(lower=0).rolling(14).sum(), (-d.clip(upper=0)).rolling(14).sum()
    cmo = (su - sd_) / (su + sd_).replace(0, np.nan)
    m1 = d.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    m2 = d.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    tsi = m1 / m2.replace(0, np.nan)
    lo14, hi14 = close.rolling(14).min(), close.rolling(14).max()
    stoch = (close - lo14) / (hi14 - lo14).replace(0, np.nan)
    mu, sd21 = r.rolling(21).mean(), r.rolling(21).std()
    atr = d.abs().pipe(wema).replace(0, np.nan)
    pdi, mdi = d.clip(lower=0).pipe(wema) / atr, (-d).clip(lower=0).pipe(wema) / atr
    dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = wema(dx)

    def trix(s):
        t = ema(ema(ema(close, s), s), s)
        return t.pct_change()

    def sroc(n, m):
        return close.pct_change(n).rolling(m).mean()

    F = {
        "mom_5d": close.pct_change(5), "mom_1m": close.pct_change(21),
        "mom_2m": close.pct_change(42), "mom_3m": close.pct_change(63),
        "mom_6m": close.pct_change(126), "mom_12_1": close.shift(21).pct_change(231),
        "rsi_14": rsi, "cmo_14": cmo, "tsi_25_13": tsi, "stoch_14": stoch,
        "updays_1m": (r > 0).rolling(21).mean().where(close.notna()),
        "tstat_1m": mu / sd21.replace(0, np.nan),
        "macd_8_17": (ema(close, 8) - ema(close, 17)) / close,
        "macd_12_26": (ema(close, 12) - ema(close, 26)) / close,
        "macd_5_35": (ema(close, 5) - ema(close, 35)) / close,
        "trix_9": trix(9), "trix_12": trix(12), "trix_15": trix(15), "trix_21": trix(21),
        "kst": 1 * sroc(10, 10) + 2 * sroc(15, 10) + 3 * sroc(20, 10) + 4 * sroc(30, 15),
        "dmi_x_adx": (pdi - mdi) * adx,
        "adx": adx,
    }
    return F


def base_score(F, feats, name):
    if name == "sharpe_6m":
        return lambda d: feats["sharpe_6m"].loc[d]
    if name == "low_vol":
        return lambda d: -feats["vol_60"].loc[d]
    if name == "adx_gate_mom":
        def f(d):
            a = F["adx"].loc[d]
            return F["mom_1m"].loc[d].where(a > a.median())
        return f
    return lambda d: F[name].loc[d]


BASES = ["mom_5d", "mom_1m", "mom_2m", "mom_3m", "mom_6m", "mom_12_1", "sharpe_6m",
         "rsi_14", "cmo_14", "tsi_25_13", "stoch_14", "updays_1m", "tstat_1m",
         "macd_8_17", "macd_12_26", "macd_5_35", "trix_9", "trix_12", "trix_15",
         "trix_21", "kst", "dmi_x_adx", "adx_gate_mom", "low_vol"]


def make_conditions(F, feats, mkt200):
    med = lambda s: s > s.median()
    conds = {
        "above_200dma": lambda d: feats["px_200dma"].loc[d] > 1,
        "above_50dma": lambda d: feats["px_50dma"].loc[d] > 1,
        "near_52wh": lambda d: feats["dist_52wh"].loc[d] > 0.90,
        "vol_below_med": lambda d: ~med(feats["vol_60"].loc[d]),
        "vol_above_med": lambda d: med(feats["vol_60"].loc[d]),
        "rsi_below_85": lambda d: F["rsi_14"].loc[d] < 85,
        "rsi_above_60": lambda d: F["rsi_14"].loc[d] > 60,
        "liq_above_med": lambda d: med(feats["turnover_60"].loc[d]),
        "mom6_positive": lambda d: feats["mom_6m"].loc[d] > 0,
        "week_positive": lambda d: F["mom_5d"].loc[d] > 0,
        "stoch_above_70": lambda d: F["stoch_14"].loc[d] > 0.70,
        "trix_positive": lambda d: F["trix_15"].loc[d] > 0,
        "macd_positive": lambda d: F["macd_12_26"].loc[d] > 0,
        "adx_above_med": lambda d: med(F["adx"].loc[d]),
    }
    gates = {"GATE_mkt_200dma": lambda d: bool(mkt200.loc[d])}
    return conds, gates


# ---------------------------------------------------------------- evaluation
class Engine:
    def __init__(self, close, volume, freqs, universes):
        self.close = close
        self.feats = compute_features(close, volume)
        self.F = build_frames(close, volume)
        univ = (1 + close.pct_change().mean(axis=1)).cumprod()
        self.mkt200 = univ > univ.rolling(200).mean()
        self.conds, self.gates = make_conditions(self.F, self.feats, self.mkt200)
        self.dates = {f: pd.DatetimeIndex([d for d in _rebal(close.index, f)
                                           if d >= pd.Timestamp(START)]) for f in freqs}
        self.masks = {}
        for f in freqs:
            for u in universes:
                m = {}
                for d in self.dates[f]:
                    ok = eligible(self.feats, d)
                    if u != "all":
                        v = self.feats["vol_60"].loc[d].where(ok)
                        mdn = v.median()
                        ok = ok & ((v <= mdn) if u == "lowvol" else (v > mdn))
                    m[d] = ok
                self.masks[(f, u)] = m
        self.cache = {}

    def eval_node(self, spec):
        key = (spec["base"], spec["freq"], spec["universe"],
               frozenset(spec["conditions"]), frozenset(spec["gates"]))
        if key in self.cache:
            return self.cache[key]
        score_at = base_score(self.F, self.feats, spec["base"])
        wbd = {}
        for d in self.dates[spec["freq"]]:
            if any(not self.gates[g](d) for g in spec["gates"]):
                wbd[d] = pd.Series(dtype=float)
                continue
            ok = self.masks[(spec["freq"], spec["universe"])][d]
            for c in spec["conditions"]:
                ok = ok & self.conds[c](d).reindex(ok.index).fillna(False)
            s = score_at(d).where(ok).dropna()
            wbd[d] = to_weights(s, N, "equal") if len(s) >= N else pd.Series(dtype=float)
        eq, tr, _ = run_backtest(self.close, wbd, START, END)
        off = eq[eq.index <= OFF_END]
        offn = 1e7 * off / off.iloc[0]
        r = offn.pct_change().dropna()
        row = {"pnl": round((offn.iloc[-1] - 1e7) / 1e7, 2),
               "mdd": round(100 * float((offn / offn.cummax() - 1).min()), 1),
               "sharpe": round(float(r.mean() / (r.std() + 1e-12) * np.sqrt(252)), 2),
               "n_trades": len(tr)}
        for label, s, e in CHUNKS:
            w = eq[(eq.index >= s) & (eq.index <= e)]
            row[f"y{label}"] = (round(100 * (w.iloc[-1] / w.iloc[0] - 1), 1)
                                if len(w) > 10 else np.nan)
        self.cache[key] = row
        return row


def node_id(spec):
    tail = "+".join(spec["conditions"] + spec["gates"])
    return f"{spec['base']}_{spec['freq']}_{spec['universe']}" + (f"|{tail}" if tail else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--freqs", default="W,2W")
    ap.add_argument("--universes", default="all,highvol,lowvol")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    freqs = args.freqs.split(",")
    universes = args.universes.split(",")

    print("loading data + precomputing signals/masks (few minutes) ...")
    close, volume = load_data(args.synthetic)
    eng = Engine(close, volume, freqs, universes)

    t0 = time.time()
    all_rows = []

    # ---- depth 0: root tournament ----
    roots = []
    for b, f, u in itertools.product(BASES, freqs, universes):
        spec = {"base": b, "freq": f, "universe": u, "conditions": [], "gates": [], "n": N}
        row = eng.eval_node(spec)
        rec = {"node": node_id(spec), "depth": 0, **row, "spec": spec}
        all_rows.append(rec)
        roots.append(rec)
    print(f"depth 0: {len(roots)} roots evaluated ({time.time()-t0:.0f}s)")
    beam = sorted([r for r in roots if r["y2026H1"] > 0],
                  key=lambda r: -r["pnl"])[:args.beam]
    print(f"  beam ({len(beam)}): " + "; ".join(f"{b['node']} {b['pnl']}cr" for b in beam[:5]) + " ...")

    # ---- expand ----
    for depth in range(1, args.steps + 1):
        children = []
        for parent in beam:
            for name in list(eng.conds) + list(eng.gates):
                if name in parent["spec"]["conditions"] or name in parent["spec"]["gates"]:
                    continue
                spec = {**parent["spec"],
                        "conditions": parent["spec"]["conditions"] + ([name] if name in eng.conds else []),
                        "gates": parent["spec"]["gates"] + ([name] if name in eng.gates else [])}
                row = eng.eval_node(spec)
                rec = {"node": node_id(spec), "depth": depth, **row, "spec": spec}
                children.append(rec)
                all_rows.append(rec)
        pool = beam + [c for c in children if c["y2026H1"] > 0]
        # dedupe by node id, keep best beam
        seen, newbeam = set(), []
        for r in sorted(pool, key=lambda r: -r["pnl"]):
            if r["node"] in seen:
                continue
            seen.add(r["node"])
            newbeam.append(r)
            if len(newbeam) == args.beam:
                break
        beam = newbeam
        print(f"depth {depth}: {len(children)} children evaluated, best now "
              f"{beam[0]['node']} = Rs{beam[0]['pnl']}cr ({time.time()-t0:.0f}s)")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "spec"} for r in all_rows])
    df = df.sort_values("pnl", ascending=False)
    df.to_csv(os.path.join(RESULTS_DIR, "tree_search.csv"), index=False)
    ok = df[df["y2026H1"] > 0]
    cols = ["node", "depth", "pnl", "mdd", "sharpe", "y2021", "y2022", "y2023",
            "y2024", "y2025", "y2026H1", "n_trades"]
    print(f"\n=== TOP 30 (of {len(df)} nodes; 2026H1>0: {len(ok)}) ===")
    with pd.option_context("display.width", 260):
        print(ok[cols].head(30).to_string(index=False))

    best = next(r for r in all_rows if r["node"] == ok.iloc[0]["node"])
    with open(os.path.join(RESULTS_DIR, "greedy_spec.json"), "w") as fh:
        json.dump(best["spec"], fh, indent=2)
    print(f"\nbest node spec -> results/greedy_spec.json "
          f"(package: python final_submission.py greedy)")


if __name__ == "__main__":
    main()
