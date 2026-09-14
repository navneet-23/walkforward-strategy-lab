"""
WALK-FORWARD CART (deployment research) - parallel, self-contained.

Every 6 months: a CART search (singles -> cross-family pairs -> condition
tree to DEPTH 2) is trained on the trailing 2 years only; its winner trades
fortnightly for the next 6 months. 2015 -> mid-2026, NIFTY100 + Midcap100.

Parallel: a process pool evaluates the ~180 candidates per retrain; each
worker builds signals once (initializer) and is reused across all windows.

Benchmarks reported (both flavours, on purpose):
  NIFTY 100 official index      (survivorship-free: index history contains
                                 the members it had at the time)
  NIFTY 100 eq-wt composite     (today's members applied backwards)
  Midcap 100 official index     (if Yahoo serves it; else skipped)
  Midcap 100 eq-wt composite
The composite-minus-index gap per tier is a visible survivorship proxy.

Usage:
    python walkforward_cart.py                     # ~data download first run
    python walkforward_cart.py --workers 8
    python walkforward_cart.py --cost 0.001        # default 0.0 per your spec
    python walkforward_cart.py --start 2015-01-01 --end 2016-12-31   # subrange
    python walkforward_cart.py --synthetic --workers 2 --end 2015-12-31
Outputs: wf_cart_log.csv, wf_cart_equity.csv, console report
(net worth from Rs 1 crore, annualised Sharpe overall and per window).
"""
import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

CAPITAL = 1e7
DL_START, DL_END = "2013-01-01", "2026-07-01"
DATA = "data_walkforward"
MIN_TURNOVER = 1e7
TRAIN_YEARS = 2
BASES = ["low_turnover", "adx_14", "macd_5_35", "macd_12_26", "trix_15",
         "kst", "mom_3m", "mom_6m", "px_sma50", "sharpe_6m"]
FAMILY = {"low_turnover": "volm", "adx_14": "adx", "macd_5_35": "macd",
          "macd_12_26": "macd", "trix_15": "macd", "kst": "macd",
          "mom_3m": "mom", "mom_6m": "mom", "sharpe_6m": "mom", "px_sma50": "ma"}
CONDS = ["rsi_below_85", "vol_above_med", "mom6_positive", "above_50dma", "near_52wh"]
NS = (4, 8)
UNIVERSES = ("all", "highvol")
TOP_SINGLES = 6
BEAM = 6
DEPTH = 2

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
INDEX_CSVS = {"nifty100": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
              "midcap100": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv"}


# ------------------------------------------------------------ data
def make_synthetic():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range(DL_START, "2026-06-30")
    n, k = len(dates), 60
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(3e-4, 0.018, (n, k)), axis=0)),
                         index=dates, columns=[f"SYN{i:03d}.NS" for i in range(k)])
    vol = pd.DataFrame(rng.integers(2e5, 5e6, (n, k)).astype(float),
                       index=dates, columns=close.columns)
    return close, vol


def get_data():
    import requests
    import yfinance as yf
    os.makedirs(DATA, exist_ok=True)
    tickers = []
    for name, url in INDEX_CSVS.items():
        local = os.path.join(DATA, f"{name}.csv")
        if not os.path.exists(local):
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            open(local, "w").write(r.text)
        df = pd.read_csv(local)
        sym = "Symbol" if "Symbol" in df.columns else df.columns[2]
        tickers += list(df[sym].astype(str).str.strip() + ".NS")
    tickers = sorted(set(tickers))
    pc = os.path.join(DATA, "close.parquet")
    if os.path.exists(pc):
        return pd.read_parquet(pc), pd.read_parquet(pc.replace("close", "volume"))
    closes, vols = [], []
    for i in range(0, len(tickers), 40):
        batch = tickers[i:i + 40]
        print(f"  downloading {i+1}-{i+len(batch)}/{len(tickers)}")
        raw = yf.download(batch, start=DL_START, end=DL_END, auto_adjust=True,
                          group_by="ticker", progress=False, threads=True)
        for t in batch:
            try:
                closes.append(raw[t]["Close"].rename(t))
                vols.append(raw[t]["Volume"].rename(t))
            except Exception:
                pass
        time.sleep(1)
    close = pd.concat(closes, axis=1).sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    vol = pd.concat(vols, axis=1).sort_index()
    vol.index = close.index
    bad = [t for t in tickers if t not in close.columns or close[t].notna().sum() < 1200]
    print(f"  retry pass: {len(bad)}")
    for t in bad:
        try:
            d = yf.download(t, start=DL_START, end=DL_END, auto_adjust=True, progress=False)
            if len(d) > 250:
                c = d["Close"].squeeze(); c.index = pd.to_datetime(c.index).tz_localize(None)
                v = d["Volume"].squeeze(); v.index = c.index
                close[t] = c.reindex(close.index); vol[t] = v.reindex(vol.index)
        except Exception:
            pass
        time.sleep(1.5)
    close.to_parquet(pc)
    vol.to_parquet(pc.replace("close", "volume"))
    return close, vol


def get_bench_index(cache_name, candidates):
    import yfinance as yf
    p = os.path.join(DATA, cache_name)
    if os.path.exists(p):
        return pd.read_parquet(p).iloc[:, 0].dropna()
    for tick in candidates:
        try:
            s = yf.download(tick, start="2014-06-01", end=DL_END,
                            auto_adjust=True, progress=False)["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            if len(s) > 500:
                s.index = pd.to_datetime(s.index).tz_localize(None)
                s.rename("b").to_frame().to_parquet(p)
                return s.dropna()
        except Exception:
            continue
    return None


# ------------------------------------------------------------ engine
class Engine:
    def __init__(self, close, volume, cost):
        self.close = close
        self.cost = cost
        self.rets = close.pct_change()
        ema = lambda s, n: s.ewm(span=n, adjust=False).mean()
        wema = lambda s: s.ewm(alpha=1 / 14, adjust=False).mean()
        r = self.rets
        d = close.diff()
        atr = d.abs().pipe(wema).replace(0, np.nan)
        pdi = d.clip(lower=0).pipe(wema) / atr
        mdi = (-d).clip(lower=0).pipe(wema) / atr
        ru, rd = d.clip(lower=0).pipe(wema), (-d.clip(upper=0)).pipe(wema)
        sroc = lambda a, b: close.pct_change(a).rolling(b).mean()
        turnover = (close * volume).rolling(60).mean()
        self.F = {"low_turnover": -np.log(turnover.replace(0, np.nan)),
                  "adx_14": wema((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)),
                  "macd_5_35": (ema(close, 5) - ema(close, 35)) / close,
                  "macd_12_26": (ema(close, 12) - ema(close, 26)) / close,
                  "trix_15": ema(ema(ema(close, 15), 15), 15).pct_change(),
                  "kst": 1 * sroc(10, 10) + 2 * sroc(15, 10) + 3 * sroc(20, 10) + 4 * sroc(30, 15),
                  "mom_3m": close.pct_change(63), "mom_6m": close.pct_change(126),
                  "px_sma50": close / close.rolling(50).mean(),
                  "sharpe_6m": close.pct_change(126) / (r.rolling(60).std() * np.sqrt(252)).replace(0, np.nan)}
        self.rsi = 100 - 100 / (1 + ru / rd.replace(0, np.nan))
        self.dist52 = close / close.rolling(252).max()
        self.aux = {"mom_12m": close.pct_change(252),
                    "vol_60": r.rolling(60).std() * np.sqrt(252),
                    "turnover_60": turnover}
        wk = pd.Series(close.index, index=close.index).groupby(close.index.to_period("W")).last()
        self.fortnights = pd.DatetimeIndex(wk.values)[::2]
        self.mask_cache, self.score_cache = {}, {}

    def cond(self, name, d):
        if name == "rsi_below_85":
            return self.rsi.loc[d] < 85
        if name == "vol_above_med":
            v = self.aux["vol_60"].loc[d]
            return v > v.median()
        if name == "mom6_positive":
            return self.F["mom_6m"].loc[d] > 0
        if name == "above_50dma":
            return self.F["px_sma50"].loc[d] > 1
        return self.dist52.loc[d] > 0.90          # near_52wh

    def eligible(self, d, universe):
        key = (d, universe)
        if key in self.mask_cache:
            return self.mask_cache[key]
        ok = self.aux["mom_12m"].loc[d].notna()
        liq = self.aux["turnover_60"].loc[d]
        ok &= liq.notna() & (liq >= MIN_TURNOVER)
        if universe == "highvol":
            v = self.aux["vol_60"].loc[d].where(ok)
            ok &= v > v.median()
        self.mask_cache[key] = ok
        return ok

    def weights(self, spec, dates):
        blend, universe, n, conds = spec
        out = {}
        for d in dates:
            key = (blend, universe, conds, d)
            if key not in self.score_cache:
                ok = self.eligible(d, universe)
                for c in conds:
                    ok = ok & self.cond(c, d).reindex(ok.index).fillna(False)
                score = None
                for b in blend:
                    z = self.F[b].loc[d].where(ok)
                    mu, sd = z.mean(), z.std()
                    z = (z - mu) / sd if sd and np.isfinite(sd) and sd > 0 else z * 0.0
                    score = z if score is None else score + z
                self.score_cache[key] = score.dropna().sort_values(ascending=False)
            s = self.score_cache[key]
            out[d] = (pd.Series(1.0 / n, index=s.head(n).index)
                      if len(s) >= n else pd.Series(dtype=float))
        return out

    def backtest(self, wbd, start, end, capital=CAPITAL):
        mask = (self.close.index >= start) & (self.close.index <= end)
        dates = self.close.index[mask]
        pos, cash = pd.Series(dtype=float), capital
        eq = np.empty(len(dates))
        wkeys = set(wbd)
        for k, dt in enumerate(dates):
            if len(pos):
                pos = pos * (1 + self.rets.loc[dt].reindex(pos.index).fillna(0.0))
            total = cash + pos.sum()
            if dt in wkeys:
                w = wbd[dt]
                allt = w.index.union(pos.index)
                cur = pos.reindex(allt).fillna(0.0)
                tgt = (w * total).reindex(allt).fillna(0.0)
                total -= self.cost * (tgt - cur).abs().sum()
                tgt = (w * total).reindex(allt).fillna(0.0)
                pos = tgt[tgt > 1e-9]
                cash = total - pos.sum()
            eq[k] = cash + pos.sum()
        return pd.Series(eq, index=dates)

    def train_ret(self, spec, t0, t1):
        dates = self.fortnights[(self.fortnights >= t0) & (self.fortnights < t1)]
        eq = self.backtest(self.weights(spec, dates), t0, t1)
        return float(eq.iloc[-1] / eq.iloc[0] - 1)


# ------------------------------------------------------------ worker pool
_ENG = None


def _init_worker(synthetic, cost):
    global _ENG
    if synthetic:
        close, volume = make_synthetic()
    else:
        close = pd.read_parquet(os.path.join(DATA, "close.parquet"))
        volume = pd.read_parquet(os.path.join(DATA, "volume.parquet"))
    _ENG = Engine(close, volume, cost)


def _eval(job):
    spec, t0, t1 = job
    return spec, _ENG.train_ret(spec, pd.Timestamp(t0), pd.Timestamp(t1))


def run_stage(pool, specs, t0, t1, scores):
    todo = [s for s in specs if s not in scores]
    for spec, ret in pool.map(_eval, [(s, str(t0), str(t1)) for s in todo], chunksize=4):
        scores[spec] = ret
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--cost", type=float, default=0.0)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if not args.synthetic:
        close, volume = get_data()
    else:
        close, volume = make_synthetic()
    print(f"panel: {close.shape[1]} tickers | cost per leg: {args.cost} "
          f"{'(NOTE: costs disabled per spec - add --cost 0.001 for the costed run)' if args.cost == 0 else ''}")
    eng = Engine(close, volume, args.cost)

    windows = []
    t = pd.Timestamp(args.start)
    end_all = pd.Timestamp(args.end)
    while t < end_all:
        te = min(t + pd.DateOffset(months=6), end_all + pd.Timedelta(days=1))
        windows.append((t - pd.DateOffset(years=TRAIN_YEARS), t, te))
        t = t + pd.DateOffset(months=6)

    log, deploy_wbd = [], {}
    t0_all = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.synthetic, args.cost)) as pool:
        for tr0, dep0, dep1 in windows:
            scores = {}
            singles = [((b,), u, n, ()) for b in BASES for u in UNIVERSES for n in NS]
            run_stage(pool, singles, tr0, dep0, scores)
            ranked = sorted(scores, key=scores.get, reverse=True)
            seen, top_singles = [], []
            for sp in ranked:
                b = sp[0][0]
                if b not in seen:
                    seen.append(b)
                    top_singles.append(sp)
                if len(top_singles) == TOP_SINGLES:
                    break
            pairs = []
            for s1, s2 in itertools.combinations(top_singles, 2):
                a, b = s1[0][0], s2[0][0]
                if FAMILY[a] == FAMILY[b]:
                    continue
                for u in UNIVERSES:
                    for n in NS:
                        pairs.append((tuple(sorted((a, b))), u, n, ()))
            run_stage(pool, pairs, tr0, dep0, scores)
            # condition tree, depth 2, beam
            beam = sorted(scores, key=scores.get, reverse=True)[:BEAM]
            for _depth in range(DEPTH):
                children = []
                for parent in beam:
                    for c in CONDS:
                        if c in parent[3]:
                            continue
                        children.append((parent[0], parent[1], parent[2],
                                         tuple(sorted(parent[3] + (c,)))))
                run_stage(pool, children, tr0, dep0, scores)
                beam = sorted(scores, key=scores.get, reverse=True)[:BEAM]
            winner = beam[0]

            dep_dates = eng.fortnights[(eng.fortnights >= dep0) & (eng.fortnights < dep1)]
            wbd = eng.weights(winner, dep_dates)
            deploy_wbd.update(wbd)
            eq_d = eng.backtest(wbd, dep0, dep1 - pd.Timedelta(days=1))
            rd = eq_d.pct_change().dropna()
            dep_ret = float(eq_d.iloc[-1] / eq_d.iloc[0] - 1)
            name = "&".join(winner[0]) + f"_{winner[1]}_n{winner[2]}"
            if winner[3]:
                name += "|" + "+".join(winner[3])
            log.append({"deploy_from": dep0.date(), "winner": name,
                        "trained_2y_%": round(100 * scores[winner], 1),
                        "deployed_6m_%": round(100 * dep_ret, 1),
                        "deployed_sharpe_ann": round(float(rd.mean() / (rd.std() + 1e-12) * np.sqrt(252)), 2)})
            print(f"  {dep0.date()} -> {name}  train {scores[winner]*100:+.0f}%  "
                  f"deployed {dep_ret*100:+.1f}%  ({time.time()-t0_all:.0f}s, "
                  f"{len(scores)} candidates)")

    eq = eng.backtest(deploy_wbd, pd.Timestamp(args.start), end_all)
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    L = pd.DataFrame(log)
    L["networth_cr"] = (1 + L["deployed_6m_%"] / 100).cumprod().round(2)

    # benchmarks
    bench = {}
    if not args.synthetic:
        idx100 = get_bench_index("bench_n100.parquet", ["^CNX100"])
        if idx100 is not None:
            bench["NIFTY100_index"] = idx100
        idxmid = get_bench_index("bench_mid.parquet",
                                 ["NIFTY_MIDCAP_100.NS", "^CNXMIDCAP", "NIFTYMIDCAP100.NS"])
        if idxmid is not None:
            bench["Midcap100_index"] = idxmid
        for lab, fn in [("NIFTY100_eqw", "nifty100.csv"), ("Midcap100_eqw", "midcap100.csv")]:
            df = pd.read_csv(os.path.join(DATA, fn))
            sym = "Symbol" if "Symbol" in df.columns else df.columns[2]
            cols = [t for t in (df[sym].astype(str).str.strip() + ".NS") if t in close.columns]
            bench[lab] = (1 + close[cols].pct_change().mean(axis=1)).cumprod().dropna()

    print("\n=== WALK-FORWARD CART, depth-2 conditions (all windows out-of-sample) ===")
    print(f"net worth      : Rs {eq.iloc[-1]/1e7:.1f} crore  (from Rs 1 crore, {args.start} start)")
    print(f"CAGR           : {100*((eq.iloc[-1]/CAPITAL)**(1/yrs)-1):.1f}%")
    print(f"annualised Sharpe (daily): {float(r.mean()/(r.std()+1e-12)*np.sqrt(252)):.2f}")
    print(f"max drawdown   : {100*float((eq/eq.cummax()-1).min()):.1f}%")
    print(f"windows positive: {int((L['deployed_6m_%']>0).sum())}/{len(L)}  "
          f"avg deployed 6m: {L['deployed_6m_%'].mean():+.1f}%")
    print("\nyearly (%): walk-forward | " + " | ".join(bench.keys()))
    for y in list(range(int(args.start[:4]), 2026)) + ["2026H1"]:
        s = pd.Timestamp(f"{y}-01-01") if y != "2026H1" else pd.Timestamp("2026-01-01")
        e = pd.Timestamp(f"{y}-12-31") if y != "2026H1" else pd.Timestamp("2026-06-30")
        w = eq[(eq.index >= s) & (eq.index <= e)]
        if len(w) < 10:
            continue
        line = f"  {y}: {100*(w.iloc[-1]/w.iloc[0]-1):+7.1f}"
        for lab, bser in bench.items():
            bw = bser[(bser.index >= s) & (bser.index <= e)].dropna()
            line += f" | {100*(bw.iloc[-1]/bw.iloc[0]-1):+6.1f}" if len(bw) > 10 else " |    n/a"
        print(line)
    print("\n(eq-wt composites use TODAY'S constituents backwards - survivorship-"
          "loaded; official indices held their then-members - the per-tier gap "
          "between the two columns is a visible survivorship proxy.)")
    print("\n=== PER-WINDOW LOG ===")
    with pd.option_context("display.width", 220):
        print(L.to_string(index=False))
    L.to_csv("wf_cart_log.csv", index=False)
    eq.to_csv("wf_cart_equity.csv")
    print("\nsaved: wf_cart_log.csv, wf_cart_equity.csv")


if __name__ == "__main__":
    main()
