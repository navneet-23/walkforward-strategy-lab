"""
WALK-FORWARD META-TEST: does the strategy-search APPROACH generalize?

Every 6 months (Jan 1 / Jul 1), a shallow version of the tree search is
re-run on the TRAILING 2 YEARS only; its winner is deployed, fortnightly,
for the NEXT 6 months. Repeated 2015 -> mid-2026 on NIFTY100+Midcap100.
Each deployment window is genuinely out-of-sample for the strategy chosen.

Shallow search pool per retrain (~100 candidates):
  bases: low_turnover, adx_14, macd_5_35, macd_12_26, trix_15, kst,
         mom_3m, mom_6m, px_sma50, sharpe_6m
  x universe {all, highvol} x N {4, 8}; + cross-family pairs of top singles.
Selection: highest training-window PnL.

Outputs: per-window winner + trained vs deployed 6m return (the honest
generalization gap), stitched 2015-2026 equity, yearly table vs universe
composite, and summary stats.

Usage:  python standalone_walkforward.py [--synthetic]
        (first run downloads 2013-2026 into data_walkforward/, ~15-25 min;
         cached after. Search phase ~10-15 min.)
"""
import argparse
import io
import itertools
import os
import time

import numpy as np
import pandas as pd

CAPITAL = 1e7
COST = 0.001
DL_START, DL_END = "2013-01-01", "2026-07-01"
DATA = "data_walkforward"
MIN_TURNOVER = 1e7
START_DEPLOY = pd.Timestamp("2015-01-01")
END_DEPLOY = pd.Timestamp("2026-06-30")
TRAIN_YEARS = 2
BASES = ["low_turnover", "adx_14", "macd_5_35", "macd_12_26", "trix_15",
         "kst", "mom_3m", "mom_6m", "px_sma50", "sharpe_6m"]
FAMILY = {"low_turnover": "volm", "adx_14": "adx", "macd_5_35": "macd",
          "macd_12_26": "macd", "trix_15": "macd", "kst": "macd",
          "mom_3m": "mom", "mom_6m": "mom", "sharpe_6m": "mom", "px_sma50": "ma"}
NS = (4, 8)
UNIVERSES = ("all", "highvol")
TOP_SINGLES_FOR_PAIRS = 6

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
INDEX_CSVS = {"nifty100": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
              "midcap100": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv"}


def get_data(synthetic):
    if synthetic:
        rng = np.random.default_rng(11)
        dates = pd.bdate_range(DL_START, "2026-06-30")
        n, k = len(dates), 60
        close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(3e-4, 0.018, (n, k)), axis=0)),
                             index=dates, columns=[f"SYN{i:03d}.NS" for i in range(k)])
        vol = pd.DataFrame(rng.integers(2e5, 5e6, (n, k)).astype(float),
                           index=dates, columns=close.columns)
        return close, vol
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
    close.to_parquet(pc); vol.to_parquet(pc.replace("close", "volume"))
    print(f"  panel: {close.shape[1]} tickers")
    return close, vol


def build_signals(close, volume):
    ema = lambda s, n: s.ewm(span=n, adjust=False).mean()
    wema = lambda s: s.ewm(alpha=1 / 14, adjust=False).mean()
    r = close.pct_change()
    d = close.diff()
    atr = d.abs().pipe(wema).replace(0, np.nan)
    pdi = d.clip(lower=0).pipe(wema) / atr
    mdi = (-d).clip(lower=0).pipe(wema) / atr
    adx = wema((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan))
    sroc = lambda a, b: close.pct_change(a).rolling(b).mean()
    turnover = (close * volume).rolling(60).mean()
    F = {"low_turnover": -np.log(turnover.replace(0, np.nan)),
         "adx_14": adx,
         "macd_5_35": (ema(close, 5) - ema(close, 35)) / close,
         "macd_12_26": (ema(close, 12) - ema(close, 26)) / close,
         "trix_15": ema(ema(ema(close, 15), 15), 15).pct_change(),
         "kst": 1 * sroc(10, 10) + 2 * sroc(15, 10) + 3 * sroc(20, 10) + 4 * sroc(30, 15),
         "mom_3m": close.pct_change(63), "mom_6m": close.pct_change(126),
         "px_sma50": close / close.rolling(50).mean(),
         "sharpe_6m": close.pct_change(126) / (r.rolling(60).std() * np.sqrt(252)).replace(0, np.nan)}
    aux = {"mom_12m": close.pct_change(252), "vol_60": r.rolling(60).std() * np.sqrt(252),
           "turnover_60": turnover}
    return F, aux


def zrow(s):
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd and np.isfinite(sd) and sd > 0 else s * 0.0


def run_backtest(close, wbd, start, end, start_capital=CAPITAL):
    mask = (close.index >= start) & (close.index <= end)
    dates = close.index[mask]
    rets = close.pct_change()
    pos, cash = pd.Series(dtype=float), start_capital
    eq = np.empty(len(dates))
    wkeys = set(wbd)
    n_tr = 0
    for k, dt in enumerate(dates):
        if len(pos):
            pos = pos * (1 + rets.loc[dt].reindex(pos.index).fillna(0.0))
        total = cash + pos.sum()
        if dt in wkeys:
            w = wbd[dt]
            allt = w.index.union(pos.index)
            cur = pos.reindex(allt).fillna(0.0)
            tgt = (w * total).reindex(allt).fillna(0.0)
            total -= COST * (tgt - cur).abs().sum()
            tgt = (w * total).reindex(allt).fillna(0.0)
            n_tr += int(((cur <= 1e-9) & (tgt > 1e-9)).sum())
            pos = tgt[tgt > 1e-9]
            cash = total - pos.sum()
        eq[k] = cash + pos.sum()
    return pd.Series(eq, index=dates), n_tr


class WF:
    def __init__(self, close, volume):
        self.close = close
        self.F, self.aux = build_signals(close, volume)
        wk = pd.Series(close.index, index=close.index).groupby(close.index.to_period("W")).last()
        self.fortnights = pd.DatetimeIndex(wk.values)[::2]
        self.mask_cache = {}
        self.score_cache = {}

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

    def weights_for(self, spec, dates):
        blend, universe, n = spec
        out = {}
        for d in dates:
            key = (blend, universe, d)
            if key not in self.score_cache:
                ok = self.eligible(d, universe)
                score = None
                for b in blend:
                    z = zrow(self.F[b].loc[d].where(ok))
                    score = z if score is None else score + z
                self.score_cache[key] = score.dropna().sort_values(ascending=False)
            s = self.score_cache[key]
            out[d] = (pd.Series(1.0 / n, index=s.head(n).index)
                      if len(s) >= n else pd.Series(dtype=float))
        return out

    def train_eval(self, spec, t0, t1):
        dates = self.fortnights[(self.fortnights >= t0) & (self.fortnights < t1)]
        wbd = self.weights_for(spec, dates)
        eq, _ = run_backtest(self.close, wbd, t0, t1)
        return float(eq.iloc[-1] / eq.iloc[0] - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    close, volume = get_data(args.synthetic)
    print(f"panel: {close.shape[1]} tickers, {close.index[0].date()} -> {close.index[-1].date()}")
    wf = WF(close, volume)

    # 6-month deployment windows
    windows = []
    t = START_DEPLOY
    while t < END_DEPLOY:
        t_end = min(t + pd.DateOffset(months=6), END_DEPLOY + pd.Timedelta(days=1))
        windows.append((t - pd.DateOffset(years=TRAIN_YEARS), t, t_end))
        t = t + pd.DateOffset(months=6)

    singles = [((b,), u, n) for b in BASES for u in UNIVERSES for n in NS]
    log = []
    deploy_wbd = {}
    t0_all = time.time()
    for tr0, dep0, dep1 in windows:
        scores = {sp: wf.train_eval(sp, tr0, dep0) for sp in singles}
        top = sorted(scores, key=scores.get, reverse=True)
        # cross-family pairs of the top single indicators (at their own u,n)
        seen_b, top_bases = [], []
        for sp in top:
            b = sp[0][0]
            if b not in seen_b:
                seen_b.append(b)
                top_bases.append(sp)
            if len(top_bases) == TOP_SINGLES_FOR_PAIRS:
                break
        for s1, s2 in itertools.combinations(top_bases, 2):
            a, b = s1[0][0], s2[0][0]
            if FAMILY[a] == FAMILY[b]:
                continue
            for u in UNIVERSES:
                for n in NS:
                    sp = (tuple(sorted((a, b))), u, n)
                    if sp not in scores:
                        scores[sp] = wf.train_eval(sp, tr0, dep0)
        winner = max(scores, key=scores.get)
        dep_dates = wf.fortnights[(wf.fortnights >= dep0) & (wf.fortnights < dep1)]
        deploy_wbd.update(wf.weights_for(winner, dep_dates))
        eq_d, _ = run_backtest(close, wf.weights_for(winner, dep_dates), dep0,
                               dep1 - pd.Timedelta(days=1))
        dep_ret = float(eq_d.iloc[-1] / eq_d.iloc[0] - 1)
        log.append({"deploy_from": dep0.date(), "winner": "&".join(winner[0]),
                    "universe": winner[1], "N": winner[2],
                    "trained_6m_ann_%": round(100 * ((1 + scores[winner]) ** 0.25 - 1) * 2, 1),
                    "trained_2y_%": round(100 * scores[winner], 1),
                    "deployed_6m_%": round(100 * dep_ret, 1)})
        print(f"  {dep0.date()} -> {'&'.join(winner[0])}_{winner[1]}_n{winner[2]}"
              f"  train2y {scores[winner]*100:+.0f}%  deployed6m {dep_ret*100:+.1f}%"
              f"  ({time.time()-t0_all:.0f}s)")

    L = pd.DataFrame(log)
    eq, n_tr = run_backtest(close, deploy_wbd, START_DEPLOY, END_DEPLOY)
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    univ = (1 + close.pct_change().mean(axis=1)).cumprod()
    print("\n=== WALK-FORWARD RESULT (every 6m window out-of-sample) ===")
    print(f"final multiple : {eq.iloc[-1]/CAPITAL:.1f}x   CAGR {100*((eq.iloc[-1]/CAPITAL)**(1/yrs)-1):.1f}%")
    print(f"max drawdown   : {100*float((eq/eq.cummax()-1).min()):.1f}%")
    print(f"Sharpe (daily) : {float(r.mean()/(r.std()+1e-12)*np.sqrt(252)):.2f}   trades entered: {n_tr}")
    pos_windows = int((L["deployed_6m_%"] > 0).sum())
    print(f"windows positive: {pos_windows}/{len(L)}   "
          f"avg deployed 6m: {L['deployed_6m_%'].mean():+.1f}%")
    print("\nyearly (%):  [walk-forward | universe eq-wt]")
    for y in list(range(2015, 2026)) + ["2026H1"]:
        s = pd.Timestamp(f"{y}-01-01") if y != "2026H1" else pd.Timestamp("2026-01-01")
        e = pd.Timestamp(f"{y}-12-31") if y != "2026H1" else pd.Timestamp("2026-06-30")
        w = eq[(eq.index >= s) & (eq.index <= e)]
        u = univ[(univ.index >= s) & (univ.index <= e)].dropna()
        if len(w) > 10:
            print(f"  {y}: {100*(w.iloc[-1]/w.iloc[0]-1):+7.1f} | {100*(u.iloc[-1]/u.iloc[0]-1):+6.1f}")
    print("\n=== PER-WINDOW LOG ===")
    with pd.option_context("display.width", 200):
        print(L.to_string(index=False))
    L.to_csv("walkforward_log.csv", index=False)
    eq.to_csv("walkforward_equity.csv")
    print("\nsaved: walkforward_log.csv, walkforward_equity.csv")


if __name__ == "__main__":
    main()
