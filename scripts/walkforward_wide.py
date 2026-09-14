"""
WIDE-STYLE WALK-FORWARD CART - many strategy styles compete every 6 months.

Style families in the pool (~34 bases):
  TREND      mom_3m, mom_6m, mom_12_1, macd_5_35, macd_12_26, trix_15, kst,
             px_sma50, px_sma200, ma_cross_50_200, adx_14, dmi_diff, dist_52wh
  CONTRARIAN rev_1m (buy 1m losers), rev_6m (buy 6m losers), rsi_oversold,
             stoch_oversold, near_52wl (buy near 52w lows), bb_lower
  DEFENSIVE  lowvol_60, lowvol_20, low_beta, low_atr, above_200_persist,
             dd_resist (shallowest 6m drawdown), updays_1m
  QUALITY    sharpe_6m, sortino_6m, calmar_6m, trend_r2 (smoothest trend)
  FLOW/LIQ   low_turnover, obv_slope, cmf_20, vol_surge
Universes {all, highvol, lowvol} x N {4, 8}; cross-family pairs of top
singles; condition tree DEPTH 2 over 7 conditions + 2 CASH GATES
(mkt>200dma, mkt 6m>0) - a gated strategy holds cash when its gate is off,
the long-only expression of a bearish view.

Retrain on trailing 2y every 6 months; deploy winner fortnightly. Parallel.

Usage:
    python walkforward_wide.py --workers 8              (~30-60 min total)
    python walkforward_wide.py --cost 0.001
    python walkforward_wide.py --synthetic --lite --workers 2 --end 2015-12-31
Outputs: wf_wide_log.csv, wf_wide_equity.csv + console report.
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
NS = (4, 8)
UNIVERSES = ("all", "highvol", "lowvol")
TOP_SINGLES = 8
BEAM = 8
DEPTH = 2
CONDS = ["rsi_below_85", "vol_above_med", "vol_below_med", "mom6_positive",
         "above_50dma", "near_52wh", "liq_above_med"]
GATES = ["gate_mkt_200dma", "gate_mkt_6m_pos"]

FAMILY = {}
def _fam(names, fam):
    for n in names.split():
        FAMILY[n] = fam
_fam("mom_3m mom_6m mom_12_1 macd_5_35 macd_12_26 trix_15 kst px_sma50 "
     "px_sma200 ma_cross_50_200 adx_14 dmi_diff dist_52wh", "trend")
_fam("rev_1m rev_6m rsi_oversold stoch_oversold near_52wl bb_lower", "contra")
_fam("lowvol_60 lowvol_20 low_beta low_atr above_200_persist dd_resist updays_1m", "defens")
_fam("sharpe_6m sortino_6m calmar_6m trend_r2", "quality")
_fam("low_turnover obv_slope cmf_20 vol_surge vol_dry pv_corr", "flow")
_fam("anti_lottery skew_neg price_low amihud", "micro")
_fam("seasonal_1m rel_str_6m", "season")
_fam("idio_vol_low vol_compress", "defens2")
_fam("eff_ratio_6m", "quality2")
BASES = list(FAMILY)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
INDEX_CSVS = {"nifty100": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
              "midcap100": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv"}


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


class Engine:
    def __init__(self, close, volume, cost, lite=False):
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
        rsi = 100 - 100 / (1 + ru / rd.replace(0, np.nan))
        mn14, mx14 = close.rolling(14).min(), close.rolling(14).max()
        stoch = (close - mn14) / (mx14 - mn14).replace(0, np.nan)
        sroc = lambda a, b: close.pct_change(a).rolling(b).mean()
        turnover = (close * volume).rolling(60).mean()
        univ_ret = r.mean(axis=1)
        beta = (r.rolling(120).cov(univ_ret)
                / univ_ret.rolling(120).var().replace(0, np.nan))
        downside = r.clip(upper=0)
        sma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
        obv = (np.sign(d) * volume).cumsum()
        clv = (2 * close - mn14 - mx14) / (mx14 - mn14).replace(0, np.nan)
        logc = np.log(close)
        tser = pd.Series(np.arange(len(close.index), dtype=float), index=close.index)
        r2 = logc.rolling(126).corr(tser) ** 2
        roll_max126 = close.rolling(126).max()
        dd126 = (close / roll_max126 - 1).rolling(126).min()

        self.F = {
            # trend
            "mom_3m": close.pct_change(63), "mom_6m": close.pct_change(126),
            "mom_12_1": close.shift(21).pct_change(231),
            "macd_5_35": (ema(close, 5) - ema(close, 35)) / close,
            "macd_12_26": (ema(close, 12) - ema(close, 26)) / close,
            "trix_15": ema(ema(ema(close, 15), 15), 15).pct_change(),
            "kst": 1 * sroc(10, 10) + 2 * sroc(15, 10) + 3 * sroc(20, 10) + 4 * sroc(30, 15),
            "px_sma50": close / close.rolling(50).mean(),
            "px_sma200": close / close.rolling(200).mean(),
            "ma_cross_50_200": close.rolling(50).mean() / close.rolling(200).mean(),
            "adx_14": wema((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)),
            "dmi_diff": pdi - mdi,
            "dist_52wh": close / close.rolling(252).max(),
            # contrarian
            "rev_1m": -close.pct_change(21), "rev_6m": -close.pct_change(126),
            "rsi_oversold": -rsi, "stoch_oversold": -stoch,
            "near_52wl": -(close / close.rolling(252).min()),
            "bb_lower": -((close - (sma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)),
            # defensive
            "lowvol_60": -(r.rolling(60).std() * np.sqrt(252)),
            "lowvol_20": -(r.rolling(20).std() * np.sqrt(252)),
            "low_beta": -beta, "low_atr": -(atr / close),
            "above_200_persist": (close > close.rolling(200).mean()).rolling(60).mean().where(close.notna()),
            "dd_resist": dd126,
            "updays_1m": (r > 0).rolling(21).mean().where(close.notna()),
            # quality
            "sharpe_6m": close.pct_change(126) / (r.rolling(60).std() * np.sqrt(252)).replace(0, np.nan),
            "sortino_6m": close.pct_change(126) / (downside.rolling(126).std() * np.sqrt(252)).replace(0, np.nan),
            "calmar_6m": close.pct_change(126) / dd126.abs().replace(0, np.nan),
            "trend_r2": (r2 * np.sign(close.pct_change(126))),
            # flow / liquidity
            "low_turnover": -np.log(turnover.replace(0, np.nan)),
            "obv_slope": obv.diff(20) / volume.rolling(60).mean().replace(0, np.nan),
            "cmf_20": (clv * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan),
            "vol_surge": (close * volume).rolling(20).mean()
            / (close * volume).rolling(120).mean().replace(0, np.nan),
        }
        # ---- widened pool: microstructure / seasonality / path-quality ----
        univ_c = (1 + univ_ret).cumprod()
        resid = r.sub(beta.mul(univ_ret, axis=0))
        mret = close.resample("ME").last().pct_change()
        seas = mret.groupby(mret.index.month, group_keys=False).apply(
            lambda g: g.shift(1).rolling(3, min_periods=2).mean())
        seas_daily = seas.reindex(close.index, method="ffill")
        self.F.update({
            "anti_lottery": -r.rolling(21).max(),
            "skew_neg": -r.rolling(126).skew(),
            "price_low": -np.log(close.replace(0, np.nan)),
            "amihud": np.log((r.abs() / (close * volume).replace(0, np.nan))
                             .rolling(60).mean().replace(0, np.nan)),
            "seasonal_1m": seas_daily,
            "rel_str_6m": close.pct_change(126).sub(
                univ_c.pct_change(126), axis=0),
            "idio_vol_low": -resid.rolling(120).std(),
            "vol_compress": -(r.rolling(20).std()
                              / r.rolling(120).std().replace(0, np.nan)),
            "eff_ratio_6m": (close.pct_change(126)
                             / r.abs().rolling(126).sum().replace(0, np.nan)),
            "vol_dry": -((close * volume).rolling(20).mean()
                         / (close * volume).rolling(120).mean().replace(0, np.nan)),
            "pv_corr": r.rolling(60).corr(volume.pct_change()),
        })
        if lite:
            keep = ["mom_6m", "kst", "adx_14", "low_turnover", "rev_1m",
                    "lowvol_60", "sharpe_6m", "near_52wl"]
            self.F = {k: v for k, v in self.F.items() if k in keep}
        self.rsi = rsi
        self.aux = {"mom_12m": close.pct_change(252),
                    "vol_60": r.rolling(60).std() * np.sqrt(252),
                    "turnover_60": turnover}
        univ = (1 + univ_ret).cumprod()
        self.gate_ok = {"gate_mkt_200dma": univ > univ.rolling(200).mean(),
                        "gate_mkt_6m_pos": univ.pct_change(126) > 0}
        wk = pd.Series(close.index, index=close.index).groupby(close.index.to_period("W")).last()
        self.fortnights = pd.DatetimeIndex(wk.values)[::2]
        self.mask_cache, self.score_cache = {}, {}

    def cond(self, name, dte):
        if name == "rsi_below_85":
            return self.rsi.loc[dte] < 85
        if name == "vol_above_med":
            v = self.aux["vol_60"].loc[dte]
            return v > v.median()
        if name == "vol_below_med":
            v = self.aux["vol_60"].loc[dte]
            return ~(v > v.median())
        if name == "mom6_positive":
            return self.close.pct_change(126).loc[dte] > 0
        if name == "above_50dma":
            return (self.close.loc[dte] / self.close.rolling(50).mean().loc[dte]) > 1
        if name == "liq_above_med":
            t = self.aux["turnover_60"].loc[dte]
            return t > t.median()
        return (self.close.loc[dte] / self.close.rolling(252).max().loc[dte]) > 0.90

    def eligible(self, dte, universe):
        key = (dte, universe)
        if key in self.mask_cache:
            return self.mask_cache[key]
        ok = self.aux["mom_12m"].loc[dte].notna()
        liq = self.aux["turnover_60"].loc[dte]
        ok &= liq.notna() & (liq >= MIN_TURNOVER)
        if universe != "all":
            v = self.aux["vol_60"].loc[dte].where(ok)
            med = v.median()
            ok &= (v > med) if universe == "highvol" else (v <= med)
        self.mask_cache[key] = ok
        return ok

    def weights(self, spec, dates):
        blend, universe, n, conds, gates = spec
        out = {}
        for dte in dates:
            if any(not bool(self.gate_ok[g].loc[dte]) for g in gates):
                out[dte] = pd.Series(dtype=float)
                continue
            key = (blend, universe, conds, dte)
            if key not in self.score_cache:
                ok = self.eligible(dte, universe)
                for c in conds:
                    ok = ok & self.cond(c, dte).reindex(ok.index).fillna(False)
                score = None
                for b in blend:
                    z = self.F[b].loc[dte].where(ok)
                    mu, sd = z.mean(), z.std()
                    z = (z - mu) / sd if sd and np.isfinite(sd) and sd > 0 else z * 0.0
                    score = z if score is None else score + z
                self.score_cache[key] = score.dropna().sort_values(ascending=False)
            s = self.score_cache[key]
            out[dte] = (pd.Series(1.0 / n, index=s.head(n).index)
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
        ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
        r = eq.pct_change().dropna()
        sharpe = float(r.mean() / (r.std() + 1e-12) * np.sqrt(252))
        mdd = float((eq / eq.cummax() - 1).min())
        calmar = ret / max(0.02, -mdd)
        return ret, sharpe, calmar


_ENG = None


def _init_worker(synthetic, cost, lite):
    global _ENG
    if synthetic:
        close, volume = make_synthetic()
    else:
        close = pd.read_parquet(os.path.join(DATA, "close.parquet"))
        volume = pd.read_parquet(os.path.join(DATA, "volume.parquet"))
    _ENG = Engine(close, volume, cost, lite)


def _eval(job):
    spec, t0, t1 = job
    return spec, _ENG.train_ret(spec, pd.Timestamp(t0), pd.Timestamp(t1))


CRIT_IX = {"return": 0, "sharpe": 1, "calmar": 2}


def run_stage(pool, specs, t0, t1, scores):
    todo = [s for s in specs if s not in scores]
    for spec, ret in pool.map(_eval, [(s, str(t0), str(t1)) for s in todo], chunksize=6):
        scores[spec] = ret
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--criterion", default="return",
                    choices=["return", "sharpe", "calmar"],
                    help="how each window's winner is crowned")
    ap.add_argument("--lite", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    close, volume = (make_synthetic() if args.synthetic else get_data())
    eng = Engine(close, volume, args.cost, args.lite)
    bases = list(eng.F)
    print(f"panel: {close.shape[1]} tickers | {len(bases)} bases across "
          f"{len(set(FAMILY[b] for b in bases))} style families | cost/leg {args.cost}")

    windows = []
    t = pd.Timestamp(args.start)
    end_all = pd.Timestamp(args.end)
    while t < end_all:
        te = min(t + pd.DateOffset(months=6), end_all + pd.Timedelta(days=1))
        windows.append((t - pd.DateOffset(years=TRAIN_YEARS), t, te))
        t = t + pd.DateOffset(months=6)

    cix = CRIT_IX[args.criterion]
    key = lambda sp: scores[sp][cix]
    print(f"selection criterion: trained {args.criterion}")
    log, deploy_wbd = [], {}
    t0_all = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.synthetic, args.cost, args.lite)) as pool:
        for tr0, dep0, dep1 in windows:
            scores = {}
            singles = [((b,), u, n, (), ()) for b in bases
                       for u in UNIVERSES for n in NS]
            run_stage(pool, singles, tr0, dep0, scores)
            ranked = sorted(scores, key=key, reverse=True)
            seen, tops = [], []
            for sp in ranked:
                b = sp[0][0]
                if b not in seen:
                    seen.append(b)
                    tops.append(sp)
                if len(tops) == TOP_SINGLES:
                    break
            pairs = []
            for s1, s2 in itertools.combinations(tops, 2):
                a, b = s1[0][0], s2[0][0]
                if FAMILY[a] == FAMILY[b]:
                    continue
                for u in UNIVERSES:
                    for n in NS:
                        pairs.append((tuple(sorted((a, b))), u, n, (), ()))
            run_stage(pool, pairs, tr0, dep0, scores)
            beam = sorted(scores, key=key, reverse=True)[:BEAM]
            for _depth in range(DEPTH):
                children = []
                for p in beam:
                    for c in CONDS:
                        if c not in p[3]:
                            children.append((p[0], p[1], p[2],
                                             tuple(sorted(p[3] + (c,))), p[4]))
                    for g in GATES:
                        if g not in p[4]:
                            children.append((p[0], p[1], p[2], p[3],
                                             tuple(sorted(p[4] + (g,)))))
                run_stage(pool, children, tr0, dep0, scores)
                beam = sorted(scores, key=key, reverse=True)[:BEAM]
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
            if winner[4]:
                name += "|" + "+".join(winner[4])
            fam = "/".join(sorted({FAMILY[b] for b in winner[0]}))
            log.append({"deploy_from": dep0.date(), "style": fam, "winner": name,
                        "trained_2y_%": round(100 * scores[winner][0], 1),
                        "trained_crit": round(scores[winner][cix], 2),
                        "deployed_6m_%": round(100 * dep_ret, 1),
                        "deployed_sharpe_ann": round(float(rd.mean() / (rd.std() + 1e-12) * np.sqrt(252)), 2)})
            print(f"  {dep0.date()} [{fam}] {name}  train {scores[winner][0]*100:+.0f}%  "
                  f"deployed {dep_ret*100:+.1f}%  ({time.time()-t0_all:.0f}s, "
                  f"{len(scores)} cands)")

    eq = eng.backtest(deploy_wbd, pd.Timestamp(args.start), end_all)
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    L = pd.DataFrame(log)
    L["networth_cr"] = (1 + L["deployed_6m_%"] / 100).cumprod().round(2)

    bench = {}
    if not args.synthetic:
        b1 = get_bench_index("bench_n100.parquet", ["^CNX100"])
        if b1 is not None:
            bench["N100_idx"] = b1
        b2 = get_bench_index("bench_mid.parquet",
                             ["NIFTY_MIDCAP_100.NS", "^CNXMIDCAP", "NIFTYMIDCAP100.NS"])
        if b2 is not None:
            bench["Mid_idx"] = b2
        for lab, fn in [("N100_eqw", "nifty100.csv"), ("Mid_eqw", "midcap100.csv")]:
            df = pd.read_csv(os.path.join(DATA, fn))
            sym = "Symbol" if "Symbol" in df.columns else df.columns[2]
            cols = [t for t in (df[sym].astype(str).str.strip() + ".NS") if t in close.columns]
            bench[lab] = (1 + close[cols].pct_change().mean(axis=1)).cumprod().dropna()

    print("\n=== WIDE-STYLE WALK-FORWARD (every window out-of-sample) ===")
    print(f"net worth : Rs {eq.iloc[-1]/1e7:.1f} cr from Rs 1 cr | "
          f"CAGR {100*((eq.iloc[-1]/CAPITAL)**(1/yrs)-1):.1f}% | "
          f"Sharpe {float(r.mean()/(r.std()+1e-12)*np.sqrt(252)):.2f} | "
          f"MDD {100*float((eq/eq.cummax()-1).min()):.1f}%")
    print(f"windows positive: {int((L['deployed_6m_%']>0).sum())}/{len(L)} | "
          f"avg deployed 6m {L['deployed_6m_%'].mean():+.1f}% | "
          f"style mix: {dict(L['style'].value_counts())}")
    print("\nyearly (%): WF | " + " | ".join(bench.keys()))
    for y in list(range(int(args.start[:4]), 2026)) + ["2026H1"]:
        s = pd.Timestamp(f"{y}-01-01") if y != "2026H1" else pd.Timestamp("2026-01-01")
        e = pd.Timestamp(f"{y}-12-31") if y != "2026H1" else pd.Timestamp("2026-06-30")
        w = eq[(eq.index >= s) & (eq.index <= e)]
        if len(w) < 10:
            continue
        line = f"  {y}: {100*(w.iloc[-1]/w.iloc[0]-1):+7.1f}"
        for bser in bench.values():
            bw = bser[(bser.index >= s) & (bser.index <= e)].dropna()
            line += f" | {100*(bw.iloc[-1]/bw.iloc[0]-1):+6.1f}" if len(bw) > 10 else " |    n/a"
        print(line)
    print("\n=== PER-WINDOW LOG ===")
    with pd.option_context("display.width", 240):
        print(L.to_string(index=False))
    L.to_csv(f"wf_wide_log_{args.criterion}.csv", index=False)
    eq.to_csv(f"wf_wide_equity_{args.criterion}.csv")
    print(f"\nsaved: wf_wide_log_{args.criterion}.csv, wf_wide_equity_{args.criterion}.csv")


if __name__ == "__main__":
    main()
