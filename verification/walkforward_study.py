"""
WALK-FORWARD STRATEGY STUDY - single script, two runs.

Run A ("current"):  universe = today's NIFTY100 + Midcap100 lists applied
                    across the whole period; benchmark = equal-weight basket
                    of the same lists (universe-matched comparison).
Run B ("pit"):      universe = approximate point-in-time membership rebuilt
                    from archived NSE constituent files; benchmark = equal-
                    weight basket of the reconstructed universe.

Both runs share the identical system: every 6 months a tree search over
~500 candidate strategies is trained on the trailing 2 years; the winner
trades a 4- or 8-stock equal-weight portfolio fortnightly for 6 months.

Usage:
    python walkforward_study.py --workers 8            # both runs
    python walkforward_study.py --mode current
    python walkforward_study.py --mode pit
    python walkforward_study.py --synthetic --workers 2 --end 2015-07-01
"""
import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# ============================================================
# BLOCK 1 - PARAMETERS
# ============================================================
CAPITAL = 1e7                    # Rs 1 crore starting capital
COST = 0.001                     # 0.1% per buy/sell leg
TRAIN_YEARS = 2                  # training lookback per retraining
START_DEPLOY = "2015-01-01"      # first deployment window
END_DEPLOY = "2026-06-30"        # end of evaluation
NS = (4, 8)                      # portfolio sizes searched
UNIVERSES = ("all", "highvol", "lowvol")   # volatility carve searched
MIN_TURNOVER = 1e7               # Rs 1 cr/day 60d avg traded value floor
TOP_SINGLES = 8                  # singles promoted to the pair stage
BEAM = 8                         # beam width in the condition tree
DEPTH = 2                        # max conditions/gates added
DL_START, DL_END = "2013-01-01", "2026-07-01"   # 2013-14 = signal warm-up
DATA = "data_walkforward"        # price/list cache (Run A)
PITDATA = "data_pit_wf"          # PIT panel + membership cache (Run B)
SNAPDIR = os.path.join(PITDATA, "snapshots")

# ============================================================
# BLOCK 2 - DATA SOURCES
#   current lists : niftyindices.com constituent CSVs
#   prices/volume : Yahoo Finance daily adjusted, via yfinance
#   archived lists: Wayback Machine captures of the NSE CSVs
# ============================================================
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
INDEX_CSVS = {"nifty100": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
              "midcap100": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv"}
WAYBACK_SOURCES = {
    "nifty100": ["niftyindices.com/IndexConstituent/ind_nifty100list.csv",
                 "www1.nseindia.com/content/indices/ind_nifty100list.csv",
                 "www.nseindia.com/content/indices/ind_nifty100list.csv"],
    "midcap100": ["niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
                  "www1.nseindia.com/content/indices/ind_niftymidcap100list.csv",
                  "www.nseindia.com/content/indices/ind_niftymidcap100list.csv"]}
ALIAS = {"MCDOWELL-N": "UNITDSPR", "SRTRANSFIN": "SHRIRAMFIN",
         "CADILAHC": "ZYDUSLIFE", "MOTHERSUMI": "MOTHERSON",
         "TATAGLOBAL": "TATACONSUM", "L&TFH": "LTF",
         "MINDTREE": "LTIM", "LTI": "LTIM", "INFRATEL": "INDUSTOWER"}


def make_synthetic():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range(DL_START, END_DEPLOY)
    n, k = len(dates), 60
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(3e-4, 0.018, (n, k)), axis=0)),
                         index=dates, columns=[f"SYN{i:03d}.NS" for i in range(k)])
    vol = pd.DataFrame(rng.integers(2e5, 5e6, (n, k)).astype(float),
                       index=dates, columns=close.columns)
    return close, vol


def download_prices(tickers, cache_dir):
    import yfinance as yf
    os.makedirs(cache_dir, exist_ok=True)
    pc = os.path.join(cache_dir, "close.parquet")
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


def current_lists():
    import requests
    os.makedirs(DATA, exist_ok=True)
    lists = {}
    for name, url in INDEX_CSVS.items():
        local = os.path.join(DATA, f"{name}.csv")
        if not os.path.exists(local):
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            open(local, "w").write(r.text)
        df = pd.read_csv(local)
        sym = "Symbol" if "Symbol" in df.columns else df.columns[2]
        lists[name] = sorted(set(df[sym].astype(str).str.strip() + ".NS"))
    return lists


# ============================================================
# BLOCK 2b - HISTORICAL MEMBERSHIP (Run B only)
#   Wayback captures of the constituent CSVs -> dated symbol sets;
#   symbol renames mapped through ALIAS; at any date the universe is
#   the UNION of each index's most recent archived list.
# ============================================================
def fetch_snapshots():
    import requests
    os.makedirs(SNAPDIR, exist_ok=True)
    out = {}
    for name, urls in WAYBACK_SOURCES.items():
        snaps = {}
        marker = os.path.join(SNAPDIR, f"{name}.fetched2")
        if not os.path.exists(marker):
            for url in urls:
                cdx = ("http://web.archive.org/cdx/search/cdx?url=" + url +
                       "&output=json&fl=timestamp&filter=statuscode:200"
                       "&collapse=timestamp:6")
                try:
                    rows = requests.get(cdx, timeout=60).json()[1:]
                except Exception:
                    rows = []
                for (ts,) in rows:
                    fp = os.path.join(SNAPDIR, f"{name}_{ts}.csv")
                    if os.path.exists(fp):
                        continue
                    try:
                        r = requests.get(f"https://web.archive.org/web/{ts}id_/https://{url}",
                                         timeout=60)
                        if r.status_code == 200 and "Symbol" in r.text[:3000]:
                            open(fp, "w", encoding="utf-8").write(r.text)
                    except Exception:
                        pass
                    time.sleep(0.6)
            open(marker, "w").write("done")
        for f in sorted(os.listdir(SNAPDIR)):
            if not (f.startswith(name + "_") and f.endswith(".csv")):
                continue
            ts = f[len(name) + 1:].split(".")[0]
            try:
                df = pd.read_csv(os.path.join(SNAPDIR, f))
                sym = "Symbol" if "Symbol" in df.columns else df.columns[2]
                tick = set(ALIAS.get(s, s) + ".NS"
                           for s in df[sym].astype(str).str.strip())
                if len(tick) > 50:
                    snaps[pd.Timestamp(ts[:8])] = tick
            except Exception:
                pass
        print(f"  {name}: {len(snaps)} archived lists "
              f"({min(snaps).date() if snaps else '-'} -> {max(snaps).date() if snaps else '-'})")
        out[name] = snaps
    return out


def build_membership(index, columns, snaps_by_idx):
    events = sorted((ts, name, tick) for name, s in snaps_by_idx.items()
                    for ts, tick in s.items())
    if not events:
        raise SystemExit("no archived lists found")
    state = {name: s[min(s)] for name, s in snaps_by_idx.items() if s}
    M = pd.DataFrame(False, index=index, columns=columns)
    start, first_snap = index[0], events[0][0]
    for ts, name, tick in events:
        seg = index[(index >= start) & (index < ts)]
        if len(seg):
            cur = set().union(*state.values())
            M.loc[seg, [c for c in columns if c in cur]] = True
        state[name], start = tick, ts
    seg = index[index >= start]
    cur = set().union(*state.values())
    M.loc[seg, [c for c in columns if c in cur]] = True
    return M, first_snap


# ============================================================
# BLOCK 3 - SIGNAL LIBRARY: 45 signals, 9 style families
#   All computed from daily adjusted close + volume only.
#   Higher score = more attractive (inversions applied via minus signs).
# ============================================================
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

CONDS = ["rsi_below_85", "vol_above_med", "vol_below_med", "mom6_positive",
         "above_50dma", "near_52wh", "liq_above_med"]
GATES = ["gate_mkt_200dma", "gate_mkt_6m_pos"]


def build_signals(close, volume):
    ema = lambda s, n: s.ewm(span=n, adjust=False).mean()
    wema = lambda s: s.ewm(alpha=1 / 14, adjust=False).mean()
    r = close.pct_change()
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
    univ_c = (1 + univ_ret).cumprod()
    beta = r.rolling(120).cov(univ_ret) / univ_ret.rolling(120).var().replace(0, np.nan)
    resid = r.sub(beta.mul(univ_ret, axis=0))
    downside = r.clip(upper=0)
    sma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
    obv = (np.sign(d) * volume).cumsum()
    clv = (2 * close - mn14 - mx14) / (mx14 - mn14).replace(0, np.nan)
    tser = pd.Series(np.arange(len(close.index), dtype=float), index=close.index)
    r2 = np.log(close).rolling(126).corr(tser) ** 2
    dd126 = (close / close.rolling(126).max() - 1).rolling(126).min()
    mret = close.resample("ME").last().pct_change()
    seas = mret.groupby(mret.index.month, group_keys=False).apply(
        lambda g: g.shift(1).rolling(3, min_periods=2).mean())

    F = {}
    # --- TREND: winners keep winning; strength and persistence of moves
    F["mom_3m"] = close.pct_change(63)
    F["mom_6m"] = close.pct_change(126)
    F["mom_12_1"] = close.shift(21).pct_change(231)
    F["macd_5_35"] = (ema(close, 5) - ema(close, 35)) / close
    F["macd_12_26"] = (ema(close, 12) - ema(close, 26)) / close
    F["trix_15"] = ema(ema(ema(close, 15), 15), 15).pct_change()
    F["kst"] = 1 * sroc(10, 10) + 2 * sroc(15, 10) + 3 * sroc(20, 10) + 4 * sroc(30, 15)
    F["px_sma50"] = close / close.rolling(50).mean()
    F["px_sma200"] = close / close.rolling(200).mean()
    F["ma_cross_50_200"] = close.rolling(50).mean() / close.rolling(200).mean()
    F["adx_14"] = wema((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan))
    F["dmi_diff"] = pdi - mdi
    F["dist_52wh"] = close / close.rolling(252).max()
    # --- CONTRARIAN: buy recent losers / oversold names
    F["rev_1m"] = -close.pct_change(21)
    F["rev_6m"] = -close.pct_change(126)
    F["rsi_oversold"] = -rsi
    F["stoch_oversold"] = -stoch
    F["near_52wl"] = -(close / close.rolling(252).min())
    F["bb_lower"] = -((close - (sma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan))
    # --- DEFENSIVE: low-risk, stable, crash-resistant names
    F["lowvol_60"] = -(r.rolling(60).std() * np.sqrt(252))
    F["lowvol_20"] = -(r.rolling(20).std() * np.sqrt(252))
    F["low_beta"] = -beta
    F["low_atr"] = -(atr / close)
    F["above_200_persist"] = (close > close.rolling(200).mean()).rolling(60).mean().where(close.notna())
    F["dd_resist"] = dd126
    F["updays_1m"] = (r > 0).rolling(21).mean().where(close.notna())
    # --- QUALITY: risk-adjusted quality of the recent path
    F["sharpe_6m"] = close.pct_change(126) / (r.rolling(60).std() * np.sqrt(252)).replace(0, np.nan)
    F["sortino_6m"] = close.pct_change(126) / (downside.rolling(126).std() * np.sqrt(252)).replace(0, np.nan)
    F["calmar_6m"] = close.pct_change(126) / dd126.abs().replace(0, np.nan)
    F["trend_r2"] = r2 * np.sign(close.pct_change(126))
    # --- FLOW / LIQUIDITY: volume, turnover, accumulation
    F["low_turnover"] = -np.log(turnover.replace(0, np.nan))
    F["obv_slope"] = obv.diff(20) / volume.rolling(60).mean().replace(0, np.nan)
    F["cmf_20"] = (clv * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    F["vol_surge"] = (close * volume).rolling(20).mean() / (close * volume).rolling(120).mean().replace(0, np.nan)
    F["vol_dry"] = -F["vol_surge"]
    F["pv_corr"] = r.rolling(60).corr(volume.pct_change())
    # --- MICROSTRUCTURE anomalies
    F["anti_lottery"] = -r.rolling(21).max()
    F["skew_neg"] = -r.rolling(126).skew()
    F["price_low"] = -np.log(close.replace(0, np.nan))
    F["amihud"] = np.log((r.abs() / (close * volume).replace(0, np.nan)).rolling(60).mean().replace(0, np.nan))
    # --- SEASONALITY / relative strength
    F["seasonal_1m"] = seas.reindex(close.index, method="ffill")
    F["rel_str_6m"] = close.pct_change(126).sub(univ_c.pct_change(126), axis=0)
    # --- DEFENSIVE-II / PATH QUALITY
    F["idio_vol_low"] = -resid.rolling(120).std()
    F["vol_compress"] = -(r.rolling(20).std() / r.rolling(120).std().replace(0, np.nan))
    F["eff_ratio_6m"] = close.pct_change(126) / r.abs().rolling(126).sum().replace(0, np.nan)

    aux = {"mom_12m": close.pct_change(252),
           "vol_60": r.rolling(60).std() * np.sqrt(252),
           "turnover_60": turnover, "rsi": rsi}
    gates = {"gate_mkt_200dma": univ_c > univ_c.rolling(200).mean(),
             "gate_mkt_6m_pos": univ_c.pct_change(126) > 0}
    return F, aux, gates


# ============================================================
# BLOCK 4 - PORTFOLIO ENGINE
#   candidate = (blend, universe, N, conditions, gates)
#   fortnightly ranking -> top-N equal weight -> costed backtest
# ============================================================
class Engine:
    def __init__(self, close, volume, member=None):
        self.close, self.member = close, member
        self.rets = close.pct_change()
        self.F, self.aux, self.gate_ok = build_signals(close, volume)
        wk = pd.Series(close.index, index=close.index).groupby(close.index.to_period("W")).last()
        self.fortnights = pd.DatetimeIndex(wk.values)[::2]
        self.mask_cache, self.score_cache = {}, {}

    def cond(self, name, dte):
        if name == "rsi_below_85":
            return self.aux["rsi"].loc[dte] < 85
        if name in ("vol_above_med", "vol_below_med"):
            v = self.aux["vol_60"].loc[dte]
            return (v > v.median()) if name == "vol_above_med" else ~(v > v.median())
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
        ok = self.aux["mom_12m"].loc[dte].notna()          # 12m history
        if self.member is not None:                        # PIT membership
            ok &= self.member.loc[dte].reindex(ok.index).fillna(False)
        liq = self.aux["turnover_60"].loc[dte]
        ok &= liq.notna() & (liq >= MIN_TURNOVER)          # liquidity floor
        if universe != "all":                              # volatility carve
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
                out[dte] = pd.Series(dtype=float)          # gate off -> cash
                continue
            key = (blend, universe, conds, dte)
            if key not in self.score_cache:
                ok = self.eligible(dte, universe)
                for c in conds:
                    ok = ok & self.cond(c, dte).reindex(ok.index).fillna(False)
                score = None
                for b in blend:                            # z-score blending
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
        dates = self.close.index[(self.close.index >= start) & (self.close.index <= end)]
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
                total -= COST * (tgt - cur).abs().sum()
                tgt = (w * total).reindex(allt).fillna(0.0)
                pos = tgt[tgt > 1e-9]
                cash = total - pos.sum()
            eq[k] = cash + pos.sum()
        return pd.Series(eq, index=dates)

    def train_ret(self, spec, t0, t1):
        dates = self.fortnights[(self.fortnights >= t0) & (self.fortnights < t1)]
        eq = self.backtest(self.weights(spec, dates), t0, t1)
        return float(eq.iloc[-1] / eq.iloc[0] - 1)


# ============================================================
# BLOCK 5 - PARALLEL WORKERS
# ============================================================
_ENG = None


def _init_worker(mode, synthetic):
    global _ENG
    if synthetic:
        close, volume = make_synthetic()
        member = _synthetic_membership(close) if mode == "pit" else None
    elif mode == "pit":
        close = pd.read_parquet(os.path.join(PITDATA, "close.parquet"))
        volume = pd.read_parquet(os.path.join(PITDATA, "volume.parquet"))
        member = pd.read_parquet(os.path.join(PITDATA, "membership.parquet"))
    else:
        close = pd.read_parquet(os.path.join(DATA, "close.parquet"))
        volume = pd.read_parquet(os.path.join(DATA, "volume.parquet"))
        member = None
    _ENG = Engine(close, volume, member)


def _synthetic_membership(close):
    m = pd.DataFrame(True, index=close.index, columns=close.columns)
    for y in range(2014, 2027):
        off = [c for i, c in enumerate(close.columns) if (i + y) % 5 == 0]
        m.loc[str(y), off] = False
    return m


def _eval(job):
    spec, t0, t1 = job
    return spec, _ENG.train_ret(spec, pd.Timestamp(t0), pd.Timestamp(t1))


def run_stage(pool, specs, t0, t1, scores):
    todo = [s for s in specs if s not in scores]
    for spec, ret in pool.map(_eval, [(s, str(t0), str(t1)) for s in todo], chunksize=6):
        scores[spec] = ret
    return scores


# ============================================================
# BLOCK 6 - TREE SEARCH (per retraining)
#   stage 1: all singles (45 x 3 universes x 2 sizes = 270)
#   stage 2: cross-family pairs of the top 8 singles (~168)
#   stage 3: beam of 8, expanded twice with one condition/gate (~140)
#   winner : highest total return on the training window
# ============================================================
def search_window(pool, tr0, dep0):
    scores = {}
    singles = [((b,), u, n, (), ()) for b in FAMILY for u in UNIVERSES for n in NS]
    run_stage(pool, singles, tr0, dep0, scores)
    ranked = sorted(scores, key=scores.get, reverse=True)
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
    beam = sorted(scores, key=scores.get, reverse=True)[:BEAM]
    for _ in range(DEPTH):
        children = []
        for p_ in beam:
            for c in CONDS:
                if c not in p_[3]:
                    children.append((p_[0], p_[1], p_[2], tuple(sorted(p_[3] + (c,))), p_[4]))
            for g in GATES:
                if g not in p_[4]:
                    children.append((p_[0], p_[1], p_[2], p_[3], tuple(sorted(p_[4] + (g,)))))
        run_stage(pool, children, tr0, dep0, scores)
        beam = sorted(scores, key=scores.get, reverse=True)[:BEAM]
    return beam[0], scores


# ============================================================
# BLOCK 7 - WALK-FORWARD LOOP + REPORTING
# ============================================================
def run_config(mode, args, close, volume, member):
    eng = Engine(close, volume, member)
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
                             initargs=(mode, args.synthetic)) as pool:
        for tr0, dep0, dep1 in windows:
            winner, scores = search_window(pool, tr0, dep0)
            dep_dates = eng.fortnights[(eng.fortnights >= dep0) & (eng.fortnights < dep1)]
            wbd = eng.weights(winner, dep_dates)
            deploy_wbd.update(wbd)
            eq_d = eng.backtest(wbd, dep0, dep1 - pd.Timedelta(days=1))
            dep_ret = float(eq_d.iloc[-1] / eq_d.iloc[0] - 1)
            name = "&".join(winner[0]) + f"_{winner[1]}_n{winner[2]}"
            if winner[3]:
                name += "|" + "+".join(winner[3])
            if winner[4]:
                name += "|" + "+".join(winner[4])
            log.append({"deploy_from": dep0.date(),
                        "style": "/".join(sorted({FAMILY[b] for b in winner[0]})),
                        "winner": name,
                        "trained_2y_%": round(100 * scores[winner], 1),
                        "deployed_6m_%": round(100 * dep_ret, 1)})
            print(f"  [{mode}] {dep0.date()} {name}  train {scores[winner]*100:+.0f}%  "
                  f"deployed {dep_ret*100:+.1f}%  ({time.time()-t0_all:.0f}s)")

    eq = eng.backtest(deploy_wbd, pd.Timestamp(args.start), end_all)
    L = pd.DataFrame(log)
    L.to_csv(f"study_log_{mode}.csv", index=False)
    eq.to_csv(f"study_equity_{mode}.csv")
    return eq, L


def report(mode, eq, L, member, close):
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if member is not None:
        m = member.shift(1).fillna(False).astype(float)
        basket = (1 + ((close.pct_change() * m).sum(axis=1)
                       / m.sum(axis=1).replace(0, np.nan)).fillna(0)).cumprod()
        blab = "eq-wt basket (PIT universe)"
    else:
        basket = (1 + close.pct_change().mean(axis=1)).cumprod()
        blab = "eq-wt basket (today's lists)"
    print(f"\n=== [{mode}] RESULT ===")
    print(f"net worth {eq.iloc[-1]/1e7:.1f} cr | CAGR {100*((eq.iloc[-1]/CAPITAL)**(1/yrs)-1):.1f}% | "
          f"Sharpe {float(r.mean()/(r.std()+1e-12)*np.sqrt(252)):.2f} | "
          f"MDD {100*float((eq/eq.cummax()-1).min()):.1f}% | "
          f"windows+ {int((L['deployed_6m_%']>0).sum())}/{len(L)}")
    print(f"yearly (%): strategy | {blab}")
    for y in list(range(int(eq.index[0].year), 2026)) + ["2026H1"]:
        s = pd.Timestamp(f"{y}-01-01") if y != "2026H1" else pd.Timestamp("2026-01-01")
        e = pd.Timestamp(f"{y}-12-31") if y != "2026H1" else pd.Timestamp("2026-06-30")
        w = eq[(eq.index >= s) & (eq.index <= e)]
        bw = basket[(basket.index >= s) & (basket.index <= e)].dropna()
        if len(w) > 10:
            print(f"  {y}: {100*(w.iloc[-1]/w.iloc[0]-1):+7.1f} | "
                  f"{100*(bw.iloc[-1]/bw.iloc[0]-1):+6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["both", "current", "pit"])
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--start", default=START_DEPLOY)
    ap.add_argument("--end", default=END_DEPLOY)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    modes = ["current", "pit"] if args.mode == "both" else [args.mode]
    for mode in modes:
        if args.synthetic:
            close, volume = make_synthetic()
            member = _synthetic_membership(close) if mode == "pit" else None
        else:
            lists = current_lists()
            tickers = sorted(set(lists["nifty100"]) | set(lists["midcap100"]))
            close, volume = download_prices(tickers, DATA)
            member = None
            if mode == "pit":
                snaps = fetch_snapshots()
                if os.path.exists(os.path.join(PITDATA, "close.parquet")):
                    close = pd.read_parquet(os.path.join(PITDATA, "close.parquet"))
                    volume = pd.read_parquet(os.path.join(PITDATA, "volume.parquet"))
                else:
                    import yfinance as yf
                    os.makedirs(PITDATA, exist_ok=True)
                    allmem = set().union(*(t for s in snaps.values() for t in s.values()))
                    extra = sorted(allmem - set(close.columns))
                    print(f"  former members to download: {len(extra)}")
                    for t in extra:
                        try:
                            dd = yf.download(t, start=DL_START, end=DL_END,
                                             auto_adjust=True, progress=False)
                            if len(dd) > 250:
                                c = dd["Close"].squeeze(); c.index = pd.to_datetime(c.index).tz_localize(None)
                                v = dd["Volume"].squeeze(); v.index = c.index
                                close[t] = c.reindex(close.index)
                                volume[t] = v.reindex(volume.index)
                        except Exception:
                            pass
                        time.sleep(1.2)
                    close.to_parquet(os.path.join(PITDATA, "close.parquet"))
                    volume.to_parquet(os.path.join(PITDATA, "volume.parquet"))
                v2 = os.path.join(PITDATA, "membership_v2.parquet")
                if os.path.exists(v2):
                    member = (pd.read_parquet(v2)
                              .reindex(index=close.index, columns=close.columns)
                              .fillna(False))
                    member.to_parquet(os.path.join(PITDATA, "membership.parquet"))
                    print(f"  membership: EVENT-BASED (membership_v2) "
                          f"avg {int(member.sum(axis=1).mean())} names/day")
                else:
                    member, first_snap = build_membership(close.index, close.columns, snaps)
                    member.to_parquet(os.path.join(PITDATA, "membership.parquet"))
                    print(f"  membership: avg {int(member.sum(axis=1).mean())} names/day; "
                          f"archive-based from {first_snap.date()}")
        print(f"\n########## RUN [{mode}]: {close.shape[1]} tickers ##########")
        eq, L = run_config(mode, args, close, volume, member)
        report(mode, eq, L, member, close)


if __name__ == "__main__":
    main()