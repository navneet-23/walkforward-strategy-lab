"""
POINT-IN-TIME UNIVERSE BACKTEST - the survivorship-bias test.

Reconstructs index membership over 2020-2026 from Wayback Machine snapshots
of the niftyindices constituent CSVs, then runs the IDENTICAL strategy
(imported from standalone_final.py) at N=7 where, at every rebalance, a
stock is eligible ONLY if it was in the NIFTY100/Midcap100/Smallcap100 lists
AS THEY STOOD ON THAT DATE. Stocks demoted from the indices since 2021 are
downloaded and included for the periods they were members.

Honest limits, printed at runtime:
  - snapshot granularity: membership changes apply from the next archived
    snapshot (typically quarterly-ish coverage), not the exact NSE
    reconstitution date;
  - names that DELISTED entirely may fail to download - their absence still
    flatters results slightly;
  - dates before the first snapshot use the first snapshot's lists.

Usage:  python standalone_pit.py            (first run: wayback queries +
                                             extra downloads, ~10-20 min;
                                             cached under data_standalone/pit/)
        python standalone_pit.py --synthetic
"""
import argparse
import io
import json
import os
import time

import numpy as np
import pandas as pd

import standalone_final as sf

PIT = os.path.join(sf.DATA, "pit")
N_PIT = 7
CDX = "http://web.archive.org/cdx/search/cdx"
WB = "https://web.archive.org/web/{ts}id_/{url}"
YEARS = ["2021", "2022", "2023", "2024", "2025", "2026H1"]


# ------------------------------------------------------------ wayback layer
def fetch_snapshots():
    """returns {index_name: {snapshot_date(Timestamp): set(tickers)}}"""
    import requests
    os.makedirs(PIT, exist_ok=True)
    cache = os.path.join(PIT, "membership.json")
    if os.path.exists(cache):
        raw = json.load(open(cache))
        return {k: {pd.Timestamp(d): set(v) for d, v in m.items()}
                for k, m in raw.items()}
    out = {}
    for name, url in sf.INDEX_CSVS.items():
        print(f"  querying wayback for {name} ...")
        try:
            r = requests.get(CDX, params={
                "url": url.replace("https://", ""), "output": "json",
                "from": "2020", "to": "2026", "filter": "statuscode:200",
                "collapse": "timestamp:6"}, timeout=60)
            rows = r.json()[1:]
        except Exception as e:
            print(f"    CDX failed ({e}) - no snapshots for {name}")
            rows = []
        snaps = {}
        for row in rows:
            ts = row[1]
            try:
                rr = requests.get(WB.format(ts=ts, url=url), timeout=60)
                df = pd.read_csv(io.StringIO(rr.text))
                sym = "Symbol" if "Symbol" in df.columns else None
                if sym is None or len(df) < 50:
                    continue
                snaps[str(pd.Timestamp(ts[:8]))] = sorted(
                    df[sym].astype(str).str.strip() + ".NS")
                print(f"    {ts[:8]}: {len(df)} constituents")
            except Exception:
                continue
            time.sleep(1)
        out[name] = snaps
    json.dump(out, open(cache, "w"))
    return {k: {pd.Timestamp(d): set(v) for d, v in m.items()} for k, m in out.items()}


def synthetic_membership(close):
    cols = list(close.columns)
    half = len(cols) * 2 // 3
    snaps = {pd.Timestamp("2020-06-01"): set(cols[:half]),
             pd.Timestamp("2023-06-01"): set(cols[len(cols) - half:])}
    return {"all3": snaps}


def membership_at(snapshots, d):
    """union of the latest snapshot of each index at date d"""
    live = set()
    for _name, snaps in snapshots.items():
        dates = sorted(snaps)
        if not dates:
            continue
        past = [x for x in dates if x <= d]
        key = past[-1] if past else dates[0]
        live |= snaps[key]
    return live


# ------------------------------------------------------------ extra tickers
def extend_panel(close, volume, snapshots):
    import yfinance as yf
    all_members = set()
    for snaps in snapshots.values():
        for s in snaps.values():
            all_members |= s
    extra = sorted(t for t in all_members if t not in close.columns)
    print(f"  {len(all_members)} distinct historical members; "
          f"{len(extra)} not in current panel (demoted/delisted names)")
    pc = os.path.join(PIT, "extra_close.parquet")
    if os.path.exists(pc):
        ec = pd.read_parquet(pc)
        ev = pd.read_parquet(pc.replace("close", "volume"))
    else:
        cs, vs, failed = [], [], []
        for t in extra:
            try:
                d = yf.download(t, start=sf.DL_START, end=sf.DL_END,
                                auto_adjust=True, progress=False)
                if len(d) > 250:
                    c = d["Close"].squeeze()
                    c.index = pd.to_datetime(c.index).tz_localize(None)
                    v = d["Volume"].squeeze()
                    v.index = c.index
                    cs.append(c.rename(t))
                    vs.append(v.rename(t))
                else:
                    failed.append(t)
            except Exception:
                failed.append(t)
            time.sleep(1.5)
        ec = pd.concat(cs, axis=1) if cs else pd.DataFrame()
        ev = pd.concat(vs, axis=1) if vs else pd.DataFrame()
        ec.to_parquet(pc)
        ev.to_parquet(pc.replace("close", "volume"))
        print(f"  downloaded {len(cs)}, unrecoverable (likely delisted): "
              f"{len(failed)} -> {', '.join(x.replace('.NS','') for x in failed[:10])}")
    if len(ec):
        close = close.join(ec.reindex(close.index))
        volume = volume.join(ev.reindex(volume.index))
    return close, volume


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        close, volume = sf.make_synthetic()
        snapshots = synthetic_membership(close)
    else:
        tickers = sf.get_universe()
        close, volume = sf.download(tickers)
        print("fetching historical constituent snapshots (wayback) ...")
        snapshots = fetch_snapshots()
        n_snaps = sum(len(s) for s in snapshots.values())
        if n_snaps < 6:
            print(f"\nWARNING: only {n_snaps} usable snapshots found - "
                  "membership granularity is coarse; results are approximate.")
        close, volume = extend_panel(close, volume, snapshots)

    os.makedirs(PIT, exist_ok=True)
    print(f"panel: {close.shape[1]} tickers")
    S = sf.build_signals(close, volume)
    dates = sf.rebal_dates(close.index)

    orig_n = sf.TOP_N
    sf.TOP_N = N_PIT
    audit = []
    wbd = {}
    for d in dates:
        live = membership_at(snapshots, d)
        wbd[d] = _pit_weights(S, d, live)
        audit.append({"date": d, "members_live": len(live),
                      "picks": ",".join(t.replace(".NS", "") for t in wbd[d].index)})
    sf.TOP_N = orig_n

    eq, tr, cost = sf.run_backtest(close, wbd, *sf.OFF)
    eq_s, tr_s, _ = sf.run_backtest(close, wbd, *sf.STRESS)
    r = eq.pct_change().dropna()
    full = pd.concat([eq, eq.iloc[-1] * eq_s / eq_s.iloc[0]])

    print("\n=== POINT-IN-TIME UNIVERSE, N=7 ===")
    print(f"PnL 2021-2025 : Rs{(eq.iloc[-1] - sf.CAPITAL) / 1e7:.2f}cr")
    print(f"MDD           : {100 * float((eq / eq.cummax() - 1).min()):.1f}%")
    print(f"Sharpe (conv) : {float(r.mean() / (r.std() + 1e-12) * np.sqrt(252)):.2f}")
    print(f"trades        : {len(tr)}")
    for y in YEARS:
        if y == "2026H1":
            s0, e0 = sf.STRESS
        else:
            s0, e0 = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        w = full[(full.index >= s0) & (full.index <= e0)]
        if len(w) > 10:
            print(f"  {y}: {100 * (w.iloc[-1] / w.iloc[0] - 1):+7.1f}%")
    pd.DataFrame(audit).to_csv(os.path.join(PIT, "membership_audit.csv"), index=False)
    tr.to_csv(os.path.join(PIT, "trades_pit.csv"), index=False)
    print(f"\nsaved: {PIT}/membership_audit.csv, trades_pit.csv")
    print("compare vs frozen-universe N=7: Rs28.05cr - the gap is the "
          "survivorship premium, quantified.")


def _pit_weights(S, d, live):
    """target_weights restricted to live members (mirror of sf.target_weights)"""
    ok = S["mom_12m"].loc[d].notna()
    ok &= pd.Series(ok.index.isin(live), index=ok.index)
    liq = S["turnover_60"].loc[d]
    ok &= liq.notna() & (liq >= sf.MIN_TURNOVER)
    if sf.UNIVERSE_HALF != "all":
        v = S["vol_60"].loc[d].where(ok)
        med = v.median()
        ok &= (v <= med) if sf.UNIVERSE_HALF == "lowvol" else (v > med)
    checks = {"rsi_below_85": S["rsi"].loc[d] < 85,
              "week_positive": S["mom_5d"].loc[d] > 0,
              "mom6_positive": S["mom_6m"].loc[d] > 0,
              "near_52wh": S["dist_52wh"].loc[d] > 0.90,
              "above_50dma": S["px_50dma"].loc[d] > 1}
    for c in sf.CONDITIONS:
        if c == "vol_above_med":
            v = S["vol_60"].loc[d].where(ok)
            ok &= v > v.median()
        else:
            ok &= checks[c].reindex(ok.index).fillna(False)
    s = S["score"].loc[d].where(ok).dropna()
    if len(s) < N_PIT:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / N_PIT, index=s.nlargest(N_PIT).index)


if __name__ == "__main__":
    main()
