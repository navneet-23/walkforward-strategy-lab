"""
PIT reconstruction audit + 2x2 cross-deployment test.
Run in the same folder after walkforward_pit.py has built its caches:
    python pit_audit.py
"""
import os
from collections import Counter

import pandas as pd

import walkforward_pit as wp

# ---------- 1) membership reconstruction quality ----------
close = pd.read_parquet(os.path.join(wp.PITDATA, "close.parquet"))
volume = pd.read_parquet(os.path.join(wp.PITDATA, "volume.parquet"))
snaps = wp.fetch_snapshots()          # cached; no re-download
panel = set(close.columns)

print("=== SNAPSHOT INVENTORY ===")
unmatched = Counter()
rows = []
for name, s in snaps.items():
    for ts in sorted(s):
        tick = s[ts]
        hit = len(tick & panel)
        rows.append({"index": name, "date": ts.date(), "symbols": len(tick),
                     "matched": hit, "match_%": round(100 * hit / len(tick), 1)})
        unmatched.update(tick - panel)
R = pd.DataFrame(rows)
per_year = R.assign(y=pd.to_datetime(R["date"]).dt.year).groupby(["index", "y"]).agg(
    snapshots=("date", "count"), avg_match=("match_%", "mean")).round(1)
print(per_year.to_string())
print(f"\nworst single snapshot match: {R['match_%'].min()}%")
print("\n=== TOP 30 UNMATCHED SYMBOLS (rename/alias candidates) ===")
for sym, n in unmatched.most_common(30):
    print(f"  {sym:<22} in {n} snapshots")

member, first_snap = wp.build_membership(close.index, close.columns, snaps)
mm = member.sum(axis=1)
print(f"\n=== POOL SIZE BY YEAR (target ~200; below ~180 = mapping leakage) ===")
print(mm.groupby(mm.index.year).mean().round(0).to_string())
print(f"true PIT from {first_snap.date()}; earlier = earliest snapshot")

# ---------- 2) 2x2 cross-deployment, 2026 H1 ----------
print("\n=== 2x2: strategy x universe, deployed Jan-Jun 2026 ===")
eng_pit = wp.Engine(close, volume, 0.001, member=member)
eng_load = wp.Engine(close, volume, 0.001, member=None)
spec_loaded = (("kst",), "all", 4, ("near_52wh", "rsi_below_85"), ())
spec_pit = (("bb_lower", "mom_3m"), "highvol", 4, ("mom6_positive", "near_52wh"), ())
d0, d1 = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-30")
for sname, spec in [("loaded-winner (kst|52wh+rsi)", spec_loaded),
                    ("PIT-winner (bb_lower&mom3m)", spec_pit)]:
    for uname, eng in [("loaded-universe", eng_load), ("PIT-universe", eng_pit)]:
        dates = eng.fortnights[(eng.fortnights >= d0) & (eng.fortnights < d1)]
        eq = eng.backtest(eng.weights(spec, dates), d0, d1)
        print(f"  {sname:<30} on {uname:<16}: {100*(eq.iloc[-1]/eq.iloc[0]-1):+6.1f}%")
