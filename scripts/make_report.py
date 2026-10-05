"""
make_report.py - final report generator for the walk-forward study.

Run AFTER walkforward_study.py has produced study_log_<mode>.csv (and with
the data caches in place), from the same folder:

    python make_report.py                          # both modes if logs exist
    python make_report.py --mode current
    python make_report.py --costs 0.001 0.0005     # base first, then lower

What it does (no re-search - fast):
  1. parses the winning strategy of every 6-month window from study_log,
  2. re-deploys the SAME winner stream at each cost level,
  3. builds strategy-vs-benchmark total-PnL charts, the winner table, and
     metrics (Sharpe, total PnL, half-yearly PnL),
  4. writes walkforward_report.html (self-contained, charts embedded)
     plus PNGs/CSVs in report_files/.
"""

import argparse
import base64
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import walkforward_study as ws

# ----- palette (validated reference set; entity-stable across figures) -----
C_BASE = "#2a78d6"   # strategy @ base cost
C_LOW = "#eb6834"    # strategy @ lower cost
C_BENCH = "#1baf7a"  # benchmark basket
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "font.family": "sans-serif",
})

OUTDIR = "report_files"


# ---------------------------------------------------------------- utilities
def parse_spec(name):
    """Invert run_config's winner-name encoding back into an Engine spec."""
    parts = name.split("|")
    conds, gates = [], []
    for extra in parts[1:]:
        for tok in extra.split("+"):
            (gates if tok in ws.GATES else conds).append(tok)
    core, nstr = parts[0].rsplit("_n", 1)
    rest, universe = core.rsplit("_", 1)
    assert universe in ws.UNIVERSES, f"bad universe in '{name}'"
    return (tuple(rest.split("&")), universe, int(nstr),
            tuple(sorted(conds)), tuple(sorted(gates)))


def load_mode_data(mode, synthetic):
    if synthetic:
        close, volume = ws.make_synthetic()
        member = ws._synthetic_membership(close) if mode == "pit" else None
    elif mode == "pit":
        close = pd.read_parquet(os.path.join(ws.PITDATA, "close.parquet"))
        volume = pd.read_parquet(os.path.join(ws.PITDATA, "volume.parquet"))
        member = pd.read_parquet(os.path.join(ws.PITDATA, "membership.parquet"))
    else:
        close = pd.read_parquet(os.path.join(ws.DATA, "close.parquet"))
        volume = pd.read_parquet(os.path.join(ws.DATA, "volume.parquet"))
        member = None
    return close, volume, member


def benchmark_curve(close, member, index_like):
    r = close.pct_change()
    if member is not None:
        m = member.shift(1).fillna(False).astype(float)
        br = ((r * m).sum(axis=1) / m.sum(axis=1).replace(0, np.nan)).fillna(0)
        lab = "eq-wt basket (point-in-time universe)"
    else:
        br = r.mean(axis=1).fillna(0)
        lab = "eq-wt basket (today's lists)"
    b = (1 + br).cumprod()
    b = b.reindex(index_like).ffill()
    return ws.CAPITAL * b / b.iloc[0], lab


def metrics(eq):
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return {
        "final_value_cr": eq.iloc[-1] / 1e7,
        "total_pnl_cr": (eq.iloc[-1] - eq.iloc[0]) / 1e7,
        "total_return_pct": 100 * (eq.iloc[-1] / eq.iloc[0] - 1),
        "CAGR_pct": 100 * ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1),
        "Sharpe_ann": float(r.mean() / (r.std() + 1e-12) * np.sqrt(252)),
        "MDD_pct": 100 * float((eq / eq.cummax() - 1).min()),
    }


def half_bounds(eq):
    out = []
    for y in range(eq.index[0].year, eq.index[-1].year + 1):
        for h, (s, e) in enumerate([(f"{y}-01-01", f"{y}-06-30"),
                                    (f"{y}-07-01", f"{y}-12-31")], 1):
            w = eq[(eq.index >= s) & (eq.index <= e)]
            if len(w) > 10:
                out.append((f"{y}H{h}", w))
    return out


def half_yearly_table(curves):
    """curves: dict label -> equity. Returns DataFrame of % return per half
    plus PnL (Rs lakh) per half for the first curve."""
    rows = {}
    first = list(curves)[0]
    for lab, eq in curves.items():
        rows[f"{lab} %"] = {p: 100 * (w.iloc[-1] / w.iloc[0] - 1)
                            for p, w in half_bounds(eq)}
    rows[f"{first} PnL (Rs lakh)"] = {p: (w.iloc[-1] - w.iloc[0]) / 1e5
                                      for p, w in half_bounds(curves[first])}
    return pd.DataFrame(rows).round(1)


def fig_to_b64(fig, path):
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def crore_axis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v/1e7:g}"))
    ax.set_ylabel("portfolio value (Rs crore, log scale)")


# ---------------------------------------------------------------- per mode
def run_mode(mode, costs, args):
    logf = f"study_log_{mode}.csv"
    if not os.path.exists(logf):
        print(f"  [{mode}] {logf} not found - skipping")
        return None
    L = pd.read_csv(logf, parse_dates=["deploy_from"])
    close, volume, member = load_mode_data(mode, args.synthetic)
    eng = ws.Engine(close, volume, member)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    # rebuild the deployed-weights stream from the logged winners
    deploy_wbd, spans = {}, []
    for _, row in L.iterrows():
        dep0 = pd.Timestamp(row["deploy_from"])
        dep1 = min(dep0 + pd.DateOffset(months=6), end + pd.Timedelta(days=1))
        spec = parse_spec(row["winner"])
        dd = eng.fortnights[(eng.fortnights >= dep0) & (eng.fortnights < dep1)]
        deploy_wbd.update(eng.weights(spec, dd))
        spans.append((dep0, dep1))
        print(f"  [{mode}] weights {dep0.date()} {row['winner']}")

    # one continuous backtest per cost level (same decisions, re-priced)
    curves = {}
    for c in costs:
        ws.COST = c
        curves[c] = eng.backtest(deploy_wbd, start, end)
        print(f"  [{mode}] cost {c*100:.3f}%/leg -> "
              f"final Rs {curves[c].iloc[-1]/1e7:.2f} cr")
    ws.COST = costs[0]
    base, low = curves[costs[0]], curves[costs[-1]]
    bench, bench_lab = benchmark_curve(close, member, base.index)

    # winner table with per-cost deployed returns from the continuous curves
    T = L.copy()
    for c in costs:
        col = f"deployed_6m_% @ {c*100:.2f}%/leg"
        vals = []
        for dep0, dep1 in spans:
            w = curves[c][(curves[c].index >= dep0) & (curves[c].index < dep1)]
            vals.append(round(100 * (w.iloc[-1] / w.iloc[0] - 1), 1)
                        if len(w) > 5 else np.nan)
        T[col] = vals

    # metrics + half-yearly
    M = pd.DataFrame({
        f"strategy @ {c*100:.2f}%/leg": metrics(curves[c]) for c in costs
    } | {"benchmark": metrics(bench)}).round(2)
    lab_base = f"strategy @ {costs[0]*100:.2f}%/leg"
    lab_low = f"strategy @ {costs[-1]*100:.2f}%/leg"
    H = half_yearly_table({lab_base: base, lab_low: low, "benchmark": bench})

    # ---- figures ----
    figs = {}
    f, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(base.index, base.values, color=C_BASE, lw=2, label=lab_base)
    if len(costs) > 1:
        ax.plot(low.index, low.values, color=C_LOW, lw=2, label=lab_low)
    ax.plot(bench.index, bench.values, color=C_BENCH, lw=2, label=bench_lab)
    ax.set_yscale("log")
    crore_axis(ax)
    ax.set_title(f"Total value of Rs 1 crore over time - {mode} universe",
                 color=INK, loc="left")
    ax.legend(loc="upper left")
    figs["equity"] = fig_to_b64(f, f"{OUTDIR}/{mode}_equity.png")

    f, ax = plt.subplots(figsize=(9.5, 4.6))
    pnl_b = (base - base.iloc[0]) / 1e7
    pnl_l = (low - low.iloc[0]) / 1e7
    pnl_x = (bench - bench.iloc[0]) / 1e7
    ax.plot(pnl_b.index, pnl_b.values, color=C_BASE, lw=2, label=lab_base)
    if len(costs) > 1:
        ax.plot(pnl_l.index, pnl_l.values, color=C_LOW, lw=2, label=lab_low)
    ax.plot(pnl_x.index, pnl_x.values, color=C_BENCH, lw=2, label=bench_lab)
    ax.set_ylabel("cumulative total PnL (Rs crore)")
    ax.set_title(f"Cumulative total PnL at every point in time - {mode} universe",
                 color=INK, loc="left")
    ax.legend(loc="upper left")
    figs["pnl"] = fig_to_b64(f, f"{OUTDIR}/{mode}_pnl.png")

    f, ax = plt.subplots(figsize=(9.5, 3.4))
    dd_s = 100 * (base / base.cummax() - 1)
    dd_b = 100 * (bench / bench.cummax() - 1)
    ax.fill_between(dd_s.index, dd_s.values, 0, color=C_BASE, alpha=0.35,
                    linewidth=0)
    ax.plot(dd_s.index, dd_s.values, color=C_BASE, lw=1.2, label=lab_base)
    ax.plot(dd_b.index, dd_b.values, color=C_BENCH, lw=1.2, label=bench_lab)
    ax.set_ylabel("drawdown (%)")
    ax.set_title(f"Drawdown from running peak - {mode} universe",
                 color=INK, loc="left")
    ax.legend(loc="lower left")
    figs["dd"] = fig_to_b64(f, f"{OUTDIR}/{mode}_drawdown.png")

    f, ax = plt.subplots(figsize=(9.5, 4.2))
    hs = H[f"{lab_base} %"].dropna()
    hb = H["benchmark %"].reindex(hs.index)
    x = np.arange(len(hs))
    ax.bar(x - 0.21, hs.values, width=0.4, color=C_BASE, label=lab_base,
           edgecolor=SURFACE, linewidth=1)
    ax.bar(x + 0.21, hb.values, width=0.4, color=C_BENCH, label=bench_lab,
           edgecolor=SURFACE, linewidth=1)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x, hs.index, rotation=60, fontsize=8)
    ax.set_ylabel("half-yearly return (%)")
    ax.set_title(f"Half-yearly PnL, strategy vs benchmark - {mode} universe",
                 color=INK, loc="left")
    ax.legend(loc="upper right")
    figs["half"] = fig_to_b64(f, f"{OUTDIR}/{mode}_halfyearly.png")

    T.to_csv(f"{OUTDIR}/{mode}_winners.csv", index=False)
    M.to_csv(f"{OUTDIR}/{mode}_metrics.csv")
    H.to_csv(f"{OUTDIR}/{mode}_halfyearly.csv")
    return {"mode": mode, "figs": figs, "winners": T, "metrics": M,
            "half": H, "bench_lab": bench_lab}


# ---------------------------------------------------------------- html
ASSUMPTIONS = """
<h2>Assumptions: charges and duties</h2>
<p>The engine charges a flat <b>{base:.2f}% per leg</b> (every buy and every
sell) on traded value, inside both the strategy-selection tournament and the
deployed backtest. That figure approximates the all-in statutory cost of
<b>delivery equity with a zero-brokerage discount broker</b>:</p>
<table>
<tr><th>Component (delivery equity)</th><th>Rate</th></tr>
<tr><td>Brokerage (discount broker, delivery)</td><td>Rs 0</td></tr>
<tr><td>Securities Transaction Tax (STT)</td><td>0.100% on buy and on sell</td></tr>
<tr><td>Exchange transaction charge (NSE)</td><td>~0.003%</td></tr>
<tr><td>SEBI turnover fee</td><td>0.0001%</td></tr>
<tr><td>Stamp duty</td><td>0.015% on buy</td></tr>
<tr><td>GST (18% on brokerage + exchange charge)</td><td>~0.0005%</td></tr>
<tr><td>DP charge on sell (flat ~Rs 13/scrip/day)</td><td>negligible at this capital</td></tr>
<tr><td><b>Approx. all-in</b></td><td><b>~0.118% buy / ~0.104% sell</b></td></tr>
</table>
<p><b>Not modelled:</b> slippage and market impact (the universe carries a
minimum traded-value floor, but impact at size is real and uncharged), capital
gains tax, and cash drag on idle balances. Dividends enter through
split-and-dividend-adjusted prices, i.e. an approximate total-return series.
Fills are assumed at the closing price of each fortnightly rebalance day.</p>
<p><b>The lower-cost reconstruction ({low:.2f}%/leg)</b> re-prices the
<i>identical</i> stream of selected strategies and trades. It represents a
plausible cheaper execution route - e.g. single-stock futures (far lower STT
and charges) or a reduced-duty regime. It is conservative in one specific way:
the strategy tournament itself was run at {base:.2f}%/leg, so at genuinely
lower costs the search would, if anything, have selected higher-turnover
candidates with more edge, not fewer.</p>
"""

MODE_INTRO = {
    "current": "<p><b>Diagnostic configuration (today's constituents applied "
               "to history - survivorship-biased). Not a performance claim.</b></p>",
    "pit": "<p>Universe membership is reconstructed point-in-time from the "
           "official index provider's press-release archive: 769 inclusion/"
           "exclusion events (2012-2026) parsed from 204 reconstitution "
           "announcements, anchored on today's official constituent lists "
           "and walked backward, then lagged one day. The benchmark is the "
           "equal-weight basket of the same point-in-time universe, so "
           "strategy and benchmark share an identical information set.</p>",
}


def html_report(sections, costs, out):
    css = """
    body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;
         background:#f9f9f7;color:#0b0b0b;max-width:980px;margin:24px auto;
         padding:0 16px;}
    h1{font-size:26px} h2{font-size:19px;margin-top:34px}
    h3{font-size:16px;margin-top:26px}
    p,li{font-size:14px;line-height:1.55;color:#2a2a28}
    table{border-collapse:collapse;font-size:12.5px;margin:12px 0;
          background:#fcfcfb}
    th,td{border:1px solid #e1e0d9;padding:5px 9px;text-align:right;
          font-variant-numeric:tabular-nums}
    th{background:#f0efec;color:#52514e;text-align:right}
    td:first-child,th:first-child{text-align:left}
    img{max-width:100%;border:1px solid #e1e0d9;border-radius:6px;
        background:#fcfcfb;margin:8px 0}
    .note{color:#898781;font-size:12px}
    """
    h = [f"<!doctype html><html><head><meta charset='utf-8'>"
         f"<title>Walk-Forward Strategy Report</title><style>{css}</style>"
         f"</head><body>"]
    h.append("<h1>Walk-Forward Strategy - Performance Report</h1>")
    h.append("<p>A 6-monthly re-selected, fortnightly-rebalanced top-N "
             "equal-weight portfolio, chosen each window by a costed tournament "
             "over 45 signals x universe variants x portfolio sizes x shallow "
             "condition trees, on the trailing 2 years. Every deployed window "
             "is fully out-of-sample for the strategy chosen in it, and the "
             "investable universe on every historical date is the index "
             "membership as it stood on that date - no information from after "
             "the trade date enters the universe, the signals, or the "
             "selection. Starting capital Rs 1 crore.</p>")
    h.append(ASSUMPTIONS.format(base=costs[0] * 100, low=costs[-1] * 100))
    for S in sections:
        if S is None:
            continue
        m = S["mode"]
        h.append("<h2>Results</h2>" if m == "pit"
                 else f"<h2>Results - {m} universe (diagnostic)</h2>")
        h.append(MODE_INTRO.get(m, ""))
        h.append("<h3>Total PnL at every point in time</h3>")
        h.append(f"<img src='data:image/png;base64,{S['figs']['pnl']}'>")
        h.append(f"<img src='data:image/png;base64,{S['figs']['equity']}'>")
        h.append(f"<img src='data:image/png;base64,{S['figs']['dd']}'>")
        h.append("<h3>Headline metrics</h3>")
        h.append(S["metrics"].to_html())
        h.append("<p class='note'>Sharpe is daily, annualised with sqrt(252). "
                 "Total PnL in Rs crore on Rs 1 crore initial capital.</p>")
        h.append("<h3>Strategy chosen at every re-evaluation</h3>")
        h.append(S["winners"].to_html(index=False))
        h.append("<p class='note'>trained_2y_% is the winner's return on its "
                 "own 2-year selection window (in-sample for selection); the "
                 "deployed columns are the following 6 months, out-of-sample, "
                 "re-priced at each cost level from the continuous equity "
                 "stream. Signal names: momentum/MACD/TRIX/KST (trend), rev/"
                 "RSI/stoch (contrarian), lowvol/beta/drawdown (defensive), "
                 "sharpe/sortino/calmar (quality), volume/liquidity (flow); "
                 "suffix _universe_nN, |conditions, |market gates.</p>")
        h.append("<h3>Half-yearly PnL</h3>")
        h.append(f"<img src='data:image/png;base64,{S['figs']['half']}'>")
        h.append(S["half"].to_html())
    h.append("<h2>Reading guard-rails</h2><ul>"
             "<li>Backtested, not live, performance; costs as stated above; "
             "slippage/impact and taxes excluded.</li>"
             "<li>38 historical members (concentrated pre-2019, skewed "
             "toward delisted failures - DHFL, Bhushan Steel, Gitanjali and "
             "similar) have no recoverable price history and are absent from "
             "the tradable universe; this residue biases results modestly "
             "upward and is enumerated in pit_events/reconcile.txt.</li>"
             "<li>The research protocol and search space are documented in "
             "the accompanying method note.</li></ul>")
    h.append("<p class='note'>Generated by make_report.py from study_log/"
             "study_equity outputs.</p></body></html>")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(h))
    print(f"\nreport written: {out}  (+ PNGs/CSVs in {OUTDIR}/)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pit", choices=["both", "current", "pit"])
    ap.add_argument("--costs", type=float, nargs="+", default=[0.001, 0.0005],
                    help="per-leg cost levels; FIRST must be the study's base")
    ap.add_argument("--start", default=ws.START_DEPLOY)
    ap.add_argument("--end", default=ws.END_DEPLOY)
    ap.add_argument("--out", default="walkforward_report.html")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    modes = ["current", "pit"] if args.mode == "both" else [args.mode]
    sections = [run_mode(m, args.costs, args) for m in modes]
    html_report(sections, args.costs, args.out)


if __name__ == "__main__":
    main()
