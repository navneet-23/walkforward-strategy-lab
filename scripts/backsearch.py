"""
Backsearch engine.

Question 1 (supervised): using ground-truth labels from 2021-2025
  label(stock, date) = 1 if the stock's NEXT-period return puts it in the
  top 10 of the eligible universe, else 0
can a classifier (logistic regression / random forest / gradient boosting)
predict top-10 membership out-of-sample on Jan-Jun 2026?

Question 2 (unsupervised): at each test date, if we cluster the universe on
technical features with no labels at all (KMeans / Gaussian mixture), do the
actual future top-10 stocks concentrate in one cluster more than chance?

Train: rebalance dates in 2021-01-01 .. 2025-12-31
Test : rebalance dates in 2026-01-01 .. 2026-06-30 (fully out-of-sample)

Usage:
    python backsearch.py                 # monthly labels, real data
    python backsearch.py --freq W        # weekly labels
    python backsearch.py --synthetic     # smoke test

Outputs: printed report + results/backsearch_report.csv
         + results/backsearch_predictions.csv (per test date, per model picks)
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture

from backtest import CAPITAL, compute_metrics, rebalance_dates, run_backtest
from features import compute_features, zscore_row
from run_search import RESULTS_DIR, load_data
from strategies import FEATURE_SETS, eligible

TRAIN_START = "2021-01-01"
TRAIN_END = "2025-12-31"
TEST_START = "2026-01-01"
TEST_END = "2026-06-30"
TOP_K = 10
FEATS = FEATURE_SETS["full_tech"]


# ---------------------------------------------------------------- dataset
def build_labelled(features, close, dates):
    """One cross-section per rebalance date: z-scored features, forward
    return to the next rebalance date, and top-10 label."""
    panels = []
    for i in range(len(dates) - 1):
        d, d_next = dates[i], dates[i + 1]
        ok = eligible(features, d)
        X = pd.DataFrame({fn: zscore_row(features[fn].loc[d].where(ok)) for fn in FEATS})
        fwd = (close.loc[d_next] / close.loc[d] - 1).where(ok)
        df = X.assign(fwd=fwd).dropna()
        if len(df) < 3 * TOP_K:
            continue
        thresh = df["fwd"].nlargest(TOP_K).iloc[-1]
        df["label"] = (df["fwd"] >= thresh).astype(int)
        df["date"] = d
        df["ticker"] = df.index
        panels.append(df.reset_index(drop=True))
    return pd.concat(panels, ignore_index=True)


def make_clf(name):
    if name == "logistic":
        return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=6,
                                      min_samples_leaf=10, class_weight="balanced",
                                      n_jobs=-1, random_state=42)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_depth=4, max_iter=200,
                                              learning_rate=0.05,
                                              class_weight="balanced",
                                              random_state=42)
    raise ValueError(name)


# ---------------------------------------------------------------- supervised
def run_supervised(data, close, test_dates):
    train = data[(data["date"] >= TRAIN_START) & (data["date"] <= TRAIN_END)]
    test = data[(data["date"] >= TEST_START) & (data["date"] <= TEST_END)]
    print(f"\ntrain: {train['date'].nunique()} dates, {len(train)} rows, "
          f"base rate {train['label'].mean():.1%}")
    print(f"test : {test['date'].nunique()} dates, {len(test)} rows, "
          f"base rate {test['label'].mean():.1%}  <- random-guess precision@10")

    report, all_preds = [], []
    for name in ["logistic", "rf", "hgb"]:
        clf = make_clf(name)
        clf.fit(train[FEATS].values, train["label"].values)
        prob = clf.predict_proba(test[FEATS].values)[:, 1]
        t = test.copy()
        t["prob"] = prob

        aucs = roc_auc_score(t["label"], t["prob"])
        ap = average_precision_score(t["label"], t["prob"])

        # per-date: model's top-10 by probability vs actual top-10
        rows = []
        weights = {}
        for d, g in t.groupby("date"):
            picks = g.nlargest(TOP_K, "prob")
            hits = int(picks["label"].sum())
            rows.append({"date": d, "hits": hits,
                         "picked_fwd": picks["fwd"].mean(),
                         "universe_fwd": g["fwd"].mean(),
                         "oracle_fwd": g.nlargest(TOP_K, "fwd")["fwd"].mean()})
            weights[pd.Timestamp(d)] = pd.Series(1.0 / TOP_K, index=picks["ticker"].values)
            for _, r in picks.iterrows():
                all_preds.append({"model": name, "date": d, "ticker": r["ticker"],
                                  "prob": r["prob"], "was_top10": int(r["label"]),
                                  "fwd_return": r["fwd"]})
        per = pd.DataFrame(rows)

        # mini backtest of the picks over H1 2026
        eq, tr, _ = run_backtest(close, weights, TEST_START, TEST_END)
        m = compute_metrics(eq, tr)

        report.append({
            "model": name, "roc_auc": round(aucs, 3), "pr_auc": round(ap, 3),
            "precision_at_10": round(per["hits"].mean() / TOP_K, 3),
            "hits_of_10_avg": round(per["hits"].mean(), 2),
            "picked_fwd_ret_avg_pct": round(100 * per["picked_fwd"].mean(), 2),
            "universe_fwd_ret_avg_pct": round(100 * per["universe_fwd"].mean(), 2),
            "oracle_fwd_ret_avg_pct": round(100 * per["oracle_fwd"].mean(), 2),
            "h1_2026_pnl_lakh": round(m["pnl"] / 1e5, 1),
            "h1_2026_sharpe": round(m["sharpe"], 2),
            "h1_2026_mdd_pct": round(m["max_drawdown_pct"], 1),
        })
        print(f"\n  {name}: AUC {aucs:.3f}  PR-AUC {ap:.3f}  "
              f"precision@10 {per['hits'].mean()/TOP_K:.1%} "
              f"({per['hits'].mean():.1f}/10 actual top-10 caught per date)")
        print(f"    picks' avg fwd return {100*per['picked_fwd'].mean():+.2f}%  "
              f"vs universe {100*per['universe_fwd'].mean():+.2f}%  "
              f"vs oracle top-10 {100*per['oracle_fwd'].mean():+.2f}%")
        print(f"    H1-2026 backtest of picks: PnL Rs{m['pnl']/1e5:.1f} lakh, "
              f"Sharpe {m['sharpe']:.2f}, MDD {m['max_drawdown_pct']:.1f}%")

    if hasattr((c := make_clf("logistic")).fit(train[FEATS].values, train["label"].values), "coef_"):
        coefs = pd.Series(c.coef_[0], index=FEATS).sort_values(key=abs, ascending=False)
        print("\n  logistic coefficients (what 'future top-10' looked like in-sample):")
        for f, v in coefs.items():
            print(f"    {f:12s} {v:+.3f}")

    return pd.DataFrame(report), pd.DataFrame(all_preds)


# ---------------------------------------------------------------- unsupervised
def run_clustering(data, ks=(5, 8, 12)):
    test = data[(data["date"] >= TEST_START) & (data["date"] <= TEST_END)]
    print("\n=== UNSUPERVISED: do future top-10 stocks cluster together? ===")
    print("lift = (share of actual top-10 landing in their modal cluster) /")
    print("       (that cluster's share of the universe).  1.0 = pure chance.\n")
    rows = []
    for algo in ["kmeans", "gmm"]:
        for k in ks:
            lifts, shares = [], []
            for d, g in test.groupby("date"):
                X = g[FEATS].values
                if algo == "kmeans":
                    lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
                else:
                    lab = GaussianMixture(n_components=k, random_state=42,
                                          covariance_type="diag").fit_predict(X)
                g = g.assign(cl=lab)
                top = g[g["label"] == 1]
                if len(top) == 0:
                    continue
                modal = top["cl"].mode().iloc[0]
                share_top = (top["cl"] == modal).mean()          # top-10 in modal cluster
                share_uni = (g["cl"] == modal).mean()            # cluster's size share
                lifts.append(share_top / share_uni if share_uni > 0 else np.nan)
                shares.append(share_top)
            rows.append({"algo": algo, "k": k,
                         "avg_top10_share_in_modal_cluster": round(np.nanmean(shares), 2),
                         "avg_lift_vs_chance": round(np.nanmean(lifts), 2),
                         "n_dates": len(lifts)})
            print(f"  {algo:6s} k={k:<3d} top-10 modal-cluster share "
                  f"{np.nanmean(shares):.0%}  lift {np.nanmean(lifts):.2f}x")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--freq", default="M", choices=["W", "M"])
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("loading data ...")
    close, volume = load_data(args.synthetic)
    features = compute_features(close, volume)
    dates = list(rebalance_dates(close.index, args.freq))

    print(f"building labelled dataset ({args.freq} frequency) ...")
    data = build_labelled(features, close, dates)
    data["date"] = pd.to_datetime(data["date"])

    test_dates = sorted(data[(data["date"] >= TEST_START) &
                             (data["date"] <= TEST_END)]["date"].unique())
    if len(test_dates) == 0:
        raise SystemExit("No test dates found in Jan-Jun 2026 - check data range.")

    print("\n=== SUPERVISED: predict next-period top-10 membership ===")
    rep, preds = run_supervised(data, close, test_dates)
    clu = run_clustering(data)

    rep.to_csv(os.path.join(RESULTS_DIR, "backsearch_report.csv"), index=False)
    clu.to_csv(os.path.join(RESULTS_DIR, "backsearch_clustering.csv"), index=False)
    preds.to_csv(os.path.join(RESULTS_DIR, "backsearch_predictions.csv"), index=False)
    print(f"\nsaved: results/backsearch_report.csv, backsearch_clustering.csv, "
          f"backsearch_predictions.csv")


if __name__ == "__main__":
    main()
