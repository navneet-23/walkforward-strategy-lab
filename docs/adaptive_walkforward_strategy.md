# Adaptive Walk-Forward Strategy Selection — Full Specification

A market-agnostic system for running a rotating long-only portfolio that
**re-selects its own strategy** from a large candidate library at a fixed
cadence — with the complete signal pool, search procedure, time
frequencies, and the evaluation protocol that keeps the backtest honest.

Works on any equity-like universe with daily close and volume data
(~100–300 liquid names with a definable historical membership source).

---

## 1. The core idea

No single trading rule works in all regimes. Instead of betting on one,
the system maintains a **library of candidate strategies** and, at a
fixed interval, re-runs a tournament on recent data to decide which
candidate trades the next interval.

1. **Search** — on the trailing training window, backtest every
   candidate, net of costs.
2. **Select** — crown the candidate with the best training-window score.
3. **Deploy** — trade that candidate, unchanged, for the deployment
   window.
4. **Repeat** at the window boundary. The winner usually changes.

Every deployed window is fully out-of-sample for the strategy chosen in
it.

---

## 2. Time frequencies (fixed a priori — never tuned on results)

| Clock | Value | Role |
|---|---|---|
| Signal warm-up | 2 years of extra price history before the first training window | so 12-month lookbacks exist on day one |
| Training window | trailing **2 years** | the tournament's scoring period |
| Re-selection cadence | every **6 months** | a fresh tournament; a fresh (possibly different) winner |
| Deployment window | **6 months** | the winner trades unchanged |
| Rebalance cadence | **fortnightly** (every 2nd weekly close) | re-rank, trade only names entering/exiting the top set |
| Transaction cost | **0.1% per leg** (reference value — set to your market) | charged on every buy and sell, in search *and* deployment |

Portfolio sizes searched: **N ∈ {4, 8}**, equal-weighted.
Eligibility floor: a minimum 60-day average daily traded value
(set to a level where your position size is a small fraction of daily
volume).

---

## 3. The signal library — 45 signals in 9 families

All signals are computed from daily close `C` and volume `V` only.
Notation: `r` = daily return, `ret(n)` = n-day return, `EMA(n)` /
`SMA(n)` = exponential / simple moving averages of close, `vol(n)` =
annualized std of daily returns over n days, `TV` = close × volume
(traded value), `univ` = equal-weight universe return series. Every
signal is oriented so **higher = more attractive**. Candidate scores are
cross-sectional z-score sums: `score = z(a) [+ z(b)]`.

**TREND — winners keep winning (13)**

| signal | formula |
|---|---|
| `mom_3m` | ret(63) |
| `mom_6m` | ret(126) |
| `mom_12_1` | 12-month return skipping the most recent month: ret(231) lagged 21d |
| `macd_5_35` | (EMA(5) − EMA(35)) / C |
| `macd_12_26` | (EMA(12) − EMA(26)) / C |
| `trix_15` | 1-day %change of triple-EMA(15) of C |
| `kst` | 1·sroc(10,10) + 2·sroc(15,10) + 3·sroc(20,10) + 4·sroc(30,15), where sroc(a,b) = b-day mean of ret(a) |
| `px_sma50` | C / SMA(50) |
| `px_sma200` | C / SMA(200) |
| `ma_cross_50_200` | SMA(50) / SMA(200) |
| `adx_14` | Wilder-smoothed \|+DI − −DI\| / (+DI + −DI) (close-only DI from signed daily changes) |
| `dmi_diff` | +DI − −DI |
| `dist_52wh` | C / 252-day max (1.0 = at 52-week high) |

**CONTRARIAN — buy recent losers / oversold (6)**

| signal | formula |
|---|---|
| `rev_1m` | −ret(21) |
| `rev_6m` | −ret(126) |
| `rsi_oversold` | −RSI(14) (Wilder) |
| `stoch_oversold` | −%K(14) on close-only 14d range |
| `near_52wl` | −(C / 252-day min) |
| `bb_lower` | −%B within Bollinger(20, 2σ) |

**DEFENSIVE — low-risk, stable, crash-resistant (7)**

| signal | formula |
|---|---|
| `lowvol_60` | −vol(60) |
| `lowvol_20` | −vol(20) |
| `low_beta` | −β vs universe (120d rolling) |
| `low_atr` | −(Wilder-smoothed \|ΔC\| / C) |
| `above_200_persist` | fraction of last 60 days spent above SMA(200) |
| `dd_resist` | rolling-126d minimum of drawdown from 126d high (least-bad drawdown scores highest) |
| `updays_1m` | fraction of up days over 21d |

**QUALITY — risk-adjusted path quality (4)**

| signal | formula |
|---|---|
| `sharpe_6m` | ret(126) / vol(60) |
| `sortino_6m` | ret(126) / downside-vol(126) |
| `calmar_6m` | ret(126) / \|max drawdown(126)\| |
| `trend_r2` | R² of log-price vs time (126d) × sign(ret(126)) |

**FLOW / LIQUIDITY (6)**

| signal | formula |
|---|---|
| `low_turnover` | −log(60d mean TV) |
| `obv_slope` | 20d change of on-balance volume / 60d mean volume |
| `cmf_20` | Chaikin money flow(20) (close-location value × V, 20d) |
| `vol_surge` | 20d mean TV / 120d mean TV |
| `vol_dry` | −`vol_surge` |
| `pv_corr` | 60d corr(returns, volume %change) |

**MICROSTRUCTURE anomalies (4)**

| signal | formula |
|---|---|
| `anti_lottery` | −(max daily return over 21d) |
| `skew_neg` | −(126d return skewness) |
| `price_low` | −log(C) (low nominal price) |
| `amihud` | log(60d mean of \|r\| / TV) — price impact per unit traded |

**SEASONALITY / RELATIVE STRENGTH (2)** — `seasonal_1m` (same-calendar-month
mean of the stock's last 3 same-month returns, lagged) · `rel_str_6m`
(ret(126) minus universe ret(126))

**DEFENSIVE-II (2)** — `idio_vol_low` (−std of 120d beta-residual
returns) · `vol_compress` (−vol(20)/vol(120))

**QUALITY-II (1)** — `eff_ratio_6m` (ret(126) / Σ\|daily moves\| over
126d — path efficiency)

**Family rule:** a blend may combine at most one signal per family —
no candidate ever stacks two measurements of the same thing.

---

## 4. The candidate grid

Each candidate = **(signal blend, universe variant, portfolio size,
conditions, gates)**.

- **Universe variants (3):** `all` · `highvol` (names above the
  universe-median 60d volatility) · `lowvol` (below it).
- **Stock-level conditions (7)** — a held name must satisfy all of the
  candidate's conditions at each rebalance: `rsi_below_85` ·
  `vol_above_med` · `vol_below_med` · `mom6_positive` (ret(126) > 0) ·
  `above_50dma` · `near_52wh` (C > 0.90 × 252d max) · `liq_above_med`.
- **Market-level gates (2)** — go to cash when false:
  `gate_mkt_200dma` (universe index above its SMA(200)) ·
  `gate_mkt_6m_pos` (universe ret(126) > 0).

---

## 5. The tournament (runs at every 6-month re-selection)

A staged tree search, ~580 backtests per window, each on the trailing
2-year training window, net of costs:

1. **Singles:** all 45 signals × 3 universes × 2 sizes = **270** runs.
2. **Pairs:** promote the top 8 single *signals* (one per family);
   form all cross-family pairs × universes × sizes ≈ **168** runs,
   scored as z(a) + z(b).
3. **Condition tree:** beam search, width 8, depth ≤ 2 — each step
   adds one unused condition or gate to each beam member ≈ **140**
   runs.
4. **Winner:** highest **total return** on the training window.

Criterion note: total return, Sharpe, and Calmar were compared once,
out-of-sample; avoid ratio criteria with small denominators (a
barely-trading strategy has a tiny drawdown and an absurd Calmar —
easy to Goodhart by accident). Decided once, never revisited.

---

## 6. Portfolio mechanics

At each fortnightly rebalance: apply the universe variant and
eligibility floor → drop names failing any condition → rank the rest by
the blend's z-score sum → hold the top N equal-weighted → if a gate is
false, hold cash instead. Trade only the entering/exiting names; charge
the per-leg cost on every trade.

---

## 7. The honest-evaluation protocol

**1. Walk-forward, not in-sample.** The in-sample champion is an
exhibit of selection bias, not a forecast.

**2. Point-in-time universe.** Running history on *today's*
constituents smuggles in survivorship — today's members are the names
that survived and grew. Reconstruct membership as it stood on each
historical date (index-provider archives, web archives of constituent
files, delisted-name recovery). In the reference implementation this
single correction removed **roughly half of the apparent annual edge**
— typically the largest lie in any equity backtest.

**3. Benchmark against the universe's own drift.** The final claim is
the strategy's return **minus** an equal-weighted basket of the same
point-in-time universe. Alpha is what survives that subtraction.

Report the full ladder — in-sample / walk-forward on today's universe /
walk-forward point-in-time / universe benchmark — so every layer of
hindsight is priced separately.

---

## 8. Reporting — per window and overall

Log for every 6-month window: training start/end, deployment start/end,
the winning candidate spec, trained return, deployed return. Overall:
CAGR, Sharpe (daily), max drawdown, count of positive deployment
windows, total trades, turnover, total costs paid — and the same for
the equal-weight universe benchmark.

---

## 9. What to expect

- **No stable identity** — the winner rotates across signal families
  with the regime; the edge, if any, is in the re-selection.
- **Trained ≫ deployed, always** — a ~580-candidate tournament crowns
  a lucky extreme; a window with an outlier trained score deserves
  suspicion, not excitement.
- **Violent drawdowns** — concentrated adaptive portfolios earn their
  premium through pain.
- **Decay** — re-validate the full ladder periodically.

---

*Research methodology, not investment advice. Every number is
conditional on point-in-time-clean data and honest costs — the
protocol exists because both usually fail silently.*
