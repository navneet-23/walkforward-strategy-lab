# The strategy universe — every candidate the tournament can choose from

At every 6-month re-selection, the search evaluates **~580 candidate
strategies** built from the components below. A candidate is a tuple:

```
(signal blend, universe variant, portfolio size, stock conditions, market gates)
```

All signals are computed from **daily close `C` and volume `V` only** — no
OHLC, no fundamentals, no analyst data. Notation: `r` = daily return,
`ret(n)` = n-day return, `EMA(n)`/`SMA(n)` = moving averages of close,
`vol(n)` = annualized std of daily returns over n days, `TV` = close ×
volume (rupee traded value), `univ` = equal-weight universe return series.

Every signal is oriented so **higher = more attractive**. A blend scores
stocks by the sum of cross-sectional z-scores: `score = z(a) [+ z(b)]`.

---

## 1. The 45 signals, in 9 style families

**Family rule:** a blend may combine at most one signal per family, so no
candidate ever stacks two measurements of the same thing.

### TREND — winners keep winning (13 signals)

| signal | formula |
|---|---|
| `mom_3m` | ret(63) |
| `mom_6m` | ret(126) |
| `mom_12_1` | 12-month return skipping the most recent month: ret(231) lagged 21d |
| `macd_5_35` | (EMA(5) − EMA(35)) / C |
| `macd_12_26` | (EMA(12) − EMA(26)) / C |
| `trix_15` | 1-day %change of triple-EMA(15) of C |
| `kst` | Know Sure Thing: 1·sroc(10,10) + 2·sroc(15,10) + 3·sroc(20,10) + 4·sroc(30,15), where sroc(a,b) = b-day mean of ret(a) |
| `px_sma50` | C / SMA(50) |
| `px_sma200` | C / SMA(200) |
| `ma_cross_50_200` | SMA(50) / SMA(200) (golden-cross distance) |
| `adx_14` | Wilder-smoothed \|+DI − −DI\| / (+DI + −DI), close-only DI from signed daily changes |
| `dmi_diff` | +DI − −DI |
| `dist_52wh` | C / 252-day max (1.0 = at the 52-week high) |

### CONTRARIAN — buy recent losers / oversold (6)

| signal | formula |
|---|---|
| `rev_1m` | −ret(21) |
| `rev_6m` | −ret(126) |
| `rsi_oversold` | −RSI(14) (Wilder) |
| `stoch_oversold` | −%K(14) on the close-only 14-day range |
| `near_52wl` | −(C / 252-day min) |
| `bb_lower` | −%B within Bollinger(20, 2σ) |

### DEFENSIVE — low-risk, stable, crash-resistant (7)

| signal | formula |
|---|---|
| `lowvol_60` | −vol(60) |
| `lowvol_20` | −vol(20) |
| `low_beta` | −β vs universe (120d rolling) |
| `low_atr` | −(Wilder-smoothed \|ΔC\| / C) |
| `above_200_persist` | fraction of last 60 days spent above SMA(200) |
| `dd_resist` | rolling-126d minimum of drawdown from the 126d high (least-bad drawdown scores highest) |
| `updays_1m` | fraction of up days over 21d |

### QUALITY — risk-adjusted path quality (4)

| signal | formula |
|---|---|
| `sharpe_6m` | ret(126) / vol(60) |
| `sortino_6m` | ret(126) / downside-vol(126) |
| `calmar_6m` | ret(126) / \|max drawdown(126)\| |
| `trend_r2` | R² of log-price vs time (126d) × sign(ret(126)) |

### FLOW / LIQUIDITY (6)

| signal | formula |
|---|---|
| `low_turnover` | −log(60d mean TV) |
| `obv_slope` | 20d change of on-balance volume / 60d mean volume |
| `cmf_20` | Chaikin money flow(20): close-location value × V, summed over 20d |
| `vol_surge` | 20d mean TV / 120d mean TV |
| `vol_dry` | −`vol_surge` |
| `pv_corr` | 60d corr(returns, volume %change) |

### MICROSTRUCTURE anomalies (4)

| signal | formula |
|---|---|
| `anti_lottery` | −(max daily return over 21d) |
| `skew_neg` | −(126d return skewness) |
| `price_low` | −log(C) — low nominal price |
| `amihud` | log(60d mean of \|r\| / TV) — price impact per rupee traded |

### SEASONALITY / RELATIVE STRENGTH (2)

| signal | formula |
|---|---|
| `seasonal_1m` | same-calendar-month mean of the stock's last 3 same-month returns, lagged |
| `rel_str_6m` | ret(126) − universe ret(126) |

### DEFENSIVE-II (2)

| signal | formula |
|---|---|
| `idio_vol_low` | −std of 120d beta-residual returns |
| `vol_compress` | −vol(20) / vol(120) |

### QUALITY-II (1)

| signal | formula |
|---|---|
| `eff_ratio_6m` | ret(126) / Σ\|daily moves\| over 126d — path efficiency |

---

## 2. Universe variants (3)

Applied before ranking, recomputed at every rebalance:

| variant | definition |
|---|---|
| `all` | every point-in-time index member passing the liquidity floor |
| `highvol` | members with 60d volatility **above** the universe median |
| `lowvol` | members with 60d volatility **below** the universe median |

Eligibility floor (all variants): minimum 60-day average daily rupee traded
value, set so the ₹1 crore book is a small fraction of daily volume.

## 3. Portfolio sizes (2)

**N ∈ {4, 8}**, equal-weighted. Concentrated by design — the tournament
decides whether concentration pays.

## 4. Stock-level conditions (7)

A held name must satisfy **all** of a candidate's conditions at each
rebalance; names failing one are dropped before ranking:

| condition | a stock passes when |
|---|---|
| `rsi_below_85` | RSI(14) < 85 (not parabolic) |
| `vol_above_med` | 60d volatility above the universe median |
| `vol_below_med` | 60d volatility below the universe median |
| `mom6_positive` | ret(126) > 0 |
| `above_50dma` | C > SMA(50) |
| `near_52wh` | C > 0.90 × 252-day max |
| `liq_above_med` | 60d rupee traded value above the universe median |

## 5. Market-level gates (2)

When a candidate's gate is false at a rebalance, the whole book goes to
**cash** until the next rebalance:

| gate | in the market when |
|---|---|
| `gate_mkt_200dma` | the equal-weight universe index is above its SMA(200) |
| `gate_mkt_6m_pos` | the equal-weight universe 126d return is positive |

---

## 6. How ~580 candidates are searched per window

A staged tree search on the trailing 2-year training window, every
backtest net of 0.1%/leg costs:

1. **Singles** — all 45 signals × 3 universes × 2 sizes = **270** backtests.
2. **Pairs** — promote the top 8 single *signals* (max one per family);
   form all cross-family pairs × universes × sizes ≈ **168** backtests,
   scored as z(a) + z(b).
3. **Condition tree** — beam search, width 8, depth ≤ 2: each step adds
   one unused condition or gate to each beam member ≈ **140** backtests.
4. **Winner** — the candidate with the highest **total return** on the
   training window. It trades, unchanged, for the next 6 months.

Selection criterion note: total return, Sharpe and Calmar were compared
once, out-of-sample, before the study was frozen. Ratio criteria with
small denominators are easy to Goodhart by accident (a barely-trading
candidate has a tiny drawdown and an absurd Calmar). Decided once, never
revisited.

## 7. Winner-name encoding

The logs encode each winner as a single string:

```
amihud&ma_cross_50_200_highvol_n4|above_50dma+vol_above_med
└─ blend ─────────────┘ └─univ─┘└N┘└─ conditions / gates ──┘
```

`&` joins blend signals, `_<universe>_n<N>` closes the core, `|` starts
the condition/gate list, `+` chains several. `scripts/make_report.py`
contains `parse_spec()`, which inverts this encoding exactly.

## 8. What the search is not allowed to touch

The meta-parameters — 2-year training window, 6-month cadence, fortnightly
rebalance, cost level, eligibility floor, beam width and depth — were fixed
a priori and are **not searched**. The tournament picks a strategy; nothing
ever picks the tournament.

See [STRATEGY_TIMELINE.md](STRATEGY_TIMELINE.md) for which candidate won
each window, and [HOLDINGS_TIMELINE.md](HOLDINGS_TIMELINE.md) for the
stocks actually held.
