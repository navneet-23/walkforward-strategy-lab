# Strategy timeline — which strategy traded when

At each 6-month boundary a tournament over ~580 candidates on the trailing
2 years crowns one winner, which then trades unchanged for the next 6 months.
This is the complete sequence of winners from the point-in-time run, with the
training-window return that won the tournament and the out-of-sample return
the strategy then actually delivered.

The gap between those two columns is the honest cost of selection bias:
the tournament crowns a lucky extreme, and reality regresses it.

| # | Deployed from | Style family | Winning strategy | Trained 2y | Deployed 6m |
|---|---|---|---|---|---|
| 1 | 2015-01-01 | micro | `price_low_all_n4` | +208.7% | -4.9% |
| 2 | 2015-07-01 | defens2/flow | `low_turnover&vol_compress_highvol_n4|mom6_positive` | +254.7% | +5.0% |
| 3 | 2016-01-01 | flow/trend | `low_turnover&ma_cross_50_200_all_n4` | +429.0% | -12.7% |
| 4 | 2016-07-01 | defens2/flow | `obv_slope&vol_compress_all_n4|rsi_below_85` | +172.0% | +31.2% |
| 5 | 2017-01-01 | micro/trend | `amihud&px_sma200_highvol_n4` | +213.7% | +34.0% |
| 6 | 2017-07-01 | micro | `amihud_highvol_n4|above_50dma+mom6_positive` | +313.5% | +41.2% |
| 7 | 2018-01-01 | flow/micro | `low_turnover&price_low_highvol_n4|above_50dma+mom6_positive` | +322.0% | -12.9% |
| 8 | 2018-07-01 | micro | `amihud_highvol_n4|above_50dma+mom6_positive` | +331.3% | +6.8% |
| 9 | 2019-01-01 | micro/trend | `amihud&ma_cross_50_200_highvol_n4|mom6_positive+rsi_below_85` | +283.8% | +9.9% |
| 10 | 2019-07-01 | contra | `bb_lower_all_n4` | +101.5% | -0.2% |
| 11 | 2020-01-01 | defens/flow | `low_atr&vol_surge_lowvol_n4` | +66.9% | +11.6% |
| 12 | 2020-07-01 | contra | `bb_lower_highvol_n4` | +136.8% | +38.4% |
| 13 | 2021-01-01 | contra/micro | `bb_lower&skew_neg_all_n4` | +226.7% | +12.0% |
| 14 | 2021-07-01 | quality2/trend | `eff_ratio_6m&ma_cross_50_200_highvol_n4|above_50dma+vol_above_med` | +297.3% | +41.3% |
| 15 | 2022-01-01 | quality/trend | `mom_3m&sortino_6m_highvol_n4|vol_above_med` | +428.4% | -16.4% |
| 16 | 2022-07-01 | micro/trend | `amihud&ma_cross_50_200_highvol_n4|above_50dma+vol_above_med` | +740.8% | +28.7% |
| 17 | 2023-01-01 | micro/trend | `amihud&ma_cross_50_200_highvol_n4|above_50dma+vol_above_med` | +562.7% | +9.5% |
| 18 | 2023-07-01 | micro/quality | `amihud&trend_r2_all_n4|liq_above_med|gate_mkt_6m_pos` | +225.9% | +30.5% |
| 19 | 2024-01-01 | micro/trend | `adx_14&price_low_highvol_n4|rsi_below_85+vol_above_med` | +327.4% | +33.3% |
| 20 | 2024-07-01 | micro/trend | `adx_14&price_low_all_n4|rsi_below_85` | +542.3% | -15.6% |
| 21 | 2025-01-01 | quality/trend | `calmar_6m&trix_15_highvol_n4|above_50dma+rsi_below_85` | +395.3% | +26.6% |
| 22 | 2025-07-01 | flow/quality2 | `eff_ratio_6m&vol_surge_highvol_n4|above_50dma+rsi_below_85` | +499.9% | +5.6% |
| 23 | 2026-01-01 | quality/trend | `macd_12_26&sortino_6m_highvol_n4|above_50dma+rsi_below_85` | +236.5% | +19.9% |

## Decoded

**1. 2015-01-01** — top 4 by **price_low** within full universe

**2. 2015-07-01** — top 4 by **low_turnover + vol_compress** within above-median volatility names; holdings must satisfy `mom6_positive`

**3. 2016-01-01** — top 4 by **low_turnover + ma_cross_50_200** within full universe

**4. 2016-07-01** — top 4 by **obv_slope + vol_compress** within full universe; holdings must satisfy `rsi_below_85`

**5. 2017-01-01** — top 4 by **amihud + px_sma200** within above-median volatility names

**6. 2017-07-01** — top 4 by **amihud** within above-median volatility names; holdings must satisfy `above_50dma`, `mom6_positive`

**7. 2018-01-01** — top 4 by **low_turnover + price_low** within above-median volatility names; holdings must satisfy `above_50dma`, `mom6_positive`

**8. 2018-07-01** — top 4 by **amihud** within above-median volatility names; holdings must satisfy `above_50dma`, `mom6_positive`

**9. 2019-01-01** — top 4 by **amihud + ma_cross_50_200** within above-median volatility names; holdings must satisfy `mom6_positive`, `rsi_below_85`

**10. 2019-07-01** — top 4 by **bb_lower** within full universe

**11. 2020-01-01** — top 4 by **low_atr + vol_surge** within below-median volatility names

**12. 2020-07-01** — top 4 by **bb_lower** within above-median volatility names

**13. 2021-01-01** — top 4 by **bb_lower + skew_neg** within full universe

**14. 2021-07-01** — top 4 by **eff_ratio_6m + ma_cross_50_200** within above-median volatility names; holdings must satisfy `above_50dma`, `vol_above_med`

**15. 2022-01-01** — top 4 by **mom_3m + sortino_6m** within above-median volatility names; holdings must satisfy `vol_above_med`

**16. 2022-07-01** — top 4 by **amihud + ma_cross_50_200** within above-median volatility names; holdings must satisfy `above_50dma`, `vol_above_med`

**17. 2023-01-01** — top 4 by **amihud + ma_cross_50_200** within above-median volatility names; holdings must satisfy `above_50dma`, `vol_above_med`

**18. 2023-07-01** — top 4 by **amihud + trend_r2** within full universe; holdings must satisfy `liq_above_med`; cash when `gate_mkt_6m_pos` is off

**19. 2024-01-01** — top 4 by **adx_14 + price_low** within above-median volatility names; holdings must satisfy `rsi_below_85`, `vol_above_med`

**20. 2024-07-01** — top 4 by **adx_14 + price_low** within full universe; holdings must satisfy `rsi_below_85`

**21. 2025-01-01** — top 4 by **calmar_6m + trix_15** within above-median volatility names; holdings must satisfy `above_50dma`, `rsi_below_85`

**22. 2025-07-01** — top 4 by **eff_ratio_6m + vol_surge** within above-median volatility names; holdings must satisfy `above_50dma`, `rsi_below_85`

**23. 2026-01-01** — top 4 by **macd_12_26 + sortino_6m** within above-median volatility names; holdings must satisfy `above_50dma`, `rsi_below_85`


## Reading notes

- `&` joins two signals into an equal-weighted z-score blend; `|` separates
  the blend from stock-level conditions and market gates; `+` chains several.
- No strategy family dominates: flow/illiquidity blends own 2016–2019,
  contrarian `bb_lower` owns the COVID windows, quality/trend blends own
  2021–2026. The rotation *is* the system.
- Trained ≫ deployed in every window, as expected from a ~580-candidate
  tournament. The claim is the sum of the deployed column, never the trained one.

Full definitions of every signal, condition and gate: [STRATEGY_UNIVERSE.md](STRATEGY_UNIVERSE.md).
