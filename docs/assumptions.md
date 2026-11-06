# JAXON v1 — Assumptions

This file separates **what the source video actually says** from **what JAXON had
to decide in order to turn it into code**. Every item in section 2 is a
*proposed implementation assumption*: configurable, untested against real
markets, and not something the video established.

## 1. Rules taken from the video

The video describes, in words only:

1. The **4-hour chart is going up**.
2. The **30-minute chart pulls down into demand**.
3. The **5-minute chart shows a bullish structure shift**.
4. **Target: VWAP.**
5. **Stop loss: at the swing low.**

The video does **not**:

- define any of these conditions mathematically,
- give any numbers (lookbacks, multipliers, windows, ratios),
- say anything about sessions, costs, sizing, or risk limits,
- establish that the setup is profitable.

## 2. Design choices made by JAXON (all configurable)

Config keys are under `strategy:` unless another section is named.

### Swings (all timeframes)
| Choice | Default | Key |
|---|---|---|
| Pivot = bars on each side | 2 left, 2 right | `*_pivot_left`, `*_pivot_right` |
| Strictness | swing high strictly above all neighbours (ties are not pivots); same for lows | — |
| Availability | only after the last right-hand bar closes; never back-dated | — |

### 4H bullish bias
| Choice | Default |
|---|---|
| "Going up" means | the two latest confirmed 4H swing highs are rising **and** the two latest confirmed swing lows are rising |
| Otherwise | neutral, no new longs |
| Not enough swing history | neutral |
| Extra filters (EMA etc.) | none |
| Bias changes between pivots | none: the bias only updates when a new pivot is confirmed |

### 30m demand zone
| Choice | Default | Key |
|---|---|---|
| Displacement candle | bullish; body ≥ 1.5 × ATR(14) as of the previous closed bar; closes above the latest 30m swing high confirmed before the candle opened | `displacement_atr_mult`, `atr_period` |
| ATR | Wilder, seeded with the simple mean of the first 14 true ranges; available after 15 bars | — |
| Demand base | most recent bearish candle (close < open) among the 5 closed 30m candles before the displacement | `base_lookback_bars` |
| Zone bounds | base candle's full low–high | — |
| Activation | when the displacement candle closes | — |
| Invalidation | a later closed 30m candle closes below the zone low | — |
| Expiry | after 20 completed 30m bars | `zone_expiry_bars` |
| Attempts | one per zone | — |
| Several zones | among valid zones that still have their attempt **and** that the touch candle overlaps, take the most recently activated. *(The spec says "most recently activated"; JAXON also skips zones whose single attempt is used, rather than blocking older valid zones.)* | — |

### Pullback and touch
| Choice | Default | Key |
|---|---|---|
| Touch | a closed 5m candle starting at/after activation with low ≤ zone high and high ≥ zone low | — |
| "Pulling down" filter | latest 2 completed 30m closes strictly declining at touch time | `pullback_declining_closes` |
| Break threshold | latest confirmed 5m swing high whose pivot bar started after zone activation, frozen at touch | — |
| No eligible swing high | skip that touch (zone stays available, attempt not used) | — |
| Touch candle trades below zone low | setup is created and immediately cancelled (attempt used). *JAXON choice: the spec cancels a setup when price trades below the zone; it does not say whether the touch candle counts.* | — |

### 5m structure shift
| Choice | Default | Key |
|---|---|---|
| Confirmation | a later 5m **close** above the frozen threshold (wicks don't count) | — |
| Window | within 6 bars after the touch candle | `confirm_window_bars` |
| Cancel when | price trades below zone low; zone invalidated or expired; 4H bias lost | — |
| Cancel and confirm on the same bar | cancel wins | — |
| Timeout | the attempt is used and the setup resets | — |
| Session end | an unconfirmed setup is cancelled at the session close | `cancel_setup_at_session_end` |

### VWAP target
| Choice | Default | Key |
|---|---|---|
| Formula | cumulative((H+L+C)/3 × volume) / cumulative(volume) on the **finest bars in the CSV**. This is a *bar-based approximation*, not tick VWAP. | — |
| Reset | at each configured session open (`session.vwap_reset: session_open`) | — |
| Unusable VWAP → no trade | volume type not in `vwap_allowed_volume_types` (default: exchange, consolidated; tick volume excluded); any missing (NaN) volume in the session; data missing at session open; any intra-session data gap; zero cumulative volume | `vwap_allowed_volume_types`, `vwap_require_complete_session` |
| Target | VWAP snapshot at the confirmation close, floored to tick, **fixed** (never trails) | — |

### Entry, stop, target
| Choice | Default | Key |
|---|---|---|
| Signal time | confirmation candle close | — |
| Entry | market at the open of the next base bar, which must start exactly at the signal time and be in the same session | — |
| Stop reference | lowest low from the touch candle through the confirmation candle. *This stands in for the video's unspecified "swing low", without waiting for a future pivot.* | — |
| Stop | reference − 1 tick | `stop_offset_ticks` |
| Minimum reward/risk | 1.5, before costs, checked at the proposed entry **and** again at the real entry price | `min_reward_risk` |
| Costs check | target proceeds after modeled round-trip costs must be positive | — |
| Rechecks at entry | bias, zone validity, entry window, daily lockout, gap below stop or above target, reward/risk, sizing, limits | — |
| Not allowed | averaging down, pyramiding, martingale, partial exits, trailing stops | — |
| Entry window | no new entries in the last 30 minutes of the session | `session.last_entry_minutes_before_close` |
| Session flatten | market exit decided at `flatten_minutes_before_close` (5) and filled at the next real bar's open. If no bar exists before the close, it fills at the next bar that does exist and the trade is flagged `flatten_filled_next_session`. | `session.flatten_minutes_before_close` |

### Sizing and limits (`risk:`)
| Choice | Default |
|---|---|
| Risk per trade | 0.25% of current marked equity |
| Risk per unit | (entry − stop) × value_per_point + exit spread/slippage + commissions on both sides |
| Rounding | down to `lot_size`; skip if below `min_quantity` (`min_size_exceeds_risk_budget`) |
| Per-order minimum commission | included: size shrinks until total stop-out cost fits the budget |
| Buying power | notional ≤ equity × `max_leverage`; if `initial_margin_per_unit` is set, margin ≤ equity × `max_margin_usage` |
| Max daily drawdown | 1% from the start-of-risk-day equity, marked to market **at each base bar close** (intrabar breaches are seen at the bar close) |
| On breach | cancel pending entries; flatten at the next bar's open; block entries until the next risk day; record overshoot |
| Entries per session | 3 filled entries per risk day |
| Open positions | 1 across the portfolio (pending entries count) |
| Aggregate open stop risk | 0.5% of equity |
| Risk day | `portfolio.risk_day_timezone` + `risk_day_reset`. With several instruments, one portfolio-wide day is used instead of each exchange's own session. |

### Execution model (`execution:` and `costs:`)
| Choice | Default |
|---|---|
| Spread | half the configured spread is paid on each side |
| Slippage | adverse ticks on market and stop fills; none on limit fills |
| Stop trigger | bar low ≤ stop (trade price). A gap open below the stop fills at the open, not at the stop |
| Target (sell limit) | fills when high − half-spread ≥ target **+ 1 tick** (`limit_requires_trade_through`); a gap open above it fills at the open |
| Same-bar stop and target | stop first, flagged `ambiguous_bar_stop_first`; finer CSV bars reduce how often this happens |
| Activation | no order fills on a bar that starts before the order was placed. Stop and target orders are active from the entry fill, so they are checked on the entry bar's high/low too |
| Intrabar fill timestamps | recorded as the bar close, flagged `intrabar_fill_time_is_bar_close` |

### Data handling (`bars:`)
| Choice | Default |
|---|---|
| CSV timestamps | bar **open** (`timestamp_convention`); naive timestamps need `naive_timestamp_timezone` |
| Out-of-order rows | sorted, with a warning |
| Exact duplicates | dropped with a warning; conflicting duplicates are an error |
| Rows outside the configured sessions | dropped with a warning (`drop_outside_session`) |
| Missing intervals | reported; higher-timeframe bars missing any base bar are **not built**; nothing is forward-filled |
| 4H alignment | `session_open` (equities: 09:30–13:30, 13:30–16:00 ET) or `utc_epoch` (crypto) |
| Stale feed | any missing base bar inside a session blocks new entries until `stale_recovery_bars` fresh bars arrive |
| Warmup | no explicit bar count: each rule waits for what it needs (2+2 confirmed 4H swings, ATR after 15 bars of 30m data, a 5m pivot after activation, VWAP from session open) |

### Evaluation
| Choice | Default |
|---|---|
| Split | development < `development_end` ≤ validation < `validation_end` ≤ out-of-sample; trades are assigned by entry time |
| Tuning | sensitivity runs feed the engine only bars before `validation_end` (asserted in code and tested) |
| Inference | below `min_trades_for_inference` (100) trades, reports say no edge can be claimed |

## 3. Not established and not claimed

- No result in this repository is evidence that the setup is profitable.
  Every bundled data file is **synthetic** and every report built from it says so.
- The scripted scenario fixture (`SYNTHETIC_scenario_5m.csv`) was hand-built to pass
  every rule. Its two trades demonstrate the mechanics, nothing more.
- Different asset classes are deliberately not treated alike: sessions, volume
  quality (consolidated vs single-venue vs tick volume), contract multipliers, lot
  sizes and margin are all instrument metadata, and VWAP is refused where the
  volume type is unsuitable.
