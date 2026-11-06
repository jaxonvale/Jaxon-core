# TradingView files

| file | what it is | how to add |
|---|---|---|
| `jaxon_signals_indicator.pine` | **Indicator**: BUY/SELL labels, entry/stop/target lines, PRO cloud, confirmation scores, supply/demand zones, alerts. Modes: Zone Tap → VWAP (default) / Zone Tap + PRO filter / PRO Breakout / PRO Breakout + 4H filter | Pine Editor → Open → **New blank indicator** → paste → Save → Add to chart |
| `jaxon_zone_tap_strategy.pine` | **Strategy**: Zone Tap → VWAP with real resting limit orders at the zone, for the Strategy Tester | Pine Editor → Open → **New blank strategy** → paste → Save → Add to chart |
| `jaxon_pro_breakout_strategy.pine` | **Strategy**: the same rules with simulated orders, for the Strategy Tester | Pine Editor → Open → **New blank strategy** → paste → Save → Add to chart |

Indicator notes:
- BUY/SELL print on closed bars only. Hover a label to see entry, stop, target 1 and the 8-point confirmation checklist (plus the 4H trend).
- **Zone Tap → VWAP**: last closed 4H candle green + 30m downtrend → BUY when price taps a 30m demand zone (entry at the zone top), stop below the zone, target VWAP. Mirror for SELLs: 4H red + 30m uptrend + tap of supply. Use a 5-minute chart.
- PRO modes use the original 5-step ATR target ladder.
- Alerts: right-click the chart → Add alert → condition "JAXON Signals" → BUY signal / SELL signal / Stop hit / Target hit.
- Backtest result (see `docs/prop_study_2025.md`): on 2025 BTC/ETH, these signals had a negative edge even before costs. Use them for study, not as proven entries.

# JAXON x PRO Breakout — TradingView strategy

File: `jaxon_pro_breakout_strategy.pine` (Pine Script v5, `strategy()`, so it runs in the Strategy Tester).

## Load it
1. TradingView → Pine Editor → **Open → New blank strategy**, delete the template, paste the file, **Save**, **Add to chart**.
2. Use a **5-minute chart**. The table (top right) warns you if you're on another timeframe.
3. Open **Strategy Tester** → Overview / List of Trades.

## Modes (Settings → 1. Mode)
| mode | entries | exits |
|---|---|---|
| JAXON (MTF VWAP bot) | 4H rising swing highs + lows → 30m demand zone touched while 30m closes fall → 5m close above the frozen swing high (within 6 bars) | stop = lowest low since touch − 1 tick; target = session VWAP (or PRO ATR target) |
| PRO Breakout | the indicator's BUY (and SELL if allowed) signals, optional 4H-bias and confirmation filters | indicator's ATR stop and target # (1–5); opposite signal closes |
| JAXON + PRO filter (default) | JAXON setups, only when the PRO cloud is up and confirmations ≥ minimum | as JAXON |

## Settings to change per market
| market | Session timezone | Flatten HHMM | No entries after | Quantity step |
|---|---|---|---|---|
| US stocks (SPY, QQQ) | America/New_York | 1555 | 1530 | 1 |
| CME futures (ES1!, NQ1!) | America/New_York | 1555 | 1530 | 1 |
| Crypto (BTCUSDT) | UTC | 2355 | 2330 | 0.001 |

Also set **Properties → Commission and Slippage** to your broker's numbers. The defaults are 0.05% per side and 2 ticks.

## Matches the Python bot
- Higher-timeframe values only come from **closed** 4H/30m bars (`[1]` + `lookahead_on`, the standard non-repainting pattern).
- Swing points use strict 2-left/2-right pivots, available only after the right-hand bars close.
- One attempt per zone, 20-bar expiry, invalidation on a 30m close below the zone, cancel beats confirm on the same bar.
- Risk 0.25% of equity per trade, max 3 entries per day, 1% max daily loss (TradingView's built-in daily lockout), flatten before session end.

## Differs from the Python bot (TradingView limits)
- **Entry rechecks:** the bot rechecks reward/risk at the real fill price. Here the entry is a buy-limit at the highest price that still gives the minimum R:R, valid for one bar only. A gap above it isn't filled. If price dips to the limit later in that bar, TradingView fills it there, which the bot wouldn't.
- **Intrabar order:** TradingView decides whether stop or target filled first. `use_bar_magnifier=true` uses lower-timeframe data where your plan allows it. The bot always assumes the stop hit first when a bar touches both.
- **VWAP:** TradingView's `ta.vwap` from chart bars, reset daily. The bot also refuses VWAP when volume is missing or unsuitable; here it doesn't.
- **Daily lockout:** TradingView's `max_intraday_loss` instead of the bot's mark-to-market check at each bar close.

## Results caveat
On real Binance BTC/ETH 1-minute data for 2025, the strict JAXON rules (VWAP target) produced **zero** trades
(`docs/results_binance_2025.md`). Expect few or no JAXON trades with "VWAP (bot rule)". "PRO ATR target" and
PRO Breakout mode trade far more often, but nothing here has been shown to be profitable.
Tune on one period and check on a later, untouched period.
