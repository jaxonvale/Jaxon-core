# Prop-firm challenge pass rate — backtest study (2025)

**Question:** run the JAXON bot with all of its rules plus the PRO Breakout indicator. What share of
funded-challenge attempts would have passed?

**Answer on the data available: 0%.** No variant passed a Topstep 50K or Apex 50K evaluation in 262
attempts (one attempt started on every trading day of 2025). The exception is a single Apex pass at 4×
the bot's risk with the 4H filter removed: 1 of 262, which is luck, not an edge.

Reproduce: `python -m jaxon prop-study configs/prop_cme_micro_crypto.yaml --out runs/prop_study`

## Setup
| item | value |
|---|---|
| Prices | REAL Binance BTCUSDT and ETHUSDT 1-minute bars, 2025, used as a **proxy** for CME Micro Bitcoin (MBT) and Micro Ether (MET) futures. No CME futures data was reachable from this environment (Yahoo/Stooq blocked). |
| Contract model | MBT 0.1 BTC, tick $0.50; MET 0.1 ETH, tick $0.05; max 50 micros; CME hours (5:00 PM–4:00 PM CT, Sun–Fri); flat by 3:10 PM CT; no entries after 3:00 PM CT |
| Costs (assumed) | 1 tick spread, 1 tick slippage, $1.25 (MBT) / $0.85 (MET) commission per side |
| Bot rules | 4H rising swings, 30m demand zone, 5m structure shift, swing-low stop, 0.25% ($125) risk per trade on the $50K start balance, 3 entries/day, $500 daily lockout, one position at a time |
| Indicator | PRO Breakout port (`jaxon/pro.py`): Signal Adjuster 10, Target Adjuster 1, Stop Adjuster 1, take profit at target #2 |
| Challenges | Topstep 50K Combine: $3,000 target, $2,000 EOD-trailing max loss checked in real time, 50% consistency, no time limit. Apex 50K (2026 rules): $3,000 target, $2,000 trailing (EOD or intraday) that stops at +$100, 30 calendar days. **Rules are from public 2026 summaries. Verify them with the firm.** |

## Results (262 attempts per row; pass rate = passed ÷ (passed + failed + expired))
| variant | risk/trade | trades in 2025 | win rate | net R/trade | Topstep 50K | Apex EOD | Apex intraday |
|---|---|---|---|---|---|---|---|
| A. Bot rules, VWAP target | 0.25% | 0 | – | – | never reaches target (262 still open) | 0% (240 expired) | 0% (240 expired) |
| B. Bot rules + PRO filter, PRO target | 0.25% | 0 | – | – | never reaches target | 0% | 0% |
| **C. PRO signals + bot rules (4H filter, risk limits)** | **0.25%** | **164** | **29.9%** | **−0.40** | **0% (205 failed, 57 open)** | **0% (240 expired)** | **0% (240 expired)** |
| C | 0.50% | 161 | 30.4% | −0.38 | 0% (235 failed) | 0% (45 failed, 195 expired) | 0% |
| C | 1.00% | 137 | 27.7% | −0.39 | 0% (241 failed) | 0% (101 failed) | 0% |
| D. PRO signals, no 4H filter | 0.25% | 526 | 28.9% | −0.37 | 0% (241 failed) | 0% | 0% |
| D | 1.00% | 347 | 21.9% | −0.35 | 0% | 0.4% (1 pass) | 0.4% (1 pass) |

At 0.25% risk the Apex attempts mostly *expire*. $125 per trade with a few trades a week can't make $3,000 in 30 days, even at breakeven.

## Where the losses come from (variant C, 0.25%)
| | trades | win rate | price move only | fees | spread + slippage | net |
|---|---|---|---|---|---|---|
| Micro BTC | 113 | 30.1% | −0.09R | −0.13R | −0.07R | −0.29R |
| Micro ETH | 51 | 29.4% | −0.06R | −0.55R | −0.04R | −0.65R |

- **The signals lose money before costs.** With a 2R target, a ~30% win rate is below the 33% needed just to break even.
- Costs then dominate, especially on Micro ETH: a $125 risk needs about 41 contracts, and per-contract fees are about half the risk.
- The original bot rules (A, B) never trigger under CME session hours, so they can't pass.

## Limits of this study
- Crypto prices standing in for CME micro crypto futures, one year (2025). E-mini/Micro ES and NQ, which most challenge traders use, are **untested**. Supply 1-minute CSVs and rerun the same command.
- Attempts that start on neighbouring days share trades. The 262 attempts are not independent.
- Commission levels and firm rules are assumptions. Each firm's optional daily loss limit isn't modeled; the bot's own $500 daily lockout is stricter than it.
- A 0% pass rate on this data does **not** say what would happen on other markets or years. It does say the current rules show no edge here.
