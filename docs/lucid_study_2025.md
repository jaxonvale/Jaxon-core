# Lucid Trading 25K / 50K — backtested pass rates (2025)

Reproduce:
```bash
python -m jaxon prop-study configs/prop_lucid_25k.yaml --out runs/lucid_25k --presets lucidflex_25k lucidpro_25k luciddaily_25k_intraday
python -m jaxon prop-study configs/prop_lucid_50k.yaml --out runs/lucid_50k --presets lucidflex_50k lucidpro_50k luciddaily_50k_intraday
```

**Result: 0 passes out of 262 attempts for every setup, plan, account size and risk level tested.**

Rules applied: `docs/lucid_rules.md` (official help center, 2026-10-20). One attempt started on every
trading day of 2025, each with a fresh $25K or $50K balance.

## Data and costs
REAL Binance BTCUSDT/ETHUSDT 1-minute bars for 2025, modeled as CME Micro Bitcoin / Micro Ether futures
(0.1 coin, $0.50 / $0.05 ticks). Lucid hours: flat by 4:45 PM ET (bot flattens 4:40 PM ET), trading day
resets 6 PM ET. Contract cap 20 (25K) / 40 (50K) micros. Assumed costs: 1-tick spread, 1-tick slippage,
$1.25 / $0.85 commission per side. **No CME futures data was available here; ES/NQ are untested.**
Lucid's allowed product list was not checked for micro crypto futures.

## Your rule: 4H green candle + 30m downtrend + tap of 30m demand → VWAP target (50K, 0.25% risk)
| | value |
|---|---|
| Trades in 2025 (BTC + ETH) | 16 |
| Win rate | 18.8% |
| Avg result | −0.60R per trade |
| Stopped out | 12 of 16, most within 3–40 minutes |
| Stop distance | 0.2–0.8% below entry (the zone is thin) |
| Planned reward/risk to VWAP | 1.6–6.8 |
| LucidFlex / LucidPro / LucidDaily | 0 passed (0.25%: all attempts still open at year end; 0.5%–1%: 38–162 failed, rest open) |

Variants tried (same 0% result):
| variant | trades | win rate | R/trade |
|---|---|---|---|
| 4H swing uptrend instead of green candle | 25–28 | 15–18% | −0.62 to −0.77 |
| 30m lower highs/lows instead of EMA | 1 | 0% | −1.0 |
| + PRO cloud must be up | 2 | 0% | −1.0 |
| PRO Breakout signals + bot rules (for comparison) | 141–169 | 28–30% | −0.38 to −0.40 |

## Why it fails here
1. **The stop is too tight for a zone tap.** Price keeps going through the thin base candle far more often than it bounces.
2. **It rarely sets up.** A green 4H candle, a falling 30m EMA, an untouched demand zone below price, and VWAP at least 1.5R overhead all lined up only ~16 times in a year on two markets.
3. With so few trades, even a positive edge could not reach a $1,250 / $3,000 target at 0.25% risk; at higher risk, the losses hit the max loss limit first.

## Not established
- Behaviour on ES/NQ (most Lucid traders' markets) or other years.
- LucidDaily's news ban is not modeled (results there would be no better).
- 16 trades is far too few to judge any edge either way.
