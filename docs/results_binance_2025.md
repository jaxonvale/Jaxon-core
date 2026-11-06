# Results: Binance spot BTCUSDT + ETHUSDT, 1-minute bars, calendar 2025

**Real historical data** (data.binance.vision, checksum-verified). Config
`configs/real_binance_btc_eth.yaml` (`real-binance-1+ab6db20efcfe0ecf`), all strategy
parameters at the proposed defaults. Reproduce:

```bash
python -m jaxon download --symbol BTCUSDT --interval 1m --start 2025-01 --end 2025-12 --out data/real/BTCUSDT_1m_2025.csv
python -m jaxon download --symbol ETHUSDT --interval 1m --start 2025-01 --end 2025-12 --out data/real/ETHUSDT_1m_2025.csv
python -m jaxon evaluate configs/real_binance_btc_eth.yaml --out runs/real_2025
```

## Data quality
| | rows | within-session gaps | sessions (UTC days) |
|---|---|---|---|
| BTCUSDT | 525,600 | 0 | 365 |
| ETHUSDT | 525,600 | 0 | 365 |

## Outcome: no trades
| period | dates | trades |
|---|---|---|
| development | 2025-01-01 → 2025-07-01 | 0 |
| validation | 2025-07-01 → 2025-10-01 | 0 |
| out-of-sample | 2025-10-01 → 2026-01-01 | 0 |

### Setup funnel (whole year, both symbols)
| stage | count |
|---|---|
| base bars processed | 1,051,200 |
| 4H bias turned bullish | 161 times (85 BTC, 76 ETH) |
| demand zones activated | 523 |
| displacement without a bearish base (no zone) | 36 |
| touch candles skipped: 30m closes not declining | 111 |
| touch candles skipped: no 5m swing high since activation | 33 |
| setups started (touch + frozen threshold) | 76 |
| cancelled: traded below zone | 33 |
| timed out: no 5m close above threshold in 6 bars | 38 |
| cancelled at session end | 3 |
| **structure shift confirmed** | **2** |
| rejected: VWAP target not above entry | 2 |
| **signals / trades** | **0** |

Both confirmations (ETH 2025-08-06 15:35 UTC, BTC 2025-09-03 17:15 UTC) had session
VWAP *below* the entry price, so "target VWAP" was not a profit target.

### Sensitivity (development + validation only; out-of-sample never fed to the engine)
Only one variant produced any trade: `displacement_atr_mult = 1.25` → 1 trade in
development, −1.0R. Every other variant: 0 trades.

## What this means
- With the stated mechanical definitions, the setup almost never fires on 2025 BTC/ETH.
  There is no sample to evaluate, so nothing can be said about profitability, either way.
- Two things make it structural, not a matter of tuning:
  1. After a bullish 30m displacement, price usually sits **above** session VWAP. A
     pullback into the base followed by a 5m break upward rarely leaves VWAP 1.5R overhead.
  2. The 6-bar confirmation window and the "no trade below the zone" rule end most
     setups (71 of 76).
- These results were **not** used to change any default. Changing the rules because of
  them would be fitting to this data; any change must be argued in development/validation
  and then judged once on a fresh out-of-sample period.

## Candidate hypotheses (untested; dev/validation only if pursued)
- The video's VWAP may mean a different anchor (prior session, weekly, or anchored to the
  4H swing low), which would sit higher during pullbacks.
- Allow entries only when price is below VWAP at the touch (a precondition rather than a
  rejection after the fact).
- Longer confirmation window, or a 30m pullback definition other than two declining closes.
