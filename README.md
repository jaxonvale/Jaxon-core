# Jaxon Core

This repo holds the core brain & memory system for Jaxon Vale — AI partner.

## JAXON trading research + paper-trading agent (`jaxon/`)

Version 1: **long-only, backtesting and paper trading only. No live orders.**

Setup (from a video; all thresholds are JAXON's own assumptions, see
[`docs/assumptions.md`](docs/assumptions.md)):

1. **4-hour chart going up**: last two confirmed 4H swing highs rising and last two swing lows rising.
2. **30-minute pullback into demand**: a demand zone (base candle before a 1.5×ATR bullish displacement), touched by a 5m candle while 30m closes are declining.
3. **5-minute bullish structure shift**: a 5m close above the swing high frozen at the touch, within 6 bars.
4. **Long entry** at the next bar's open, after rechecking every rule at the real fill price.
5. **Target = session VWAP** snapshot at confirmation. **Stop = swing low** (lowest low from touch to confirmation, minus 1 tick).

> Nothing here shows the setup makes money. The bundled data is synthetic and labeled so.
> Real-data results are in [`docs/results_binance_2025.md`](docs/results_binance_2025.md).

### Install
```bash
python3 -m pip install -r requirements.txt      # Python 3.11+
python3 -m pytest -q                            # full test suite
```

### Commands
```bash
# validate a config and its CSV data
python -m jaxon validate-config configs/example_scenario.yaml
python -m jaxon validate-data   configs/example_equity.yaml

# CSV backtest (no credentials) -> trades/setups CSV+JSON, equity, journal, summary, HTML chart
python -m jaxon backtest configs/example_scenario.yaml --out runs/scenario

# chronological dev / validation / out-of-sample split + parameter sensitivity (dev+validation only)
python -m jaxon evaluate configs/example_equity.yaml --out runs/equity_eval

# deterministic paper replay, persisted and resumable (kill it any time, rerun the same command)
python -m jaxon replay configs/example_scenario.yaml --state-dir runs/paper --max-bars 1000
python -m jaxon replay configs/example_scenario.yaml --state-dir runs/paper
python -m jaxon status --state-dir runs/paper

# explain a logged setup (deterministic record; --llm adds prose via the Anthropic API, read-only)
python -m jaxon explain --run runs/scenario --setup <setup_id from setups.csv> [--llm]

# regenerate the labeled synthetic fixtures
python -m jaxon make-fixtures
```

### Real historical data (public, no credentials)
```bash
python -m jaxon download --source binance_vision --symbol BTCUSDT --interval 1m \
    --start 2025-01 --end 2025-12 --out data/real/BTCUSDT_1m_2025.csv
python -m jaxon download --source binance_vision --symbol ETHUSDT --interval 1m \
    --start 2025-01 --end 2025-12 --out data/real/ETHUSDT_1m_2025.csv
python -m jaxon evaluate configs/real_binance_btc_eth.yaml --out runs/real_2025
```
Binance archives are checked against their published SHA-256, and the checksums are written into the CSV header.
`--source coinbase --symbol BTC-USD` uses Coinbase's public candles instead.
Real data files are not committed (`data/real/` is git-ignored).

### Live paper trading on a real feed (no orders sent)
```bash
python -m jaxon paper-live configs/live_coinbase_btc.yaml --state-dir runs/live_paper
```
It polls Coinbase's public 5-minute candles and uses only closed candles. It loads 12 days of history to warm up the indicators, then
runs each new candle through the same engine as the backtest. State is saved after every candle, so it resumes after a restart.
A feed that goes quiet is logged as stale, and new entries are blocked until fresh bars arrive.

### Layout
| module | role |
|---|---|
| `jaxon/config.py` | strict YAML config; rejects missing instrument metadata and unknown keys |
| `jaxon/calendar.py` | exchange sessions (DST-aware, overnight sessions, holidays, early closes), risk day |
| `jaxon/data/csv_source.py` | CSV import, UTC normalization, validation report, provenance |
| `jaxon/data/download.py` | real-data downloaders (Binance archive, Coinbase) |
| `jaxon/data/fixtures.py` | synthetic fixtures (random and scripted scenario) |
| `jaxon/bars.py` | 5m/30m/4H aggregation; only complete, closed bars |
| `jaxon/indicators.py` | strict pivots, Wilder ATR, session VWAP, stale-feed monitor |
| `jaxon/strategy.py` | the rules and the per-symbol state machine (no broker access) |
| `jaxon/risk.py` | position sizing and limit checks |
| `jaxon/execution/` | adapter interfaces + deterministic paper execution model |
| `jaxon/engine.py` | the one event loop used by backtest, replay and live paper |
| `jaxon/replay.py`, `jaxon/live_paper.py` | persisted paper modes with restart recovery |
| `jaxon/evaluation.py`, `jaxon/report.py`, `jaxon/chart.py` | splits, sensitivity, metrics, exports, chart |
| `jaxon/explain.py` | optional LLM explanation of logged decisions (cannot change any decision) |

Docs: [`assumptions.md`](docs/assumptions.md) (video rules vs. design choices) ·
[`csv_schema.md`](docs/csv_schema.md) · [`integration.md`](docs/integration.md) (broker/feed interfaces for later).
