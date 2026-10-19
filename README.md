# Jaxon Core

This repo holds the core brain & memory system for Jaxon Vale — AI partner.

## Trading bot (`trading_bot/`)

Long-only multi-timeframe setup:

1. **4H going up** — close > EMA50, EMA20 > EMA50, latest 4H swing low higher than the last one.
2. **30m pulling back** — close < EMA21 and (EMA9 < EMA21 or a 30m lower high).
3. **5m structure break** — the 5m makes a lower high during the pullback, then a 5m candle closes above it.
4. **VWAP entry** — within 1h of the break, the first 5m candle that dips to session VWAP and closes back above it. Fill at next candle open.
5. **Stop under the lows** — lowest low of the pullback leg minus 0.1×ATR. Skipped if the stop is <0.2% or >3% away.
6. **Exits** — 2R target, stop to breakeven at +1R, 24h max hold. Risk 1% of equity per trade.

Higher-timeframe candles are only used after they close (tested — no lookahead). All settings live in `trading_bot/config.py`.

```bash
pip install -r requirements.txt

python -m trading_bot backtest --demo --chart demo.html                 # synthetic data, no network
python -m trading_bot backtest --symbol BTC/USDT --days 90 --chart btc.html --trades btc.csv
python -m trading_bot backtest --symbol SPY --source yfinance --tz America/New_York
python -m trading_bot backtest --csv my_5m.csv                          # timestamp,open,high,low,close,volume

python -m trading_bot scan --symbols BTC/USDT ETH/USDT SOL/USDT         # where each chart is in the setup
python -m trading_bot live --symbol BTC/USDT                            # paper trading
EXCHANGE_API_KEY=... EXCHANGE_SECRET=... python -m trading_bot live --symbol BTC/USDT --live   # real orders
```

`--chart` writes an HTML chart with candles, VWAP, entries, stops and targets.
Live mode sends market orders and manages stop/target from the bot (checks price every 15s), so keep it running. State and trade log go to `bot_state/`.
For stocks use `--tz America/New_York` so VWAP resets at the US session. yfinance only gives ~60 days of 5m history.
