# Integration interfaces (for later versions)

v1 does **not** submit live orders. Nothing in this repository talks to a broker
account. These are the seams a later version would use.

## Market data — `jaxon.execution.base.MarketDataAdapter`
```python
class MarketDataAdapter(ABC):
    def bars(self) -> Iterator[Bar]: ...          # CLOSED base bars, strict time order
    def is_stale(self, now) -> bool: ...          # wall-clock staleness for live feeds
```
Implemented:
- the CSV loader (`jaxon.data.csv_source`, used by backtests and paper replay),
- `jaxon.live_paper.CoinbaseCandleFeed`: polls Coinbase's public candle endpoint, emits
  only fully closed candles, no credentials.

Requirements for any new feed:
- never emit a bar before it has closed; never emit a bar twice (the engine also drops duplicates),
- never fill gaps; report them (the engine treats a gap as a stale feed),
- stamp `Bar.session_id` with the instrument's calendar (`TradingCalendar.session_for`).

## Execution — `jaxon.execution.base.ExecutionAdapter`
```python
class ExecutionAdapter(ABC):
    def submit(self, order: Order) -> None: ...
    def cancel(self, order_id: str, reason: str = "") -> None: ...
    def open_orders(self, symbol=None) -> list[Order]: ...
    def process_bar(self, bar: Bar) -> list[Fill]: ...
```
`PaperBroker` is the only implementation. `LiveBrokerNotImplemented` raises on every call.

A real broker adapter would have to:
- turn `Order` into the broker's order types (market entry; a stop and a limit as an OCO pair),
- report **actual** fills (price, quantity, fee, time) instead of computing them; it must never invent a fill,
- reconcile positions and open orders against the engine after reconnecting, before any new entry,
- respect the `active_from` field (no order sent before its activation time).

## Engine contract
- `TradingEngine.process(bar)` is the only entry point. Backtest, paper replay and live paper all use it.
- State persists through `PaperReplay.save()` / `restore()` (pickle + manifest with config hash,
  engine version, data hashes and snapshot SHA-256). Only load state directories you created.
- The LLM explainer (`jaxon.explain`) reads exported records after the fact. It has no write path
  into the engine.
