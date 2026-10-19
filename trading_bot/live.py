"""Live / paper runner (ccxt). Paper mode by default — nothing is sent to the exchange
unless --live is passed AND API keys are set in EXCHANGE_API_KEY / EXCHANGE_SECRET."""
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

from .config import StrategyConfig
from .data import fetch_ccxt
from .strategy import STAGES, generate

log = logging.getLogger("trading_bot.live")


class LiveRunner:
    def __init__(self, symbol: str, exchange: str = "binance", cfg: StrategyConfig | None = None,
                 paper: bool = True, paper_equity: float = 10_000.0, poll_seconds: int = 15,
                 state_dir: str = "bot_state"):
        import ccxt

        self.symbol, self.exchange_id, self.cfg = symbol, exchange, cfg or StrategyConfig()
        self.paper, self.poll = paper, poll_seconds
        opts = {"enableRateLimit": True}
        if not paper:
            key, secret = os.environ.get("EXCHANGE_API_KEY"), os.environ.get("EXCHANGE_SECRET")
            if not key or not secret:
                raise SystemExit("Live mode needs EXCHANGE_API_KEY and EXCHANGE_SECRET env vars.")
            opts.update(apiKey=key, secret=secret)
        self.ex = getattr(ccxt, exchange)(opts)
        self.dir = Path(state_dir)
        self.dir.mkdir(exist_ok=True)
        tag = symbol.replace("/", "_")
        self.state_file = self.dir / f"{tag}_state.json"
        self.trades_file = self.dir / f"{tag}_trades.csv"
        self.state = {"position": None, "last_signal": None, "paper_equity": paper_equity}
        if self.state_file.exists():
            self.state.update(json.loads(self.state_file.read_text()))

    # ---------- helpers ----------
    def _save(self):
        self.state_file.write_text(json.dumps(self.state, indent=2, default=str))

    def _equity(self) -> float:
        if self.paper:
            return self.state["paper_equity"]
        quote = self.symbol.split("/")[1].split(":")[0]
        return float(self.ex.fetch_balance()["free"].get(quote, 0))

    def _price(self) -> float:
        return float(self.ex.fetch_ticker(self.symbol)["last"])

    def _record(self, row: dict):
        pd.DataFrame([row]).to_csv(self.trades_file, mode="a", header=not self.trades_file.exists(), index=False)

    # ---------- trading ----------
    def check_entry(self):
        df5 = fetch_ccxt(self.symbol, "5m", self.exchange_id, limit=700)
        df30 = fetch_ccxt(self.symbol, "30m", self.exchange_id, limit=300)
        df4h = fetch_ccxt(self.symbol, "4h", self.exchange_id, limit=300)
        f, signals = generate(df5, self.cfg, df30, df4h)
        last_t = df5.index[-1]
        log.info("%s  %s  close=%.4f vwap=%.4f  stage: %s", self.symbol, last_t, f.close.iloc[-1],
                 f.vwap.iloc[-1], STAGES[int(f.stage.iloc[-1])])
        if not signals or signals[-1].time != last_t or str(last_t) == self.state["last_signal"]:
            return
        s = signals[-1]
        self.state["last_signal"] = str(last_t)
        price = self._price()
        if price <= s.stop:
            log.warning("Signal skipped: price %.4f already under stop %.4f", price, s.stop)
            return self._save()
        risk = price - s.stop
        equity = self._equity()
        qty = min(equity * self.cfg.risk_per_trade / risk, equity / price)
        if not self.paper:
            qty = float(self.ex.amount_to_precision(self.symbol, qty))
            order = self.ex.create_market_buy_order(self.symbol, qty)
            price = float(order.get("average") or price)
        self.state["position"] = {
            "entry": price, "qty": qty, "stop": s.stop, "risk": risk, "be_moved": False,
            "target": price + self.cfg.reward_r * risk, "opened": str(pd.Timestamp.now(tz="UTC")),
            "bars": 0,
        }
        log.info("BUY %s qty=%.6f @ %.4f  stop=%.4f  target=%.4f %s", self.symbol, qty, price, s.stop,
                 self.state["position"]["target"], "(paper)" if self.paper else "")
        self._save()

    def manage(self, new_bar: bool):
        p = self.state["position"]
        price = self._price()
        if new_bar:
            p["bars"] += 1
        reason = None
        if price <= p["stop"]:
            reason = "breakeven" if p["be_moved"] else "stop"
        elif price >= p["target"]:
            reason = "target"
        elif self.cfg.max_hold_bars and p["bars"] >= self.cfg.max_hold_bars:
            reason = "time"
        elif self.cfg.breakeven_at_r and not p["be_moved"] and price >= p["entry"] + self.cfg.breakeven_at_r * p["risk"]:
            p["stop"], p["be_moved"] = p["entry"], True
            log.info("Stop moved to breakeven %.4f", p["entry"])
        if reason:
            if not self.paper:
                order = self.ex.create_market_sell_order(self.symbol, float(self.ex.amount_to_precision(self.symbol, p["qty"])))
                price = float(order.get("average") or price)
            fees = (p["entry"] + price) * p["qty"] * self.cfg.fee_rate
            pnl = (price - p["entry"]) * p["qty"] - fees
            if self.paper:
                self.state["paper_equity"] += pnl
            log.info("SELL %s @ %.4f  %s  pnl=%.2f  R=%.2f", self.symbol, price, reason, pnl,
                     (price - p["entry"]) / p["risk"])
            self._record({**p, "exit": price, "reason": reason, "pnl": pnl,
                          "closed": str(pd.Timestamp.now(tz="UTC"))})
            self.state["position"] = None
        self._save()

    def run(self):
        log.info("Starting %s on %s (%s)", self.symbol, self.exchange_id, "PAPER" if self.paper else "LIVE")
        last_bar = None
        while True:
            try:
                now = pd.Timestamp.now(tz="UTC")
                bar = now.floor("5min")
                new_bar = bar != last_bar and (now - bar).total_seconds() >= 5   # let the candle settle
                if self.state["position"]:
                    self.manage(new_bar)
                elif new_bar:
                    self.check_entry()
                if new_bar:
                    last_bar = bar
            except Exception as e:  # keep the bot alive through network hiccups
                log.exception("loop error: %s", e)
            time.sleep(self.poll)
