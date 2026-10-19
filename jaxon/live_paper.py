"""Live PAPER trading on a real public data feed. No orders are sent anywhere.

CoinbaseCandleFeed polls Coinbase Exchange's public candle endpoint (no
credentials), emits only candles that have fully closed, and never fills gaps.
LivePaper bootstraps indicator history from recent candles, then processes each
new closed candle through the same TradingEngine as the backtest, persisting
state after every cycle (restart-safe, same manifest checks as PaperReplay).
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .backtest import LoadedData
from .calendar import TradingCalendar
from .config import Config, InstrumentConfig
from .engine import TradingEngine
from .execution.base import MarketDataAdapter
from .models import Bar
from .replay import PaperReplay

URL = "https://api.exchange.coinbase.com/products/{p}/candles?granularity={g}&start={s}&end={e}"


def _http_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "jaxon-research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


class CoinbaseCandleFeed(MarketDataAdapter):
    def __init__(self, product: str, inst: InstrumentConfig, fetch: Callable = _http_json):
        self.product, self.inst, self._fetch = product, inst, fetch
        self.minutes = inst.bars.base_interval_minutes
        if self.minutes not in (1, 5):
            raise ValueError("Coinbase feed supports 1m or 5m base bars")
        self.step = pd.Timedelta(minutes=self.minutes)
        self.cal = TradingCalendar(inst.session)
        self.last_start: Optional[pd.Timestamp] = None

    def _candles(self, start: pd.Timestamp, end: pd.Timestamp, now: pd.Timestamp) -> list[Bar]:
        rows, s, chunk = [], start, self.step * 300
        while s < end:
            e = min(s + chunk, end)
            rows += self._fetch(URL.format(p=self.product, g=self.minutes * 60, s=s.isoformat(), e=e.isoformat()))
            s = e
            time.sleep(0.35)
        bars = {}
        for t, lo, hi, op, cl, vol in rows:
            st = pd.Timestamp(int(t), unit="s", tz="UTC")
            if st + self.step > now:            # still forming: never emit
                continue
            sess = self.cal.session_for(st)
            if sess is None or st + self.step > sess.close:
                continue
            bars[st] = Bar(self.inst.symbol, st, st + self.step, float(op), float(hi), float(lo), float(cl),
                           float(vol), self.minutes, sess.session_id)
        return [bars[k] for k in sorted(bars)]

    def bootstrap(self, days: float, now: pd.Timestamp | None = None) -> list[Bar]:
        now = now or pd.Timestamp.now(tz="UTC")
        start = (now - pd.Timedelta(days=days)).floor("D")
        out = self._candles(start, now, now)
        if out:
            self.last_start = out[-1].start
        return out

    def poll(self, now: pd.Timestamp | None = None) -> list[Bar]:
        now = now or pd.Timestamp.now(tz="UTC")
        start = (self.last_start + self.step) if self.last_start is not None else now - self.step * 50
        new = [b for b in self._candles(start, now, now) if self.last_start is None or b.start > self.last_start]
        if new:
            self.last_start = new[-1].start
        return new

    def bars(self):
        return iter(self.poll())

    def is_stale(self, now: pd.Timestamp) -> bool:
        if self.last_start is None:
            return True
        return now - (self.last_start + self.step) > self.step * 2 + pd.Timedelta(seconds=60)


class LivePaper(PaperReplay):
    def __init__(self, cfg: Config, state_dir: str | Path, feeds: dict[str, CoinbaseCandleFeed]):
        self.cfg = cfg
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.snap = self.dir / "engine_state.pkl"
        self.manifest = self.dir / "manifest.json"
        self.journal_path = self.dir / "journal.jsonl"
        self.feeds = feeds
        prov = {s: {"path": f"live:coinbase:{f.product}", "sha256": f"live:coinbase:{f.product}",
                    "synthetic": False, "rows_in_file": 0, "header_comments": [], "symbol": s}
                for s, f in feeds.items()}
        self.data = LoadedData([], prov, {}, False)
        self.engine: Optional[TradingEngine] = None
        self.resumed = False
        self._stale_logged: set = set()

    def _process(self, bars: list[Bar]) -> int:
        n = 0
        for b in sorted(bars, key=lambda x: (x.start, x.symbol)):
            last = self.engine.rt[b.symbol].last_start
            if last is not None and b.start <= last:
                continue
            if self.engine.last_event_time is not None and b.start < self.engine.last_event_time:
                continue                      # late bar for one symbol after another moved on: drop, never reorder
            self.engine.process(b)
            n += 1
        return n

    def start(self, bootstrap_days: float = 12) -> dict:
        if not self.restore():
            self.engine = TradingEngine(self.cfg, self.data.provenance)
            hist = [b for f in self.feeds.values() for b in f.bootstrap(bootstrap_days)]
            n = self._process(hist)
            self.engine.journal.log(self.engine.last_event_time, "live_bootstrap", bars=n, days=bootstrap_days)
        else:   # resume: feeds continue after the last processed bar
            for s, f in self.feeds.items():
                f.last_start = self.engine.rt[s].last_start
        self.save()
        return self.status()

    def cycle(self, now: pd.Timestamp | None = None) -> dict:
        now = now or pd.Timestamp.now(tz="UTC")
        new = [b for f in self.feeds.values() for b in f.poll(now)]
        n = self._process(new)
        for s, f in self.feeds.items():
            if f.is_stale(now) and s not in self._stale_logged:
                self.engine.journal.log(now, "feed_stale_wallclock", s, last_bar=f.last_start)
                self._stale_logged.add(s)
            elif not f.is_stale(now):
                self._stale_logged.discard(s)
        self.save()
        out = self.status()
        out["new_bars"] = n
        return out

    def status(self) -> dict:
        e = self.engine
        return {"bars_total": e.processed, "last_bar": {s: str(rt.last_start) for s, rt in e.rt.items()},
                "states": {s: rt.core.state.value for s, rt in e.rt.items()},
                "bias": {s: rt.core.bias for s, rt in e.rt.items()},
                "open_trades": [rt.trade.as_record() for rt in e.rt.values() if rt.trade is not None],
                "closed_trades": len(e.trades), "equity": round(e.equity(), 2), "resumed": self.resumed}

    def run_forever(self, max_cycles: int | None = None, printer=print):
        printer(json.dumps(self.start(), default=str))
        step = min(f.step for f in self.feeds.values())
        k = 0
        while max_cycles is None or k < max_cycles:
            now = pd.Timestamp.now(tz="UTC")
            nxt = now.floor(step) + step + pd.Timedelta(seconds=8)   # a few seconds after the candle closes
            time.sleep(max(0.0, (nxt - now).total_seconds()))
            try:
                printer(json.dumps(self.cycle(), default=str))
            except Exception as ex:      # network trouble: keep going, the feed-stale logic blocks entries
                printer(json.dumps({"error": f"{type(ex).__name__}: {ex}"}))
            k += 1
