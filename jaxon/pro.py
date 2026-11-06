"""Port of the "PRO Breakout Signals" TradingView indicator (MPL-2.0) trend logic.

Computed bar by bar on closed 5-minute bars, matching Pine's built-ins:
  ta.ema  : SMA-seeded exponential average
  ta.rma  : SMA-seeded Wilder average (used by ta.rsi and ta.atr)
  ta.stdev: population standard deviation
  ta.highest/lowest include the current bar

breakout  = (RSI14 > 60 or RSI14 < 40) and close > EMA(hlc3,L) + stdev(EMA,L) * 0.25*adj
            and low > EMA and high >= highest(high, L)            (L = 4 * adj)
breakdown = mirror image. Trend flips to +1 on breakout, -1 on breakdown.
BUY = flip from -1 to +1; SELL = flip from +1 to -1.
Stop = close - stop_adj * ATR(L); target #n = close + n * target_adj * ATR(L).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional


class _Sma:
    def __init__(self, n):
        self.n, self.buf, self.s = n, deque(), 0.0

    def update(self, x):
        self.buf.append(x)
        self.s += x
        if len(self.buf) > self.n:
            self.s -= self.buf.popleft()
        return self.s / self.n if len(self.buf) == self.n else None


class _Ema:
    def __init__(self, n, wilder=False):
        self.alpha = 1.0 / n if wilder else 2.0 / (n + 1)
        self.sma = _Sma(n)
        self.v: Optional[float] = None

    def update(self, x):
        if x is None:
            return self.v
        if self.v is None:
            self.v = self.sma.update(x)
        else:
            self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v


@dataclass
class ProState:
    trend: int
    long_entry: bool
    short_entry: bool
    atr: Optional[float]
    ema: Optional[float]


class ProBreakout:
    def __init__(self, trend_adj: int = 10):
        self.adj = trend_adj
        self.L = 4 * trend_adj
        self.ema = _Ema(self.L)
        self.ema_hist: deque = deque(maxlen=self.L)
        self.highs: deque = deque(maxlen=self.L)
        self.lows: deque = deque(maxlen=self.L)
        self.up, self.dn = _Ema(14, wilder=True), _Ema(14, wilder=True)
        self.atr = _Ema(self.L, wilder=True)
        self.prev_close: Optional[float] = None
        self.trend = 0
        self.prev_trend = 0

    def update(self, bar) -> ProState:
        hlc3 = (bar.high + bar.low + bar.close) / 3.0
        e = self.ema.update(hlc3)
        if e is not None:
            self.ema_hist.append(e)
        self.highs.append(bar.high)
        self.lows.append(bar.low)
        rsi = None
        if self.prev_close is not None:
            ch = bar.close - self.prev_close
            u, d = self.up.update(max(ch, 0.0)), self.dn.update(max(-ch, 0.0))
            if u is not None and d is not None:
                rsi = 100.0 if d == 0 else (0.0 if u == 0 else 100 - 100 / (1 + u / d))
            tr = max(bar.high - bar.low, abs(bar.high - self.prev_close), abs(bar.low - self.prev_close))
        else:
            tr = bar.high - bar.low
        atr = self.atr.update(tr)
        self.prev_close = bar.close

        brk = dwn = False
        if e is not None and rsi is not None and len(self.ema_hist) == self.L and len(self.highs) == self.L:
            m = sum(self.ema_hist) / self.L
            sd = math.sqrt(max(sum((x - m) ** 2 for x in self.ema_hist) / self.L, 0.0))
            band = sd * 0.25 * self.adj
            rsi_ok = rsi > 60 or rsi < 40
            brk = rsi_ok and bar.close > e + band and bar.low > e and bar.high >= max(self.highs)
            dwn = rsi_ok and bar.close < e - band and bar.high < e and bar.low <= min(self.lows)
        self.prev_trend = self.trend
        self.trend = 1 if brk else (-1 if dwn else self.trend)
        return ProState(self.trend, self.trend == 1 and self.prev_trend == -1,
                        self.trend == -1 and self.prev_trend == 1, atr, e)
