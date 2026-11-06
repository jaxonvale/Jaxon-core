"""Incremental indicators. Each is fed closed bars in time order only."""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import pandas as pd

from .models import Bar, Pivot


class PivotTracker:
    """Strict pivots: the middle bar's high (low) must be strictly greater (less)
    than every one of the `left` + `right` neighbours. Known only once the last
    right-hand bar has closed."""

    def __init__(self, left: int = 2, right: int = 2, timeframe: int = 0):
        self.left, self.right, self.tf = left, right, timeframe
        self.window: deque[Bar] = deque(maxlen=left + right + 1)
        self.highs: list[Pivot] = []
        self.lows: list[Pivot] = []

    def on_bar(self, bar: Bar) -> list[Pivot]:
        self.window.append(bar)
        if len(self.window) < self.window.maxlen:
            return []
        mid = self.window[self.left]
        others = [b for i, b in enumerate(self.window) if i != self.left]
        new = []
        if all(mid.high > b.high for b in others):
            p = Pivot("high", mid.high, mid.start, bar.end, self.tf)
            self.highs.append(p)
            new.append(p)
        if all(mid.low < b.low for b in others):
            p = Pivot("low", mid.low, mid.start, bar.end, self.tf)
            self.lows.append(p)
            new.append(p)
        return new

    def last(self, kind: str, n: int, as_of: pd.Timestamp) -> list[Pivot]:
        src = self.highs if kind == "high" else self.lows
        out = []
        for p in reversed(src):
            if p.available_at <= as_of:
                out.append(p)
                if len(out) == n:
                    break
        return out[::-1]


class WilderATR:
    """ATR seeded with the simple mean of the first `period` true ranges, then
    Wilder-smoothed. Available after period + 1 bars (each TR needs a prior close)."""

    def __init__(self, period: int = 14):
        self.period = period
        self.prev_close: Optional[float] = None
        self.trs: list[float] = []
        self.value: Optional[float] = None
        self.available_at: Optional[pd.Timestamp] = None

    def update(self, bar: Bar) -> Optional[float]:
        if self.prev_close is not None:
            tr = max(bar.high - bar.low, abs(bar.high - self.prev_close), abs(bar.low - self.prev_close))
            if self.value is None:
                self.trs.append(tr)
                if len(self.trs) == self.period:
                    self.value = sum(self.trs) / self.period
            else:
                self.value = (self.value * (self.period - 1) + tr) / self.period
            if self.value is not None:
                self.available_at = bar.end
        self.prev_close = bar.close
        return self.value


class SessionVWAP:
    """Bar-based VWAP approximation: cum(typical price * volume) / cum(volume),
    reset at each session open, built from the finest bars available."""

    def __init__(self, require_complete_session: bool = True, volume_allowed: bool = True):
        self.require_complete = require_complete_session
        self.volume_allowed = volume_allowed
        self.session_id: Optional[str] = None
        self.cum_pv = 0.0
        self.cum_v = 0.0
        self.invalid_reason = ""
        self.last_end: Optional[pd.Timestamp] = None

    def on_base(self, bar: Bar, session_open: pd.Timestamp) -> None:
        if bar.session_id != self.session_id:
            self.session_id, self.cum_pv, self.cum_v = bar.session_id, 0.0, 0.0
            self.invalid_reason = "" if self.volume_allowed else "volume_type_not_allowed"
            if self.require_complete and bar.start != session_open and not self.invalid_reason:
                self.invalid_reason = "session_open_data_missing"
        elif self.require_complete and self.last_end is not None and bar.start != self.last_end \
                and not self.invalid_reason:
            self.invalid_reason = "intra_session_data_gap"
        if math.isnan(bar.volume) and not self.invalid_reason:
            self.invalid_reason = "missing_volume"
        if not math.isnan(bar.volume):
            tp = (bar.high + bar.low + bar.close) / 3.0
            self.cum_pv += tp * bar.volume
            self.cum_v += bar.volume
        self.last_end = bar.end

    def value(self) -> tuple[Optional[float], str]:
        if self.invalid_reason:
            return None, self.invalid_reason
        if self.cum_v <= 0:
            return None, "zero_cumulative_volume"
        return self.cum_pv / self.cum_v, ""


class FeedMonitor:
    """Flags stale data: a gap of missing base bars inside a session (backtest/replay)
    or no new bar for too long against a wall clock (live paper)."""

    def __init__(self, base_minutes: int, stale_gap_bars: int = 1, recovery_bars: int = 6):
        self.step = pd.Timedelta(minutes=base_minutes)
        self.stale_gap_bars, self.recovery = stale_gap_bars, recovery_bars
        self.last_end: Optional[pd.Timestamp] = None
        self.last_session: Optional[str] = None
        self.fresh = recovery_bars
        self.last_gap: Optional[tuple] = None

    def on_base(self, bar: Bar) -> bool:
        """Returns True if this bar arrived after a stale gap."""
        stale = False
        if self.last_end is not None and bar.session_id == self.last_session:
            missing = (bar.start - self.last_end) / self.step
            if missing > 0 and missing >= self.stale_gap_bars:
                stale = True
                self.fresh = 0
                self.last_gap = (self.last_end, bar.start)
        if not stale:
            self.fresh += 1
        self.last_end, self.last_session = bar.end, bar.session_id
        return stale

    @property
    def entries_blocked(self) -> bool:
        return self.fresh < self.recovery

    def is_stale(self, now: pd.Timestamp, max_age: pd.Timedelta) -> bool:
        return self.last_end is None or now - self.last_end > max_age
