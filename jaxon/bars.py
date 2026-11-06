"""Incremental resampling of base bars into 5m / 30m / 4h bars.

A higher-timeframe bar is emitted only when the base bar that closes its
bucket arrives AND every expected base bar was present. Buckets with missing
base bars are dropped (reported), never filled.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import math
import pandas as pd

from .calendar import TradingCalendar
from .models import Bar


@dataclass
class DroppedBucket:
    symbol: str
    minutes: int
    start: pd.Timestamp
    end: pd.Timestamp
    have: int
    expected: int


class BarAggregator:
    def __init__(self, symbol: str, minutes: int, base_minutes: int, calendar: TradingCalendar,
                 alignment: str = "session_open"):
        assert minutes % base_minutes == 0
        self.symbol, self.minutes, self.base = symbol, minutes, base_minutes
        self.cal, self.alignment = calendar, alignment
        self.tf = pd.Timedelta(minutes=minutes)
        self._key = None
        self._acc = None
        self.dropped: list[DroppedBucket] = []

    def bucket(self, bar: Bar) -> tuple[pd.Timestamp, pd.Timestamp]:
        s = self.cal.session_by_close_date(dt.date.fromisoformat(bar.session_id))
        if self.alignment == "utc_epoch":
            start = bar.start.floor(self.tf)
            end = min(start + self.tf, s.close)
            start = max(start, s.open)
        else:
            k = (bar.start - s.open) // self.tf
            start = s.open + k * self.tf
            end = min(start + self.tf, s.close)
        return start, end

    def _drop_current(self):
        if self._acc is not None:
            a = self._acc
            self.dropped.append(DroppedBucket(self.symbol, self.minutes, a["start"], a["end"], a["n"], a["expected"]))
        self._acc, self._key = None, None

    def on_base(self, bar: Bar) -> Bar | None:
        if self.minutes == self.base:
            return bar
        start, end = self.bucket(bar)
        if self._key != (start, end):
            self._drop_current()
            expected = int((end - start) / pd.Timedelta(minutes=self.base))
            self._key = (start, end)
            self._acc = {"start": start, "end": end, "expected": expected, "n": 0, "o": bar.open,
                         "h": bar.high, "l": bar.low, "c": bar.close, "v": 0.0, "sid": bar.session_id}
        a = self._acc
        a["n"] += 1
        a["h"] = max(a["h"], bar.high)
        a["l"] = min(a["l"], bar.low)
        a["c"] = bar.close
        a["v"] = a["v"] + bar.volume if not math.isnan(a["v"]) else a["v"]   # NaN propagates
        if bar.end >= end:
            if a["n"] == a["expected"]:
                out = Bar(self.symbol, start, end, a["o"], a["h"], a["l"], a["c"], a["v"], self.minutes, a["sid"])
                self._acc, self._key = None, None
                return out
            self._drop_current()
        return None


def resample_all(bars: list[Bar], minutes: int, base_minutes: int, cal: TradingCalendar,
                 alignment: str = "session_open") -> list[Bar]:
    agg = BarAggregator(bars[0].symbol if bars else "", minutes, base_minutes, cal, alignment)
    out = []
    for b in bars:
        r = agg.on_base(b)
        if r is not None:
            out.append(r)
    return out
