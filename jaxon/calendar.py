"""Trading calendar: exchange sessions in local time, converted to UTC (DST-aware)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd

from .config import PortfolioConfig, SessionConfig


@dataclass(frozen=True)
class Session:
    session_id: str          # ISO date of the session's close (local)
    open: pd.Timestamp       # UTC
    close: pd.Timestamp      # UTC
    flatten_at: pd.Timestamp
    last_entry_at: pd.Timestamp


def _t(s: str) -> tuple[int, int]:
    h, m = (int(x) for x in s.split(":"))
    return h, m


class TradingCalendar:
    def __init__(self, cfg: SessionConfig):
        self.cfg = cfg
        self.tz = ZoneInfo(cfg.timezone)
        self.holidays = {str(d) for d in cfg.holidays}
        self.early = {str(k): v for k, v in cfg.early_closes.items()}
        self._cache: dict[dt.date, Session | None] = {}

    def _local(self, d: dt.date, hm: tuple[int, int]) -> pd.Timestamp:
        h, m = hm
        if h == 24:
            d, h = d + dt.timedelta(days=1), 0
        return pd.Timestamp(dt.datetime(d.year, d.month, d.day, h, m), tz=self.tz).tz_convert("UTC")

    def session_by_close_date(self, d: dt.date) -> Session | None:
        if d in self._cache:
            return self._cache[d]
        s = None
        if d.weekday() in self.cfg.trading_weekdays and d.isoformat() not in self.holidays:
            o, c = _t(self.cfg.open), _t(self.cfg.close)
            close_hm = _t(self.early[d.isoformat()]) if d.isoformat() in self.early else c
            overnight = c[0] != 24 and o >= c
            open_ts = self._local(d - dt.timedelta(days=1) if overnight else d, o)
            close_ts = self._local(d, close_hm)
            if close_ts > open_ts:
                s = Session(
                    session_id=d.isoformat(), open=open_ts, close=close_ts,
                    flatten_at=close_ts - pd.Timedelta(minutes=self.cfg.flatten_minutes_before_close),
                    last_entry_at=close_ts - pd.Timedelta(minutes=self.cfg.last_entry_minutes_before_close),
                )
        self._cache[d] = s
        return s

    def session_for(self, ts: pd.Timestamp) -> Session | None:
        """The session containing instant `ts` (open <= ts < close), else None."""
        local_date = ts.tz_convert(self.tz).date()
        for d in (local_date, local_date + dt.timedelta(days=1), local_date - dt.timedelta(days=1)):
            s = self.session_by_close_date(d)
            if s is not None and s.open <= ts < s.close:
                return s
        return None

    def next_session_after(self, ts: pd.Timestamp, max_days: int = 15) -> Session | None:
        d = ts.tz_convert(self.tz).date()
        for k in range(max_days):
            s = self.session_by_close_date(d + dt.timedelta(days=k))
            if s is not None and s.open >= ts:
                return s
        return None


class RiskDay:
    """Portfolio-level day used by the daily-loss and entries-per-session limits."""

    def __init__(self, cfg: PortfolioConfig):
        self.tz = ZoneInfo(cfg.risk_day_timezone)
        h, m = _t(cfg.risk_day_reset)
        self.offset = pd.Timedelta(hours=h, minutes=m)

    def day_id(self, ts: pd.Timestamp) -> str:
        local = ts.tz_convert(self.tz).tz_localize(None) - self.offset
        return local.date().isoformat()
