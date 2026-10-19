import math

import pandas as pd
import pytest

from jaxon.bars import BarAggregator
from jaxon.calendar import TradingCalendar
from jaxon.config import SessionConfig
from jaxon.indicators import FeedMonitor, PivotTracker, SessionVWAP, WilderATR
from tests.helpers import bar, mid_bar, ts

NY = SessionConfig(timezone="America/New_York", open="09:30", close="16:00", trading_weekdays=[0, 1, 2, 3, 4])
UTC24 = SessionConfig(timezone="UTC", open="00:00", close="24:00", trading_weekdays=list(range(7)))


def session_bars(cal, day, step=5, mid=100.0):
    s = cal.session_by_close_date(pd.Timestamp(day).date())
    out, t = [], s.open
    while t < s.close:
        out.append(bar(t, mid, mid + 0.1, mid - 0.1, mid, minutes=step, session=s.session_id))
        t += pd.Timedelta(minutes=step)
    return out


# ---------------------------------------------------------------- calendar / resampling
def test_equity_session_dst_and_4h_session_alignment():
    cal = TradingCalendar(NY)
    winter = cal.session_by_close_date(pd.Timestamp("2025-03-07").date())
    summer = cal.session_by_close_date(pd.Timestamp("2025-03-10").date())   # after DST change
    assert winter.open == ts("2025-03-07 14:30") and summer.open == ts("2025-03-10 13:30")
    agg = BarAggregator("TEST", 240, 5, cal, "session_open")
    out = [b for b in (agg.on_base(x) for x in session_bars(cal, "2025-03-07")) if b]
    assert [(b.start, b.end) for b in out] == [(ts("2025-03-07 14:30"), ts("2025-03-07 18:30")),
                                               (ts("2025-03-07 18:30"), ts("2025-03-07 21:00"))]


def test_overnight_session_and_holiday():
    fut = SessionConfig(timezone="America/Chicago", open="17:00", close="16:00", trading_weekdays=[0, 1, 2, 3, 4],
                        holidays=["2025-01-20"])
    cal = TradingCalendar(fut)
    s = cal.session_for(ts("2025-01-14 00:00"))   # Mon 18:00 CT -> session closing Tue
    assert s.session_id == "2025-01-14" and s.open == ts("2025-01-13 23:00")
    assert cal.session_by_close_date(pd.Timestamp("2025-01-20").date()) is None
    assert cal.session_for(ts("2025-01-14 22:30")) is None   # 16:30 CT maintenance break


def test_utc_epoch_4h_alignment():
    cal = TradingCalendar(UTC24)
    agg = BarAggregator("TEST", 240, 5, cal, "utc_epoch")
    out = [b for b in (agg.on_base(x) for x in session_bars(cal, "2025-01-01")) if b]
    assert [b.start.hour for b in out] == [0, 4, 8, 12, 16, 20]


def test_htf_bar_only_emitted_when_complete_and_closed():
    cal = TradingCalendar(UTC24)
    base = session_bars(cal, "2025-01-01")[:12]          # 00:00 - 01:00
    agg = BarAggregator("TEST", 30, 5, cal)
    emitted = [(b.end, agg.on_base(b)) for b in base]
    got = [(t, x) for t, x in emitted if x is not None]
    assert [x.end for _, x in got] == [ts("2025-01-01 00:30"), ts("2025-01-01 01:00")]
    assert all(t == x.end for t, x in got)               # emitted exactly at its close, never earlier
    agg2 = BarAggregator("TEST", 30, 5, cal)
    holed = [b for i, b in enumerate(base) if i != 3]    # one missing 5m bar
    out2 = [x for x in (agg2.on_base(b) for b in holed) if x]
    assert [x.start for x in out2] == [ts("2025-01-01 00:30")]   # first bucket dropped, not filled
    assert agg2.dropped[0].have == 5 and agg2.dropped[0].expected == 6


# ---------------------------------------------------------------- pivots
def _highs(seq, left=2, right=2):
    t = PivotTracker(left, right)
    start = ts("2025-01-01")
    log = []
    for i, h in enumerate(seq):
        b = bar(start + pd.Timedelta(minutes=5 * i), h - 0.2, h, h - 1, h - 0.1)
        log.append((i, t.on_bar(b)))
    return t, log


def test_pivot_confirmed_only_after_right_bars():
    t, log = _highs([1, 2, 5, 3, 2, 1])
    new = {i: p for i, p in log if p}
    assert list(new) == [4]                               # visible at bar 4, sits at bar 2
    p = [x for x in new[4] if x.kind == "high"][0]
    assert p.bar_start == ts("2025-01-01 00:10") and p.available_at == ts("2025-01-01 00:25")
    assert t.last("high", 1, ts("2025-01-01 00:20")) == []   # not known one bar earlier


def test_pivot_strictness_and_configurable_lengths():
    t, _ = _highs([1, 2, 5, 5, 2, 1])                     # equal neighbouring highs -> no pivot
    assert t.highs == []
    t, _ = _highs([1, 2, 3, 5, 4, 1], left=3, right=1)
    assert [p.price for p in t.highs] == [5]


# ---------------------------------------------------------------- ATR
def test_atr_warmup_and_value():
    a = WilderATR(3)
    start = ts("2025-01-01")
    vals = [a.update(mid_bar(start + pd.Timedelta(minutes=5 * i), 100, half=0.5)) for i in range(5)]
    assert vals[:3] == [None, None, None] and vals[3] == pytest.approx(1.0) and vals[4] == pytest.approx(1.0)


# ---------------------------------------------------------------- VWAP
def test_vwap_reset_and_formula():
    v = SessionVWAP()
    d1 = ts("2025-01-01")
    v.on_base(bar(d1, 10, 12, 8, 10, v=100), d1)            # tp 10
    v.on_base(bar(d1 + pd.Timedelta(minutes=5), 20, 22, 18, 20, v=300), d1)   # tp 20
    assert v.value()[0] == pytest.approx((10 * 100 + 20 * 300) / 400)
    d2 = ts("2025-01-02")
    v.on_base(bar(d2, 50, 51, 49, 50, v=10), d2)
    assert v.value()[0] == pytest.approx(50)                 # new session, prior volume gone


def test_vwap_zero_and_missing_volume():
    d = ts("2025-01-01")
    v = SessionVWAP()
    v.on_base(bar(d, 10, 11, 9, 10, v=0), d)
    assert v.value() == (None, "zero_cumulative_volume")
    v.on_base(bar(d + pd.Timedelta(minutes=5), 20, 21, 19, 20, v=5), d)
    assert v.value()[0] == pytest.approx(20)                 # zero-volume bar adds nothing
    v.on_base(bar(d + pd.Timedelta(minutes=10), 20, 21, 19, 20, v=float("nan")), d)
    assert v.value() == (None, "missing_volume")
    assert SessionVWAP(volume_allowed=False).value()[0] is None


def test_vwap_requires_session_open_and_no_gaps():
    d = ts("2025-01-01")
    v = SessionVWAP()
    v.on_base(bar(d + pd.Timedelta(hours=1), 10, 11, 9, 10), d)
    assert v.value() == (None, "session_open_data_missing")
    v2 = SessionVWAP()
    v2.on_base(bar(d, 10, 11, 9, 10), d)
    v2.on_base(bar(d + pd.Timedelta(minutes=15), 10, 11, 9, 10), d)
    assert v2.value() == (None, "intra_session_data_gap")
    v3 = SessionVWAP(require_complete_session=False)
    v3.on_base(bar(d + pd.Timedelta(hours=1), 10, 11, 9, 10), d)
    assert v3.value()[0] == pytest.approx(10)


# ---------------------------------------------------------------- stale feed
def test_feed_monitor_blocks_after_gap_until_recovered():
    f = FeedMonitor(5, stale_gap_bars=1, recovery_bars=3)
    d = ts("2025-01-01")
    assert not f.on_base(bar(d, 1, 1, 1, 1)) and not f.entries_blocked
    assert f.on_base(bar(d + pd.Timedelta(minutes=15), 1, 1, 1, 1))     # two bars missing
    assert f.entries_blocked
    for k in range(3):
        f.on_base(bar(d + pd.Timedelta(minutes=20 + 5 * k), 1, 1, 1, 1))
    assert not f.entries_blocked
    assert f.is_stale(d + pd.Timedelta(hours=2), pd.Timedelta(minutes=10))
    assert not f.is_stale(d + pd.Timedelta(minutes=35), pd.Timedelta(minutes=10))
