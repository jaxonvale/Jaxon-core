"""Test helpers: tiny configs, bar builders, and the scripted scenario."""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from jaxon.backtest import LoadedData, load_data, run
from jaxon.config import from_dict, load_config
from jaxon.journal import IdFactory, Journal
from jaxon.models import Bar
from jaxon.strategy import FiveMinContext, StrategyCore

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_CFG = ROOT / "configs" / "example_scenario.yaml"


def base_cfg_dict(path="unused.csv", **inst_over) -> dict:
    inst = {
        "symbol": "TEST", "asset_class": "crypto", "currency": "USD", "tick_size": 0.01, "lot_size": 1,
        "min_quantity": 1, "value_per_point": 1.0, "volume_type": "exchange",
        "session": {"timezone": "UTC", "open": "00:00", "close": "24:00", "trading_weekdays": list(range(7))},
        "bars": {"base_interval_minutes": 5, "four_hour_alignment": "utc_epoch"},
        "costs": {"spread_ticks": 0, "slippage_ticks": 0},
        "data": {"path": str(path), "synthetic": True},
    }
    inst.update(inst_over)
    return {"config_version": "test", "example": True,
            "portfolio": {"starting_equity": 100000, "currency": "USD"}, "instruments": [inst]}


def cfg(**inst_over):
    return from_dict(base_cfg_dict(**inst_over))


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def bar(start, o, h, l, c, v=1000.0, minutes=5, symbol="TEST", session=None) -> Bar:
    st = ts(start) if isinstance(start, str) else start
    sid = session or st.tz_convert("UTC").date().isoformat()
    return Bar(symbol, st, st + pd.Timedelta(minutes=minutes), o, h, l, c, v, minutes, sid)


def mid_bar(start, mid, half=0.5, minutes=5, v=1000.0) -> Bar:
    return bar(start, mid, mid + half, mid - half, mid, v, minutes)


# ---------------------------------------------------------------- strategy-core fixture
class CoreScenario:
    """Builds a StrategyCore with bullish 4H bias and one demand zone.

    30m bars (times are 30m bar opens):
      15 warm-up bars rising 0.1/bar, range 1.0 -> ATR ~ 1
      peak bar P (high 102.0), P+1 bearish, base B (bearish, 100.5-101.0), displacement D (100.6 -> 103.0)
    Zone = [100.5, 101.0], activated at D close.
    """

    def __init__(self, strategy_overrides=None):
        c = cfg()
        for k, v in (strategy_overrides or {}).items():
            setattr(c.strategy, k, v)
        self.cfg = c
        self.j = Journal()
        self.core = StrategyCore("TEST", c.strategy, self.j, IdFactory())
        self.t0 = ts("2025-01-10 00:00")
        self._bias()

    def _bias(self):
        mids = [10, 11, 12, 11, 10, 11, 12, 13, 12, 11, 12, 13, 14]
        start = ts("2025-01-01 00:00")
        for i, m in enumerate(mids):
            self.core.on_4h(mid_bar(start + pd.Timedelta(hours=4 * i), m, minutes=240))
        assert self.core.bias == "bullish", self.core.bias_detail

    def t30(self, i):
        return self.t0 + pd.Timedelta(minutes=30 * i)

    def feed30(self, i, o, h, l, c):
        b = bar(self.t30(i), o, h, l, c, minutes=30)
        self.core.on_30m(b)
        return b

    def build_zone(self, displacement_close=103.0, base_bearish=True, warmup=15):
        for i in range(warmup):
            m = 99.0 + 0.1 * i
            self.feed30(i, m - 0.05, m + 0.5, m - 0.5, m + 0.05)
        p = warmup
        self.feed30(p, 101.3, 102.0, 101.0, 101.5)          # peak (swing high 102.0)
        if base_bearish:
            self.feed30(p + 1, 101.5, 101.6, 101.0, 101.2)  # lower, bearish
            self.feed30(p + 2, 101.0, 101.0, 100.5, 100.6)  # base: most recent bearish candle
        else:                                               # no bearish candle in the 5 before D
            self.feed30(p + 1, 101.0, 101.6, 101.0, 101.2)
            self.feed30(p + 2, 100.6, 101.0, 100.5, 101.0)
        d = self.feed30(p + 3, 100.6, displacement_close + 0.1, 100.5, displacement_close)
        self.d_index = p + 3
        self.activation = d.end
        return d

    def zone(self):
        return self.core.zones[-1] if self.core.zones else None

    def after_zone_30m(self, closes=(102.5, 101.8)):
        """Two more 30m bars with declining closes (pullback filter)."""
        bars = []
        for k, c in enumerate(closes):
            o = closes[k - 1] if k else 103.0
            bars.append(self.feed30(self.d_index + 1 + k, o, max(o, c) + 0.1, min(o, c) - 0.1, c))
        self.t5 = self.t30(self.d_index + 1 + len(closes))
        return bars

    def feed5(self, o, h, l, c, can_start=True, session_close=None):
        b = bar(self.t5, o, h, l, c)
        self.t5 = b.end
        ctx = FiveMinContext(can_start, session_close or ts("2030-01-01"))
        return b, self.core.on_5m(b, ctx)

    def five_min_swing_high(self, peak=102.8):
        """Rise to a strict 5m swing high, then fall toward the zone."""
        seq = [peak - 0.8, peak - 0.4, peak, peak - 0.3, peak - 0.6, peak - 0.9, peak - 1.2]
        for m in seq:
            self.feed5(m, m + 0.05, m - 0.05, m)
        return peak + 0.05


def scenario_cfg():
    return load_config(SCENARIO_CFG)


def scenario_data(c=None) -> LoadedData:
    return load_data(c or scenario_cfg())


def modified_data(data: LoadedData, fn) -> LoadedData:
    """Apply fn(list_of_bars) -> list_of_bars to a copy."""
    d = copy.copy(data)
    d.bars = fn(list(data.bars))
    return d


def replace_bar(b: Bar, **kw) -> Bar:
    vals = dict(symbol=b.symbol, start=b.start, end=b.end, open=b.open, high=b.high, low=b.low, close=b.close,
                volume=b.volume, minutes=b.minutes, session_id=b.session_id)
    vals.update(kw)
    return Bar(**vals)


def run_scenario(c=None, data=None, end=None):
    c = c or scenario_cfg()
    return run(c, data or scenario_data(c), end=end)
