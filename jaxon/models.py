"""Plain data objects shared by every module.

Every derived value carries `available_at`: the earliest UTC time at which a
live system could have known it. Strategy code may only use a value when
`available_at <= decision time`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import pandas as pd


class State(str, Enum):
    IDLE = "IDLE"
    BULLISH_BIAS = "BULLISH_BIAS"
    ZONE_ACTIVE = "ZONE_ACTIVE"
    ZONE_TOUCHED = "ZONE_TOUCHED"
    SHIFT_CONFIRMED = "SHIFT_CONFIRMED"
    ENTRY_PENDING = "ENTRY_PENDING"
    POSITION_OPEN = "POSITION_OPEN"
    CLOSED = "CLOSED"
    INVALIDATED = "INVALIDATED"
    DAILY_LOCKOUT = "DAILY_LOCKOUT"


@dataclass(frozen=True)
class Bar:
    symbol: str
    start: pd.Timestamp          # UTC, inclusive
    end: pd.Timestamp            # UTC, exclusive; also the bar's availability time
    open: float
    high: float
    low: float
    close: float
    volume: float                # NaN = volume unknown
    minutes: int
    session_id: str

    @property
    def available_at(self) -> pd.Timestamp:
        return self.end

    @property
    def volume_ok(self) -> bool:
        return not math.isnan(self.volume) and self.volume >= 0


@dataclass(frozen=True)
class Pivot:
    kind: str                    # "high" | "low"
    price: float
    bar_start: pd.Timestamp      # where it sits on the chart
    available_at: pd.Timestamp   # close of the last right-hand bar
    timeframe: int


@dataclass
class Zone:
    zone_id: str
    symbol: str
    lower: float
    upper: float
    activated_at: pd.Timestamp       # displacement candle close
    base_bar_start: pd.Timestamp
    displacement_bar_start: pd.Timestamp
    displacement_body: float
    atr_at_displacement: float
    broken_swing_high: float
    bars_since_activation: int = 0
    status: str = "active"           # active | invalidated | expired | used
    status_at: Optional[pd.Timestamp] = None
    attempt_used: bool = False

    @property
    def is_valid(self) -> bool:
        return self.status in ("active", "used")


@dataclass
class Setup:
    setup_id: str
    symbol: str
    zone_id: str
    zone_lower: float
    zone_upper: float
    zone_activated_at: pd.Timestamp
    touch_at: pd.Timestamp                 # close time of the touch candle
    touch_bar_start: pd.Timestamp
    break_threshold: float                 # frozen at touch, never moved
    threshold_pivot_start: pd.Timestamp
    threshold_pivot_available_at: pd.Timestamp
    lowest_low: float
    htf_swings: dict
    bias_at_touch: str
    bars_after_touch: int = 0
    status: str = "touched"                # touched | confirmed | cancelled | timeout | rejected | pending | filled | closed
    confirm_at: Optional[pd.Timestamp] = None
    vwap_snapshot: Optional[float] = None
    vwap_note: str = ""
    proposed_entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    proposed_rr: Optional[float] = None
    signal_id: Optional[str] = None
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    actual_entry: Optional[float] = None
    quantity: Optional[float] = None
    actual_rr: Optional[float] = None
    est_round_trip_cost: Optional[float] = None
    outcome: str = ""                       # rejection reason or exit reason
    outcome_at: Optional[pd.Timestamp] = None

    def as_record(self) -> dict:
        d = asdict(self)
        return {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in d.items()}


@dataclass
class Signal:
    signal_id: str
    setup_id: str
    symbol: str
    created_at: pd.Timestamp       # confirmation candle close
    proposed_entry: float
    stop: float
    target: float
    zone_id: str


@dataclass
class Order:
    order_id: str
    symbol: str
    side: str                 # buy | sell
    kind: str                 # market | stop | limit
    quantity: float
    active_from: pd.Timestamp # order cannot fill on any bar starting before this
    price: Optional[float] = None
    purpose: str = ""         # entry | stop | target | flatten | lockout
    oco_group: Optional[str] = None
    status: str = "open"      # open | filled | cancelled | rejected
    signal_id: Optional[str] = None
    setup_id: Optional[str] = None


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float              # executable price incl. spread + slippage
    reference_price: float    # price before spread/slippage
    fee: float
    time: pd.Timestamp
    purpose: str
    flags: list = field(default_factory=list)

    @property
    def friction(self) -> float:
        """Cash lost to spread + slippage, before value_per_point scaling."""
        return abs(self.price - self.reference_price) * self.quantity


@dataclass
class Trade:
    trade_id: str
    symbol: str
    setup_id: str
    signal_id: str
    zone_id: str
    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    stop: float
    target: float
    planned_risk_cash: float
    entry_fee: float
    entry_friction_cash: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    exit_fee: float = 0.0
    exit_friction_cash: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    r_multiple: float = 0.0
    flags: list = field(default_factory=list)

    def as_record(self) -> dict:
        d = asdict(self)
        d["flags"] = ";".join(self.flags)
        return {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in d.items()}
