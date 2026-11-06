"""Strategy rules (pure logic, no broker or account access).

4H bullish bias -> 30m demand zone -> 5m touch + frozen break threshold ->
5m close above threshold (structure shift) -> signal with VWAP target and
swing-low stop. See docs/assumptions.md for which parts come from the video
and which are proposed definitions.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from .config import StrategyConfig
from .indicators import PivotTracker, WilderATR
from .journal import IdFactory, Journal
from .models import Bar, Setup, State, Zone


def floor_tick(x: float, tick: float) -> float:
    return round(math.floor(x / tick + 1e-9) * tick, 10)


def ceil_tick(x: float, tick: float) -> float:
    return round(math.ceil(x / tick - 1e-9) * tick, 10)


@dataclass
class FiveMinContext:
    can_start_setup: bool          # engine: no position / pending entry / lockout / entry limit
    session_close: pd.Timestamp


@dataclass
class SignalPlan:
    ok: bool
    reason: str
    proposed_entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    rr: float = 0.0


class StrategyCore:
    def __init__(self, symbol: str, cfg: StrategyConfig, journal: Journal, ids: IdFactory):
        self.symbol, self.cfg, self.j, self.ids = symbol, cfg, journal, ids
        self.p4 = PivotTracker(cfg.htf_pivot_left, cfg.htf_pivot_right, 240)
        self.p30 = PivotTracker(cfg.mtf_pivot_left, cfg.mtf_pivot_right, 30)
        self.p5 = PivotTracker(cfg.ltf_pivot_left, cfg.ltf_pivot_right, 5)
        self.atr30 = WilderATR(cfg.atr_period)
        self.recent30: deque[Bar] = deque(maxlen=max(cfg.base_lookback_bars, cfg.pullback_declining_closes) + 1)
        self.zones: list[Zone] = []
        self.setup: Optional[Setup] = None
        self.setups: list[Setup] = []
        self.touch_skips: Counter = Counter()
        self.bias = "neutral"
        self.bias_detail: dict = {"reason": "insufficient_swing_history"}
        self.state = State.IDLE
        self.locked = False
        self.position_open = False
        self.entry_pending = False
        # zone_tap inputs (all from CLOSED bars)
        from .pro import _Ema
        self.last4h_green = False
        self.ema30 = _Ema(cfg.tap_ema_len)
        self.ema30_down = False
        self.swings30_down = False

    # ------------------------------------------------------------ state
    def _to(self, new: State, ts: pd.Timestamp, reason: str = "", **ids):
        if new == self.state:
            return
        self.j.log(ts, "transition", self.symbol, frm=self.state, to=new, reason=reason, **ids)
        self.state = new

    def rest_state(self) -> State:
        if self.locked:
            return State.DAILY_LOCKOUT
        if self.position_open:
            return State.POSITION_OPEN
        if self.entry_pending:
            return State.ENTRY_PENDING
        if self.setup is not None:
            return State.ZONE_TOUCHED
        if self.bias != "bullish":
            return State.IDLE
        if any(z.status == "active" and not z.attempt_used for z in self.zones):
            return State.ZONE_ACTIVE
        return State.BULLISH_BIAS

    def settle(self, ts: pd.Timestamp, reason: str = ""):
        self._to(self.rest_state(), ts, reason)

    def zone(self, zone_id: str) -> Zone:
        return next(z for z in self.zones if z.zone_id == zone_id)

    # ------------------------------------------------------------ 4H
    def htf_swings(self, as_of: pd.Timestamp) -> dict:
        hs, ls = self.p4.last("high", 2, as_of), self.p4.last("low", 2, as_of)
        return {"highs": [(p.bar_start.isoformat(), p.price, p.available_at.isoformat()) for p in hs],
                "lows": [(p.bar_start.isoformat(), p.price, p.available_at.isoformat()) for p in ls]}

    def on_4h(self, bar: Bar):
        self.p4.on_bar(bar)
        t = bar.end
        hs, ls = self.p4.last("high", 2, t), self.p4.last("low", 2, t)
        if len(hs) < 2 or len(ls) < 2:
            bias, detail = "neutral", {"reason": "insufficient_swing_history"}
        else:
            hh, hl = hs[1].price > hs[0].price, ls[1].price > ls[0].price
            bias = "bullish" if hh and hl else "neutral"
            detail = {"rising_highs": hh, "rising_lows": hl, **self.htf_swings(t)}
        if bias != self.bias:
            self.j.log(t, "bias", self.symbol, bias=bias, previous=self.bias, **detail)
        self.bias, self.bias_detail = bias, detail
        self.last4h_green = bar.close > bar.open
        self.settle(t, f"4h_bias_{bias}")

    # ------------------------------------------------------------ 30m
    def on_30m(self, bar: Bar):
        t, c = bar.end, self.cfg
        # 1. maintain zones activated before this bar
        for z in self.zones:
            if z.status not in ("active",):
                continue
            z.bars_since_activation += 1
            if bar.close < z.lower:
                z.status, z.status_at = "invalidated", t
                self.j.log(t, "zone_invalidated", self.symbol, zone_id=z.zone_id, close=bar.close, lower=z.lower)
            elif z.bars_since_activation >= c.zone_expiry_bars:
                z.status, z.status_at = "expired", t
                self.j.log(t, "zone_expired", self.symbol, zone_id=z.zone_id, bars=z.bars_since_activation)

        # 2. displacement candle -> new zone (uses only info known before this bar)
        atr_prev = self.atr30.value
        body = bar.close - bar.open
        if atr_prev is not None and body > 0 and body >= c.displacement_atr_mult * atr_prev:
            sh = self.p30.last("high", 1, bar.start)
            if not sh:
                self.j.log(t, "zone_skipped", self.symbol, reason="no_prior_30m_swing_high", bar=bar.start)
            elif bar.close <= sh[0].price:
                pass  # not a displacement through structure; not logged to keep the journal readable
            else:
                prior = list(self.recent30)[-c.base_lookback_bars:]
                base = next((b for b in reversed(prior) if b.close < b.open), None)
                if base is None:
                    self.j.log(t, "zone_skipped", self.symbol, reason="no_bearish_base", bar=bar.start)
                else:
                    z = Zone(self.ids.new("Z", self.symbol, t), self.symbol, base.low, base.high, t, base.start,
                             bar.start, body, atr_prev, sh[0].price)
                    self.zones.append(z)
                    self.j.log(t, "zone_activated", self.symbol, zone_id=z.zone_id, lower=z.lower, upper=z.upper,
                               base_bar=base.start, displacement_bar=bar.start, body=body, atr=atr_prev,
                               broken_swing_high=sh[0].price)

        # 3. update indicators with this bar
        self.atr30.update(bar)
        self.p30.on_bar(bar)
        self.recent30.append(bar)
        prev = self.ema30.v
        e = self.ema30.update(bar.close)
        self.ema30_down = e is not None and prev is not None and bar.close < e and e < prev
        hs, ls = self.p30.last("high", 2, t), self.p30.last("low", 2, t)
        self.swings30_down = len(hs) == 2 and len(ls) == 2 and hs[1].price < hs[0].price and ls[1].price < ls[0].price
        self.settle(t)

    def tap_long_allowed(self) -> bool:
        """Zone-tap filters: 4H rule says up AND the 30m chart is pulling back (downtrend)."""
        c = self.cfg
        green, swing = self.last4h_green, self.bias == "bullish"
        htf = green if c.tap_htf_filter == "green_candle" else swing if c.tap_htf_filter == "swing_trend" else (green and swing)
        mtf = (self.ema30_down if c.tap_mtf_trend == "ema" else self.swings30_down if c.tap_mtf_trend == "swings"
               else (self.ema30_down or self.swings30_down))
        return htf and mtf

    # ------------------------------------------------------------ 5m
    def _pullback_ok(self) -> bool:
        n = self.cfg.pullback_declining_closes
        if len(self.recent30) < n:
            return False
        closes = [b.close for b in list(self.recent30)[-n:]]
        return all(b < a for a, b in zip(closes, closes[1:]))

    def _end_setup(self, s: Setup, status: str, reason: str, t: pd.Timestamp):
        s.status, s.outcome, s.outcome_at = status, reason, t
        self.setup = None
        self._to(State.INVALIDATED, t, reason, setup_id=s.setup_id, zone_id=s.zone_id)
        self.settle(t)

    def on_5m(self, bar: Bar, ctx: FiveMinContext) -> Optional[Setup]:
        t = bar.end
        self.p5.on_bar(bar)
        s = self.setup
        if s is not None and s.status == "touched":
            s.bars_after_touch += 1
            s.lowest_low = min(s.lowest_low, bar.low)
            z = self.zone(s.zone_id)
            cancel = None
            if bar.low < s.zone_lower:
                cancel = "cancelled_traded_below_zone"
            elif z.status == "invalidated":
                cancel = "cancelled_zone_invalidated"
            elif z.status == "expired":
                cancel = "cancelled_zone_expired"
            elif self.bias != "bullish":
                cancel = "cancelled_bias_lost"
            if cancel:                          # cancellation wins over a same-bar confirmation
                self._end_setup(s, "cancelled", cancel, t)
                return None
            if bar.close > s.break_threshold:
                s.status, s.confirm_at = "confirmed", t
                self._to(State.SHIFT_CONFIRMED, t, "close_above_frozen_threshold", setup_id=s.setup_id,
                         close=bar.close, threshold=s.break_threshold)
                return s
            if s.bars_after_touch >= self.cfg.confirm_window_bars:
                self._end_setup(s, "timeout", "timeout_no_structure_shift", t)
            elif self.cfg.cancel_setup_at_session_end and t >= ctx.session_close:
                self._end_setup(s, "cancelled", "cancelled_session_end", t)
            return None

        if s is not None or not ctx.can_start_setup or self.bias != "bullish":
            return None

        # look for a touch of the most recently activated eligible zone
        cands = [z for z in self.zones if z.status == "active" and not z.attempt_used
                 and z.activated_at <= bar.start and bar.low <= z.upper and bar.high >= z.lower]
        if not cands:
            return None
        z = max(cands, key=lambda q: (q.activated_at, q.zone_id))
        if not self._pullback_ok():
            self.touch_skips["pullback_filter_failed"] += 1
            self.j.log(t, "touch_skipped", self.symbol, zone_id=z.zone_id, reason="pullback_filter_failed",
                       closes_30m=[b.close for b in self.recent30][-self.cfg.pullback_declining_closes:])
            return None
        sh = [p for p in self.p5.highs if p.bar_start >= z.activated_at and p.available_at <= t]
        if not sh:
            self.touch_skips["no_eligible_5m_swing_high"] += 1
            self.j.log(t, "touch_skipped", self.symbol, zone_id=z.zone_id, reason="no_eligible_5m_swing_high")
            return None
        piv = sh[-1]
        z.attempt_used = True
        s = Setup(self.ids.new("S", self.symbol, t), self.symbol, z.zone_id, z.lower, z.upper, z.activated_at,
                  t, bar.start, piv.price, piv.bar_start, piv.available_at, bar.low, self.htf_swings(t), self.bias)
        self.setups.append(s)
        self.setup = s
        self._to(State.ZONE_TOUCHED, t, "5m_touch", setup_id=s.setup_id, zone_id=z.zone_id,
                 threshold=piv.price, zone=[z.lower, z.upper])
        if bar.low < z.lower:
            self._end_setup(s, "cancelled", "cancelled_traded_below_zone", t)
        elif self.cfg.cancel_setup_at_session_end and t >= ctx.session_close:
            self._end_setup(s, "cancelled", "cancelled_session_end", t)
        return None

    # ------------------------------------------------------------ signal rules
    def plan(self, s: Setup, confirm_close: float, vwap: Optional[float], vwap_reason: str, tick: float,
             net_per_unit: Callable[[float, float], float]) -> SignalPlan:
        """Stop/target/RR rules evaluated at the confirmation close."""
        stop = floor_tick(s.lowest_low - self.cfg.stop_offset_ticks * tick, tick)
        if vwap is None:
            return SignalPlan(False, f"vwap_unavailable:{vwap_reason}", confirm_close, stop)
        target = floor_tick(vwap, tick)
        entry = confirm_close
        if not stop < entry:
            return SignalPlan(False, "stop_not_below_entry", entry, stop, target)
        if not target > entry:
            return SignalPlan(False, "target_not_above_entry", entry, stop, target)
        rr = (target - entry) / (entry - stop)
        if rr < self.cfg.min_reward_risk:
            return SignalPlan(False, "rr_below_min", entry, stop, target, rr)
        if net_per_unit(entry, target) <= 0:
            return SignalPlan(False, "costs_exceed_reward", entry, stop, target, rr)
        return SignalPlan(True, "", entry, stop, target, rr)

    # ------------------------------------------------------------ engine callbacks
    def reject(self, s: Setup, reason: str, t: pd.Timestamp):
        self.entry_pending = False
        s.status, s.outcome, s.outcome_at = "rejected", reason, t
        if self.setup is s:
            self.setup = None
        self._to(State.INVALIDATED, t, reason, setup_id=s.setup_id)
        self.settle(t)

    def mark_pending(self, s: Setup, t: pd.Timestamp):
        self.setup = None
        self.entry_pending = True
        s.status = "pending"
        self._to(State.ENTRY_PENDING, t, "entry_order_submitted", setup_id=s.setup_id, order_id=s.order_id)

    def mark_filled(self, s: Setup, t: pd.Timestamp):
        self.entry_pending, self.position_open = False, True
        s.status = "filled"
        self._to(State.POSITION_OPEN, t, "entry_filled", setup_id=s.setup_id, trade_id=s.trade_id)

    def mark_closed(self, s: Setup, reason: str, t: pd.Timestamp):
        self.position_open = False
        s.status, s.outcome, s.outcome_at = "closed", reason, t
        self._to(State.CLOSED, t, reason, setup_id=s.setup_id, trade_id=s.trade_id)
        self.settle(t)

    def lock(self, t: pd.Timestamp):
        if self.setup is not None:
            s = self.setup
            s.status, s.outcome, s.outcome_at = "cancelled", "cancelled_daily_lockout", t
            self.setup = None
        self.locked = True
        self._to(State.DAILY_LOCKOUT, t, "daily_loss_limit")

    def unlock(self, t: pd.Timestamp):
        if self.locked:
            self.locked = False
            self.settle(t, "new_risk_day")
