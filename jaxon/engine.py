"""Event-driven engine. Backtest and paper replay both call `process(bar)` with
closed base bars in time order; there is no other code path.

Per base bar, in this order:
  1. duplicate / out-of-order guard, risk-day and session rollover, feed check
  2. pending entry: recheck rules at this bar's open, size, submit
  3. paper fills for this bar (entry, then exits that are active from the fill)
  4. VWAP update (finest bars)
  5. 4H, 30m, 5m bars that closed at this bar's end -> strategy
  6. mark-to-market, daily-loss check, session-end flatten
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .bars import BarAggregator
from .calendar import RiskDay, Session, TradingCalendar
from .config import Config, InstrumentConfig
from .execution.paper import CostModel, PaperBroker
from .indicators import FeedMonitor, SessionVWAP
from .journal import IdFactory, Journal
from .models import Bar, Fill, Order, Setup, Signal, State, Trade
from .risk import SizeResult, size_position
from .strategy import FiveMinContext, StrategyCore

ENGINE_VERSION = "jaxon-engine-1.0"


@dataclass
class Pending:
    setup: Setup
    signal: Signal
    order: Order
    created_at: pd.Timestamp
    session_id: str


@dataclass
class SymbolRuntime:
    inst: InstrumentConfig
    cal: TradingCalendar
    cm: CostModel
    core: StrategyCore
    agg5: BarAggregator
    agg30: BarAggregator
    agg4h: BarAggregator
    vwap: SessionVWAP
    feed: FeedMonitor
    session: Optional[Session] = None
    last_start: Optional[pd.Timestamp] = None
    last_close: Optional[float] = None
    pending: Optional[Pending] = None
    trade: Optional[Trade] = None
    trade_setup: Optional[Setup] = None
    exit_order_ids: list = field(default_factory=list)
    flatten_order_id: Optional[str] = None
    bars_5m: int = 0
    planned: Optional[SizeResult] = None


class TradingEngine:
    def __init__(self, cfg: Config, provenance: dict | None = None):
        self.cfg = cfg
        self.provenance = provenance or {}
        self.journal = Journal()
        self.ids = IdFactory()
        self.broker = PaperBroker({i.symbol: i for i in cfg.instruments}, cfg.execution)
        self.riskday = RiskDay(cfg.portfolio)
        self.rt: dict[str, SymbolRuntime] = {}
        s = cfg.strategy
        for inst in cfg.instruments:
            cal = TradingCalendar(inst.session)
            base = inst.bars.base_interval_minutes
            self.rt[inst.symbol] = SymbolRuntime(
                inst=inst, cal=cal, cm=self.broker.costs[inst.symbol],
                core=StrategyCore(inst.symbol, s, self.journal, self.ids),
                agg5=BarAggregator(inst.symbol, 5, base, cal),
                agg30=BarAggregator(inst.symbol, 30, base, cal),
                agg4h=BarAggregator(inst.symbol, 240, base, cal, inst.bars.four_hour_alignment),
                vwap=SessionVWAP(s.vwap_require_complete_session, inst.volume_type in s.vwap_allowed_volume_types),
                feed=FeedMonitor(base, inst.bars.stale_gap_bars, inst.bars.stale_recovery_bars),
            )
        self.cash = cfg.portfolio.starting_equity
        self.trades: list[Trade] = []
        self.signals: list[Signal] = []
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self.lockouts: list[dict] = []
        self.risk_day: Optional[str] = None
        self.day_start_equity = self.cash
        self.entries_today = 0
        self.locked = False
        self.processed = 0
        self.duplicates_ignored = 0
        self.last_event_time: Optional[pd.Timestamp] = None
        self.finalized = False

    # ------------------------------------------------------------ helpers
    def equity(self) -> float:
        eq = self.cash
        for rt in self.rt.values():
            if rt.trade is not None and rt.last_close is not None:
                px = rt.cm.sell_market(rt.last_close)
                eq += rt.trade.quantity * px * rt.cm.vpp - rt.cm.commission(rt.trade.quantity, px)
        return eq

    def open_risk_cash(self) -> float:
        return sum(rt.trade.planned_risk_cash for rt in self.rt.values() if rt.trade is not None)

    def _positions_and_pending(self) -> int:
        return sum((rt.trade is not None) + (rt.pending is not None) for rt in self.rt.values())

    def _reject(self, rt: SymbolRuntime, s: Setup, reason: str, t: pd.Timestamp):
        rt.core.reject(s, reason, t)
        self.journal.log(t, "signal_rejected", rt.inst.symbol, setup_id=s.setup_id, reason=reason)

    # ------------------------------------------------------------ main entry
    def process(self, bar: Bar) -> bool:
        rt = self.rt.get(bar.symbol)
        if rt is None:
            raise KeyError(f"no instrument configured for {bar.symbol}")
        if rt.last_start is not None and bar.start <= rt.last_start:
            self.duplicates_ignored += 1
            self.journal.log(bar.end, "duplicate_bar_ignored", bar.symbol, bar_start=bar.start)
            return False
        if self.last_event_time is not None and bar.start < self.last_event_time:
            raise ValueError("bars must be fed in time order across instruments")
        self.processed += 1
        self.last_event_time = bar.start

        self._risk_day_roll(bar.start)
        if rt.session is None or rt.session.session_id != bar.session_id:
            rt.session = rt.cal.session_by_close_date(dt.date.fromisoformat(bar.session_id))
            self.journal.log(bar.start, "session_start", bar.symbol, session_id=bar.session_id)
        if rt.feed.on_base(bar):
            self.journal.log(bar.start, "feed_stale", bar.symbol, gap_from=rt.feed.last_gap[0],
                             gap_to=rt.feed.last_gap[1], recovery_bars=rt.feed.recovery)

        if rt.pending is not None:
            self._try_entry(rt, bar)
        entry_filled = self._handle_fills(rt, self.broker.process_bar(bar), bar)
        if entry_filled:   # protective orders are active from the entry fill: check this same bar
            self._handle_fills(rt, self.broker.process_bar(bar), bar)

        rt.vwap.on_base(bar, rt.session.open)
        rt.last_close = bar.close

        b4, b30, b5 = rt.agg4h.on_base(bar), rt.agg30.on_base(bar), rt.agg5.on_base(bar)
        core = rt.core
        if b4 is not None:
            core.on_4h(b4)
        if b30 is not None:
            core.on_30m(b30)
        if b5 is not None:
            rt.bars_5m += 1
            can = (not self.locked and self.entries_today < self.cfg.risk.max_entries_per_session
                   and self._positions_and_pending() < self.cfg.risk.max_open_positions)
            confirmed = core.on_5m(b5, FiveMinContext(can, rt.session.close))
            if confirmed is not None:
                self._on_confirm(rt, confirmed, b5)

        eq = self.equity()
        self.equity_curve.append((bar.end, eq))
        if not self.locked and eq <= self.day_start_equity * (1 - self.cfg.risk.max_daily_drawdown_pct):
            self._lockout(bar.end, eq)
        if rt.trade is not None and rt.flatten_order_id is None and bar.end >= rt.session.flatten_at:
            self._submit_exit(rt, bar.end, "session_flatten")
        rt.last_start = bar.start
        return True

    # ------------------------------------------------------------ risk day
    def _risk_day_roll(self, ts: pd.Timestamp):
        day = self.riskday.day_id(ts)
        if day == self.risk_day:
            return
        self.risk_day = day
        self.day_start_equity = self.equity()
        self.entries_today = 0
        if self.locked:
            self.locked = False
            for rt in self.rt.values():
                rt.core.unlock(ts)
        self.journal.log(ts, "risk_day_start", risk_day=day, start_equity=self.day_start_equity)

    def _lockout(self, t: pd.Timestamp, eq: float):
        self.locked = True
        rec = {"time": t.isoformat(), "risk_day": self.risk_day, "start_equity": self.day_start_equity,
               "equity_at_breach": eq, "limit_pct": self.cfg.risk.max_daily_drawdown_pct,
               "drawdown_pct_at_breach": 1 - eq / self.day_start_equity, "flatten_fill": None,
               "overshoot_pct": None}
        self.lockouts.append(rec)
        self.journal.log(t, "daily_lockout", **rec)
        for rt in self.rt.values():
            if rt.pending is not None:
                p, rt.pending = rt.pending, None
                p.order.status = "cancelled"
                self._reject(rt, p.setup, "daily_lockout", t)
            rt.core.lock(t)
            if rt.trade is not None and rt.flatten_order_id is None:
                self._submit_exit(rt, t, "daily_lockout_flatten")

    # ------------------------------------------------------------ signals
    def _on_confirm(self, rt: SymbolRuntime, s: Setup, b5: Bar):
        t, inst, cm = b5.end, rt.inst, rt.cm
        vwap, why = rt.vwap.value()
        s.vwap_snapshot = vwap
        s.vwap_note = why or "bar_based_vwap_approximation"
        plan = rt.core.plan(s, b5.close, vwap, why, inst.tick_size, cm.net_target_per_unit)
        s.proposed_entry, s.stop, s.target, s.proposed_rr = plan.proposed_entry, plan.stop, plan.target, plan.rr
        if plan.target:
            e = cm.buy_market(plan.proposed_entry)
            s.est_round_trip_cost = ((e - plan.proposed_entry) * cm.vpp + cm.commission_per_unit(e)
                                     + cm.commission_per_unit(plan.target))
        reason = plan.reason
        if not reason:
            if rt.feed.entries_blocked:
                reason = "stale_feed"
            elif self.locked:
                reason = "daily_lockout"
            elif t >= rt.session.last_entry_at:
                reason = "outside_entry_window"
            elif self.entries_today >= self.cfg.risk.max_entries_per_session:
                reason = "entry_limit_reached"
            elif self._positions_and_pending() >= self.cfg.risk.max_open_positions:
                reason = "portfolio_position_limit"
        if reason:
            self._reject(rt, s, reason, t)
            return
        sig = Signal(self.ids.new("G", inst.symbol, t), s.setup_id, inst.symbol, t, plan.proposed_entry,
                     plan.stop, plan.target, s.zone_id)
        order = Order(self.ids.new("O", inst.symbol, t), inst.symbol, "buy", "market", 0.0, t,
                      purpose="entry", signal_id=sig.signal_id, setup_id=s.setup_id)
        s.signal_id, s.order_id = sig.signal_id, order.order_id
        self.signals.append(sig)
        rt.pending = Pending(s, sig, order, t, b5.session_id)
        self.journal.log(t, "signal", inst.symbol, signal_id=sig.signal_id, setup_id=s.setup_id,
                         entry=plan.proposed_entry, stop=plan.stop, target=plan.target, rr=plan.rr,
                         vwap=vwap, threshold=s.break_threshold, lowest_low=s.lowest_low)
        rt.core.mark_pending(s, t)

    def _try_entry(self, rt: SymbolRuntime, bar: Bar):
        p, rt.pending = rt.pending, None
        s, inst, cm, cfg = p.setup, rt.inst, rt.cm, self.cfg
        t = bar.start
        exec_px = cm.buy_market(bar.open)
        reason = ""
        if bar.start != p.created_at:
            reason = "entry_bar_missing"
        elif bar.session_id != p.session_id:
            reason = "entry_bar_next_session"
        elif t >= rt.session.last_entry_at:
            reason = "outside_entry_window"
        elif self.locked:
            reason = "daily_lockout"
        elif rt.core.bias != "bullish":
            reason = "bias_lost_before_entry"
        elif not rt.core.zone(s.zone_id).is_valid:
            reason = "zone_invalid_before_entry"
        elif bar.open <= s.stop or exec_px <= s.stop:
            reason = "gapped_below_stop"
        elif exec_px >= s.target:
            reason = "gapped_above_target"
        elif (s.target - exec_px) / (exec_px - s.stop) < cfg.strategy.min_reward_risk:
            reason = "entry_rr_below_min"
        size: Optional[SizeResult] = None
        if not reason:
            size = size_position(self.equity(), exec_px, s.stop, inst, cm, cfg.risk, self.open_risk_cash())
            if not size.ok:
                reason = size.reason
            else:
                q = size.quantity
                net = (s.target - exec_px) * q * cm.vpp - cm.commission(q, exec_px) - cm.commission(q, s.target)
                if net <= 0:
                    reason = "entry_costs_exceed_reward"
        s.actual_rr = (s.target - exec_px) / (exec_px - s.stop) if exec_px > s.stop else None
        if reason:
            p.order.status = "rejected"
            self._reject(rt, s, reason, t)
            return
        p.order.quantity, p.order.active_from = size.quantity, bar.start
        s.quantity = size.quantity
        rt.planned = size
        self.broker.submit(p.order)
        self.journal.log(t, "order_submitted", inst.symbol, order_id=p.order.order_id, qty=size.quantity,
                         expected_price=exec_px, planned_risk=size.planned_risk_cash, notional=size.notional)
        rt.trade_setup = s

    # ------------------------------------------------------------ fills
    def _submit_exit(self, rt: SymbolRuntime, t: pd.Timestamp, purpose: str):
        o = Order(self.ids.new("O", rt.inst.symbol, t), rt.inst.symbol, "sell", "market", rt.trade.quantity, t,
                  purpose=purpose, setup_id=rt.trade.setup_id)
        self.broker.submit(o)
        rt.flatten_order_id = o.order_id
        self.journal.log(t, "order_submitted", rt.inst.symbol, order_id=o.order_id, purpose=purpose,
                         qty=o.quantity)

    def _handle_fills(self, rt: SymbolRuntime, fills: list[Fill], bar: Bar) -> bool:
        entry_filled = False
        for f in fills:
            self.journal.log(f.time, "fill", f.symbol, order_id=f.order_id, side=f.side, qty=f.quantity,
                             price=f.price, reference=f.reference_price, fee=f.fee, purpose=f.purpose, flags=f.flags)
            if f.purpose == "entry":
                self._open_trade(rt, f)
                entry_filled = True
            else:
                self._close_trade(rt, f, bar)
        return entry_filled

    def _open_trade(self, rt: SymbolRuntime, f: Fill):
        s, cm = rt.trade_setup, rt.cm
        size = rt.planned
        tr = Trade(self.ids.new("T", f.symbol, f.time), f.symbol, s.setup_id, s.signal_id, s.zone_id, f.time,
                   f.price, f.quantity, s.stop, s.target, size.planned_risk_cash, f.fee, f.friction * cm.vpp)
        self.cash -= f.quantity * f.price * cm.vpp + f.fee
        rt.trade = tr
        s.trade_id, s.actual_entry = tr.trade_id, f.price
        self.entries_today += 1
        grp = f"OCO-{tr.trade_id}"
        stop = Order(self.ids.new("O", f.symbol, f.time), f.symbol, "sell", "stop", f.quantity, f.time, s.stop,
                     "stop", grp, setup_id=s.setup_id)
        tgt = Order(self.ids.new("O", f.symbol, f.time), f.symbol, "sell", "limit", f.quantity, f.time, s.target,
                    "target", grp, setup_id=s.setup_id)
        self.broker.submit(stop)
        self.broker.submit(tgt)
        rt.exit_order_ids = [stop.order_id, tgt.order_id]
        rt.core.mark_filled(s, f.time)

    def _close_trade(self, rt: SymbolRuntime, f: Fill, bar: Bar):
        tr, cm = rt.trade, rt.cm
        if tr is None:
            return
        self.cash += f.quantity * f.price * cm.vpp - f.fee
        tr.exit_time, tr.exit_price, tr.exit_fee = f.time, f.price, f.fee
        tr.exit_friction_cash = f.friction * cm.vpp
        tr.exit_reason = {"stop": "stop", "target": "target"}.get(f.purpose, f.purpose)
        tr.flags = sorted(set(tr.flags + f.flags))
        entry_sess = rt.cal.session_for(tr.entry_time)
        if f.purpose.endswith("flatten") and entry_sess is not None and bar.session_id != entry_sess.session_id:
            tr.flags.append("flatten_filled_next_session")
        tr.gross_pnl = (tr.exit_price - tr.entry_price) * tr.quantity * cm.vpp
        tr.net_pnl = tr.gross_pnl - tr.entry_fee - tr.exit_fee
        tr.r_multiple = tr.net_pnl / tr.planned_risk_cash if tr.planned_risk_cash else 0.0
        if tr.exit_reason == "stop" and tr.net_pnl < -tr.planned_risk_cash * 1.0001:
            tr.flags.append("loss_exceeded_planned_risk")
        for oid in rt.exit_order_ids + ([rt.flatten_order_id] if rt.flatten_order_id else []):
            self.broker.cancel(oid, "position_closed")
        self.trades.append(tr)
        s = rt.trade_setup
        rt.trade, rt.trade_setup, rt.exit_order_ids, rt.flatten_order_id = None, None, [], None
        rt.core.mark_closed(s, tr.exit_reason, f.time)
        if f.purpose == "daily_lockout_flatten" and self.lockouts:
            lk = self.lockouts[-1]
            eq = self.equity()
            dd = 1 - eq / lk["start_equity"]
            lk["flatten_fill"] = {"time": f.time.isoformat(), "price": f.price, "equity_after": eq}
            lk["overshoot_pct"] = max(0.0, dd - lk["limit_pct"])
        self.journal.log(f.time, "trade_closed", f.symbol, trade_id=tr.trade_id, reason=tr.exit_reason,
                         net_pnl=tr.net_pnl, r=tr.r_multiple, flags=tr.flags)

    # ------------------------------------------------------------ end of data
    def finalize(self) -> list[dict]:
        """Close the books without inventing fills: open positions are reported as open,
        marked at the last close minus modeled exit costs."""
        open_positions = []
        t = self.last_event_time or pd.Timestamp.now(tz="UTC")
        for rt in self.rt.values():
            if rt.pending is not None:
                p, rt.pending = rt.pending, None
                self._reject(rt, p.setup, "end_of_data", t)
            if rt.core.setup is not None:
                s = rt.core.setup
                s.status, s.outcome, s.outcome_at = "cancelled", "end_of_data", t
                rt.core.setup = None
            if rt.trade is not None:
                tr = rt.trade
                mark = rt.cm.sell_market(rt.last_close)
                open_positions.append({**tr.as_record(), "mark_price": mark,
                                       "unrealized_pnl": (mark - tr.entry_price) * tr.quantity * rt.cm.vpp - tr.entry_fee,
                                       "status": "open_at_end_of_data"})
        self.finalized = True
        self.open_positions = open_positions
        return open_positions

    # ------------------------------------------------------------ views
    def setups(self) -> list[Setup]:
        out = []
        for rt in self.rt.values():
            out.extend(rt.core.setups)
        return sorted(out, key=lambda s: (s.touch_at, s.setup_id))

    def zones(self):
        out = []
        for rt in self.rt.values():
            out.extend(rt.core.zones)
        return out

    def touch_skips(self) -> dict:
        from collections import Counter
        c = Counter()
        for rt in self.rt.values():
            c.update(rt.core.touch_skips)
        return dict(c)

    def state(self, symbol: str) -> State:
        return self.rt[symbol].core.state
