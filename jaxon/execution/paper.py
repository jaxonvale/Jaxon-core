"""Deterministic paper execution model over OHLCV bars.

Rules:
- Market orders fill at the open of the first bar starting at/after activation,
  plus half the spread and slippage (adverse).
- Sell stops trigger when bar low <= stop. If the bar OPENS at/below the stop
  (gap), the fill is at the open, not the stop price.
- Sell limit (target) fills when bar high - half spread reaches the limit
  (plus one tick if limit_requires_trade_through). A gap open above fills at
  the open - half spread.
- If stop and target are both reachable inside one bar and the open does not
  decide it, the stop fills first and the fill is flagged ambiguous.
- No order fills on a bar that starts before the order's activation time.
"""
from __future__ import annotations

from ..config import ExecutionConfig, InstrumentConfig
from ..models import Bar, Fill, Order
from .base import ExecutionAdapter


class CostModel:
    def __init__(self, inst: InstrumentConfig):
        self.inst = inst
        self.tick = inst.tick_size
        self.half_spread = inst.costs.spread_ticks * inst.tick_size / 2.0
        self.slip = inst.costs.slippage_ticks * inst.tick_size
        self.vpp = inst.value_per_point

    def buy_market(self, ref: float) -> float:
        return ref + self.half_spread + self.slip

    def sell_market(self, ref: float) -> float:
        return ref - self.half_spread - self.slip

    def commission(self, qty: float, price: float) -> float:
        if qty <= 0:
            return 0.0
        c = self.inst.costs
        fee = qty * c.commission_per_unit + c.commission_pct_notional * qty * price * self.vpp
        return max(fee, c.min_commission_per_order)

    def commission_per_unit(self, price: float) -> float:
        c = self.inst.costs
        return c.commission_per_unit + c.commission_pct_notional * price * self.vpp

    def net_target_per_unit(self, ref_entry: float, target: float) -> float:
        """Cash per unit if bought at market from ref_entry and sold at the target limit, after fees."""
        entry = self.buy_market(ref_entry)
        return (target - entry) * self.vpp - self.commission_per_unit(entry) - self.commission_per_unit(target)

    def stop_exit_cost_per_unit(self, stop: float) -> float:
        return (self.half_spread + self.slip) * self.vpp + self.commission_per_unit(stop)


class PaperBroker(ExecutionAdapter):
    def __init__(self, instruments: dict[str, InstrumentConfig], exec_cfg: ExecutionConfig):
        self.costs = {s: CostModel(i) for s, i in instruments.items()}
        self.cfg = exec_cfg
        self.orders: dict[str, Order] = {}
        self.cancelled: list[tuple[str, str]] = []

    def submit(self, order: Order) -> None:
        if order.order_id in self.orders:
            raise ValueError(f"duplicate order id {order.order_id}")
        if order.quantity <= 0:
            raise ValueError("order quantity must be positive")
        self.orders[order.order_id] = order

    def cancel(self, order_id: str, reason: str = "") -> None:
        o = self.orders.get(order_id)
        if o and o.status == "open":
            o.status = "cancelled"
            self.cancelled.append((order_id, reason))

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        return [o for o in self.orders.values() if o.status == "open" and (symbol is None or o.symbol == symbol)]

    def _fill(self, o: Order, price: float, ref: float, bar: Bar, at_open: bool, flags=None) -> Fill:
        cm = self.costs[o.symbol]
        o.status = "filled"
        t = bar.start if at_open else bar.end
        return Fill(o.order_id, o.symbol, o.side, o.quantity, price, ref, cm.commission(o.quantity, price), t,
                    o.purpose, list(flags or []) + ([] if at_open else ["intrabar_fill_time_is_bar_close"]))

    def process_bar(self, bar: Bar) -> list[Fill]:
        cm = self.costs[bar.symbol]
        live = [o for o in self.open_orders(bar.symbol) if o.active_from <= bar.start]
        fills: list[Fill] = []

        for o in [o for o in live if o.kind == "market"]:
            px = cm.buy_market(bar.open) if o.side == "buy" else cm.sell_market(bar.open)
            fills.append(self._fill(o, px, bar.open, bar, True))
            if o.side == "sell":    # a market exit closes the position: drop its protective orders
                for other in self.open_orders(bar.symbol):
                    if other.kind in ("stop", "limit") and other.side == "sell":
                        self.cancel(other.order_id, "position_closed_by_market_exit")

        groups: dict[str, list[Order]] = {}
        for o in live:
            if o.status == "open" and o.kind in ("stop", "limit"):
                groups.setdefault(o.oco_group or o.order_id, []).append(o)
        tt = cm.tick if self.cfg.limit_requires_trade_through else 0.0
        for grp in groups.values():
            stop = next((o for o in grp if o.kind == "stop"), None)
            lim = next((o for o in grp if o.kind == "limit"), None)
            hit = None
            if stop and bar.open <= stop.price:
                hit = (stop, cm.sell_market(bar.open), bar.open, True, ["gap_through_stop"])
            elif lim and bar.open - cm.half_spread >= lim.price:
                hit = (lim, bar.open - cm.half_spread, bar.open, True, ["gap_above_target"])
            else:
                s_hit = stop is not None and bar.low <= stop.price
                t_hit = lim is not None and bar.high - cm.half_spread >= lim.price + tt
                if s_hit and t_hit:
                    hit = (stop, cm.sell_market(stop.price), stop.price, False, ["ambiguous_bar_stop_first"])
                elif s_hit:
                    hit = (stop, cm.sell_market(stop.price), stop.price, False, [])
                elif t_hit:
                    hit = (lim, lim.price, lim.price, False, [])
            if hit:
                o, px, ref, at_open, flags = hit
                fills.append(self._fill(o, px, ref, bar, at_open, flags))
                for other in grp:
                    if other is not o:
                        self.cancel(other.order_id, "oco")
        return fills
