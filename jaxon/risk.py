"""Position sizing and limit checks. Deterministic; no discretion."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import InstrumentConfig, RiskConfig
from .execution.paper import CostModel


def floor_lot(q: float, lot: float) -> float:
    if q <= 0:
        return 0.0
    return round(math.floor(q / lot * (1 + 1e-9) + 1e-12) * lot, 10)   # tolerate float noise


@dataclass
class SizeResult:
    quantity: float
    risk_per_unit: float
    planned_risk_cash: float
    notional: float
    reason: str = ""          # empty = ok

    @property
    def ok(self) -> bool:
        return not self.reason and self.quantity > 0


def total_risk(q: float, entry: float, stop: float, cm: CostModel) -> float:
    """Cash lost if the stop fills at its price: move + exit friction + both commissions."""
    return (q * (entry - stop) * cm.vpp + q * (cm.half_spread + cm.slip) * cm.vpp
            + cm.commission(q, entry) + cm.commission(q, stop))


def size_position(equity: float, entry: float, stop: float, inst: InstrumentConfig, cm: CostModel,
                  rc: RiskConfig, open_risk_cash: float = 0.0) -> SizeResult:
    if not stop < entry:
        return SizeResult(0, 0, 0, 0, "stop_not_below_entry")
    lot = inst.lot_size
    rpu = (entry - stop) * cm.vpp + cm.stop_exit_cost_per_unit(stop) + cm.commission_per_unit(entry)
    budget = equity * rc.risk_per_trade_pct

    def max_lots_within(cash: float, cap_lots: int) -> int:
        """Largest lot count whose total stop-out cost fits in `cash` (min commissions are per order)."""
        lo, hi = 0, max(cap_lots, 0)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if total_risk(mid * lot, entry, stop, cm) <= cash * (1 + 1e-9):
                lo = mid
            else:
                hi = mid - 1
        return lo

    q_risk = floor_lot(max_lots_within(budget, int(round(floor_lot(budget / rpu, lot) / lot))) * lot, lot)
    if q_risk < inst.min_quantity:
        return SizeResult(0, rpu, 0, 0, "min_size_exceeds_risk_budget")

    q = min(q_risk, floor_lot(equity * rc.max_leverage / (entry * cm.vpp), lot))
    if inst.initial_margin_per_unit:
        q = min(q, floor_lot(equity * rc.max_margin_usage / inst.initial_margin_per_unit, lot))
    if inst.max_quantity:
        q = min(q, floor_lot(inst.max_quantity, lot))
    room = equity * rc.max_open_risk_pct - open_risk_cash
    q = floor_lot(max_lots_within(room, int(round(q / lot))) * lot, lot)
    if q < inst.min_quantity:
        reason = ("open_risk_limit" if total_risk(inst.min_quantity, entry, stop, cm) > room * (1 + 1e-9)
                  else "buying_power_limit")
        return SizeResult(0, rpu, 0, 0, reason)
    notional = q * entry * cm.vpp
    if notional < inst.min_notional:
        return SizeResult(0, rpu, 0, notional, "below_min_notional")
    return SizeResult(q, rpu, total_risk(q, entry, stop, cm), notional)
