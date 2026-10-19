import pandas as pd
import pytest

from jaxon.config import ExecutionConfig, RiskConfig
from jaxon.execution.base import LiveBrokerNotImplemented
from jaxon.execution.paper import CostModel, PaperBroker
from jaxon.models import Order
from jaxon.risk import size_position, total_risk
from tests.helpers import bar, cfg, ts


def broker(spread=2, slip=1, per_unit=0.01, pct=0.0, min_fee=0.0, trade_through=True):
    c = cfg(costs={"spread_ticks": spread, "slippage_ticks": slip, "commission_per_unit": per_unit,
                   "commission_pct_notional": pct, "min_commission_per_order": min_fee})
    inst = c.instruments[0]
    return PaperBroker({"TEST": inst}, ExecutionConfig(limit_requires_trade_through=trade_through)), inst


def oco(b, stop=99.0, target=102.0, active=ts("2025-01-01 00:00"), qty=100):
    b.submit(Order("S", "TEST", "sell", "stop", qty, active, stop, "stop", "G"))
    b.submit(Order("L", "TEST", "sell", "limit", qty, active, target, "target", "G"))


# ---------------------------------------------------------------- fills
def test_market_fill_costs_and_fees():
    b, _ = broker(spread=2, slip=1, per_unit=0.01)
    b.submit(Order("E", "TEST", "buy", "market", 100, ts("2025-01-01 00:00"), purpose="entry"))
    f = b.process_bar(bar("2025-01-01 00:00", 100, 101, 99.5, 100.5))[0]
    assert f.price == pytest.approx(100 + 0.01 + 0.01)     # half spread (1 tick) + 1 tick slippage
    assert f.fee == pytest.approx(1.0) and f.time == ts("2025-01-01 00:00")


def test_order_cannot_fill_before_activation():
    b, _ = broker()
    oco(b, active=ts("2025-01-01 00:10"))
    assert b.process_bar(bar("2025-01-01 00:05", 100, 105, 95, 100)) == []   # bar starts before activation
    assert len(b.process_bar(bar("2025-01-01 00:10", 100, 105, 95, 100))) == 1


def test_gap_through_stop_fills_at_open_not_stop():
    b, _ = broker(spread=0, slip=1)
    oco(b)
    f = b.process_bar(bar("2025-01-01 00:00", 97.0, 97.5, 96.5, 97.2))[0]
    assert f.purpose == "stop" and f.price == pytest.approx(96.99) and "gap_through_stop" in f.flags
    assert b.orders["L"].status == "cancelled"


def test_ambiguous_bar_is_stop_first_and_flagged():
    b, _ = broker(spread=0, slip=0)
    oco(b)
    f = b.process_bar(bar("2025-01-01 00:00", 100, 103, 98, 100))
    assert len(f) == 1 and f[0].purpose == "stop" and "ambiguous_bar_stop_first" in f[0].flags


def test_finer_bars_resolve_ambiguity():
    b, _ = broker(spread=0, slip=0)
    oco(b)
    ones = [bar("2025-01-01 00:00", 100, 102.5, 99.8, 102.2, minutes=1),   # target first
            bar("2025-01-01 00:01", 102.2, 102.3, 98.0, 98.5, minutes=1)]
    fills = [f for x in ones for f in b.process_bar(x)]
    assert [f.purpose for f in fills] == ["target"] and not any("ambiguous" in fl for fl in fills[0].flags)


def test_target_requires_trade_through_and_spread():
    b, _ = broker(spread=0, slip=0, trade_through=True)
    oco(b)
    assert b.process_bar(bar("2025-01-01 00:00", 100, 102.0, 99.5, 101)) == []   # touch only
    assert b.process_bar(bar("2025-01-01 00:05", 101, 102.01, 100.5, 101))[0].price == 102.0
    b2, _ = broker(spread=4, slip=0, trade_through=False)    # half spread = 0.02
    oco(b2)
    assert b2.process_bar(bar("2025-01-01 00:00", 100, 102.01, 99.5, 101)) == []  # bid only 101.99
    assert b2.process_bar(bar("2025-01-01 00:05", 101, 102.02, 100.5, 101))[0].purpose == "target"


def test_gap_above_target_fills_at_open():
    b, _ = broker(spread=0, slip=0)
    oco(b)
    f = b.process_bar(bar("2025-01-01 00:00", 103, 104, 102.5, 103.5))[0]
    assert f.price == 103 and "gap_above_target" in f.flags


def test_live_broker_refuses():
    with pytest.raises(NotImplementedError):
        LiveBrokerNotImplemented().submit(None)


# ---------------------------------------------------------------- sizing
def _inst(**kw):
    base = dict(costs={"spread_ticks": 0, "slippage_ticks": 0})
    base.update(kw)
    return cfg(**base).instruments[0]


def test_equity_sizing_rounds_down_to_lot():
    inst = _inst()
    r = size_position(100_000, 100.0, 99.0, inst, CostModel(inst), RiskConfig())
    assert r.ok and r.quantity == 250 and r.planned_risk_cash == pytest.approx(250)
    inst = _inst(lot_size=100, min_quantity=100)
    r = size_position(100_000, 100.0, 99.0, inst, CostModel(inst), RiskConfig())
    assert r.quantity == 200                                  # 250 rounded down to the 100-lot


def test_futures_multiplier_sizing():
    inst = _inst(asset_class="future", tick_size=0.25, value_per_point=50.0, initial_margin_per_unit=15000)
    cm = CostModel(inst)
    r = size_position(100_000, 5000.0, 4990.0, inst, cm, RiskConfig(max_leverage=20))
    assert not r.ok and r.reason == "min_size_exceeds_risk_budget"      # 1 contract risks $500 > $250
    r = size_position(1_000_000, 5000.0, 4990.0, inst, cm, RiskConfig(max_leverage=20))
    assert r.quantity == 5 and r.planned_risk_cash == pytest.approx(2500)  # $500/contract, $2,500 budget
    r = size_position(1_000_000, 5000.0, 4990.0, inst, cm, RiskConfig(max_leverage=20, max_margin_usage=0.05))
    assert r.quantity == 3                                                 # margin cap: 50k / 15k


def test_forex_lot_sizing():
    inst = _inst(asset_class="forex", tick_size=0.00001, lot_size=1000, min_quantity=1000)
    r = size_position(100_000, 1.10000, 1.09800, inst, CostModel(inst), RiskConfig(max_leverage=100))
    assert r.quantity == 125_000                                           # $250 / 0.002


def test_costs_and_buying_power_limit_size():
    inst = _inst(costs={"spread_ticks": 2, "slippage_ticks": 1, "commission_per_unit": 0.0,
                        "min_commission_per_order": 5.0})
    cm = CostModel(inst)
    r = size_position(100_000, 100.0, 99.0, inst, cm, RiskConfig())
    assert total_risk(r.quantity, 100.0, 99.0, cm) <= 250 and r.quantity < 250
    inst = _inst()
    r = size_position(100_000, 100.0, 99.95, inst, CostModel(inst), RiskConfig(max_leverage=1.0))
    assert r.quantity == 1000                                             # notional cap $100k at $100


def test_open_risk_limit():
    inst = _inst()
    r = size_position(100_000, 100.0, 99.0, inst, CostModel(inst), RiskConfig(), open_risk_cash=500)
    assert not r.ok and r.reason == "open_risk_limit"
