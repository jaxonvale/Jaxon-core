import copy

import numpy as np
import pandas as pd
import pytest

from jaxon.backtest import run
from jaxon.challenge import PRESETS, ChallengeRules, simulate
from jaxon.pro import ProBreakout
from tests.helpers import bar, scenario_cfg, scenario_data, ts


# ---------------------------------------------------------------- PRO Breakout port
def _series(prices, start="2025-01-01"):
    t = ts(start)
    out = []
    for i, p in enumerate(prices):
        out.append(bar(t + pd.Timedelta(minutes=5 * i), p, p + 0.2, p - 0.2, p))
    return out


def test_pro_flips_and_signals_only_on_reversal():
    pro = ProBreakout(trend_adj=2)                      # L = 8
    down = [100 - 0.5 * i for i in range(60)]
    up = [down[-1] + 1.0 * i for i in range(1, 60)]
    states = [pro.update(b) for b in _series(down + up)]
    trend = [s.trend for s in states]
    assert -1 in trend[:60] and trend[-1] == 1
    buys = [i for i, s in enumerate(states) if s.long_entry]
    assert len(buys) == 1 and buys[0] >= 60                 # one BUY, after the turn
    assert not any(s.long_entry for s in states[:60])       # first trend (0 -> -1) is not a SELL either
    assert states[-1].atr == pytest.approx(1.0 + 0.0, abs=0.5)


def test_pro_no_lookahead():
    prices = list(np.cumsum(np.random.default_rng(1).normal(0, 1, 400)) + 100)
    bars = _series(prices)
    pf = ProBreakout(3)
    full = [pf.update(b) for b in bars]
    p = ProBreakout(3)
    part = [p.update(b) for b in bars[:250]]
    assert [(s.trend, s.long_entry, s.atr) for s in part] == [(s.trend, s.long_entry, s.atr) for s in full[:250]]


# ---------------------------------------------------------------- modes on the scenario
def test_modes_run_end_to_end():
    c = scenario_cfg()
    d = scenario_data(c)
    for mode, tm in (("jaxon_pro", "pro_atr"), ("pro_breakout", "pro_atr"), ("jaxon", "pro_atr")):
        c2 = copy.deepcopy(c)
        c2.strategy.mode, c2.strategy.target_mode = mode, tm
        r = run(c2, d)
        for t in r.engine.trades:
            assert t.stop < t.entry_price
        assert all(s.outcome != "" or s.status in ("touched", "confirmed") for s in r.engine.setups())


def test_fixed_starting_equity_sizing():
    c = scenario_cfg()
    c.risk.size_from = "starting_equity"
    r = run(c, scenario_data(c))
    assert all(t.planned_risk_cash <= 250 + 1e-6 for t in r.engine.trades)


# ---------------------------------------------------------------- challenge rules
def days_from(pnls, intraday=None, start="2025-01-06"):
    """Daily closes from cumulative P&L; optional intraday path overrides per day."""
    out, eq = [], 0.0
    for k, p in enumerate(pnls):
        path = np.array(intraday[k]) if intraday and k in intraday else np.array([eq, eq + p])
        out.append({"day": pd.Timestamp(start) + pd.Timedelta(days=k), "open": eq, "path": path,
                    "close": path[-1], "trades": 1})
        eq = float(path[-1])
    return out


TS = PRESETS["topstep_50k"]


def test_steady_profits_pass():
    a = simulate(days_from([600] * 10), TS)[0]
    assert a.outcome == "passed" and a.calendar_days == 5


def test_consistency_raises_target():
    a = simulate(days_from([3000, 100, 100]), TS)[0]
    assert a.outcome == "unresolved"            # needs 3000/0.5 = 6000 total
    b = simulate(days_from([3000] + [500] * 6), TS)[0]
    assert b.outcome == "passed" and b.calendar_days == 7


def test_eod_trailing_threshold_breach_intraday():
    days = days_from([1000, -100], intraday={1: [1000, -999, 900]})
    # day 2 dips to 49,001 while the threshold is 51,000 - 2,000 = 49,000 -> survives
    assert simulate(days, TS)[0].outcome == "unresolved"
    days = days_from([1000, -100], intraday={1: [1000, -1000.5, 900]})
    assert simulate(days, TS)[0].outcome == "failed"   # 48,999.5 <= 49,000


def test_intraday_trailing_with_cap_and_expiry():
    rules = PRESETS["apex_50k_intraday"]
    days = days_from([0], intraday={0: [0, 2500, 50]})    # peak 52,500 -> threshold capped at 50,100; 50,050 breaches
    assert simulate(days, rules)[0].outcome == "failed"
    rules2 = ChallengeRules("x", 50_000, 3_000, 2_000, "intraday", None, None, 30)
    days = days_from([0], intraday={0: [0, 2500, 600]})   # uncapped threshold 50,500 < 50,600 -> survives
    assert simulate(days, rules2)[0].outcome == "unresolved"
    assert simulate(days_from([10] * 40), PRESETS["apex_50k_eod"])[0].outcome == "expired"


def test_scale_multiplies_pnl():
    assert simulate(days_from([300] * 20), TS, scale=2.0)[0].calendar_days == 5


# ---------------------------------------------------------------- zone tap + Lucid
def test_buy_limit_fills_only_when_touched():
    from jaxon.config import ExecutionConfig
    from jaxon.execution.paper import PaperBroker
    from jaxon.models import Order
    from tests.helpers import cfg
    inst = cfg(costs={"spread_ticks": 0, "slippage_ticks": 0}).instruments[0]
    b = PaperBroker({"TEST": inst}, ExecutionConfig())
    b.submit(Order("E", "TEST", "buy", "limit", 10, ts("2025-01-01"), 100.0, "entry"))
    assert b.process_bar(bar("2025-01-01 00:00", 101, 102, 100.5, 101)) == []
    f = b.process_bar(bar("2025-01-01 00:05", 101, 101, 99.5, 100))[0]
    assert f.price == 100.0 and f.purpose == "entry"
    b.submit(Order("E2", "TEST", "buy", "limit", 10, ts("2025-01-01"), 100.0, "entry"))
    f2 = b.process_bar(bar("2025-01-01 00:10", 98, 99, 97, 98))[0]          # gap below: better fill at open
    assert f2.price == 98.0


def test_zone_tap_filters():
    from tests.helpers import CoreScenario
    s = CoreScenario()
    core = s.core
    core.last4h_green, core.ema30_down = True, True
    assert core.tap_long_allowed()
    core.last4h_green = False
    assert not core.tap_long_allowed()
    core.cfg.tap_htf_filter = "swing_trend"
    assert core.tap_long_allowed()                    # scenario has a bullish 4H swing bias
    core.ema30_down = False
    assert not core.tap_long_allowed()


def test_zone_tap_mode_runs_and_uses_limit_entries():
    c = scenario_cfg()
    c.strategy.mode = "zone_tap"
    r = run(c, scenario_data(c))
    for s in r.engine.setups():
        assert s.proposed_entry == s.zone_upper or abs(s.proposed_entry - s.zone_upper) < 0.011
        assert s.stop < s.zone_lower and s.target > s.proposed_entry


def test_lucid_presets_lock_at_start_plus_100():
    r = PRESETS["lucidpro_25k"]
    # +1,200 day lifts EOD high to 26,200 -> threshold min(25,200, 25,100) = 25,100
    days = days_from([1200, 0], intraday={1: [1200, 101, 150]})     # dips to 25,101 -> survives
    assert simulate(days, r)[0].outcome == "unresolved"
    days = days_from([1200, 0], intraday={1: [1200, 100, 150]})     # touches 25,100 -> breach
    assert simulate(days, r)[0].outcome == "failed"
    assert simulate(days_from([700, 600]), PRESETS["lucidflex_25k"])[0].outcome == "unresolved"   # 700/1300 > 40%
    assert simulate(days_from([400, 400, 450]), PRESETS["lucidflex_25k"])[0].outcome == "passed"
