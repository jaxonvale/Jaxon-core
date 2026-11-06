"""Temporal leakage: appending future bars must not change anything already decided.

For several cut times we run the engine on bars before the cut (no finalize) and
compare against the full run: emitted signals, journal records, pivots, zones,
setups decided before the cut, closed trades and the equity curve must match.
"""
from dataclasses import asdict

import pytest

from jaxon.config import load_config
from jaxon.backtest import load_data
from jaxon.engine import TradingEngine
from tests.helpers import ROOT, scenario_cfg, ts


def run_until(cfg, data, cut=None) -> TradingEngine:
    e = TradingEngine(cfg, data.provenance)
    for b in data.bars:
        if cut is not None and b.start >= cut:
            break
        e.process(b)
    return e


def known(seq, cut, key):
    return [asdict(x) if hasattr(x, "__dataclass_fields__") else x for x in seq if key(x) <= cut]


def compare(full: TradingEngine, part: TradingEngine, cut):
    assert part.last_event_time < cut
    sym = list(full.rt)[0]
    fc, pc = full.rt[sym].core, part.rt[sym].core
    assert known(full.signals, cut, lambda s: s.created_at) == known(part.signals, cut, lambda s: s.created_at)
    for name in ("p4", "p30", "p5"):
        for kind in ("highs", "lows"):
            f = getattr(getattr(fc, name), kind)
            p = getattr(getattr(pc, name), kind)
            assert known(f, cut, lambda q: q.available_at) == known(p, cut, lambda q: q.available_at)
    fz = [(z.zone_id, z.lower, z.upper, z.activated_at) for z in fc.zones if z.activated_at <= cut]
    pz = [(z.zone_id, z.lower, z.upper, z.activated_at) for z in pc.zones]
    assert fz == pz
    done_f = [s.as_record() for s in fc.setups if s.outcome_at is not None and s.outcome_at < cut]
    done_p = [s.as_record() for s in pc.setups if s.outcome_at is not None and s.outcome_at < cut]
    assert done_f == done_p
    frozen = lambda s: (s.setup_id, s.break_threshold, s.touch_at, s.zone_id, s.vwap_snapshot if s.confirm_at
                        and s.confirm_at < cut else None)
    assert [frozen(s) for s in fc.setups if s.touch_at < cut] == [frozen(s) for s in pc.setups]
    assert [t.as_record() for t in full.trades if t.exit_time < cut] == \
           [t.as_record() for t in part.trades if t.exit_time < cut]
    jf = [r for r in full.journal.records if r["ts"] < cut.isoformat()]
    jp = [r for r in part.journal.records if r["ts"] < cut.isoformat()]
    assert jf == jp
    assert [x for x in full.equity_curve if x[0] < cut] == [x for x in part.equity_curve if x[0] < cut]


SCENARIO_CUTS = ["2025-01-09 00:02", "2025-01-11 09:31", "2025-01-11 12:17", "2025-01-11 12:36",
                 "2025-01-11 12:41", "2025-01-12 13:00", "2025-01-13 18:00"]


@pytest.fixture(scope="module")
def scenario_full():
    c = scenario_cfg()
    d = load_data(c)
    return c, d, run_until(c, d)


@pytest.mark.parametrize("cut", SCENARIO_CUTS)
def test_scenario_no_leakage(scenario_full, cut):
    c, d, full = scenario_full
    compare(full, run_until(c, d, ts(cut)), ts(cut))


def test_random_fixture_no_leakage():
    c = load_config(ROOT / "configs" / "example_crypto.yaml")
    d = load_data(c)
    d.bars = d.bars[: 288 * 60]                       # 60 days keeps the test fast
    full = run_until(c, d)
    assert full.zones() and full.setups()            # the comparison is not vacuous
    for cut in ("2025-01-20 07:33", "2025-02-11 15:01", "2025-02-27 23:59"):
        compare(full, run_until(c, d, ts(cut)), ts(cut))
