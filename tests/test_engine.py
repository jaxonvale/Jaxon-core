import copy
import math

import pandas as pd
import pytest

from jaxon.backtest import run
from jaxon.engine import TradingEngine
from jaxon.models import State
from jaxon.replay import PaperReplay, RecoveryError
from tests.helpers import modified_data, replace_bar, run_scenario, scenario_cfg, scenario_data, ts

DAY5_SIGNAL = ts("2025-01-11 12:35")
DAY6_SIGNAL = ts("2025-01-12 12:35")

ALLOWED = {
    State.IDLE: {State.BULLISH_BIAS, State.DAILY_LOCKOUT},
    State.BULLISH_BIAS: {State.ZONE_ACTIVE, State.IDLE, State.DAILY_LOCKOUT, State.ZONE_TOUCHED},
    State.ZONE_ACTIVE: {State.ZONE_TOUCHED, State.BULLISH_BIAS, State.IDLE, State.DAILY_LOCKOUT},
    State.ZONE_TOUCHED: {State.SHIFT_CONFIRMED, State.INVALIDATED, State.DAILY_LOCKOUT},
    State.SHIFT_CONFIRMED: {State.ENTRY_PENDING, State.INVALIDATED},
    State.ENTRY_PENDING: {State.POSITION_OPEN, State.INVALIDATED, State.DAILY_LOCKOUT},
    State.POSITION_OPEN: {State.CLOSED, State.DAILY_LOCKOUT},
    State.CLOSED: {State.IDLE, State.BULLISH_BIAS, State.ZONE_ACTIVE, State.DAILY_LOCKOUT},
    State.INVALIDATED: {State.IDLE, State.BULLISH_BIAS, State.ZONE_ACTIVE, State.DAILY_LOCKOUT},
    State.DAILY_LOCKOUT: {State.IDLE, State.BULLISH_BIAS, State.ZONE_ACTIVE, State.POSITION_OPEN, State.CLOSED},
}


@pytest.fixture(scope="module")
def base():
    c = scenario_cfg()
    d = scenario_data(c)
    return c, d, run(c, d)


def setup_at(r, t):
    return next(s for s in r.engine.setups() if s.confirm_at == t)


# ---------------------------------------------------------------- end to end
def test_scenario_trades_and_records(base):
    _, d, r = base
    assert d.synthetic
    e = r.engine
    assert [t.exit_reason for t in e.trades] == ["target", "stop"]
    win, loss = e.trades
    assert win.entry_time == DAY5_SIGNAL and win.exit_price == win.target and win.r_multiple > 1.5
    assert loss.r_multiple == pytest.approx(-1.0, abs=0.01)
    assert win.planned_risk_cash <= 0.0025 * 100_000 + 1e-6
    s = setup_at(r, DAY5_SIGNAL)
    assert s.target == math.floor(s.vwap_snapshot * 100) / 100       # fixed VWAP snapshot, floored to tick
    assert s.stop == round(s.lowest_low - 0.01, 2)
    assert s.vwap_note == "bar_based_vwap_approximation"


def test_transitions_are_legal_and_ids_unique(base):
    _, _, r = base
    tr = r.engine.journal.transitions("SCENARIO")
    for x in tr:
        assert State(x["to"]) in ALLOWED[State(x["frm"])], x
    ids = [s.setup_id for s in r.engine.setups()] + [z.zone_id for z in r.engine.zones()] + \
          [t.trade_id for t in r.engine.trades] + list(r.engine.broker.orders)
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------- duplicates / restart
def test_duplicate_events_do_not_duplicate_entries(base):
    c, d, r = base
    eng = TradingEngine(c, d.provenance)
    for b in d.bars:
        eng.process(b)
        assert eng.process(b) is False                # re-delivered bar ignored
    eng.finalize()
    assert eng.duplicates_ignored == len(d.bars)
    assert [t.as_record() for t in eng.trades] == [t.as_record() for t in r.engine.trades]


def _comparable(journal):
    return [{k: v for k, v in r.items() if k != "seq"} for r in journal if r["kind"] != "recovered"]


# crash_at N: bars 1..N-1 were checkpointed, bar N was processed but its state was lost
@pytest.mark.parametrize("crash_at,state", [(700, State.IDLE), (1588, State.ZONE_TOUCHED),
                                            (1592, State.ENTRY_PENDING), (1593, State.POSITION_OPEN)])
def test_restart_recovery_matches_uninterrupted_run(tmp_path, base, crash_at, state):
    c, d, _ = base
    full = PaperReplay(c, tmp_path / "a", d)
    full.run(checkpoint_every=50)
    rp = PaperReplay(c, tmp_path / "b", d)
    assert rp.run(checkpoint_every=1, stop_before=crash_at) == {"crashed_after_bars": crash_at}
    rp2 = PaperReplay(c, tmp_path / "b", d)          # new process: restore from disk
    assert rp2.restore() and rp2.engine.state("SCENARIO") == state
    out = rp2.run(checkpoint_every=25)
    assert out["resumed"] is True
    assert [t.as_record() for t in rp2.engine.trades] == [t.as_record() for t in full.engine.trades]
    assert _comparable(rp2.engine.journal.records) == _comparable(full.engine.journal.records)
    lines = (tmp_path / "b" / "journal.jsonl").read_text().splitlines()
    assert len(lines) == len(rp2.engine.journal.records)
    again = PaperReplay(c, tmp_path / "b", d).run()   # nothing new: idempotent
    assert again["bars_this_run"] == 0 and again["trades_closed"] == 2


def test_recovery_refuses_changed_config_or_corrupt_state(tmp_path, base):
    c, d, _ = base
    PaperReplay(c, tmp_path, d).run(max_bars=100)
    c2 = copy.deepcopy(c)
    c2.strategy.min_reward_risk = 2.0
    with pytest.raises(RecoveryError, match="config changed"):
        PaperReplay(c2, tmp_path, d).run()
    snap = tmp_path / "engine_state.pkl"
    snap.write_bytes(snap.read_bytes()[:-10])
    with pytest.raises(RecoveryError, match="checksum"):
        PaperReplay(c, tmp_path, d).run()


# ---------------------------------------------------------------- entry rechecks / gaps
def _with_entry_open(d, open_px):
    def fn(bars):
        out = []
        for b in bars:
            if b.start == DAY5_SIGNAL:
                b = replace_bar(b, open=open_px, high=max(b.high, open_px), low=min(b.low, open_px))
            out.append(b)
        return out
    return modified_data(d, fn)


@pytest.mark.parametrize("open_px,reason", [(116.0, "entry_rr_below_min"), (112.0, "gapped_below_stop"),
                                            (117.0, "gapped_above_target")])
def test_entry_gap_invalidates(base, open_px, reason):
    c, d, _ = base
    r = run(c, _with_entry_open(d, open_px))
    s = setup_at(r, DAY5_SIGNAL)
    assert s.status == "rejected" and s.outcome == reason and s.trade_id is None
    assert DAY5_SIGNAL not in [t.entry_time for t in r.engine.trades]


def test_stale_feed_blocks_entry(base):
    c, d, _ = base
    gap = ts("2025-01-11 12:20")
    r = run(c, modified_data(d, lambda bars: [b for b in bars if b.start != gap]))
    assert any(rec["kind"] == "feed_stale" for rec in r.engine.journal.records)
    assert DAY5_SIGNAL not in [t.entry_time for t in r.engine.trades]
    outcomes = {s.outcome for s in r.engine.setups()}
    assert outcomes & {"stale_feed", "vwap_unavailable:intra_session_data_gap"}


# ---------------------------------------------------------------- daily lockout
def test_daily_lockout_flattens_and_blocks(base):
    c, d, _ = base
    c2 = copy.deepcopy(c)
    c2.risk.max_daily_drawdown_pct = 0.001
    r = run(c2, d)
    e = r.engine
    assert len(e.lockouts) == 1
    lk = e.lockouts[0]
    assert lk["risk_day"] == "2025-01-12" and lk["flatten_fill"] is not None
    assert lk["overshoot_pct"] is not None and lk["overshoot_pct"] >= 0
    assert e.trades[-1].exit_reason == "daily_lockout_flatten"
    tr = e.journal.transitions("SCENARIO")
    assert any(x["to"] == "DAILY_LOCKOUT" for x in tr)
    assert any(x["frm"] == "DAILY_LOCKOUT" and x["reason"] == "new_risk_day" for x in tr)
    t_lock = pd.Timestamp(lk["time"])
    assert not [s for s in e.signals if t_lock < s.created_at < ts("2025-01-13")]


# ---------------------------------------------------------------- missing / unsuitable data
def test_no_trade_without_volume(base):
    c, d, _ = base
    r = run(c, modified_data(d, lambda bars: [replace_bar(b, volume=float("nan")) for b in bars]))
    assert r.engine.trades == []
    assert "vwap_unavailable:missing_volume" in {s.outcome for s in r.engine.setups()}


def test_no_trade_with_unsuitable_volume_type(base):
    c, d, _ = base
    c2 = copy.deepcopy(c)
    c2.instruments[0].volume_type = "tick"
    r = run(c2, d)
    assert r.engine.trades == []
    assert "vwap_unavailable:volume_type_not_allowed" in {s.outcome for s in r.engine.setups()}


def test_no_trade_without_4h_history(base):
    c, d, _ = base
    r = run(c, modified_data(d, lambda bars: [b for b in bars if b.start >= ts("2025-01-10")]))
    assert r.engine.trades == [] and r.engine.signals == []


# ---------------------------------------------------------------- session end
def test_session_end_flatten_uses_real_bar(base):
    c, d, _ = base

    def flat(bars):
        out = []
        for b in bars:
            if DAY5_SIGNAL < b.start < ts("2025-01-12"):
                b = replace_bar(b, open=113.8, high=113.85, low=113.75, close=113.8)
            out.append(b)
        return out
    r = run(c, modified_data(d, flat))
    t = next(t for t in r.engine.trades if t.entry_time == DAY5_SIGNAL)
    assert t.exit_reason == "session_flatten" and t.exit_time == ts("2025-01-11 23:55")
    assert t.exit_price == pytest.approx(113.8 - 0.005 - 0.01)


def test_entries_per_session_limit(base):
    c, d, _ = base
    e = TradingEngine(c, d.provenance)
    for b in d.bars:
        if b.start == ts("2025-01-11 06:00"):
            e.entries_today = c.risk.max_entries_per_session      # limit already used up today
        e.process(b)
    assert [t.entry_time for t in e.trades] == [DAY6_SIGNAL]      # day 5 blocked, day 6 resets
