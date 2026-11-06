import pandas as pd

from jaxon.models import State
from tests.helpers import CoreScenario, mid_bar, ts


def ready(**over):
    s = CoreScenario(over)
    s.build_zone()
    s.after_zone_30m()
    return s


def touch(s):
    s.five_min_swing_high()                        # threshold 102.85
    return s.feed5(101.6, 101.65, 100.9, 101.0)     # overlaps zone [100.5, 101.0]


# ---------------------------------------------------------------- 4H bias
def test_bias_needs_two_rising_highs_and_lows():
    s = CoreScenario()
    assert s.core.bias == "bullish"
    # a lower high then lower low flips bias to neutral once confirmed
    start = ts("2025-01-03 04:00")
    for i, m in enumerate([13, 12.5, 11, 9, 10, 11, 12]):
        s.core.on_4h(mid_bar(start + pd.Timedelta(hours=4 * i), m, minutes=240))
    assert s.core.bias == "neutral"


def test_missing_swing_history_is_neutral():
    from jaxon.journal import IdFactory, Journal
    from jaxon.strategy import StrategyCore
    from tests.helpers import cfg
    core = StrategyCore("TEST", cfg().strategy, Journal(), IdFactory())
    for i, m in enumerate([10, 11, 12, 11, 10]):
        core.on_4h(mid_bar(ts("2025-01-01") + pd.Timedelta(hours=4 * i), m, minutes=240))
    assert core.bias == "neutral" and core.bias_detail["reason"] == "insufficient_swing_history"


# ---------------------------------------------------------------- zones
def test_zone_activates_at_displacement_close():
    s = CoreScenario()
    d = s.build_zone()
    z = s.zone()
    assert z is not None and z.activated_at == d.end
    assert (z.lower, z.upper) == (100.5, 101.0)
    assert z.broken_swing_high == 102.0
    assert z.atr_at_displacement < (d.close - d.open) / 1.5


def test_no_zone_when_body_too_small_or_no_break_or_no_base():
    s = CoreScenario()
    s.build_zone(displacement_close=101.9)          # does not close above the 102.0 swing high
    assert s.zone() is None
    s = CoreScenario({"displacement_atr_mult": 5.0})
    s.build_zone()
    assert s.zone() is None
    s = CoreScenario()
    s.build_zone(base_bearish=False)
    assert s.zone() is None
    assert any(r["kind"] == "zone_skipped" and r["reason"] == "no_bearish_base" for r in s.j.records)


def test_swing_high_confirmed_by_displacement_itself_is_not_used():
    s = CoreScenario()
    for i in range(15):
        m = 99.0 + 0.1 * i
        s.feed30(i, m - 0.05, m + 0.5, m - 0.5, m + 0.05)
    s.feed30(15, 101.3, 102.0, 101.0, 101.5)        # peak
    s.feed30(16, 101.0, 101.0, 100.5, 100.6)        # bearish; peak needs one more bar to confirm
    s.feed30(17, 100.6, 103.1, 100.5, 103.0)        # would-be displacement confirms the peak itself
    assert s.zone() is None
    assert any(r.get("reason") == "no_prior_30m_swing_high" for r in s.j.records)


def test_zone_invalidated_on_30m_close_below_and_expires():
    s = CoreScenario()
    s.build_zone()
    s.feed30(s.d_index + 1, 103.0, 103.1, 100.3, 100.4)   # close below 100.5
    assert s.zone().status == "invalidated" and s.zone().status_at == s.t30(s.d_index + 2)
    s = CoreScenario({"zone_expiry_bars": 3})
    s.build_zone()
    for k in range(1, 4):
        s.feed30(s.d_index + k, 103.0, 103.2, 102.8, 103.0)
    assert s.zone().status == "expired" and s.zone().bars_since_activation == 3


def test_most_recent_zone_chosen():
    s = ready()
    older = s.zone()
    from jaxon.models import Zone
    newer = Zone("Z-new", "TEST", 100.6, 101.1, older.activated_at + pd.Timedelta(minutes=30),
                 older.base_bar_start, older.displacement_bar_start, 2, 1, 102)
    s.core.zones.append(newer)
    touch(s)
    assert s.core.setup.zone_id == "Z-new"


# ---------------------------------------------------------------- touch / pullback
def test_touch_must_follow_activation_and_pullback_filter():
    s = CoreScenario()
    s.build_zone()
    s.after_zone_30m(closes=(102.5, 102.9))         # rising 30m closes: filter fails
    touch(s)
    assert s.core.setup is None and s.core.touch_skips["pullback_filter_failed"] == 1
    assert not s.zone().attempt_used                 # skipped touch does not consume the attempt


def test_touch_without_eligible_swing_high_is_skipped():
    s = ready()
    s.feed5(101.6, 101.65, 100.9, 101.0)            # no 5m swing high since activation
    assert s.core.setup is None and s.core.touch_skips["no_eligible_5m_swing_high"] == 1


def test_touch_freezes_threshold_and_tracks_low():
    s = ready()
    thr = s.five_min_swing_high()
    b, _ = s.feed5(101.6, 101.65, 100.9, 101.0)
    st = s.core.setup
    assert st is not None and s.core.state == State.ZONE_TOUCHED
    assert st.break_threshold == thr and st.touch_at == b.end and st.lowest_low == 100.9
    assert st.threshold_pivot_available_at <= b.end
    assert s.zone().attempt_used


# ---------------------------------------------------------------- structure shift
def test_threshold_frozen_wick_not_enough_close_confirms():
    s = ready()
    touch(s)
    thr = s.core.setup.break_threshold
    # form a new, lower 5m swing high after the touch: threshold must NOT move
    for m in (101.2, 101.6, 101.3, 101.1, 101.15):
        s.feed5(m, m + 0.05, m - 0.05, m)
    assert s.core.setup.break_threshold == thr
    s2 = ready()
    touch(s2)
    _, res = s2.feed5(101.0, 103.5, 100.9, 102.5)   # wick above threshold, close below
    assert res is None and s2.core.setup.status == "touched"
    _, res = s2.feed5(102.5, 103.2, 102.4, 103.0)   # close above
    assert res is not None and res.status == "confirmed" and s2.core.state == State.SHIFT_CONFIRMED
    assert res.lowest_low == 100.9


def test_confirmation_timeout_consumes_attempt():
    s = ready()
    touch(s)
    for _ in range(6):
        s.feed5(101.2, 101.4, 101.1, 101.3)
    assert s.core.setup is None
    st = s.core.setups[-1]
    assert st.status == "timeout" and st.outcome == "timeout_no_structure_shift"
    touch(s)                                          # zone touched again: no new attempt
    assert len(s.core.setups) == 1


def test_simultaneous_cancel_and_confirm_is_cancel():
    s = ready()
    touch(s)
    _, res = s.feed5(101.0, 103.5, 100.4, 103.2)      # trades below zone AND closes above threshold
    assert res is None
    assert s.core.setups[-1].outcome == "cancelled_traded_below_zone"
    assert s.core.state != State.SHIFT_CONFIRMED


def test_cancel_on_zone_invalidation_and_bias_loss():
    s = ready()
    touch(s)
    s.core.on_30m(mid_bar(s.t5, 100.2, half=0.2, minutes=30))   # 30m close below zone
    s.feed5(101.0, 101.2, 100.6, 101.1)
    assert s.core.setups[-1].outcome == "cancelled_zone_invalidated"
    s = ready()
    touch(s)
    s.core.bias = "neutral"
    s.feed5(101.0, 101.2, 100.6, 101.1)
    assert s.core.setups[-1].outcome == "cancelled_bias_lost"


def test_plan_rules():
    s = ready()
    touch(s)
    _, st = s.feed5(102.5, 103.2, 102.4, 103.0)
    net = lambda e, t: (t - e)
    p = s.core.plan(st, 103.0, None, "missing_volume", 0.01, net)
    assert not p.ok and p.reason == "vwap_unavailable:missing_volume"
    p = s.core.plan(st, 103.0, 102.0, "", 0.01, net)
    assert p.reason == "target_not_above_entry"
    p = s.core.plan(st, 103.0, 104.0, "", 0.01, net)            # RR = 1 / 2.11
    assert p.reason == "rr_below_min"
    p = s.core.plan(st, 103.0, 107.0, "", 0.01, lambda e, t: -1)
    assert p.reason == "costs_exceed_reward"
    p = s.core.plan(st, 103.0, 107.0, "", 0.01, net)
    assert p.ok and p.stop == 100.89 and p.target == 107.0      # stop = lowest low 100.9 - 1 tick
