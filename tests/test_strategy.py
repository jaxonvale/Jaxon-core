import numpy as np
import pandas as pd

from trading_bot.backtest import backtest
from trading_bot.config import StrategyConfig
from trading_bot.data import resample, synthetic
from trading_bot.indicators import session_vwap, swings
from trading_bot.strategy import generate


def test_signals_have_no_lookahead():
    """Signals found on truncated data must match the full-data run exactly."""
    df = synthetic(days=60, seed=3)
    _, full = generate(df)
    assert full, "demo data should produce signals"
    cut = df.index[len(df) * 2 // 3]
    _, part = generate(df[df.index < cut])
    expected = [s for s in full if s.time < cut - pd.Timedelta("5min")]
    got = [s for s in part if s.time < cut - pd.Timedelta("5min")]
    assert [s.as_dict() for s in got] == [s.as_dict() for s in expected]


def test_every_signal_matches_rules():
    cfg = StrategyConfig()
    df = synthetic(days=60, seed=11)
    f, sigs = generate(df, cfg)
    for s in sigs:
        row = f.loc[s.time]
        assert row.htf_up                                  # 4H up
        assert row.close > row.vwap                        # closed above VWAP
        assert row.low <= row.vwap * (1 + cfg.vwap_touch_tol)  # tagged VWAP
        assert s.stop < s.swing_low < s.close_price        # stop under the lows
        assert cfg.min_stop_pct <= (s.close_price - s.stop) / s.close_price <= cfg.max_stop_pct


def test_no_longs_in_downtrend():
    df = synthetic(days=40, seed=5)
    flipped = df.copy()
    flipped[["open", "close"]] = 1e4 / df[["open", "close"]]
    flipped["high"], flipped["low"] = 1e4 / df["low"], 1e4 / df["high"]
    f, _ = generate(flipped)
    # steady downtrend: 4H trend should almost never read as up
    assert f.htf_up.mean() < 0.35


def test_swing_confirmed_after_right_bars():
    idx = pd.date_range("2026-01-01", periods=9, freq="5min", tz="UTC")
    high = [1, 2, 3, 9, 3, 2, 1, 1, 1]
    df = pd.DataFrame({"high": high, "low": np.array(high) - 0.5}, index=idx)
    sw = swings(df, 2)
    assert np.isnan(sw["sh"].iloc[4])           # pivot at bar 3 not known yet
    assert sw["sh"].iloc[5] == 9                # known 2 bars later
    assert sw["sh_time"].iloc[5] == idx[3]


def test_vwap_resets_each_session():
    idx = pd.date_range("2026-01-01 23:50", periods=4, freq="5min", tz="UTC")
    df = pd.DataFrame({"high": [10, 10, 20, 20], "low": [10, 10, 20, 20], "close": [10, 10, 20, 20],
                       "volume": [1, 1, 1, 1]}, index=idx)
    v = session_vwap(df)
    assert v.iloc[1] == 10 and v.iloc[2] == 20  # new day starts fresh


def test_resample_drops_unfinished_bar():
    df = synthetic(days=1)
    df = df.iloc[:-3]                            # last 4H bar incomplete
    r = resample(df, "4h")
    assert r.index[-1] + pd.Timedelta("4h") <= df.index[-1] + pd.Timedelta("5min")


def test_backtest_runs_and_risk_is_bounded():
    trades, st, _ = backtest(synthetic(days=60, seed=3))
    assert st["trades"] > 0
    assert trades.r.min() > -1.5                 # a stop costs ~1R, plus fees/slip
    assert set(trades.reason) <= {"stop", "target", "breakeven", "time", "open_at_end"}
    # trades never overlap
    assert (trades.entry_time.iloc[1:].values > trades.exit_time.iloc[:-1].values).all()
