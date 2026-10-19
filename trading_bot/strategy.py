"""Multi-timeframe long setup.

  1. 4H trend is UP      -> close > EMA50, EMA20 > EMA50, last 4H swing low higher than the one before
  2. 30m is pulling BACK -> close < EMA21 and (EMA9 < EMA21 or 30m lower high)
  3. 5m structure breaks -> 5m made a lower high during the pullback, then a 5m close breaks above it
  4. VWAP entry          -> after the break, first 5m bar that dips to session VWAP and closes back above it
  5. Stop                -> below the lowest low of the pullback leg (swing high -> entry), minus an ATR buffer

Higher-timeframe bars are only used once they have CLOSED, so there is no lookahead.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .data import resample
from .indicators import atr, ema, session_vwap, swings

STAGES = {0: "no 4H uptrend", 1: "4H up, waiting for 30m pullback", 2: "30m pulling back, waiting 5m break",
          3: "5m structure broke, waiting VWAP entry", 4: "ENTRY"}

FIVE_MIN = pd.Timedelta("5min")


@dataclass
class Signal:
    time: pd.Timestamp        # open time of the 5m signal bar (entry happens at the next bar's open)
    close_price: float
    stop: float
    vwap: float
    bos_level: float
    swing_low: float

    def as_dict(self):
        return self.__dict__.copy()


def _htf_features(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    sw = swings(df, cfg.htf_pivot)
    f = pd.DataFrame(index=df.index)
    f["htf_close"] = df["close"]
    f["htf_ema_fast"] = ema(df["close"], cfg.htf_ema_fast)
    f["htf_ema_slow"] = ema(df["close"], cfg.htf_ema_slow)
    up = (f["htf_close"] > f["htf_ema_slow"]) & (f["htf_ema_fast"] > f["htf_ema_slow"])
    if cfg.htf_require_higher_low:
        up &= sw["sl"] > sw["sl_prev"]
    f["htf_up"] = up
    return f


def _mtf_features(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    sw = swings(df, cfg.mtf_pivot)
    f = pd.DataFrame(index=df.index)
    fast, slow = ema(df["close"], cfg.mtf_ema_fast), ema(df["close"], cfg.mtf_ema_slow)
    f["mtf_close"] = df["close"]
    f["mtf_ema_slow"] = slow
    lower_high = sw["sh"] < sw["sh_prev"]
    f["mtf_down"] = (df["close"] < slow) & ((fast < slow) | lower_high)
    return f


def _align(df5: pd.DataFrame, htf: pd.DataFrame, period: str) -> pd.DataFrame:
    """Attach each 5m bar to the latest higher-TF bar that had CLOSED by the 5m bar's close."""
    right = htf.copy()
    right.index = right.index + pd.Timedelta(period)          # time the HTF bar becomes known
    right = right.reset_index(names="avail").sort_values("avail")
    left = pd.DataFrame({"avail": df5.index + FIVE_MIN})       # 5m bar close time
    merged = pd.merge_asof(left, right, on="avail", direction="backward")
    merged.index = df5.index
    return merged.drop(columns="avail")


def build_features(df5: pd.DataFrame, cfg: StrategyConfig, df30: pd.DataFrame | None = None,
                   df4h: pd.DataFrame | None = None) -> pd.DataFrame:
    df30 = df30 if df30 is not None else resample(df5, "30min")
    df4h = df4h if df4h is not None else resample(df5, "4h")

    f = df5.copy()
    f["vwap"] = session_vwap(df5, cfg.vwap_session_tz)
    f["atr"] = atr(df5, 14)
    sw = swings(df5, cfg.ltf_pivot)
    f = f.join(sw)
    f = f.join(_align(df5, _htf_features(df4h, cfg), "4h"))
    f = f.join(_align(df5, _mtf_features(df30, cfg), "30min"))
    f["htf_up"] = f["htf_up"].astype("boolean").fillna(False).astype(bool)
    f["mtf_down"] = f["mtf_down"].astype("boolean").fillna(False).astype(bool)
    return f


def run_state_machine(f: pd.DataFrame, cfg: StrategyConfig) -> tuple[pd.Series, list[Signal]]:
    """Walk 5m bars once. Returns the stage of every bar and the list of entry signals."""
    idx = f.index
    o, h, l, c = (f[k].to_numpy() for k in ("open", "high", "low", "close"))
    vwap, a = f["vwap"].to_numpy(), f["atr"].to_numpy()
    sh, sh_prev = f["sh"].to_numpy(), f["sh_prev"].to_numpy()
    sh_time = f["sh_time"]
    up, down = f["htf_up"].to_numpy(), f["mtf_down"].to_numpy()

    stage = np.zeros(len(f), dtype=int)
    signals: list[Signal] = []

    pb_start = None          # time the current 30m pullback was first seen
    pb_last_i = None         # last bar the 30m pullback was active
    bos_i = None             # bar of 5m break of structure
    bos_level = leg_start_i = None
    lookback = pd.Timedelta("90min")   # 5m swing high may have formed a bit before the 30m bar confirmed

    for i in range(len(f)):
        if not up[i]:
            pb_start = pb_last_i = bos_i = None
            continue
        stage[i] = 1

        if down[i]:
            if pb_last_i is None:
                pb_start = idx[i]
            pb_last_i = i
        armed = pb_last_i is not None and (i - pb_last_i) <= cfg.pullback_expiry_bars
        if not armed and bos_i is None:
            pb_start = pb_last_i = None
            continue
        stage[i] = 2

        # 3. break of 5m structure: close above the last 5m lower high formed during the pullback
        if bos_i is None:
            t_sh = sh_time.iloc[i]
            ok = (not np.isnan(sh[i]) and pd.notna(t_sh) and t_sh >= pb_start - lookback
                  and (np.isnan(sh_prev[i]) or sh[i] < sh_prev[i]))
            if ok and c[i] > sh[i]:
                bos_i, bos_level = i, sh[i]
                leg_start_i = idx.get_loc(t_sh)
            else:
                continue

        stage[i] = 3
        leg_low = l[leg_start_i:i + 1].min()

        # setup dead: broke the pullback low or waited too long
        if c[i] < leg_low or (i - bos_i) > cfg.entry_window_bars:
            bos_i = None
            stage[i] = 2 if armed else 1
            continue

        # 4. VWAP play: dip to VWAP, close back above it
        touched = l[i] <= vwap[i] * (1 + cfg.vwap_touch_tol)
        if touched and c[i] > vwap[i] and c[i] > bos_level * 0.995:
            stop = leg_low - cfg.stop_buffer_atr * a[i]
            risk_pct = (c[i] - stop) / c[i]
            if cfg.min_stop_pct <= risk_pct <= cfg.max_stop_pct:
                stage[i] = 4
                signals.append(Signal(idx[i], float(c[i]), float(stop), float(vwap[i]),
                                      float(bos_level), float(leg_low)))
            # one shot per pullback either way
            bos_i = pb_start = pb_last_i = None

    return pd.Series(stage, index=idx, name="stage"), signals


def generate(df5: pd.DataFrame, cfg: StrategyConfig | None = None, df30=None, df4h=None):
    cfg = cfg or StrategyConfig()
    f = build_features(df5, cfg, df30, df4h)
    stage, signals = run_state_machine(f, cfg)
    f["stage"] = stage
    return f, signals
