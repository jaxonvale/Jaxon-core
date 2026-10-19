"""Indicators: EMA, ATR, session VWAP and confirmed swing points (no lookahead)."""
import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def session_vwap(df: pd.DataFrame, tz: str = "UTC") -> pd.Series:
    """VWAP that resets at midnight of `tz` each day."""
    idx = df.index
    local = idx.tz_convert(tz) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(tz)
    session = pd.Series(local.normalize(), index=idx)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)
    pv = (tp * vol).groupby(session).cumsum()
    vv = vol.groupby(session).cumsum()
    vwap = pv / vv.replace(0, np.nan)
    # zero-volume bars at session open: fall back to typical price
    return vwap.fillna(tp)


def swings(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Swing highs/lows using an n-left / n-right pivot.

    A pivot at bar i is only *known* at bar i+n, so every output column is
    shifted to the confirmation bar. Columns (as of each bar):
      sh, sh_prev   last / previous confirmed swing-high price
      sl, sl_prev   last / previous confirmed swing-low price
      sh_time       open time of the bar that made the last swing high
      sl_time       open time of the bar that made the last swing low
    """
    win = 2 * n + 1
    hi, lo = df["high"], df["low"]
    is_ph = hi.eq(hi.rolling(win, center=True).max())
    is_pl = lo.eq(lo.rolling(win, center=True).min())

    times = pd.Series(df.index, index=df.index)
    ph = hi.where(is_ph).shift(n)
    pl = lo.where(is_pl).shift(n)
    ph_t = times.where(is_ph).shift(n)
    pl_t = times.where(is_pl).shift(n)

    out = pd.DataFrame(index=df.index)
    out["sh"] = ph.ffill()
    out["sl"] = pl.ffill()
    out["sh_prev"] = ph.dropna().shift(1).reindex(df.index).ffill()
    out["sl_prev"] = pl.dropna().shift(1).reindex(df.index).ffill()
    out["sh_time"] = ph_t.ffill()
    out["sl_time"] = pl_t.ffill()
    return out
