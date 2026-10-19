"""Data loading: CSV, ccxt (crypto), yfinance (stocks), synthetic demo, resampling."""
import numpy as np
import pandas as pd

COLS = ["open", "high", "low", "close", "volume"]
TF = {"5m": "5min", "30m": "30min", "4h": "4h"}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    df = df[COLS].astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna()


def load_csv(path: str) -> pd.DataFrame:
    """CSV with a timestamp/date/time column (or epoch ms) + open,high,low,close,volume."""
    raw = pd.read_csv(path)
    raw.columns = [c.lower().strip() for c in raw.columns]
    tcol = next(c for c in raw.columns if c in ("timestamp", "datetime", "date", "time"))
    t = raw[tcol]
    if np.issubdtype(t.dtype, np.number):
        idx = pd.to_datetime(t, unit="ms" if t.iloc[0] > 1e11 else "s", utc=True)
    else:
        idx = pd.to_datetime(t, utc=True)
    raw.index = pd.DatetimeIndex(idx)
    return _clean(raw)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Bars labelled by their OPEN time. Drops the trailing bar if it is still forming."""
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    if len(df):
        last_close = df.index[-1] + pd.Timedelta(TF["5m"])
        if len(out) and out.index[-1] + pd.Timedelta(rule) > last_close:
            out = out.iloc[:-1]
    return out


def fetch_ccxt(symbol: str, timeframe: str, exchange: str = "binance", days: int = 30,
               limit: int | None = None) -> pd.DataFrame:
    import ccxt  # optional dependency

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    now = ex.milliseconds()
    since = now - (limit * tf_ms if limit else days * 86_400_000)
    rows = []
    while since < now:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        nxt = batch[-1][0] + tf_ms
        if nxt <= since:
            break
        since = nxt
    df = pd.DataFrame(rows, columns=["ts"] + COLS)
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = _clean(df)
    # drop the candle that is still forming
    if len(df) and df.index[-1] + pd.Timedelta(tf_ms, "ms") > pd.Timestamp.now(tz="UTC"):
        df = df.iloc[:-1]
    return df


def fetch_yfinance(symbol: str, timeframe: str, days: int = 59) -> pd.DataFrame:
    import yfinance as yf  # optional dependency

    interval = {"5m": "5m", "30m": "30m", "4h": "1h"}[timeframe]
    period = f"{min(days, 59 if timeframe != '4h' else 729)}d"
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = _clean(df)
    if timeframe == "4h":
        df = df.resample("4h", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return df


def synthetic(days: int = 30, seed: int = 7, start: str = "2026-01-01") -> pd.DataFrame:
    """Trending random walk with pullbacks, 5m bars. For demos/tests only."""
    rng = np.random.default_rng(seed)
    n = days * 288
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    t = np.arange(n)
    drift = 0.00004 + 0.00025 * np.sin(t / 288 * 2 * np.pi / 3)  # multi-day swings
    ret = drift + rng.normal(0, 0.0018, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.0009, n)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0009, n)) * close
    vol = rng.lognormal(10, 0.4, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
