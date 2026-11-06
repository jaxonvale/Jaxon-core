"""Real historical OHLCV download -> JAXON CSV schema. No API credentials.

Sources (market-data adapters; the strategy never imports this module):
  * binance_vision: Binance's official public archive (data.binance.vision),
    monthly spot kline zips, each verified against the published SHA-256.
  * coinbase: Coinbase Exchange public candles API (60/300/900 s granularity).

Every output file starts with '# SOURCE:' comment lines naming the venue,
symbol, interval, date range, download time and per-archive checksums, so the
loader records it as HISTORICAL (not synthetic) data with provenance.

Volume is single-venue volume, not a consolidated market volume. VWAP built
from it is that venue's VWAP.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UA = {"User-Agent": "jaxon-research/1.0"}
BINANCE = "https://data.binance.vision/data/spot/monthly/klines/{sym}/{iv}/{sym}-{iv}-{ym}.zip"
COINBASE = "https://api.exchange.coinbase.com/products/{sym}/candles?granularity={g}&start={s}&end={e}"


def _get(url: str, retries: int = 4) -> bytes:
    for k in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(2 ** (k + 1))


def _months(start: str, end: str) -> list[str]:
    a = pd.Period(start[:7], "M")
    b = pd.Period(end[:7], "M")
    return [str(p) for p in pd.period_range(a, b, freq="M")]


def binance_vision(symbol: str, interval: str, start: str, end: str, cache: Path) -> tuple[pd.DataFrame, list[str]]:
    """Months [start, end] inclusive (YYYY-MM). Returns frame + provenance lines."""
    cache.mkdir(parents=True, exist_ok=True)
    frames, prov = [], []
    for ym in _months(start, end):
        url = BINANCE.format(sym=symbol, iv=interval, ym=ym)
        zpath = cache / Path(url).name
        expected = _get(url + ".CHECKSUM").decode().split()[0]
        if not zpath.exists() or hashlib.sha256(zpath.read_bytes()).hexdigest() != expected:
            zpath.write_bytes(_get(url))
        got = hashlib.sha256(zpath.read_bytes()).hexdigest()
        if got != expected:
            raise ValueError(f"checksum mismatch for {url}: {got} != {expected}")
        with zipfile.ZipFile(zpath) as z:
            raw = z.read(z.namelist()[0])
        df = pd.read_csv(io.BytesIO(raw), header=None, usecols=range(6),
                         names=["t", "open", "high", "low", "close", "volume"])
        if not str(df.iloc[0, 0]).isdigit():          # some archives carry a header row
            df = df.iloc[1:].astype({"t": "int64"})
        t = df["t"].astype("int64")
        unit = "us" if t.iloc[0] > 10 ** 14 else "ms"  # Binance spot switched to microseconds in 2025
        df["timestamp"] = pd.to_datetime(t, unit=unit, utc=True)
        frames.append(df.drop(columns="t"))
        prov.append(f"{Path(url).name} sha256={got}")
    out = pd.concat(frames, ignore_index=True)
    return out, prov


def coinbase(product: str, granularity: int, start: str, end: str) -> tuple[pd.DataFrame, list[str]]:
    """Coinbase returns at most 300 candles per call, newest first: [time, low, high, open, close, volume]."""
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    step = pd.Timedelta(seconds=granularity * 300)
    rows = []
    while s < e:
        chunk_end = min(s + step, e)
        url = COINBASE.format(sym=product, g=granularity, s=s.isoformat(), e=chunk_end.isoformat())
        rows += json.loads(_get(url))
        s = chunk_end
        time.sleep(0.35)                               # stay under public rate limits
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.drop(columns="t").drop_duplicates("timestamp").sort_values("timestamp")
    return df, [f"coinbase exchange public candles {product} granularity={granularity}s"]


def write_csv(df: pd.DataFrame, path: str | Path, source_lines: list[str], header: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("timestamp")
    out = pd.DataFrame({"timestamp": df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
                        "volume": df["volume"]})
    with open(path, "w") as fh:
        fh.write(f"# SOURCE: {header}\n")
        fh.write(f"# DOWNLOADED: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}\n")
        fh.write("# HISTORICAL MARKET DATA (single venue). Not adjusted, not consolidated.\n")
        for line in source_lines:
            fh.write(f"# {line}\n")
        out.to_csv(fh, index=False)
    return path


def download(source: str, symbol: str, interval: str, start: str, end: str, out: str | Path,
             cache: str | Path = "data/cache") -> Path:
    if source == "binance_vision":
        df, prov = binance_vision(symbol, interval, start, end, Path(cache))
        header = f"Binance spot {symbol} {interval} klines, data.binance.vision, months {start[:7]}..{end[:7]}"
    elif source == "coinbase":
        g = {"1m": 60, "5m": 300, "15m": 900}[interval]
        df, prov = coinbase(symbol, g, start, end)
        header = f"Coinbase Exchange {symbol} {interval} candles {start}..{end}"
    else:
        raise ValueError(f"unknown source {source}")
    return write_csv(df, out, prov, header)
