"""CSV import and validation. No API credentials needed.

Schema (see docs/csv_schema.md): timestamp, open, high, low, close, volume.
Lines starting with '#' are comments; a comment containing 'SYNTHETIC' marks
the file as a synthetic fixture.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..calendar import TradingCalendar
from ..config import InstrumentConfig
from ..models import Bar

REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]


class DataValidationError(ValueError):
    def __init__(self, msg: str, report: "QualityReport"):
        super().__init__(msg)
        self.report = report


@dataclass
class Provenance:
    symbol: str
    path: str
    sha256: str
    rows_in_file: int
    synthetic: bool
    header_comments: list
    first_bar: str = ""
    last_bar: str = ""

    def as_dict(self):
        return asdict(self)


@dataclass
class QualityReport:
    symbol: str
    rows_in_file: int = 0
    rows_used: int = 0
    exact_duplicates_dropped: int = 0
    conflicting_duplicates: list = field(default_factory=list)
    out_of_order_rows: int = 0
    misaligned_timestamps: list = field(default_factory=list)
    invalid_ohlc_rows: list = field(default_factory=list)
    non_numeric_rows: list = field(default_factory=list)
    negative_volume_rows: list = field(default_factory=list)
    missing_volume_rows: int = 0
    zero_volume_rows: int = 0
    outside_session_rows_dropped: int = 0
    gaps: list = field(default_factory=list)            # within-session missing intervals
    sessions: int = 0
    sessions_missing_open: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def as_dict(self):
        d = asdict(self)
        for k in ("conflicting_duplicates", "misaligned_timestamps", "invalid_ohlc_rows",
                  "non_numeric_rows", "negative_volume_rows"):
            d[k] = d[k][:50]
        d["gaps_total"] = len(self.gaps)
        d["gaps"] = d["gaps"][:200]
        return d


def _read(path: str) -> tuple[pd.DataFrame, list[str], bytes]:
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8-sig")
    comments = [ln[1:].strip() for ln in text.splitlines() if ln.startswith("#")]
    body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))
    df = pd.read_csv(io.StringIO(body), dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df, comments, raw


def _parse_ts(col: pd.Series, naive_tz: str | None, rep: QualityReport) -> pd.DatetimeIndex:
    s = col.str.strip()
    if s.str.fullmatch(r"\d{9,13}").all():
        n = s.astype("int64")
        unit = "ms" if n.iloc[0] > 10**11 else "s"
        return pd.DatetimeIndex(pd.to_datetime(n, unit=unit, utc=True))
    has_offset = s.str.contains(r"(?:Z|[+-]\d\d:?\d\d)$", regex=True)
    if has_offset.all():
        return pd.DatetimeIndex(pd.to_datetime(s, utc=True, format="ISO8601"))
    if has_offset.any():
        rep.errors.append("timestamps mix timezone-aware and naive values")
        raise DataValidationError(rep.errors[-1], rep)
    if not naive_tz:
        rep.errors.append("timestamps have no UTC offset and bars.naive_timestamp_timezone is not set")
        raise DataValidationError(rep.errors[-1], rep)
    local = pd.to_datetime(s, format="ISO8601")
    return pd.DatetimeIndex(local.dt.tz_localize(naive_tz, ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC"))


def load_instrument_csv(inst: InstrumentConfig, declared_synthetic: bool | None = None):
    """Return (list[Bar] base bars, Provenance, QualityReport). Raises DataValidationError."""
    rep = QualityReport(symbol=inst.symbol)
    df, comments, raw = _read(inst.data.path)
    rep.rows_in_file = len(df)
    synthetic = bool(inst.data.synthetic or declared_synthetic or any("SYNTHETIC" in c.upper() for c in comments))
    prov = Provenance(inst.symbol, inst.data.path, hashlib.sha256(raw).hexdigest(), len(df), synthetic, comments)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        rep.errors.append(f"missing columns {missing}; required {REQUIRED}")
        raise DataValidationError(rep.errors[-1], rep)
    if df.empty:
        rep.errors.append("file has no rows")
        raise DataValidationError(rep.errors[-1], rep)

    idx = _parse_ts(df["timestamp"], inst.bars.naive_timestamp_timezone, rep)
    step = pd.Timedelta(minutes=inst.bars.base_interval_minutes)
    if inst.bars.timestamp_convention == "close":
        idx = idx - step

    num = {}
    for c in ["open", "high", "low", "close"]:
        num[c] = pd.to_numeric(df[c].str.strip(), errors="coerce").to_numpy()
    vol = pd.to_numeric(df["volume"].str.strip().replace({"": np.nan}), errors="coerce").to_numpy()
    frame = pd.DataFrame(num, index=idx)
    frame["volume"] = vol
    frame["_row"] = np.arange(len(df)) + 2  # 1-based file line incl. header (comments excluded)

    bad_num = frame[["open", "high", "low", "close"]].isna().any(axis=1)
    rep.non_numeric_rows = frame.loc[bad_num, "_row"].tolist()
    if rep.non_numeric_rows:
        rep.errors.append(f"{len(rep.non_numeric_rows)} rows with missing/non-numeric prices")

    # ordering
    order_breaks = int((np.diff(frame.index.asi8) < 0).sum())
    rep.out_of_order_rows = order_breaks
    if order_breaks:
        rep.warnings.append(f"{order_breaks} out-of-order rows; data was sorted by timestamp")
        frame = frame.sort_index(kind="stable")

    # duplicates
    dup_mask = frame.index.duplicated(keep=False)
    if dup_mask.any():
        dups = frame[dup_mask]
        for ts, g in dups.groupby(level=0):
            vals = g[["open", "high", "low", "close", "volume"]]
            if len(vals.drop_duplicates()) > 1:
                rep.conflicting_duplicates.append(ts.isoformat())
        exact = frame.assign(_ts=frame.index).duplicated(subset=["_ts", "open", "high", "low", "close", "volume"])
        rep.exact_duplicates_dropped = int(exact.sum())
        frame = frame[~exact.to_numpy()]
        if rep.conflicting_duplicates:
            rep.errors.append(f"{len(rep.conflicting_duplicates)} timestamps have conflicting duplicate rows")

    # alignment to base interval
    mis = ((frame.index - pd.Timestamp(0, tz="UTC")) % step) != pd.Timedelta(0)
    rep.misaligned_timestamps = [t.isoformat() for t in frame.index[mis]]
    if mis.any():
        rep.errors.append(f"{int(mis.sum())} timestamps not aligned to the {inst.bars.base_interval_minutes}m grid")

    # OHLC sanity
    o, h, l, c = (frame[k] for k in ("open", "high", "low", "close"))
    bad = (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (l <= 0) | (h < l)
    bad &= frame[["open", "high", "low", "close"]].notna().all(axis=1)
    rep.invalid_ohlc_rows = frame.loc[bad, "_row"].tolist()
    if rep.invalid_ohlc_rows:
        rep.errors.append(f"{len(rep.invalid_ohlc_rows)} rows violate OHLC rules (low<=min(o,c)<=max(o,c)<=high, prices>0)")

    neg = frame["volume"] < 0
    rep.negative_volume_rows = frame.loc[neg, "_row"].tolist()
    if neg.any():
        rep.errors.append(f"{int(neg.sum())} rows with negative volume")
    rep.missing_volume_rows = int(frame["volume"].isna().sum())
    rep.zero_volume_rows = int((frame["volume"] == 0).sum())
    if rep.missing_volume_rows:
        rep.warnings.append(f"{rep.missing_volume_rows} rows with missing volume; VWAP is disabled for those sessions")

    if rep.errors:
        raise DataValidationError("; ".join(rep.errors), rep)

    # sessions
    cal = TradingCalendar(inst.session)
    sess_ids, keep = [], []
    for ts in frame.index:
        s = cal.session_for(ts)
        ok = s is not None and ts + step <= s.close
        keep.append(ok)
        sess_ids.append(s.session_id if s else "")
    keep = np.array(keep)
    rep.outside_session_rows_dropped = int((~keep).sum())
    if rep.outside_session_rows_dropped:
        msg = f"{rep.outside_session_rows_dropped} rows outside configured sessions"
        if not inst.bars.drop_outside_session:
            rep.errors.append(msg)
            raise DataValidationError(msg, rep)
        rep.warnings.append(msg + " were dropped")
    frame = frame[keep]
    sess_arr = [sid for sid, k in zip(sess_ids, keep) if k]

    bars: list[Bar] = []
    prev_end, prev_sess = None, None
    seen_sessions = []
    for row, sid in zip(frame.itertuples(), sess_arr):
        ts = row.Index
        s = cal.session_by_close_date(dt.date.fromisoformat(sid))
        if sid != prev_sess:
            seen_sessions.append(sid)
            if ts > s.open:
                rep.sessions_missing_open.append({"session": sid, "first_bar": ts.isoformat(),
                                                  "session_open": s.open.isoformat()})
        elif ts > prev_end:
            rep.gaps.append({"session": sid, "from": prev_end.isoformat(), "to": ts.isoformat(),
                             "missing_bars": int((ts - prev_end) / step)})
        bars.append(Bar(inst.symbol, ts, ts + step, float(row.open), float(row.high), float(row.low),
                        float(row.close), float(row.volume), inst.bars.base_interval_minutes, sid))
        prev_end, prev_sess = ts + step, sid
    rep.sessions = len(seen_sessions)
    rep.rows_used = len(bars)
    if rep.gaps:
        rep.warnings.append(f"{len(rep.gaps)} within-session gaps; affected higher-timeframe bars are not built")
    if not bars:
        rep.errors.append("no bars inside configured sessions")
        raise DataValidationError(rep.errors[-1], rep)
    prov.first_bar, prov.last_bar = bars[0].start.isoformat(), bars[-1].start.isoformat()
    return bars, prov, rep
