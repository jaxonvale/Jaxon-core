"""Append-only decision journal and deterministic IDs."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Optional

import pandas as pd


class IdFactory:
    """Deterministic IDs: same event sequence -> same IDs (needed for replay/recovery)."""

    def __init__(self):
        self.counters = defaultdict(int)

    def new(self, prefix: str, symbol: str, ts: pd.Timestamp) -> str:
        self.counters[prefix] += 1
        return f"{prefix}-{symbol}-{ts:%Y%m%dT%H%M}-{self.counters[prefix]:05d}"


def _clean(v):
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "value") and hasattr(v, "name"):   # Enum
        return v.value
    return v


class Journal:
    def __init__(self):
        self.records: list[dict] = []
        self.sink_path: Optional[str] = None
        self._written = 0

    def log(self, ts: pd.Timestamp, kind: str, symbol: str = "", **details) -> dict:
        rec = {"seq": len(self.records) + 1, "ts": _clean(ts), "kind": kind, "symbol": symbol}
        rec.update({k: _clean(v) for k, v in details.items()})
        self.records.append(rec)
        return rec

    def transitions(self, symbol: str | None = None) -> list[dict]:
        return [r for r in self.records if r["kind"] == "transition" and (symbol is None or r["symbol"] == symbol)]

    def flush(self, path: str | None = None) -> int:
        """Append unwritten records to a JSONL file. Returns total records written."""
        path = path or self.sink_path
        if not path:
            return self._written
        with open(path, "a") as fh:
            for r in self.records[self._written:]:
                fh.write(json.dumps(r, default=str) + "\n")
        self._written = len(self.records)
        return self._written

    def __getstate__(self):
        return self.__dict__.copy()
