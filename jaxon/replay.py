"""Deterministic paper replay with persisted state and restart recovery.

Bars are fed one at a time through the same TradingEngine the backtest uses.
After every `checkpoint_every` bars (and at the end) the engine state is
written atomically together with a manifest; the journal is appended to a
JSONL file. On restart:
  * the manifest's config hash, engine version and data hashes must match,
  * the snapshot checksum must match,
  * the JSONL journal is truncated/rebuilt to exactly the snapshot's records,
  * bars at or before each symbol's last processed bar are skipped, so
    re-delivering bars can never create duplicate entries.

The snapshot is a Python pickle. Only load state directories you created.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Optional

from .backtest import LoadedData, load_data
from .config import Config
from .engine import ENGINE_VERSION, TradingEngine


class RecoveryError(RuntimeError):
    pass


class PaperReplay:
    def __init__(self, cfg: Config, state_dir: str | Path, data: LoadedData | None = None):
        self.cfg = cfg
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.snap = self.dir / "engine_state.pkl"
        self.manifest = self.dir / "manifest.json"
        self.journal_path = self.dir / "journal.jsonl"
        self.data = data or load_data(cfg)
        self.engine: Optional[TradingEngine] = None
        self.resumed = False

    def _data_hashes(self) -> dict:
        return {s: p["sha256"] for s, p in self.data.provenance.items()}

    # ------------------------------------------------------------ persistence
    def save(self):
        eng = self.engine
        eng.journal.flush(str(self.journal_path))
        blob = pickle.dumps(eng, protocol=pickle.HIGHEST_PROTOCOL)
        tmp = self.snap.with_suffix(".tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, self.snap)
        man = {
            "engine_version": ENGINE_VERSION,
            "config_version": self.cfg.version_tag,
            "data_sha256": self._data_hashes(),
            "snapshot_sha256": hashlib.sha256(blob).hexdigest(),
            "journal_records": len(eng.journal.records),
            "bars_processed": eng.processed,
            "last_bar": {s: (rt.last_start.isoformat() if rt.last_start is not None else None)
                         for s, rt in eng.rt.items()},
            "synthetic_data": self.data.synthetic,
            "saved_at_wallclock": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmpm = self.manifest.with_suffix(".tmp")
        tmpm.write_text(json.dumps(man, indent=2))
        os.replace(tmpm, self.manifest)

    def restore(self) -> bool:
        if not self.snap.exists() or not self.manifest.exists():
            return False
        man = json.loads(self.manifest.read_text())
        if man["engine_version"] != ENGINE_VERSION:
            raise RecoveryError(f"state was written by {man['engine_version']}, this is {ENGINE_VERSION}")
        if man["config_version"] != self.cfg.version_tag:
            raise RecoveryError(f"config changed since state was saved ({man['config_version']} -> "
                                f"{self.cfg.version_tag}); start a new state dir or use --reset")
        if man["data_sha256"] != self._data_hashes():
            raise RecoveryError("data files differ from the ones this state was built from")
        blob = self.snap.read_bytes()
        if hashlib.sha256(blob).hexdigest() != man["snapshot_sha256"]:
            raise RecoveryError("snapshot checksum mismatch (corrupt or partially written state)")
        eng: TradingEngine = pickle.loads(blob)
        n = len(eng.journal.records)
        lines = self.journal_path.read_text().splitlines() if self.journal_path.exists() else []
        if len(lines) != n:   # crash between journal append and snapshot: make the file match the snapshot
            with open(self.journal_path, "w") as fh:
                for r in eng.journal.records:
                    fh.write(json.dumps(r, default=str) + "\n")
        eng.journal._written = n
        eng.journal.log(eng.last_event_time, "recovered", bars_processed=eng.processed,
                        journal_records=n)
        self.engine = eng
        self.resumed = True
        return True

    # ------------------------------------------------------------ run
    def run(self, max_bars: int | None = None, checkpoint_every: int = 1, delay_seconds: float = 0.0,
            stop_before: int | None = None) -> dict:
        """Process up to `max_bars` new bars. `stop_before` simulates a crash before
        the N-th checkpoint (tests only)."""
        if self.engine is None and not self.restore():
            self.engine = TradingEngine(self.cfg, self.data.provenance)
        eng = self.engine
        done = 0
        for b in self.data.bars:
            last = eng.rt[b.symbol].last_start
            if last is not None and b.start <= last:
                continue
            if max_bars is not None and done >= max_bars:
                break
            eng.process(b)
            done += 1
            if done % checkpoint_every == 0:
                if stop_before is not None and done >= stop_before:
                    return {"crashed_after_bars": done}
                self.save()
            if delay_seconds:
                time.sleep(delay_seconds)
        finished = max_bars is None or done < max_bars
        self.save()
        return {"bars_this_run": done, "bars_total": eng.processed, "resumed": self.resumed,
                "data_exhausted": finished, "trades_closed": len(eng.trades),
                "states": {s: rt.core.state.value for s, rt in eng.rt.items()}}
