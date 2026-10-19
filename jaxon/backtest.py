"""CSV backtest: load + validate data, feed bars through the engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .config import Config, validate
from .data.csv_source import load_instrument_csv
from .engine import TradingEngine
from .models import Bar


@dataclass
class LoadedData:
    bars: list[Bar]
    provenance: dict
    quality: dict
    synthetic: bool


@dataclass
class RunResult:
    engine: TradingEngine
    data: LoadedData
    config_warnings: list = field(default_factory=list)
    end: Optional[pd.Timestamp] = None


def load_data(cfg: Config) -> LoadedData:
    allb, prov, qual = [], {}, {}
    for inst in cfg.instruments:
        bars, p, q = load_instrument_csv(inst)
        allb.extend(bars)
        prov[inst.symbol] = p.as_dict()
        qual[inst.symbol] = q.as_dict()
    allb.sort(key=lambda b: (b.start, b.symbol))
    return LoadedData(allb, prov, qual, any(p["synthetic"] for p in prov.values()))


def run(cfg: Config, data: LoadedData | None = None, end: pd.Timestamp | None = None) -> RunResult:
    """Run the engine over all bars starting before `end` (the engine never sees later bars)."""
    warnings = validate(cfg)
    data = data or load_data(cfg)
    eng = TradingEngine(cfg, data.provenance)
    for b in data.bars:
        if end is not None and b.start >= end:
            break
        eng.process(b)
    eng.finalize()
    return RunResult(eng, data, warnings, end)
