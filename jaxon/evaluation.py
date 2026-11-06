"""Chronological evaluation: development / validation / untouched out-of-sample.

Parameter sensitivity runs only on data BEFORE `validation_end`: the engine is
never fed out-of-sample bars during tuning. The out-of-sample period is
reported once, for the configuration as given.
"""
from __future__ import annotations

import pandas as pd

from .backtest import LoadedData, load_data, run
from .config import Config, ConfigError, with_strategy_overrides
from .report import summarize

KEYS = ("trades", "win_rate", "expectancy_r", "expectancy_cash", "profit_factor", "net_pnl", "max_drawdown_pct")


def _pick(m: dict) -> dict:
    return {k: m.get(k) for k in KEYS}


def sensitivity(cfg: Config, data: LoadedData | None = None) -> dict:
    ev = cfg.evaluation
    if not ev.validation_end or not ev.development_end:
        raise ConfigError("evaluation.development_end and validation_end are required for sensitivity analysis")
    data = data or load_data(cfg)
    cutoff = pd.Timestamp(ev.validation_end, tz="UTC")
    rows = []
    for param, values in ev.sensitivity.items():
        base_val = getattr(cfg.strategy, param)
        for v in values:
            c = with_strategy_overrides(cfg, **{param: v})
            r = run(c, data, end=cutoff)
            assert all(b.start < cutoff for b in data.bars[:r.engine.processed]), "sensitivity saw OOS data"
            s = summarize(r)
            rows.append({"parameter": param, "value": v, "is_baseline": v == base_val,
                         "development": _pick(s["periods"].get("development", {})),
                         "validation": _pick(s["periods"].get("validation", {})),
                         "bars_seen_until": data.bars[r.engine.processed - 1].start.isoformat()
                         if r.engine.processed else None})
    return {"data_used_until_exclusive": cutoff.isoformat(),
            "note": "One-at-a-time variation around the configured baseline. Out-of-sample data was not used.",
            "rows": rows}


def evaluate(cfg: Config, data: LoadedData | None = None, with_sensitivity: bool = True):
    data = data or load_data(cfg)
    sens = sensitivity(cfg, data) if with_sensitivity and cfg.evaluation.sensitivity else None
    result = run(cfg, data)
    return result, summarize(result, sens)
