"""Prop-challenge pass-rate study: backtest strategy variants, then replay a
challenge attempt starting on every trading day under each firm's rules."""
from __future__ import annotations

import copy
import json
import multiprocessing as mp
from pathlib import Path

import pandas as pd

from .backtest import load_data, run
from .challenge import PRESETS, build_days, simulate, summarize
from .config import Config, load_config
from .report import _trades_df, metrics

VARIANTS = {
    "Z_zone_tap_green4h_ema30": {"mode": "zone_tap", "tap_htf_filter": "green_candle", "tap_mtf_trend": "ema"},
    "Z_zone_tap_swing4h_ema30": {"mode": "zone_tap", "tap_htf_filter": "swing_trend", "tap_mtf_trend": "ema"},
    "Z_zone_tap_green4h_swings30": {"mode": "zone_tap", "tap_htf_filter": "green_candle", "tap_mtf_trend": "swings"},
    "Z_zone_tap_plus_PRO": {"mode": "zone_tap_pro", "tap_htf_filter": "green_candle", "tap_mtf_trend": "ema"},
    "A_bot_rules_vwap_target": {"mode": "jaxon", "target_mode": "vwap"},
    "B_bot_plus_PRO_filter": {"mode": "jaxon_pro", "target_mode": "pro_atr"},
    "C_PRO_signals_with_bot_rules": {"mode": "pro_breakout", "target_mode": "pro_atr", "pro_require_bias": True},
    "D_PRO_signals_no_4h_filter": {"mode": "pro_breakout", "target_mode": "pro_atr", "pro_require_bias": False},
}
RISKS = [0.0025, 0.005, 0.01]

_DATA = None
_CFG = None
_PRESETS = None


def _job(args):
    name, over, risk = args
    cfg = copy.deepcopy(_CFG)
    for k, v in over.items():
        setattr(cfg.strategy, k, v)
    cfg.risk.risk_per_trade_pct = risk
    cfg.risk.max_open_risk_pct = max(cfg.risk.max_open_risk_pct, risk)
    r = run(cfg, _DATA)
    e = r.engine
    tdf = _trades_df(e.trades)
    stats = metrics(tdf, None, cfg.portfolio.starting_equity, 0, 100)
    stats.pop("exit_reasons", None)
    out = {"variant": name, "risk_per_trade_pct": risk * 100, "trades": len(e.trades),
           "signals": len(e.signals), "setups": len(e.setups()), "zones": len(e.zones()),
           "trade_stats": stats, "challenges": {}}
    for key, rules in PRESETS.items():
        if _PRESETS and key not in _PRESETS:
            continue
        days = build_days(e.equity_curve, e.trades, rules)
        out["challenges"][key] = summarize(simulate(days, rules), rules)
    return out


def study(config_path: str, out_dir: str, variants=None, risks=None, workers: int = 4, presets=None) -> list[dict]:
    global _DATA, _CFG, _PRESETS
    _PRESETS = presets
    _CFG = load_config(config_path)
    _DATA = load_data(_CFG)
    jobs = [(n, o, r) for n, o in (variants or VARIANTS).items() for r in (risks or RISKS)]
    ctx = mp.get_context("fork")
    with ctx.Pool(min(workers, len(jobs))) as pool:
        results = pool.map(_job, jobs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prop_study.json").write_text(json.dumps({"config": _CFG.version_tag, "data": _DATA.provenance,
                                                     "results": results}, indent=2, default=str))
    rows = []
    for r in results:
        for key, c in r["challenges"].items():
            ts = r["trade_stats"]
            rows.append({"variant": r["variant"], "risk_%": r["risk_per_trade_pct"], "trades": r["trades"],
                         "win_rate": ts.get("win_rate"), "exp_R": ts.get("expectancy_r"),
                         "net_$": ts.get("net_pnl"), "challenge": key, "attempts": c["attempts_started"],
                         "passed": c["passed"], "failed": c["failed"], "expired": c["expired"],
                         "unresolved": c["unresolved_at_data_end"], "pass_rate_%": c["pass_rate_pct"],
                         "median_days_to_pass": c["median_days_to_pass"]})
    pd.DataFrame(rows).to_csv(out / "prop_study.csv", index=False)
    return results
