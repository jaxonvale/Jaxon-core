"""Metrics and exports. Reports only what the engine recorded; nothing is estimated
beyond the documented mark-to-market of open positions."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .engine import ENGINE_VERSION

SYNTHETIC_BANNER = ("SYNTHETIC DATA — these results come from generated test fixtures, not market "
                    "history. They are not evidence of profitability.")


def _trades_df(trades) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["trade_id", "symbol", "entry_time", "exit_time", "net_pnl", "r_multiple"])
    df = pd.DataFrame([t.as_record() for t in trades])
    for c in ("entry_time", "exit_time"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def metrics(trades: pd.DataFrame, equity: pd.Series, start_equity: float, total_minutes: float,
            min_trades: int) -> dict:
    n = len(trades)
    out = {"trades": n}
    if equity is not None and len(equity):
        curve = np.r_[start_equity, equity.to_numpy()]
        peak = np.maximum.accumulate(curve)
        out["max_drawdown_pct"] = round(float(((curve - peak) / peak).min()) * 100, 3)
        out["end_equity"] = round(float(curve[-1]), 2)
        out["net_return_pct"] = round((curve[-1] / start_equity - 1) * 100, 3)
    if n == 0:
        out["note"] = "no closed trades"
        return out
    wins, losses = trades[trades.net_pnl > 0], trades[trades.net_pnl <= 0]
    gl = -losses.net_pnl.sum()
    fees = trades.entry_fee.sum() + trades.exit_fee.sum()
    friction = trades.entry_friction_cash.sum() + trades.exit_friction_cash.sum()
    hold = (trades.exit_time - trades.entry_time).dt.total_seconds() / 60
    flags = Counter(f for s in trades["flags"] for f in str(s).split(";") if f)
    out.update({
        "win_rate": round(len(wins) / n, 4),
        "avg_win": round(float(wins.net_pnl.mean()), 2) if len(wins) else None,
        "avg_loss": round(float(losses.net_pnl.mean()), 2) if len(losses) else None,
        "expectancy_cash": round(float(trades.net_pnl.mean()), 2),
        "expectancy_r": round(float(trades.r_multiple.mean()), 4),
        "profit_factor": round(float(wins.net_pnl.sum() / gl), 3) if gl > 0 else None,
        "net_pnl": round(float(trades.net_pnl.sum()), 2),
        "total_fees": round(float(fees), 2),
        "total_spread_slippage": round(float(friction), 2),
        "total_costs": round(float(fees + friction), 2),
        "avg_holding_minutes": round(float(hold.mean()), 1),
        "max_holding_minutes": round(float(hold.max()), 1),
        "exposure_pct": round(float(hold.sum() / total_minutes * 100), 3) if total_minutes else None,
        "exit_reasons": trades.exit_reason.value_counts().to_dict(),
        "execution_flags": dict(flags),
    })
    out["sample_warning"] = (f"Only {n} trades (< {min_trades}). No statistical edge can be claimed."
                             if n < min_trades else "")
    return out


def by_group(trades: pd.DataFrame, key) -> dict:
    if trades.empty:
        return {}
    out = {}
    for k, g in trades.groupby(key):
        out[str(k)] = {"trades": len(g), "net_pnl": round(float(g.net_pnl.sum()), 2),
                       "win_rate": round(float((g.net_pnl > 0).mean()), 4),
                       "expectancy_r": round(float(g.r_multiple.mean()), 4)}
    return out


def skip_counts(engine) -> dict:
    c = Counter()
    for s in engine.setups():
        if s.status in ("cancelled", "timeout", "rejected"):
            c[f"setup:{s.outcome}"] += 1
    for k, v in engine.touch_skips().items():
        c[f"touch_candle:{k}"] += v
    for r in engine.journal.records:
        if r["kind"] == "zone_skipped":
            c[f"zone:{r['reason']}"] += 1
    return dict(sorted(c.items()))


def periods(cfg, first: pd.Timestamp, last: pd.Timestamp) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    ev = cfg.evaluation
    if not ev.development_end or not ev.validation_end:
        return [("full", first, last + pd.Timedelta(days=1))]
    d = pd.Timestamp(ev.development_end, tz="UTC")
    v = pd.Timestamp(ev.validation_end, tz="UTC")
    return [("development", first, d), ("validation", d, v), ("out_of_sample", v, last + pd.Timedelta(days=1))]


def summarize(result, sensitivity: dict | None = None) -> dict:
    eng, cfg, data = result.engine, result.engine.cfg, result.data
    tdf = _trades_df(eng.trades)
    eq = pd.Series([e for _, e in eng.equity_curve], index=[t for t, _ in eng.equity_curve])
    eq = eq[~eq.index.duplicated(keep="last")]
    bars = data.bars if result.end is None else [b for b in data.bars if b.start < result.end]
    step = min(i.bars.base_interval_minutes for i in cfg.instruments)
    total_min = len({b.start for b in bars}) * step
    first, last = bars[0].start, bars[-1].start
    mt = cfg.evaluation.min_trades_for_inference

    split = {}
    for name, a, b in periods(cfg, first, last):
        sub = tdf[(tdf.entry_time >= a) & (tdf.entry_time < b)] if len(tdf) else tdf
        eqs = eq[(eq.index > a) & (eq.index <= b)]
        start_eq = float(eq[eq.index <= a].iloc[-1]) if (eq.index <= a).any() else cfg.portfolio.starting_equity
        mins = len({x.start for x in bars if a <= x.start < b}) * step
        split[name] = {"from": a.isoformat(), "to": b.isoformat(), **metrics(sub, eqs, start_eq, mins, mt)}

    s = {
        "jaxon_version": __version__,
        "engine_version": ENGINE_VERSION,
        "config_version": cfg.version_tag,
        "config_file": cfg.source_path,
        "example_config": cfg.example,
        "data_kind": "SYNTHETIC" if data.synthetic else "HISTORICAL",
        "disclaimer": SYNTHETIC_BANNER if data.synthetic else
        "Historical simulation with modeled costs. Past simulated results do not predict future results.",
        "data_provenance": data.provenance,
        "bars_processed": eng.processed,
        "duplicates_ignored": eng.duplicates_ignored,
        "overall": metrics(tdf, eq, cfg.portfolio.starting_equity, total_min, mt),
        "periods": split,
        "by_month": by_group(tdf, tdf.entry_time.dt.strftime("%Y-%m")) if len(tdf) else {},
        "by_instrument": by_group(tdf, "symbol") if len(tdf) else {},
        "setups_total": len(eng.setups()),
        "signals_total": len(eng.signals),
        "zones_total": len(eng.zones()),
        "skipped_by_reason": skip_counts(eng),
        "daily_lockouts": eng.lockouts,
        "open_positions_at_end": getattr(eng, "open_positions", []),
        "config_warnings": result.config_warnings,
    }
    if sensitivity is not None:
        s["parameter_sensitivity"] = sensitivity
    return s


def setup_records(result) -> list[dict]:
    eng = result.engine
    prov = {k: f"{v['path']}#sha256={v['sha256'][:12]}{' [SYNTHETIC]' if v['synthetic'] else ''}"
            for k, v in result.data.provenance.items()}
    trades = {t.trade_id: t for t in eng.trades}
    out = []
    for s in eng.setups():
        r = s.as_record()
        t = trades.get(s.trade_id)
        if t is not None:
            r.update(exit_reason=t.exit_reason, exit_price=t.exit_price, net_pnl=t.net_pnl, r_multiple=t.r_multiple,
                     fees=t.entry_fee + t.exit_fee, spread_slippage=t.entry_friction_cash + t.exit_friction_cash,
                     execution_flags=";".join(t.flags))
        r["htf_swings"] = json.dumps(r["htf_swings"])
        r["data_provenance"] = prov.get(s.symbol, "")
        r["config_version"] = eng.cfg.version_tag
        out.append(r)
    return out


def export(result, out_dir: str | Path, summary: dict, chart: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eng = result.engine
    paths = {}
    trades = [t.as_record() for t in eng.trades]
    pd.DataFrame(trades).to_csv(out / "trades.csv", index=False)
    (out / "trades.json").write_text(json.dumps(trades, indent=2, default=str))
    setups = setup_records(result)
    pd.DataFrame(setups).to_csv(out / "setups.csv", index=False)
    (out / "setups.json").write_text(json.dumps(setups, indent=2, default=str))
    pd.DataFrame(eng.equity_curve, columns=["time", "equity"]).drop_duplicates("time", keep="last") \
        .to_csv(out / "equity.csv", index=False)
    with open(out / "journal.jsonl", "w") as fh:
        for r in eng.journal.records:
            fh.write(json.dumps(r, default=str) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "data_quality.json").write_text(json.dumps(result.data.quality, indent=2, default=str))
    paths.update({k: str(out / f) for k, f in [("trades_csv", "trades.csv"), ("trades_json", "trades.json"),
                                               ("setups_csv", "setups.csv"), ("setups_json", "setups.json"),
                                               ("equity", "equity.csv"), ("journal", "journal.jsonl"),
                                               ("summary", "summary.json"), ("quality", "data_quality.json")]})
    if chart:
        from .chart import write_chart
        for inst in eng.cfg.instruments:
            p = out / f"chart_{inst.symbol}.html"
            write_chart(result, inst.symbol, p)
            paths[f"chart_{inst.symbol}"] = str(p)
    return paths
