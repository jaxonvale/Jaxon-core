"""Command line interface: python -m jaxon <command> ..."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def cmd_validate_config(a):
    from .config import load_config, validate
    cfg = load_config(a.config)
    _print({"ok": True, "config_version": cfg.version_tag, "example": cfg.example,
            "instruments": [i.symbol for i in cfg.instruments], "warnings": validate(cfg)})


def cmd_validate_data(a):
    from .backtest import load_data
    from .config import load_config
    from .data.csv_source import DataValidationError
    cfg = load_config(a.config)
    try:
        d = load_data(cfg)
    except DataValidationError as e:
        _print({"ok": False, "error": str(e), "report": e.report.as_dict()})
        return 2
    _print({"ok": True, "synthetic": d.synthetic, "provenance": d.provenance, "quality": d.quality})


def _headline(summary: dict):
    o = summary["overall"]
    print(f"\n[{summary['data_kind']}] {summary['disclaimer']}")
    print(f"config {summary['config_version']}  bars {summary['bars_processed']}  zones {summary['zones_total']}  "
          f"setups {summary['setups_total']}  signals {summary['signals_total']}  trades {o['trades']}")
    for k in ("win_rate", "expectancy_r", "expectancy_cash", "profit_factor", "net_pnl", "net_return_pct",
              "max_drawdown_pct", "total_costs"):
        if k in o:
            print(f"  {k:18s} {o[k]}")
    if o.get("sample_warning"):
        print("  NOTE:", o["sample_warning"])
    print("  skipped:", summary["skipped_by_reason"])


def cmd_backtest(a):
    from .backtest import run
    from .config import load_config
    from .report import export, summarize
    cfg = load_config(a.config)
    end = pd.Timestamp(a.until, tz="UTC") if a.until else None
    r = run(cfg, end=end)
    s = summarize(r)
    paths = export(r, a.out, s, chart=not a.no_chart)
    _headline(s)
    print("\nwrote:")
    for k, v in paths.items():
        print(f"  {k:16s} {v}")


def cmd_evaluate(a):
    from .config import load_config
    from .evaluation import evaluate
    from .report import export
    cfg = load_config(a.config)
    r, s = evaluate(cfg, with_sensitivity=not a.no_sensitivity)
    paths = export(r, a.out, s, chart=not a.no_chart)
    _headline(s)
    for name, p in s["periods"].items():
        print(f"\n  {name:14s} {p['from'][:10]} -> {p['to'][:10]}  trades {p['trades']}  "
              f"expectancy_r {p.get('expectancy_r')}  net {p.get('net_pnl')}")
    if s.get("parameter_sensitivity"):
        print("\n  sensitivity (development / validation only):")
        for row in s["parameter_sensitivity"]["rows"]:
            d, v = row["development"], row["validation"]
            print(f"    {row['parameter']:24s} {str(row['value']):6s}{' *' if row['is_baseline'] else '  '} "
                  f"dev trades {d.get('trades')} R {d.get('expectancy_r')} | val trades {v.get('trades')} "
                  f"R {v.get('expectancy_r')}")
    print("\nsummary:", paths["summary"])


def cmd_replay(a):
    from .config import load_config
    from .replay import PaperReplay
    cfg = load_config(a.config)
    if a.reset and Path(a.state_dir).exists():
        shutil.rmtree(a.state_dir)
    rp = PaperReplay(cfg, a.state_dir)
    _print(rp.run(max_bars=a.max_bars, checkpoint_every=a.checkpoint_every, delay_seconds=a.delay))


def cmd_status(a):
    man = Path(a.state_dir) / "manifest.json"
    if not man.exists():
        print("no state in", a.state_dir)
        return 1
    _print(json.loads(man.read_text()))


def cmd_explain(a):
    from .explain import explain
    print(explain(a.run, a.setup, use_llm=a.llm, model=a.model))


def cmd_fixtures(a):
    from .data.fixtures import build_default_fixtures
    for p in build_default_fixtures(a.out):
        print("wrote", p)


def cmd_paper_live(a):
    from .config import load_config
    from .live_paper import CoinbaseCandleFeed, LivePaper
    cfg = load_config(a.config)
    if a.reset and Path(a.state_dir).exists():
        shutil.rmtree(a.state_dir)
    products = dict(x.split("=") for x in a.product) if a.product else {}
    feeds = {i.symbol: CoinbaseCandleFeed(products.get(i.symbol, i.symbol), i) for i in cfg.instruments}
    print("LIVE PAPER: real Coinbase candles, simulated fills. No orders are sent.", flush=True)
    LivePaper(cfg, a.state_dir, feeds).run_forever(a.max_cycles, printer=lambda s: print(s, flush=True))


def cmd_prop_study(a):
    from .prop_study import RISKS, VARIANTS, study
    import pandas as pd
    risks = [float(x) / 100 for x in a.risk] if a.risk else RISKS
    variants = {k: v for k, v in VARIANTS.items() if not a.variant or k in a.variant}
    study(a.config, a.out, variants, risks, a.workers)
    df = pd.read_csv(Path(a.out) / "prop_study.csv")
    print(df.to_string(index=False))


def cmd_download(a):
    from .data.download import download
    p = download(a.source, a.symbol, a.interval, a.start, a.end, a.out, a.cache)
    print("wrote", p)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="jaxon", description="JAXON research + paper-trading agent (v1, long-only, no live orders)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("validate-config"); s.add_argument("config"); s.set_defaults(f=cmd_validate_config)
    s = sub.add_parser("validate-data"); s.add_argument("config"); s.set_defaults(f=cmd_validate_data)

    s = sub.add_parser("backtest", help="CSV backtest with exports and chart")
    s.add_argument("config"); s.add_argument("--out", default="runs/backtest")
    s.add_argument("--until", help="ISO date; feed only bars before it")
    s.add_argument("--no-chart", action="store_true"); s.set_defaults(f=cmd_backtest)

    s = sub.add_parser("evaluate", help="dev/validation/out-of-sample split + parameter sensitivity")
    s.add_argument("config"); s.add_argument("--out", default="runs/evaluate")
    s.add_argument("--no-sensitivity", action="store_true"); s.add_argument("--no-chart", action="store_true")
    s.set_defaults(f=cmd_evaluate)

    s = sub.add_parser("replay", help="deterministic paper replay with persisted, resumable state")
    s.add_argument("config"); s.add_argument("--state-dir", default="runs/paper")
    s.add_argument("--max-bars", type=int); s.add_argument("--checkpoint-every", type=int, default=1)
    s.add_argument("--delay", type=float, default=0.0, help="seconds to sleep between bars")
    s.add_argument("--reset", action="store_true", help="delete existing state first")
    s.set_defaults(f=cmd_replay)

    s = sub.add_parser("status"); s.add_argument("--state-dir", default="runs/paper"); s.set_defaults(f=cmd_status)

    s = sub.add_parser("explain", help="explain a logged setup (read-only)")
    s.add_argument("--run", required=True, help="output dir of a backtest/evaluate run")
    s.add_argument("--setup", required=True); s.add_argument("--llm", action="store_true")
    s.add_argument("--model", default="claude-opus-5-5"); s.set_defaults(f=cmd_explain)

    s = sub.add_parser("make-fixtures", help="regenerate the labeled synthetic fixtures")
    s.add_argument("--out", default="fixtures"); s.set_defaults(f=cmd_fixtures)

    s = sub.add_parser("paper-live", help="paper trade on live public Coinbase candles (no orders sent)")
    s.add_argument("config"); s.add_argument("--state-dir", default="runs/live_paper")
    s.add_argument("--product", action="append", help="SYMBOL=COINBASE_PRODUCT if they differ")
    s.add_argument("--max-cycles", type=int, help="stop after N candle cycles (default: run forever)")
    s.add_argument("--reset", action="store_true"); s.set_defaults(f=cmd_paper_live)

    s = sub.add_parser("prop-study", help="backtest variants and replay prop-firm challenges from every start day")
    s.add_argument("config"); s.add_argument("--out", default="runs/prop_study")
    s.add_argument("--risk", nargs="*", help="risk per trade in %, e.g. 0.25 0.5 1")
    s.add_argument("--variant", nargs="*"); s.add_argument("--workers", type=int, default=4)
    s.set_defaults(f=cmd_prop_study)

    s = sub.add_parser("download", help="download REAL historical OHLCV (public, no credentials) to CSV")
    s.add_argument("--source", choices=["binance_vision", "coinbase"], default="binance_vision")
    s.add_argument("--symbol", required=True, help="BTCUSDT for binance_vision, BTC-USD for coinbase")
    s.add_argument("--interval", default="1m", choices=["1m", "5m"])
    s.add_argument("--start", required=True, help="YYYY-MM (binance) or ISO date (coinbase)")
    s.add_argument("--end", required=True)
    s.add_argument("--out", required=True); s.add_argument("--cache", default="data/cache")
    s.set_defaults(f=cmd_download)

    a = p.parse_args(argv)
    try:
        return a.f(a) or 0
    except Exception as e:
        from .config import ConfigError
        from .data.csv_source import DataValidationError
        from .replay import RecoveryError
        if isinstance(e, (ConfigError, DataValidationError, RecoveryError, FileNotFoundError)):
            print(f"error: {e}", file=sys.stderr)
            return 2
        raise
