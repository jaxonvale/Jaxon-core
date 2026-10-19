"""CLI.

  python -m trading_bot backtest --demo
  python -m trading_bot backtest --symbol BTC/USDT --exchange binance --days 90 --chart out.html
  python -m trading_bot backtest --csv my_5m_data.csv
  python -m trading_bot backtest --symbol SPY --source yfinance --tz America/New_York
  python -m trading_bot scan --symbols BTC/USDT ETH/USDT SOL/USDT
  python -m trading_bot live --symbol BTC/USDT            # paper
  python -m trading_bot live --symbol BTC/USDT --live     # real orders (needs API keys)
"""
import argparse
import json
import logging

from .backtest import backtest
from .config import StrategyConfig
from .strategy import STAGES, generate


def _load(args):
    from . import data

    if args.demo:
        return data.synthetic(days=args.days), None, None
    if args.csv:
        return data.load_csv(args.csv), None, None
    if args.source == "yfinance":
        return (data.fetch_yfinance(args.symbol, "5m", args.days), data.fetch_yfinance(args.symbol, "30m", args.days),
                data.fetch_yfinance(args.symbol, "4h", max(args.days, 120)))
    df5 = data.fetch_ccxt(args.symbol, "5m", args.exchange, days=args.days)
    df30 = data.fetch_ccxt(args.symbol, "30m", args.exchange, days=args.days + 5)
    df4h = data.fetch_ccxt(args.symbol, "4h", args.exchange, days=args.days + 60)  # warm up EMA50
    return df5, df30, df4h


def _cfg(args) -> StrategyConfig:
    cfg = StrategyConfig()
    if args.tz:
        cfg.vwap_session_tz = args.tz
    if args.rr:
        cfg.reward_r = args.rr
    if args.risk:
        cfg.risk_per_trade = args.risk
    return cfg


def main():
    p = argparse.ArgumentParser(prog="trading_bot", description="4H up / 30m pullback / 5m break / VWAP entry bot")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--exchange", default="binance")
        sp.add_argument("--source", choices=["ccxt", "yfinance"], default="ccxt")
        sp.add_argument("--tz", help="VWAP session timezone (default UTC; stocks: America/New_York)")
        sp.add_argument("--rr", type=float, help="reward:risk target (default 2)")
        sp.add_argument("--risk", type=float, help="fraction of equity risked per trade (default 0.01)")

    b = sub.add_parser("backtest")
    common(b)
    b.add_argument("--symbol", default="BTC/USDT")
    b.add_argument("--csv")
    b.add_argument("--demo", action="store_true", help="synthetic data, no network")
    b.add_argument("--days", type=int, default=60)
    b.add_argument("--equity", type=float, default=10_000)
    b.add_argument("--chart", help="write HTML chart to this path")
    b.add_argument("--trades", help="write trades CSV to this path")

    s = sub.add_parser("scan", help="show where each symbol is in the setup right now")
    common(s)
    s.add_argument("--symbols", nargs="+", required=True)
    s.add_argument("--days", type=int, default=5)

    lv = sub.add_parser("live")
    common(lv)
    lv.add_argument("--symbol", required=True)
    lv.add_argument("--live", action="store_true", help="send real orders")
    lv.add_argument("--equity", type=float, default=10_000, help="paper equity")

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = _cfg(args)

    if args.cmd == "backtest":
        df5, df30, df4h = _load(args)
        trades, st, f = backtest(df5, cfg, args.equity, df30, df4h)
        print(json.dumps(st, indent=2, default=str))
        if not trades.empty:
            print(trades[["entry_time", "entry", "stop", "target", "exit", "reason", "r", "pnl"]]
                  .tail(20).to_string(index=False))
        if args.trades:
            trades.to_csv(args.trades, index=False)
        if args.chart:
            from .chart import write_chart
            print("chart:", write_chart(f, trades, args.chart, args.symbol if not args.demo else "demo"))

    elif args.cmd == "scan":
        from . import data
        for sym in args.symbols:
            try:
                if args.source == "yfinance":
                    d = [data.fetch_yfinance(sym, tf, 30 if tf != "4h" else 120) for tf in ("5m", "30m", "4h")]
                else:
                    d = [data.fetch_ccxt(sym, "5m", args.exchange, limit=700),
                         data.fetch_ccxt(sym, "30m", args.exchange, limit=300),
                         data.fetch_ccxt(sym, "4h", args.exchange, limit=300)]
                f, sigs = generate(d[0], cfg, d[1], d[2])
                last = f.iloc[-1]
                print(f"{sym:12s} {STAGES[int(last.stage)]:45s} close={last.close:.4f} vwap={last.vwap:.4f}"
                      + (f"  last signal {sigs[-1].time} stop={sigs[-1].stop:.4f}" if sigs else ""))
            except Exception as e:
                print(f"{sym:12s} error: {e}")

    elif args.cmd == "live":
        from .live import LiveRunner
        LiveRunner(args.symbol, args.exchange, cfg, paper=not args.live, paper_equity=args.equity).run()


if __name__ == "__main__":
    main()
