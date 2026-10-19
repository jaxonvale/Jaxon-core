"""Bar-by-bar backtest of the signals. Entry at next 5m open, stop/target checked intrabar
(if both are hit in the same bar the stop is assumed first — conservative)."""
import numpy as np
import pandas as pd

from .config import StrategyConfig
from .strategy import generate


def backtest(df5: pd.DataFrame, cfg: StrategyConfig | None = None, equity: float = 10_000.0,
             df30=None, df4h=None, max_leverage: float = 1.0):
    cfg = cfg or StrategyConfig()
    f, signals = generate(df5, cfg, df30, df4h)
    idx = f.index
    o, h, l, c = (f[k].to_numpy() for k in ("open", "high", "low", "close"))

    trades = []
    busy_until = -1
    eq = equity
    for s in signals:
        i = idx.get_loc(s.time) + 1          # fill on next bar open
        if i >= len(f) or i <= busy_until:
            continue
        entry = o[i] * (1 + cfg.slippage)
        stop = s.stop
        if entry <= stop:
            continue                          # gapped through the stop
        risk = entry - stop
        target = entry + cfg.reward_r * risk
        qty = min(eq * cfg.risk_per_trade / risk, eq * max_leverage / entry)

        exit_px = reason = None
        be_moved = False
        j = i
        last = len(f) - 1 if not cfg.max_hold_bars else min(len(f) - 1, i + cfg.max_hold_bars)
        while j <= last:
            if l[j] <= stop:
                exit_px, reason = min(o[j], stop) if j > i else stop, "breakeven" if be_moved else "stop"
                break
            if h[j] >= target:
                exit_px, reason = max(o[j], target) if j > i else target, "target"
                break
            if cfg.breakeven_at_r and not be_moved and h[j] >= entry + cfg.breakeven_at_r * risk:
                stop, be_moved = entry, True
            j += 1
        if exit_px is None:
            j = last
            exit_px, reason = c[j], "time" if cfg.max_hold_bars and last < len(f) - 1 else "open_at_end"
        exit_px *= (1 - cfg.slippage) if reason != "target" else 1

        fees = (entry + exit_px) * qty * cfg.fee_rate
        pnl = (exit_px - entry) * qty - fees
        eq += pnl
        busy_until = j
        trades.append({
            "entry_time": idx[i], "exit_time": idx[j], "entry": entry, "stop": s.stop, "target": target,
            "exit": exit_px, "reason": reason, "qty": qty, "r": pnl / (risk * qty),  # net of fees
            "pnl": pnl, "equity": eq, "vwap": s.vwap, "bos_level": s.bos_level,
        })

    t = pd.DataFrame(trades)
    return t, stats(t, equity), f


def stats(t: pd.DataFrame, start_equity: float) -> dict:
    if t.empty:
        return {"trades": 0}
    wins = t[t.pnl > 0]
    losses = t[t.pnl <= 0]
    curve = np.r_[start_equity, t.equity.to_numpy()]
    peak = np.maximum.accumulate(curve)
    dd = ((curve - peak) / peak).min()
    gross_loss = -losses.pnl.sum()
    return {
        "trades": len(t),
        "win_rate": round(len(wins) / len(t), 3),
        "avg_r": round(t.r.mean(), 3),
        "total_r": round(t.r.sum(), 2),
        "profit_factor": round(wins.pnl.sum() / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "net_pnl": round(t.pnl.sum(), 2),
        "return_pct": round((curve[-1] / start_equity - 1) * 100, 2),
        "max_drawdown_pct": round(dd * 100, 2),
        "exits": t.reason.value_counts().to_dict(),
    }
