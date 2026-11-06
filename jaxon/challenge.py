"""Prop-firm evaluation ("funded challenge") simulator.

Takes the bar-by-bar equity curve and trade list of ONE engine run (sized on a
fixed starting balance, so every trade's P&L is independent of when an
attempt began) and replays a fresh challenge attempt starting at every
trading day. For each attempt it applies the firm's rules to the equity path:

  * trailing max-loss threshold (end-of-day or intraday trailing, optional cap),
    breached by the marked-to-market equity at ANY bar (unrealized included)
  * profit target, checked on end-of-day balance
  * consistency rule (best day <= pct of total profit; otherwise the target rises)
  * minimum trading days, calendar-day time limit

Rule presets are taken from public 2026 descriptions and DO change. Check the
firm's current rules before relying on any number. Attempts that start on
consecutive days share most of their trades, so they are not independent
samples.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass
class ChallengeRules:
    name: str
    account_size: float
    profit_target: float
    max_loss: float
    trailing: str = "eod"                     # eod | intraday | static
    trail_cap_over_start: Optional[float] = None   # threshold stops rising at start + this (Apex: +100)
    consistency_pct: Optional[float] = None   # best day <= pct * total profit
    max_calendar_days: Optional[int] = None
    min_trading_days: int = 1
    day_tz: str = "America/Chicago"
    day_reset: str = "17:00"
    notes: str = ""


PRESETS = {
    "topstep_50k": ChallengeRules(
        "Topstep 50K Trading Combine", 50_000, 3_000, 2_000, "eod", None, 0.50, None, 2,
        notes="MLL $2,000 trails end-of-day balance, breach checked in real time; consistency 50% "
              "(some 2026 sources say 55%); no time limit; flat by 3:10 PM CT; max 50 micros. "
              "Optional daily loss limit not modeled (the bot's own 1% = $500 daily lockout is tighter)."),
    "apex_50k_eod": ChallengeRules(
        "Apex 50K (EOD trailing, 2026 rules)", 50_000, 3_000, 2_000, "eod", 100.0, None, 30, 1,
        notes="$2,000 EOD trailing drawdown that stops trailing at start+$100; 30 calendar days; "
              "$1,000 DLL not modeled (bot's $500 daily lockout is tighter); max 60 micros."),
    "apex_50k_intraday": ChallengeRules(
        "Apex 50K (intraday trailing, 2026 rules)", 50_000, 3_000, 2_000, "intraday", 100.0, None, 30, 1,
        notes="$2,000 drawdown trails the highest unrealized equity in real time, stops at start+$100; 30 days."),
    # ---- Lucid Trading (support.lucidtrading.com, read 2026-10-20). Trading day resets 6:00 PM ET;
    # flat by 4:45 PM ET; MLL trails the end-of-day balance and locks at start+$100 once the
    # balance passes start+MLL+$100. Daily loss limits are SOFT (pause until next session) and are
    # covered by the bot's own tighter daily lockout, so they are not modeled separately.
    "lucidflex_25k": ChallengeRules(
        "LucidFlex 25K evaluation", 25_000, 1_250, 1_000, "eod", 100.0, 0.40, None, 1, "America/New_York", "18:00",
        notes="Target $1,250; MLL $1,000 EOD trailing, locks at $25,100; consistency 40% (50% with paid add-on); "
              "DLL optional; max 2 minis / 20 micros; no time limit."),
    "lucidflex_50k": ChallengeRules(
        "LucidFlex 50K evaluation", 50_000, 3_000, 2_000, "eod", 100.0, 0.40, None, 1, "America/New_York", "18:00",
        notes="Target $3,000; MLL $2,000 EOD trailing, locks at $50,100; consistency 40% (50% add-on); "
              "DLL optional; max 4 minis / 40 micros; no time limit."),
    "lucidpro_25k": ChallengeRules(
        "LucidPro 25K evaluation", 25_000, 1_250, 1_000, "eod", 100.0, None, None, 1, "America/New_York", "18:00",
        notes="Target $1,250; MLL $1,000 EOD trailing, locks at $25,100; no DLL; no evaluation consistency rule; "
              "max 2 minis / 20 micros; can pass in one day."),
    "lucidpro_50k": ChallengeRules(
        "LucidPro 50K evaluation", 50_000, 3_000, 2_000, "eod", 100.0, None, None, 1, "America/New_York", "18:00",
        notes="Target $3,000; MLL $2,000 EOD trailing, locks at $50,100; $1,200 soft DLL; no evaluation consistency "
              "rule; max 4 minis / 40 micros."),
    "luciddaily_25k_intraday": ChallengeRules(
        "LucidDaily 25K evaluation (intraday drawdown)", 25_000, 1_250, 1_000, "intraday", 100.0, 0.50, None, 1,
        "America/New_York", "18:00",
        notes="Target $1,250; MLL $1,000 intraday trailing (EOD available at higher price); 50% consistency; "
              "$600 soft DLL; red-folder NEWS TRADING IS A HARD BREACH (not modeled)."),
    "luciddaily_50k_intraday": ChallengeRules(
        "LucidDaily 50K evaluation (intraday drawdown)", 50_000, 3_000, 2_000, "intraday", 100.0, 0.50, None, 1,
        "America/New_York", "18:00",
        notes="Target $3,000; MLL $2,000 intraday trailing; 50% consistency; $1,200 soft DLL; "
              "red-folder NEWS TRADING IS A HARD BREACH (not modeled)."),
}


@dataclass
class Attempt:
    start_day: str
    outcome: str = "unresolved"   # passed | failed | expired | unresolved
    end_day: str = ""
    calendar_days: int = 0
    trading_days: int = 0
    trades: int = 0
    final_balance: float = 0.0
    best_day: float = 0.0


def _day_ids(times: pd.DatetimeIndex, tz: str, reset: str) -> np.ndarray:
    h, m = (int(x) for x in reset.split(":"))
    local = times.tz_convert(ZoneInfo(tz)).tz_localize(None) - pd.Timedelta(hours=h, minutes=m)
    return local.normalize().to_numpy()


def build_days(equity_curve: list, trades: list, rules: ChallengeRules) -> list[dict]:
    """Split the run's equity curve into trading days (by the firm's day reset)."""
    eq = pd.Series([e for _, e in equity_curve], index=pd.DatetimeIndex([t for t, _ in equity_curve]))
    eq = eq[~eq.index.duplicated(keep="last")]
    ids = _day_ids(eq.index, rules.day_tz, rules.day_reset)
    tr_days = pd.Series(1, index=pd.DatetimeIndex([t.exit_time for t in trades])) if trades else pd.Series(dtype=int)
    tr_ids = _day_ids(tr_days.index, rules.day_tz, rules.day_reset) if len(tr_days) else np.array([])
    counts = pd.Series(1, index=tr_ids).groupby(level=0).sum() if len(tr_ids) else pd.Series(dtype=int)
    days = []
    vals = eq.to_numpy()
    uniq, first = np.unique(ids, return_index=True)
    bounds = list(first) + [len(vals)]
    prev_end = None
    for k, d in enumerate(uniq):
        seg = vals[bounds[k]:bounds[k + 1]]
        open_eq = prev_end if prev_end is not None else seg[0]
        days.append({"day": pd.Timestamp(d), "open": open_eq, "path": seg, "close": seg[-1],
                     "trades": int(counts.get(d, 0))})
        prev_end = seg[-1]
    return days


def simulate(days: list[dict], rules: ChallengeRules, scale: float = 1.0) -> list[Attempt]:
    """`scale` multiplies P&L (e.g. 2.0 = double the position size). Only valid for
    runs sized on a fixed balance; rounding to whole contracts makes it approximate."""
    out = []
    for i in range(len(days)):
        a = Attempt(start_day=days[i]["day"].date().isoformat())
        base = days[i]["open"]
        start = rules.account_size
        cap = start + rules.trail_cap_over_start if rules.trail_cap_over_start is not None else np.inf
        eod_high, peak = start, start
        profit_days = []
        for j in range(i, len(days)):
            d = days[j]
            a.calendar_days = (d["day"] - days[i]["day"]).days + 1
            if rules.max_calendar_days and a.calendar_days > rules.max_calendar_days:
                a.outcome, a.end_day = "expired", d["day"].date().isoformat()
                break
            path = start + (d["path"] - base) * scale
            day_open = start + (d["open"] - base) * scale
            if rules.trailing == "intraday":
                run = np.maximum.accumulate(np.maximum(path, peak))
                thr = np.minimum(run - rules.max_loss, cap)      # cap = start + trail cap (inf if none)
                breach = bool((path <= thr).any())
                peak = float(run[-1])
            else:
                thr = eod_high - rules.max_loss if rules.trailing == "eod" else start - rules.max_loss
                thr = min(thr, cap)
                breach = bool(path.min() <= thr)
            a.trades += d["trades"]
            a.trading_days += 1 if d["trades"] else 0
            close = float(path[-1])
            if breach:
                a.outcome, a.end_day, a.final_balance = "failed", d["day"].date().isoformat(), close
                break
            day_pnl = close - day_open
            if d["trades"]:
                profit_days.append(day_pnl)
            eod_high = max(eod_high, close)
            total = close - start
            need = rules.profit_target
            best = max([p for p in profit_days if p > 0], default=0.0)
            if rules.consistency_pct:
                need = max(need, best / rules.consistency_pct)
            a.final_balance, a.best_day = close, best
            if total >= need and a.trading_days >= rules.min_trading_days:
                a.outcome, a.end_day = "passed", d["day"].date().isoformat()
                break
        out.append(a)
    return out


def summarize(attempts: list[Attempt], rules: ChallengeRules) -> dict:
    df = pd.DataFrame([asdict(a) for a in attempts])
    c = df.outcome.value_counts().to_dict()
    resolved = c.get("passed", 0) + c.get("failed", 0) + c.get("expired", 0)
    passed = df[df.outcome == "passed"]
    failed = df[df.outcome == "failed"]
    return {
        "challenge": rules.name,
        "attempts_started": len(df),
        "passed": c.get("passed", 0), "failed": c.get("failed", 0), "expired": c.get("expired", 0),
        "unresolved_at_data_end": c.get("unresolved", 0),
        "pass_rate_pct": round(100 * c.get("passed", 0) / resolved, 1) if resolved else None,
        "median_days_to_pass": float(passed.calendar_days.median()) if len(passed) else None,
        "median_days_to_fail": float(failed.calendar_days.median()) if len(failed) else None,
        "median_trades_per_attempt": float(df.trades.median()) if len(df) else None,
        "rules": asdict(rules),
    }
