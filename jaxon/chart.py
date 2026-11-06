"""Standalone HTML chart: 5m candles, session VWAP, demand zones, touches,
confirmations, entries, stops, targets, exits, and the equity curve."""
from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

import pandas as pd

from .bars import resample_all
from .calendar import TradingCalendar
from .indicators import SessionVWAP

_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JAXON {symbol}</title>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{{--bg:#ffffff;--fg:#1d2330;--muted:#5d6676;--grid:#eceff3;--warn-bg:#fff4d6;--warn-fg:#6b4e00}}
@media (prefers-color-scheme: dark){{:root{{--bg:#11141a;--fg:#e4e7ec;--muted:#98a1b2;--grid:#1f242d;--warn-bg:#3a2f0f;--warn-fg:#f3d58a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}}
header{{padding:10px 16px}} h1{{font-size:16px;margin:0 0 4px}}
.warn{{background:var(--warn-bg);color:var(--warn-fg);padding:8px 16px;font-weight:600}}
.legend{{color:var(--muted);font-size:12px}} #price{{height:62vh}} #equity{{height:24vh}}
</style></head><body>
{banner}
<header><h1>JAXON — {symbol} (5-minute)</h1>
<div class="legend">Orange: session VWAP (bar-based approximation) · Teal bands: demand zones (activation → end) ·
● touch · ◆ structure shift · ▲ entry · ▼ exit · red/blue segments: stop/target · config {config}</div></header>
<div id="price"></div><div id="equity"></div>
<script>
const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const base = {{layout:{{background:{{color: dark?'#11141a':'#ffffff'}}, textColor: dark?'#e4e7ec':'#1d2330'}},
  grid:{{vertLines:{{color: dark?'#1f242d':'#eceff3'}}, horzLines:{{color: dark?'#1f242d':'#eceff3'}}}},
  timeScale:{{timeVisible:true}}, autoSize:true}};
const pc = LightweightCharts.createChart(document.getElementById('price'), base);
const candles = pc.addCandlestickSeries();
candles.setData({candles});
pc.addLineSeries({{color:'#f08c00', lineWidth:2, priceLineVisible:false, lastValueVisible:false}}).setData({vwap});
const seg = (pts, color, width, style) => pc.addLineSeries({{color, lineWidth:width, lineStyle:style||0,
  priceLineVisible:false, lastValueVisible:false, crosshairMarkerVisible:false}}).setData(pts);
for (const z of {zones}) {{ seg(z.top,'#12b886',1,2); seg(z.bottom,'#12b886',1,2); }}
for (const t of {levels}) {{ seg(t.stop,'#e03131',2); seg(t.target,'#1c7ed6',2); }}
candles.setMarkers({markers});
const ec = LightweightCharts.createChart(document.getElementById('equity'), base);
ec.addAreaSeries({{lineColor:'#7048e8', topColor:'rgba(112,72,232,.3)', bottomColor:'rgba(112,72,232,0)'}}).setData({equity});
const focus = {focus};
if (focus) {{ pc.timeScale().setVisibleRange(focus); }} else {{ pc.timeScale().fitContent(); }}
const sync = r => {{ if (r) ec.timeScale().setVisibleRange(r); }};
sync(pc.timeScale().getVisibleRange());
pc.timeScale().subscribeVisibleTimeRangeChange(sync);
</script></body></html>"""


def _t(ts) -> int:
    return int(pd.Timestamp(ts).timestamp())


def write_chart(result, symbol: str, path: str | Path) -> Path:
    eng = result.engine
    inst = eng.cfg.instrument(symbol)
    cal = TradingCalendar(inst.session)
    base = [b for b in result.data.bars if b.symbol == symbol and (result.end is None or b.start < result.end)]
    five = resample_all(base, 5, inst.bars.base_interval_minutes, cal)
    candles = [{"time": _t(b.start), "open": b.open, "high": b.high, "low": b.low, "close": b.close} for b in five]

    vw = SessionVWAP(eng.cfg.strategy.vwap_require_complete_session, True)
    ends = {b.end for b in five}
    vwap = []
    for b in base:
        s = cal.session_by_close_date(dt.date.fromisoformat(b.session_id))
        vw.on_base(b, s.open)
        if b.end in ends:
            v, _ = vw.value()
            if v is not None:
                vwap.append({"time": _t(b.end - pd.Timedelta(minutes=5)), "value": round(v, 6)})

    zones = []
    for z in eng.rt[symbol].core.zones:
        end = z.status_at or (z.activated_at + pd.Timedelta(minutes=30 * eng.cfg.strategy.zone_expiry_bars))
        a, b = _t(z.activated_at), _t(end)
        if b <= a:
            continue
        zones.append({"top": [{"time": a, "value": z.upper}, {"time": b, "value": z.upper}],
                      "bottom": [{"time": a, "value": z.lower}, {"time": b, "value": z.lower}]})

    markers, levels = [], []
    for s in eng.rt[symbol].core.setups:
        markers.append({"time": _t(s.touch_bar_start), "position": "belowBar", "color": "#fab005", "shape": "circle",
                        "text": "touch"})
        if s.confirm_at is not None:
            markers.append({"time": _t(s.confirm_at - pd.Timedelta(minutes=5)), "position": "aboveBar",
                            "color": "#4c6ef5", "shape": "square", "text": f"shift>{s.break_threshold:g}"})
    for t in eng.trades + []:
        if t.symbol != symbol:
            continue
        a, b = _t(t.entry_time), _t(t.exit_time)
        markers.append({"time": a, "position": "belowBar", "color": "#2b8a3e", "shape": "arrowUp",
                        "text": f"BUY {t.entry_price:g}"})
        markers.append({"time": _t(pd.Timestamp(t.exit_time).floor("5min")), "position": "aboveBar",
                        "color": "#2b8a3e" if t.net_pnl > 0 else "#c92a2a", "shape": "arrowDown",
                        "text": f"{t.exit_reason} {t.r_multiple:+.2f}R"})
        b = max(b, a + 300)
        levels.append({"stop": [{"time": a, "value": t.stop}, {"time": b, "value": t.stop}],
                       "target": [{"time": a, "value": t.target}, {"time": b, "value": t.target}]})
    markers.sort(key=lambda m: m["time"])

    eq, seen = [], set()
    for ts, v in eng.equity_curve:
        k = _t(ts)
        if k in seen:
            eq[-1] = {"time": k, "value": round(v, 2)}
        else:
            eq.append({"time": k, "value": round(v, 2)})
            seen.add(k)
    eq = eq[::max(1, len(eq) // 5000)]

    events = sorted(m["time"] for m in markers)
    focus = {"from": events[0] - 6 * 3600, "to": events[0] + 6 * 3600} if events else None

    banner = ""
    if result.data.synthetic:
        banner = ('<div class="warn">SYNTHETIC DATA — generated test fixture, not market history. '
                  'Nothing on this chart is evidence of profitability.</div>')
    out = _TEMPLATE.format(symbol=html.escape(symbol), banner=banner, config=html.escape(eng.cfg.version_tag),
                           candles=json.dumps(candles), vwap=json.dumps(vwap), zones=json.dumps(zones),
                           levels=json.dumps(levels), markers=json.dumps(markers), equity=json.dumps(eq),
                           focus=json.dumps(focus))
    path = Path(path)
    path.write_text(out)
    return path
