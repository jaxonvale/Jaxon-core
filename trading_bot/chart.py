"""Write a standalone HTML chart: 5m candles, VWAP, entries, stops, targets, exits."""
import json

import pandas as pd

TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>body{{margin:0;background:#0f1115;color:#ddd;font:14px system-ui}}#c{{height:100vh}}
#h{{position:absolute;top:8px;left:12px;z-index:2}}</style></head><body>
<div id="h">{title} — orange: VWAP · green ▲ entry · red line stop · blue line target</div><div id="c"></div>
<script>
const chart = LightweightCharts.createChart(document.getElementById('c'), {{
  layout: {{background: {{color: '#0f1115'}}, textColor: '#ddd'}},
  grid: {{vertLines: {{color: '#1c1f26'}}, horzLines: {{color: '#1c1f26'}}}},
  timeScale: {{timeVisible: true}}}});
const candles = chart.addCandlestickSeries();
candles.setData({candles});
chart.addLineSeries({{color: '#f5a623', lineWidth: 2}}).setData({vwap});
candles.setMarkers({markers});
for (const t of {trades}) {{
  chart.addLineSeries({{color:'#e5484d', lineWidth:1, lastValueVisible:false, priceLineVisible:false}})
    .setData([{{time:t.a, value:t.stop}}, {{time:t.b, value:t.stop}}]);
  chart.addLineSeries({{color:'#3e8ef7', lineWidth:1, lastValueVisible:false, priceLineVisible:false}})
    .setData([{{time:t.a, value:t.target}}, {{time:t.b, value:t.target}}]);
}}
</script></body></html>"""


def write_chart(f: pd.DataFrame, trades: pd.DataFrame, path: str, title: str = "MTF VWAP bot"):
    ts = (f.index.astype("int64") // 10**9).tolist()
    candles = [{"time": t, "open": o, "high": h, "low": l, "close": c}
               for t, o, h, l, c in zip(ts, f.open, f.high, f.low, f.close)]
    vwap = [{"time": t, "value": v} for t, v in zip(ts, f.vwap)]
    markers, tr = [], []
    for _, r in (trades.iterrows() if not trades.empty else []):
        a, b = int(r.entry_time.timestamp()), int(r.exit_time.timestamp())
        markers.append({"time": a, "position": "belowBar", "color": "#30a46c", "shape": "arrowUp",
                        "text": f"BUY {r.entry:.2f}"})
        markers.append({"time": b, "position": "aboveBar", "color": "#30a46c" if r.pnl > 0 else "#e5484d",
                        "shape": "arrowDown", "text": f"{r.reason} {r.r:+.2f}R"})
        tr.append({"a": a, "b": b, "stop": r.stop, "target": r.target})
    markers.sort(key=lambda m: m["time"])
    html = TEMPLATE.format(title=title, candles=json.dumps(candles), vwap=json.dumps(vwap),
                           markers=json.dumps(markers), trades=json.dumps(tr))
    with open(path, "w") as fh:
        fh.write(html)
    return path
