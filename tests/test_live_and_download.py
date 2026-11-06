import io
import zipfile

import pandas as pd
import pytest

from jaxon.data import download as dl
from jaxon.data.csv_source import load_instrument_csv
from jaxon.live_paper import CoinbaseCandleFeed, LivePaper
from tests.helpers import cfg, ts


def _candles(start, n, step=300, skip=()):
    """Coinbase-style rows, newest first: [time, low, high, open, close, volume]."""
    t0 = int(ts(start).timestamp())
    rows = [[t0 + k * step, 99.0, 101.0, 100.0, 100.5, 10.0] for k in range(n) if k not in skip]
    return rows[::-1]


def fake_api(rows):
    def fetch(url):
        q = dict(p.split("=") for p in url.split("?")[1].split("&"))
        s, e = pd.Timestamp(q["start"]), pd.Timestamp(q["end"])
        return [r for r in rows if s.timestamp() <= r[0] < e.timestamp()]
    return fetch


def test_feed_emits_only_closed_candles_once(monkeypatch):
    monkeypatch.setattr("jaxon.live_paper.time.sleep", lambda s: None)
    rows = _candles("2025-01-01 00:00", 10)
    f = CoinbaseCandleFeed("BTC-USD", cfg().instruments[0], fetch=fake_api(rows))
    now = ts("2025-01-01 00:47")                      # candle 00:45 still forming
    bars = f.bootstrap(1, now=now)
    assert [b.start.minute for b in bars][-1] == 40 and len(bars) == 9
    assert f.poll(now) == []                          # nothing new, nothing repeated
    later = f.poll(ts("2025-01-01 00:51"))
    assert [b.start for b in later] == [ts("2025-01-01 00:45")]
    assert not f.is_stale(ts("2025-01-01 00:51"))
    assert f.is_stale(ts("2025-01-01 01:30"))


def test_feed_does_not_fill_gaps(monkeypatch):
    monkeypatch.setattr("jaxon.live_paper.time.sleep", lambda s: None)
    rows = _candles("2025-01-01 00:00", 10, skip={4, 5})
    f = CoinbaseCandleFeed("BTC-USD", cfg().instruments[0], fetch=fake_api(rows))
    bars = f.bootstrap(1, now=ts("2025-01-01 01:00"))
    assert len(bars) == 8 and ts("2025-01-01 00:20") not in [b.start for b in bars]


def test_live_paper_persists_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr("jaxon.live_paper.time.sleep", lambda s: None)
    rows = _candles("2025-01-01 00:00", 40)
    c = cfg()
    mk = lambda: {"TEST": CoinbaseCandleFeed("BTC-USD", c.instruments[0], fetch=fake_api(rows))}
    lp = LivePaper(c, tmp_path, mk())
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls, tz=None: ts("2025-01-01 02:00")))
    first = lp.start(bootstrap_days=1)
    assert first["bars_total"] == 24 and not first["resumed"]
    out = lp.cycle(ts("2025-01-01 03:21"))
    assert out["new_bars"] == 16
    lp2 = LivePaper(c, tmp_path, mk())
    again = lp2.start()
    assert again["resumed"] and again["bars_total"] == 40
    assert lp2.cycle(ts("2025-01-01 03:30"))["new_bars"] == 0


def test_binance_archive_parsing_and_checksum(tmp_path, monkeypatch):
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as z:     # 2025+ archives use microsecond timestamps
        z.writestr("BTCUSDT-1m-2025-01.csv", "1735689600000000,1,2,0.5,1.5,10,0,0,0,0,0,0\n"
                                             "1735689660000000,1.5,2,1,1.2,5,0,0,0,0,0,0\n")
    blob = raw.getvalue()
    import hashlib
    good = hashlib.sha256(blob).hexdigest()

    def get(url, retries=4):
        return f"{good}  x.zip".encode() if url.endswith(".CHECKSUM") else blob
    monkeypatch.setattr(dl, "_get", get)
    df, prov = dl.binance_vision("BTCUSDT", "1m", "2025-01", "2025-01", tmp_path / "cache")
    assert list(df.timestamp) == [ts("2025-01-01 00:00"), ts("2025-01-01 00:01")]
    assert good in prov[0]
    out = dl.write_csv(df, tmp_path / "o.csv", prov, "test")
    c = cfg(path=str(out), bars={"base_interval_minutes": 1})
    c.instruments[0].data.synthetic = False
    bars, p, _ = load_instrument_csv(c.instruments[0])
    assert len(bars) == 2 and p.synthetic is False and any("SOURCE" in h for h in p.header_comments)

    def bad(url, retries=4):
        return b"0" * 64 + b"  x.zip" if url.endswith(".CHECKSUM") else blob
    monkeypatch.setattr(dl, "_get", bad)
    with pytest.raises(ValueError, match="checksum"):
        dl.binance_vision("BTCUSDT", "1m", "2025-01", "2025-01", tmp_path / "cache2")
