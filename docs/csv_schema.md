# CSV schema

One file per instrument, one row per bar, at the instrument's `bars.base_interval_minutes`
(1 or 5). No API credentials are needed to backtest from CSV.

| column | type | rules |
|---|---|---|
| `timestamp` | ISO-8601 with offset (`2025-03-07T14:30:00Z`), epoch seconds/milliseconds, or naive ISO | Marks the bar **open** by default (`bars.timestamp_convention: close` if your file marks the close). Naive timestamps require `bars.naive_timestamp_timezone`. Everything is converted to UTC. Must sit on the base-interval grid. |
| `open`, `high`, `low`, `close` | number > 0 | `low ≤ min(open, close) ≤ max(open, close) ≤ high` |
| `volume` | number ≥ 0, or empty | Empty = unknown. VWAP is disabled for any session containing unknown volume, so that session cannot produce a trade. |

Header names are case-insensitive. Lines starting with `#` are comments.

## Provenance comments
- A comment containing `SYNTHETIC` marks the file as a synthetic fixture. Every report
  built from it is labeled SYNTHETIC. `data.synthetic: true` in the config does the same.
- Files written by `python -m jaxon download` start with `# SOURCE:` lines naming the venue,
  symbol, interval, download time and each archive's SHA-256.
- The loader records each file's own SHA-256 in every run summary and setup record.

## Validation (python -m jaxon validate-data CONFIG)
| Problem | Handling |
|---|---|
| Missing column, non-numeric price, OHLC violation, negative volume, off-grid timestamp, conflicting duplicate timestamps | **error**, nothing runs |
| Out-of-order rows | sorted, warning |
| Exact duplicate rows | dropped, warning |
| Rows outside configured sessions | dropped, warning (or error if `drop_outside_session: false`) |
| Missing intervals inside a session | reported in `gaps`. Nothing is filled. Higher-timeframe bars containing the gap are not built, VWAP is disabled for that session, and new entries are blocked until the feed recovers. |
| Session starting late | reported in `sessions_missing_open`; VWAP is disabled for that session |

## Bundled files (all SYNTHETIC)
| file | what it is |
|---|---|
| `fixtures/SYNTHETIC_scenario_5m.csv` | Hand-scripted 8-day path that passes through every rule: one VWAP-target win, one stop-out, one cancelled setup |
| `fixtures/SYNTHETIC_equity_1m.csv` | Random 1-minute equity-like bars (NY session, DST change, two holidays, one deliberate 3-minute gap) |
| `fixtures/SYNTHETIC_crypto_5m.csv` | Random 5-minute 24/7 bars |

Regenerate them with `python -m jaxon make-fixtures`.
