import json

import pandas as pd
import pytest

from jaxon.cli import main
from jaxon.evaluation import sensitivity
from jaxon.explain import explain
from jaxon.report import export, metrics, summarize
from tests.helpers import SCENARIO_CFG, run_scenario, scenario_cfg, scenario_data


def test_metrics_math():
    t = pd.DataFrame({
        "net_pnl": [200.0, -100.0, 300.0, -100.0], "r_multiple": [2.0, -1.0, 3.0, -1.0],
        "entry_fee": [1.0] * 4, "exit_fee": [1.0] * 4, "entry_friction_cash": [0.5] * 4,
        "exit_friction_cash": [0.5] * 4, "exit_reason": ["target", "stop", "target", "stop"], "flags": [""] * 4,
        "entry_time": pd.to_datetime(["2025-01-01 10:00"] * 4, utc=True),
        "exit_time": pd.to_datetime(["2025-01-01 11:00"] * 4, utc=True)})
    eq = pd.Series([100200, 100100, 100400, 100300.0])
    m = metrics(t, eq, 100000, total_minutes=480, min_trades=100)
    assert m["win_rate"] == 0.5 and m["profit_factor"] == 2.5 and m["expectancy_r"] == 0.75
    assert m["avg_win"] == 250 and m["avg_loss"] == -100 and m["total_costs"] == 12
    assert m["exposure_pct"] == 50.0 and m["max_drawdown_pct"] == pytest.approx(-0.0998, abs=1e-3)
    assert "No statistical edge" in m["sample_warning"]


def test_summary_labels_synthetic_and_splits_periods():
    r = run_scenario()
    s = summarize(r)
    assert s["data_kind"] == "SYNTHETIC" and "not evidence" in s["disclaimer"]
    assert set(s["periods"]) == {"development", "validation", "out_of_sample"}
    assert sum(p["trades"] for p in s["periods"].values()) == s["overall"]["trades"] == 2
    assert s["skipped_by_reason"]["setup:cancelled_traded_below_zone"] == 1
    assert s["config_version"].startswith("example-scenario-1+")


def test_exports_and_chart(tmp_path):
    r = run_scenario()
    paths = export(r, tmp_path, summarize(r))
    for k in ("trades_csv", "trades_json", "setups_csv", "setups_json", "equity", "journal", "summary"):
        assert (tmp_path / paths[k].split("/")[-1]).exists()
    setups = json.loads((tmp_path / "setups.json").read_text())
    s = next(x for x in setups if x.get("exit_reason") == "target")
    for field in ("htf_swings", "zone_lower", "zone_upper", "zone_activated_at", "touch_at", "confirm_at",
                  "break_threshold", "vwap_snapshot", "proposed_entry", "actual_entry", "stop", "target",
                  "quantity", "proposed_rr", "actual_rr", "est_round_trip_cost", "fees", "outcome",
                  "data_provenance", "config_version"):
        assert s.get(field) not in (None, ""), field
    assert "[SYNTHETIC]" in s["data_provenance"]
    html = (tmp_path / "chart_SCENARIO.html").read_text()
    assert "SYNTHETIC DATA" in html and "lightweight-charts@4.2.0" in html and "BUY" in html


def test_sensitivity_never_sees_out_of_sample():
    c = scenario_cfg()
    d = scenario_data(c)
    out = sensitivity(c, d)
    cutoff = pd.Timestamp(c.evaluation.validation_end, tz="UTC")
    assert all(pd.Timestamp(r["bars_seen_until"]) < cutoff for r in out["rows"])
    assert {r["parameter"] for r in out["rows"]} == set(c.evaluation.sensitivity)


def test_explain_is_deterministic_and_llm_optional(tmp_path):
    r = run_scenario()
    export(r, tmp_path, summarize(r), chart=False)
    sid = r.engine.trades[0].setup_id
    txt = explain(tmp_path, sid)
    assert "frozen break threshold" in txt and "VWAP snapshot" in txt
    txt2 = explain(tmp_path, sid, use_llm=True)       # no SDK/credentials here: must degrade, not fail
    assert txt2.startswith(txt) and ("LLM explanation unavailable" in txt2 or "LLM EXPLANATION" in txt2)


def test_cli_backtest_and_replay(tmp_path, capsys):
    assert main(["backtest", str(SCENARIO_CFG), "--out", str(tmp_path / "bt"), "--no-chart"]) == 0
    assert "SYNTHETIC" in capsys.readouterr().out
    assert main(["replay", str(SCENARIO_CFG), "--state-dir", str(tmp_path / "pp"), "--max-bars", "500",
                 "--checkpoint-every", "100"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["bars_this_run"] == 500 and not first["resumed"]
    assert main(["replay", str(SCENARIO_CFG), "--state-dir", str(tmp_path / "pp")]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["resumed"] and second["bars_total"] == 2304 and second["trades_closed"] == 2
    assert main(["validate-config", str(tmp_path / "missing.yaml")]) == 2
