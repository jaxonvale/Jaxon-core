"""Memory: CRUD for every table, recall, finance math, goals, tasks, sessions, snapshot."""

from __future__ import annotations

import json
import threading
from datetime import date

import pytest

from jaxon.memory import PRIORITIES, TASK_STATUSES, Memory, goal_progress


# -- construction ---------------------------------------------------------
def test_file_db_creates_parent_dirs_and_persists(tmp_path):
    path = tmp_path / "nested" / "dir" / "j.db"
    m = Memory(str(path))
    m.remember("persisted fact")
    m.close()
    assert path.exists()
    m2 = Memory(str(path))
    assert [r["content"] for r in m2.recall("persisted")] == ["persisted fact"]
    m2.close()


def test_threaded_writes_are_safe(memory):
    def worker(n):
        for i in range(20):
            memory.remember(f"fact {n}-{i}")
    threads = [threading.Thread(target=worker, args=(n,)) for n in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert memory._one("SELECT COUNT(*) AS c FROM memories")["c"] == 100


# -- memories -------------------------------------------------------------
def test_remember_and_recall_roundtrip(memory):
    mid = memory.remember("  Acme pays annually  ", kind="customer", tags=["acme", "billing"])
    rows = memory.recall("acme")
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == mid
    assert row["content"] == "Acme pays annually"  # stripped
    assert row["kind"] == "customer"
    assert row["tags"] == "acme billing"
    assert row["created_at"]


def test_remember_rejects_empty(memory):
    with pytest.raises(ValueError):
        memory.remember("   ")


def test_recall_matches_tags(memory):
    memory.remember("They prefer email", tags=["founder"])
    assert [r["content"] for r in memory.recall("founder")] == ["They prefer email"]


def test_recall_ranking_puts_best_match_first(memory):
    memory.remember("pricing discussion with the board")
    memory.remember("enterprise pricing pricing pricing tiers for enterprise customers")
    memory.remember("unrelated lunch note")
    rows = memory.recall("enterprise pricing")
    assert [r["content"] for r in rows][0].startswith("enterprise pricing")
    assert len(rows) == 2  # the unrelated note does not match
    assert all("lunch" not in r["content"] for r in rows)


def test_recall_or_semantics_and_punctuation_safe(memory):
    memory.remember("hiring plan for Q3")
    memory.remember("budget for marketing")
    # punctuation and FTS operators in the query must not break the MATCH
    rows = memory.recall('hiring OR "budget" AND (NOT) -*')
    contents = {r["content"] for r in rows}
    assert contents == {"hiring plan for Q3", "budget for marketing"}


def test_recall_kind_filter(memory):
    memory.remember("Alice is our designer", kind="person")
    memory.remember("Alice likes dark mode", kind="preference")
    assert [r["kind"] for r in memory.recall("alice", kind="person")] == ["person"]
    assert [r["kind"] for r in memory.recall("", kind="preference")] == ["preference"]


def test_recall_empty_query_returns_most_recent_first(memory):
    ids = [memory.remember(f"note {i}") for i in range(5)]
    rows = memory.recall("", limit=3)
    assert [r["id"] for r in rows] == list(reversed(ids))[:3]


def test_recall_limit_is_clamped(memory):
    for i in range(60):
        memory.remember(f"note {i}")
    assert len(memory.recall("", limit=0)) == 1
    assert len(memory.recall("", limit=1000)) == 50
    assert len(memory.recall("note", limit="5")) == 5


def test_forget_removes_from_table_and_index(memory):
    mid = memory.remember("obsolete pricing fact")
    assert memory.forget(mid) is True
    assert memory.recall("obsolete") == []
    assert memory.recall("") == []
    assert memory.forget(mid) is False


# -- profile --------------------------------------------------------------
def test_profile_upsert_normalises_key(memory):
    memory.set_profile(" Name ", " Acme ")
    memory.set_profile("name", "Acme Inc")
    memory.set_profile("stage", "seed")
    assert memory.profile() == {"name": "Acme Inc", "stage": "seed"}


# -- tasks ----------------------------------------------------------------
def test_create_and_get_task(memory):
    t = memory.create_task("  Ship v1  ", details="d", owner="cto", priority="high", due="2026-12-01")
    assert t["title"] == "Ship v1"
    assert (t["owner"], t["priority"], t["status"], t["due"], t["notes"]) == ("cto", "high", "open", "2026-12-01", [])
    assert memory.get_task(t["id"]) == t
    assert memory.get_task(9999) is None


def test_create_task_defaults(memory):
    t = memory.create_task("x", due="")
    assert (t["owner"], t["priority"], t["status"], t["due"]) == ("jaxon", "medium", "open", None)


def test_create_task_bad_priority(memory):
    with pytest.raises(ValueError, match="priority"):
        memory.create_task("x", priority="urgent")


def test_update_task_fields_and_notes(memory):
    t = memory.create_task("x")
    u = memory.update_task(t["id"], status="in_progress", note="started", priority="critical",
                           due="2026-11-30", owner="founder")
    assert (u["status"], u["priority"], u["due"], u["owner"]) == ("in_progress", "critical", "2026-11-30", "founder")
    assert [n["note"] for n in u["notes"]] == ["started"]
    u = memory.update_task(t["id"], note="halfway")
    assert [n["note"] for n in u["notes"]] == ["started", "halfway"]
    assert u["status"] == "in_progress"  # unchanged fields kept


@pytest.mark.parametrize("kwargs, exc", [
    ({"status": "finished"}, ValueError),
    ({"priority": "urgent"}, ValueError),
])
def test_update_task_validation(memory, kwargs, exc):
    t = memory.create_task("x")
    with pytest.raises(exc):
        memory.update_task(t["id"], **kwargs)


def test_update_missing_task(memory):
    with pytest.raises(KeyError):
        memory.update_task(42, status="done")


def test_task_ordering_by_priority_then_due_then_id(memory):
    a = memory.create_task("low", priority="low", due="2026-01-01")
    b = memory.create_task("med-nodue", priority="medium")
    c = memory.create_task("med-late", priority="medium", due="2026-12-01")
    d = memory.create_task("med-early", priority="medium", due="2026-06-01")
    e = memory.create_task("critical", priority="critical")
    f = memory.create_task("high", priority="high", due="2027-01-01")
    g = memory.create_task("med-nodue-2", priority="medium")
    order = [t["id"] for t in memory.list_tasks()]
    assert order == [e["id"], f["id"], d["id"], c["id"], b["id"], g["id"], a["id"]]


def test_list_tasks_filters(memory):
    open_ = memory.create_task("open", owner="cfo")
    prog = memory.create_task("prog", owner="cto")
    blocked = memory.create_task("blocked", owner="cfo")
    done = memory.create_task("done", owner="cfo")
    cancelled = memory.create_task("cancelled")
    memory.update_task(prog["id"], status="in_progress")
    memory.update_task(blocked["id"], status="blocked")
    memory.update_task(done["id"], status="done")
    memory.update_task(cancelled["id"], status="cancelled")

    ids = lambda rows: {r["id"] for r in rows}
    assert ids(memory.list_tasks("active")) == {open_["id"], prog["id"], blocked["id"]}
    assert ids(memory.list_tasks("done")) == {done["id"]}
    assert ids(memory.list_tasks()) == ids([open_, prog, blocked, done, cancelled])
    assert ids(memory.list_tasks("active", owner="cfo")) == {open_["id"], blocked["id"]}
    assert all(isinstance(r["notes"], list) for r in memory.list_tasks())


def test_constants():
    assert TASK_STATUSES == ("open", "in_progress", "blocked", "done", "cancelled")
    assert PRIORITIES == ("low", "medium", "high", "critical")


# -- decisions ------------------------------------------------------------
def test_decisions_newest_first_and_limit(memory):
    d1 = memory.log_decision("Pricing", "Charge $49", "market", "free tier")
    assert (d1["title"], d1["decision"], d1["rationale"], d1["alternatives"]) == ("Pricing", "Charge $49", "market", "free tier")
    memory.log_decision("Hire", "Hire a designer")
    memory.log_decision("Infra", "Stay on AWS")
    assert [d["title"] for d in memory.list_decisions()] == ["Infra", "Hire", "Pricing"]
    assert [d["title"] for d in memory.list_decisions(limit=1)] == ["Infra"]


# -- goals ----------------------------------------------------------------
def test_goal_progress_function():
    assert goal_progress([]) == 0.0
    assert goal_progress([{"target": 100, "current": 50}]) == 50.0
    assert goal_progress([{"target": 10, "current": 20}]) == 100.0  # capped
    assert goal_progress([{"target": 10, "current": -5}]) == 0.0  # floored
    assert goal_progress([{"target": 0, "current": 0}]) == 100.0  # zero target counts as met
    assert goal_progress([{"target": 4, "current": 1}, {"target": 3, "current": 2}]) == round((25 + 200 / 3) / 2, 1)


def test_set_goal_normalises_key_results(memory):
    g = memory.set_goal("Grow", [{"description": "MRR", "target": "10000", "current": 2500, "unit": "$"},
                                 {"description": "Logos"}], period="2026-Q4")
    assert g["objective"] == "Grow" and g["period"] == "2026-Q4" and g["status"] == "active"
    assert g["key_results"] == [
        {"description": "MRR", "target": 10000.0, "current": 2500.0, "unit": "$"},
        {"description": "Logos", "target": 100.0, "current": 0.0, "unit": ""},
    ]
    assert g["progress"] == 12.5


def test_update_key_result_and_errors(memory):
    g = memory.set_goal("Grow", [{"description": "MRR", "target": 200}])
    g2 = memory.update_key_result(g["id"], 0, 150)
    assert g2["key_results"][0]["current"] == 150.0
    assert g2["progress"] == 75.0
    with pytest.raises(IndexError):
        memory.update_key_result(g["id"], 1, 1)
    with pytest.raises(IndexError):
        memory.update_key_result(g["id"], -1, 1)
    with pytest.raises(KeyError):
        memory.update_key_result(999, 0, 1)


def test_goal_status_and_listing(memory):
    a = memory.set_goal("A", [{"description": "x", "target": 1}])
    b = memory.set_goal("B", [{"description": "y", "target": 1}])
    assert memory.set_goal_status(a["id"], "achieved")["status"] == "achieved"
    assert [g["id"] for g in memory.list_goals()] == [b["id"]]
    assert [g["id"] for g in memory.list_goals("achieved")] == [a["id"]]
    assert [g["id"] for g in memory.list_goals(None)] == [a["id"], b["id"]]
    with pytest.raises(ValueError):
        memory.set_goal_status(a["id"], "won")
    with pytest.raises(KeyError):
        memory.set_goal_status(999, "missed")


# -- metrics --------------------------------------------------------------
def test_metrics_history_and_latest(memory):
    memory.record_metric(" MRR ", 100, "$", "2026-01-01")
    memory.record_metric("mrr", 300, "$", "2026-03-01")
    memory.record_metric("mrr", 200, "$", "2026-02-01")
    r = memory.record_metric("nps", "42", note="survey")
    assert r["name"] == "nps" and r["value"] == 42.0 and r["recorded_on"] == date.today().isoformat()
    hist = memory.metric_history("MRR")
    assert [h["value"] for h in hist] == [100, 200, 300]  # chronological
    assert [h["value"] for h in memory.metric_history("mrr", limit=2)] == [200, 300]  # most recent two
    latest = {m["name"]: m["value"] for m in memory.latest_metrics()}
    assert latest == {"mrr": 200.0, "nps": 42.0}  # latest *recorded* row per name


def test_record_metric_rejects_non_numeric(memory):
    with pytest.raises(ValueError):
        memory.record_metric("mrr", "lots")


# -- finances -------------------------------------------------------------
def test_transactions_normalise_category(memory):
    t = memory.record_transaction("-50.5", " Marketing ", "ads", "2026-01-03")
    assert (t["amount"], t["category"], t["description"], t["occurred_on"]) == (-50.5, "marketing", "ads", "2026-01-03")
    assert memory.record_transaction(1, "revenue")["occurred_on"] == date.today().isoformat()


def test_financial_summary_empty(memory):
    s = memory.financial_summary()
    assert s["cash"] == 0 and s["monthly_burn"] == 0 and s["runway_months"] is None
    assert s["months"] == [] and s["by_category"] == []


def test_financial_summary_burn_and_runway(memory):
    memory.set_profile("cash_balance", "$100,000")
    memory.record_transaction(-12000, "payroll", occurred_on="2026-01-15")
    memory.record_transaction(2000, "revenue", occurred_on="2026-01-20")
    memory.record_transaction(-9000, "payroll", occurred_on="2026-02-15")
    memory.record_transaction(1000, "revenue", occurred_on="2026-02-20")
    memory.record_transaction(-7000, "payroll", occurred_on="2026-03-15")
    memory.record_transaction(1000, "revenue", occurred_on="2026-03-20")
    s = memory.financial_summary()
    # nets: -10000, -8000, -6000 -> avg -8000 -> burn 8000
    assert s["opening_balance"] == 100000.0
    assert s["cash"] == 100000 - 24000
    assert s["monthly_burn"] == 8000.0
    assert s["avg_monthly_net"] == -8000.0
    assert s["runway_months"] == round(76000 / 8000, 1)
    assert [m["month"] for m in s["months"]] == ["2026-01", "2026-02", "2026-03"]  # chronological
    jan = s["months"][0]
    assert (jan["revenue"], jan["expenses"], jan["net"]) == (2000, 12000, -10000)
    cats = {c["category"]: (c["total"], c["n"]) for c in s["by_category"]}
    assert cats == {"payroll": (-28000, 3), "revenue": (4000, 3)}
    assert s["by_category"][0]["category"] == "payroll"  # most negative first


def test_financial_summary_window_uses_latest_months(memory):
    memory.set_profile("cash_balance", "50000")
    memory.record_transaction(-30000, "x", occurred_on="2025-12-01")  # outside a 2-month window
    memory.record_transaction(-1000, "x", occurred_on="2026-01-01")
    memory.record_transaction(-3000, "x", occurred_on="2026-02-01")
    s = memory.financial_summary(months=2)
    assert s["monthly_burn"] == 2000.0
    assert s["cash"] == 16000.0
    assert s["runway_months"] == 8.0
    # months is clamped to at least 1
    assert memory.financial_summary(months=0)["monthly_burn"] == 3000.0


def test_financial_summary_profitable_means_no_runway(memory):
    memory.set_profile("cash_balance", "1,000")
    memory.record_transaction(5000, "revenue", occurred_on="2026-01-01")
    memory.record_transaction(-1000, "rent", occurred_on="2026-01-02")
    s = memory.financial_summary()
    assert s["monthly_burn"] == 0.0
    assert s["runway_months"] is None
    assert s["avg_monthly_net"] == 4000.0
    assert s["cash"] == 5000.0


def test_financial_summary_unparseable_cash_balance(memory):
    memory.set_profile("cash_balance", "about a million")
    memory.record_transaction(-100, "x", occurred_on="2026-01-01")
    s = memory.financial_summary()
    assert s["opening_balance"] == 0.0
    assert s["cash"] == -100.0


def test_financial_summary_cash_balance_with_decimals(memory):
    memory.set_profile("cash_balance", "$1,234,567.89")
    assert memory.financial_summary()["cash"] == 1234567.89


# -- sessions -------------------------------------------------------------
def test_sessions_save_load_list_delete(memory):
    assert memory.load_session("nope") == []
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}]
    memory.save_session("s1", msgs)
    assert memory.load_session("s1") == msgs
    memory.save_session("s1", msgs + msgs)  # upsert
    assert memory.load_session("s1") == msgs + msgs
    memory.save_session("s2", [])
    assert {s["id"] for s in memory.list_sessions()} == {"s1", "s2"}
    assert memory.delete_session("s1") is True
    assert memory.delete_session("s1") is False
    assert memory.load_session("s1") == []


def test_session_created_at_preserved_on_update(memory):
    memory.save_session("s", [])
    created = memory._one("SELECT created_at FROM sessions WHERE id='s'")["created_at"]
    memory.save_session("s", [{"role": "user", "content": "x"}])
    assert memory._one("SELECT created_at FROM sessions WHERE id='s'")["created_at"] == created


# -- snapshot -------------------------------------------------------------
def test_snapshot_shape(memory):
    memory.set_profile("name", "Acme")
    memory.set_profile("cash_balance", "10000")
    for i in range(17):
        memory.create_task(f"t{i}")
    done = memory.create_task("finished")
    memory.update_task(done["id"], status="done")
    memory.set_goal("Grow", [{"description": "x", "target": 10, "current": 5}], "Q4")
    memory.record_metric("mrr", 10)
    memory.record_transaction(-1000, "rent", occurred_on="2026-01-01")
    for i in range(7):
        memory.log_decision(f"d{i}", "x")

    s = memory.snapshot()
    json.dumps(s)  # serialisable
    assert s["date"] == date.today().isoformat()
    assert s["profile"]["name"] == "Acme"
    assert s["active_task_count"] == 17
    assert len(s["active_tasks"]) == 15
    assert set(s["active_tasks"][0]) == {"id", "title", "owner", "priority", "status", "due"}
    assert s["goals"] == [{"id": 1, "objective": "Grow", "period": "Q4", "progress": 50.0}]
    assert s["latest_metrics"][0]["name"] == "mrr"
    assert s["finance"] == {"cash": 9000.0, "monthly_burn": 1000.0, "runway_months": 9.0}
    assert [d["title"] for d in s["recent_decisions"]] == ["d6", "d5", "d4", "d3", "d2"]
