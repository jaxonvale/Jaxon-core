"""Every tool through run_tool, including error paths. run_tool must never raise."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from jaxon.memory import PRIORITIES, TASK_STATUSES
from jaxon.tools import Tool, ToolError, build_tools, run_tool

ALL_TOOLS = {
    "remember", "recall", "forget", "update_company_profile", "get_company_snapshot", "create_task",
    "update_task", "list_tasks", "log_decision", "list_decisions", "set_goal", "update_key_result",
    "set_goal_status", "list_goals", "record_metric", "get_metrics", "record_transaction",
    "financial_summary", "get_current_time",
}
READ_ONLY = {"recall", "get_company_snapshot", "list_tasks", "list_decisions", "list_goals", "get_metrics",
             "financial_summary", "get_current_time"}


@pytest.fixture
def tools(memory):
    return build_tools(memory)


def ok(tools, name, args=None):
    out, err = run_tool(tools, name, {} if args is None else args)
    assert err is False, out
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return out


def fail(tools, name, args):
    out, err = run_tool(tools, name, args)
    assert err is True
    assert isinstance(out, str) and out
    return out


# -- registry / definitions -----------------------------------------------
def test_tool_registry(tools):
    assert set(tools) == ALL_TOOLS
    assert {n for n, t in tools.items() if t.read_only} == READ_ONLY


def test_definitions_are_valid_schemas(tools):
    for name, tool in tools.items():
        d = tool.definition()
        assert set(d) == {"name", "description", "input_schema"}
        assert d["name"] == name
        assert d["description"].strip()
        schema = d["input_schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) <= set(schema["properties"])


def test_enums_in_schemas(tools):
    props = tools["update_task"].schema["properties"]
    assert props["status"]["enum"] == list(TASK_STATUSES)
    assert props["priority"]["enum"] == list(PRIORITIES)
    assert tools["create_task"].schema["properties"]["priority"]["enum"] == list(PRIORITIES)


# -- generic run_tool error paths -----------------------------------------
def test_unknown_tool(tools):
    out = fail(tools, "launch_rockets", {})
    assert "Unknown tool" in out and "launch_rockets" in out


@pytest.mark.parametrize("args", [None, "a string", ["list"], 42])
def test_non_object_input(tools, args):
    assert "JSON object" in fail(tools, "remember", args)


def test_unexpected_handler_exception_is_contained():
    def boom(a):
        raise RuntimeError("disk on fire")
    tools = {"boom": Tool("boom", "d", {"type": "object"}, boom)}
    out, err = run_tool(tools, "boom", {})
    assert err is True and "RuntimeError" in out and "disk on fire" in out


def test_tool_error_message_passed_through():
    def h(a):
        raise ToolError("please fix X")
    out, err = run_tool({"t": Tool("t", "d", {}, h)}, "t", {})
    assert (out, err) == ("please fix X", True)


def test_string_results_not_json_encoded():
    out, err = run_tool({"t": Tool("t", "d", {}, lambda a: "plain")}, "t", {})
    assert (out, err) == ("plain", False)


# -- memory tools ---------------------------------------------------------
def test_remember_recall_forget(tools, memory):
    r = ok(tools, "remember", {"content": "Acme renews in March", "kind": "customer", "tags": ["acme"]})
    assert r["saved"] is True
    rows = ok(tools, "recall", {"query": "acme"})
    assert [x["id"] for x in rows] == [r["id"]]
    assert rows[0]["kind"] == "customer"
    assert ok(tools, "recall", {"query": "acme", "kind": "person"}) == []
    assert len(ok(tools, "recall", {"query": ""})) == 1
    assert len(ok(tools, "recall", {})) == 1  # query omitted -> recent
    assert ok(tools, "forget", {"id": r["id"]}) == {"deleted": True}
    assert ok(tools, "forget", {"id": r["id"]}) == {"deleted": False}


def test_remember_defaults_kind(tools, memory):
    ok(tools, "remember", {"content": "x", "kind": None, "tags": None})
    assert memory.recall("")[0]["kind"] == "fact"


@pytest.mark.parametrize("args", [{}, {"content": ""}, {"content": None}])
def test_remember_missing_content(tools, args):
    assert "content" in fail(tools, "remember", args)


def test_remember_whitespace_content_is_error(tools):
    fail(tools, "remember", {"content": "   "})


def test_recall_bad_limit(tools):
    assert "recall" in fail(tools, "recall", {"query": "x", "limit": "many"})


@pytest.mark.parametrize("args", [{}, {"id": "abc"}])
def test_forget_bad_input(tools, args):
    fail(tools, "forget", args)


# -- profile / snapshot ---------------------------------------------------
def test_update_company_profile(tools):
    assert ok(tools, "update_company_profile", {"field": "Name", "value": "Acme"}) == {"name": "Acme"}
    out = fail(tools, "update_company_profile", {"field": "name"})
    assert "value" in out


def test_get_company_snapshot(tools):
    ok(tools, "update_company_profile", {"field": "name", "value": "Acme"})
    snap = ok(tools, "get_company_snapshot")
    assert snap["profile"] == {"name": "Acme"}
    assert {"date", "active_tasks", "goals", "latest_metrics", "finance", "recent_decisions"} <= set(snap)


# -- tasks ----------------------------------------------------------------
def test_create_task_and_defaults(tools):
    t = ok(tools, "create_task", {"title": "Ship"})
    assert (t["owner"], t["priority"], t["due"], t["status"]) == ("jaxon", "medium", None, "open")
    t2 = ok(tools, "create_task", {"title": "Plan", "owner": "cfo", "priority": "high", "due": "2026-12-01",
                                   "details": "budget"})
    assert (t2["owner"], t2["priority"], t2["due"], t2["details"]) == ("cfo", "high", "2026-12-01", "budget")


def test_create_task_errors(tools):
    assert "title" in fail(tools, "create_task", {})
    assert "priority" in fail(tools, "create_task", {"title": "x", "priority": "urgent"})


def test_update_task(tools):
    t = ok(tools, "create_task", {"title": "x"})
    u = ok(tools, "update_task", {"id": t["id"], "status": "done", "note": "shipped"})
    assert u["status"] == "done" and u["notes"][0]["note"] == "shipped"


@pytest.mark.parametrize("args, needle", [
    ({"id": 999, "status": "done"}, "not found"),
    ({"status": "done"}, "id"),
    ({"id": "seven"}, "invalid literal"),
])
def test_update_task_errors(tools, args, needle):
    assert needle in fail(tools, "update_task", args)


@pytest.mark.parametrize("field, value", [("status", "finished"), ("priority", "urgent")])
def test_update_task_bad_enum(tools, field, value):
    t = ok(tools, "create_task", {"title": "x"})
    assert field in fail(tools, "update_task", {"id": t["id"], field: value})


def test_list_tasks_statuses(tools):
    a = ok(tools, "create_task", {"title": "a", "owner": "cfo"})
    b = ok(tools, "create_task", {"title": "b"})
    ok(tools, "update_task", {"id": b["id"], "status": "done"})
    assert [t["id"] for t in ok(tools, "list_tasks")] == [a["id"]]  # default active
    assert {t["id"] for t in ok(tools, "list_tasks", {"status": "all"})} == {a["id"], b["id"]}
    assert [t["id"] for t in ok(tools, "list_tasks", {"status": "done"})] == [b["id"]]
    assert [t["id"] for t in ok(tools, "list_tasks", {"status": "all", "owner": "cfo"})] == [a["id"]]


# -- decisions ------------------------------------------------------------
def test_decisions(tools):
    d = ok(tools, "log_decision", {"title": "Price", "decision": "$49", "rationale": "comps"})
    assert d["alternatives"] == ""
    ok(tools, "log_decision", {"title": "Hire", "decision": "yes"})
    assert [x["title"] for x in ok(tools, "list_decisions")] == ["Hire", "Price"]
    assert [x["title"] for x in ok(tools, "list_decisions", {"limit": 1})] == ["Hire"]
    assert "decision" in fail(tools, "log_decision", {"title": "x"})
    assert "title" in fail(tools, "log_decision", {"decision": "x"})


# -- goals ----------------------------------------------------------------
def test_goal_lifecycle(tools):
    g = ok(tools, "set_goal", {"objective": "Grow", "period": "2026-Q4",
                               "key_results": [{"description": "MRR", "target": 100, "current": 25}]})
    assert g["progress"] == 25.0
    g = ok(tools, "update_key_result", {"goal_id": g["id"], "kr_index": 0, "current": 50})
    assert g["progress"] == 50.0
    assert [x["id"] for x in ok(tools, "list_goals")] == [g["id"]]
    assert ok(tools, "set_goal_status", {"goal_id": g["id"], "status": "achieved"})["status"] == "achieved"
    assert ok(tools, "list_goals") == []
    assert [x["id"] for x in ok(tools, "list_goals", {"status": "all"})] == [g["id"]]
    assert [x["id"] for x in ok(tools, "list_goals", {"status": "achieved"})] == [g["id"]]


@pytest.mark.parametrize("args, needle", [
    ({"key_results": [{"description": "x", "target": 1}]}, "objective"),
    ({"objective": "x"}, "key result"),
    ({"objective": "x", "key_results": []}, "key result"),
    ({"objective": "x", "key_results": [{"target": 1}]}, "set_goal"),  # KR without description
    ({"objective": "x", "key_results": [{"description": "d", "target": "lots"}]}, "set_goal"),
])
def test_set_goal_errors(tools, args, needle):
    assert needle in fail(tools, "set_goal", args)


def test_update_key_result_errors(tools):
    g = ok(tools, "set_goal", {"objective": "x", "key_results": [{"description": "d", "target": 1}]})
    assert "out of range" in fail(tools, "update_key_result", {"goal_id": g["id"], "kr_index": 3, "current": 1})
    assert "not found" in fail(tools, "update_key_result", {"goal_id": 99, "kr_index": 0, "current": 1})
    fail(tools, "update_key_result", {"goal_id": g["id"], "kr_index": 0})
    fail(tools, "update_key_result", {"goal_id": g["id"], "kr_index": 0, "current": "high"})


def test_set_goal_status_errors(tools):
    g = ok(tools, "set_goal", {"objective": "x", "key_results": [{"description": "d", "target": 1}]})
    assert "status" in fail(tools, "set_goal_status", {"goal_id": g["id"], "status": "won"})
    assert "not found" in fail(tools, "set_goal_status", {"goal_id": 99, "status": "missed"})
    fail(tools, "set_goal_status", {"goal_id": g["id"]})


# -- metrics --------------------------------------------------------------
def test_metrics(tools):
    ok(tools, "record_metric", {"name": "MRR", "value": 100, "unit": "$", "recorded_on": "2026-01-01"})
    ok(tools, "record_metric", {"name": "mrr", "value": 150, "recorded_on": "2026-02-01"})
    ok(tools, "record_metric", {"name": "nps", "value": 40})
    r = ok(tools, "get_metrics", {"name": "mrr"})
    assert [h["value"] for h in r["history"]] == [100, 150]
    assert r["change_pct"] == 50.0
    one = ok(tools, "get_metrics", {"name": "nps"})
    assert "change_pct" not in one and len(one["history"]) == 1
    assert {m["name"] for m in ok(tools, "get_metrics")} == {"mrr", "nps"}
    assert ok(tools, "get_metrics", {"name": "missing"}) == {"history": []}


def test_get_metrics_zero_baseline_has_no_change(tools):
    ok(tools, "record_metric", {"name": "users", "value": 0, "recorded_on": "2026-01-01"})
    ok(tools, "record_metric", {"name": "users", "value": 10, "recorded_on": "2026-02-01"})
    assert "change_pct" not in ok(tools, "get_metrics", {"name": "users"})


def test_get_metrics_negative_baseline(tools):
    ok(tools, "record_metric", {"name": "net", "value": -100, "recorded_on": "2026-01-01"})
    ok(tools, "record_metric", {"name": "net", "value": -50, "recorded_on": "2026-02-01"})
    assert ok(tools, "get_metrics", {"name": "net"})["change_pct"] == 50.0


@pytest.mark.parametrize("args", [{"value": 1}, {"name": "mrr"}, {"name": "mrr", "value": "lots"}])
def test_record_metric_errors(tools, args):
    fail(tools, "record_metric", args)


# -- finance --------------------------------------------------------------
def test_finance_tools(tools):
    ok(tools, "update_company_profile", {"field": "cash_balance", "value": "$10,000"})
    tx = ok(tools, "record_transaction", {"amount": -2000, "category": "Payroll", "occurred_on": "2026-01-05"})
    assert tx["category"] == "payroll"
    s = ok(tools, "financial_summary")
    assert s["cash"] == 8000.0 and s["monthly_burn"] == 2000.0 and s["runway_months"] == 4.0
    assert ok(tools, "financial_summary", {"months": 1})["monthly_burn"] == 2000.0


@pytest.mark.parametrize("args", [{"amount": 5}, {"category": "x"}, {"amount": "five", "category": "x"}])
def test_record_transaction_errors(tools, args):
    fail(tools, "record_transaction", args)


def test_get_current_time(tools):
    out, err = run_tool(tools, "get_current_time", {})
    assert err is False
    assert datetime.fromisoformat(out).tzinfo is not None


# -- never raises ---------------------------------------------------------
@pytest.mark.parametrize("name", sorted(ALL_TOOLS))
@pytest.mark.parametrize("args", [{}, {"id": None, "goal_id": "x", "kr_index": [], "amount": {}, "value": []}])
def test_no_tool_raises_on_garbage(tools, name, args):
    out, err = run_tool(tools, name, args)
    assert isinstance(out, str) and isinstance(err, bool)
