"""The delegate tool: parallel department heads with read-only tools."""

from __future__ import annotations

import json
import threading

import pytest
from conftest import resp, text, tool_use

from jaxon.departments import DEPARTMENTS, DEPT_SYSTEM
from jaxon.tools import run_tool

CFO = "You are the Chief Financial Officer"
CMO = "You are the Chief Marketing Officer"
CTO = "You are the Chief Technology Officer"
COUNSEL = "You are the General Counsel"
READ_ONLY = {"recall", "get_company_snapshot", "list_tasks", "list_decisions", "list_goals", "get_metrics",
             "financial_summary", "get_current_time"}


def delegate_turn(client, briefs, final="Decision made."):
    """Script the CEO: one delegate call, then a final answer."""
    client.script(resp(tool_use("del_1", "delegate", {"briefs": briefs}), stop="tool_use"), resp(text(final)))


def delegate_result(client):
    """The tool_result the CEO got back for the delegate call."""
    ceo_calls = client.calls_for("You are Jaxon Vale")
    results = ceo_calls[-1]["messages"][-1]["content"]
    assert len(results) == 1 and results[0]["tool_use_id"] == "del_1"
    return results[0]


def test_departments_catalogue():
    assert set(DEPARTMENTS) == {"cfo", "cmo", "cto", "coo", "cpo", "counsel"}
    for info in DEPARTMENTS.values():
        prompt = DEPT_SYSTEM.format(**info)
        assert prompt.startswith(f"You are the {info['title']} of the company")
        assert info["focus"] in prompt


def test_delegate_schema(jaxon):
    tool = jaxon.tools["delegate"]
    item = tool.schema["properties"]["briefs"]["items"]
    assert item["properties"]["department"]["enum"] == list(DEPARTMENTS)
    assert tool.schema["required"] == ["briefs"]
    for key, info in DEPARTMENTS.items():
        assert key in tool.description and info["title"] in tool.description


def test_parallel_briefs(jaxon, client):
    barrier = threading.Barrier(3, timeout=5)  # cfo, cmo and counsel must all be in flight at once

    def memo(name):
        def respond(kwargs):
            barrier.wait()
            return resp(text(f"{name} memo"))
        return respond

    client.script(memo("CFO"), system_prefix=CFO)
    client.script(memo("CMO"), system_prefix=CMO)
    client.script(memo("Counsel"), system_prefix=COUNSEL)
    delegate_turn(client, [
        {"department": "cfo", "brief": "Can we afford two hires?"},
        {"department": "cmo", "brief": "Plan the launch."},
        {"department": "counsel", "brief": "Any GDPR exposure?"},
    ])
    _, result = jaxon.chat("Should we hire?")
    assert result.text == "Decision made."
    r = delegate_result(client)
    assert r["is_error"] is False
    assert json.loads(r["content"]) == {"memos": {"cfo": "CFO memo", "cmo": "CMO memo", "counsel": "Counsel memo"}}
    assert client.remaining() == 0

    cfo_call = client.calls_for(CFO)[0]
    assert cfo_call["messages"] == [{"role": "user", "content": "Can we afford two hires?"}]
    assert cfo_call["system"] == DEPT_SYSTEM.format(**DEPARTMENTS["cfo"])
    assert client.calls_for(COUNSEL)[0]["system"].startswith("You are the General Counsel of the company")


def test_sub_agents_get_only_read_only_tools(jaxon, client):
    client.script(resp(text("memo")), system_prefix=CFO)
    delegate_turn(client, [{"department": "cfo", "brief": "Runway?"}])
    jaxon.chat("x")
    tools = client.calls_for(CFO)[0]["tools"]
    names = {t["name"] for t in tools}
    assert names == READ_ONLY | {"web_search"}
    assert "delegate" not in names and "remember" not in names and "create_task" not in names


def test_sub_agent_cannot_call_write_tools(jaxon, client, memory):
    client.script(
        resp(tool_use("c1", "create_task", {"title": "sneaky"}), tool_use("c2", "financial_summary", {}),
             stop="tool_use"),
        resp(text("Runway is fine.")),
        system_prefix=CFO,
    )
    delegate_turn(client, [{"department": "cfo", "brief": "Runway?"}])
    jaxon.chat("x")
    assert memory.list_tasks("all") == []
    results = client.calls_for(CFO)[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["c1", "c2"]
    assert results[0]["is_error"] is True and "Unknown tool" in results[0]["content"]
    assert results[1]["is_error"] is False and "monthly_burn" in results[1]["content"]
    assert json.loads(delegate_result(client)["content"])["memos"]["cfo"] == "Runway is fine."


def test_department_effort_used(make_jaxon, client):
    j = make_jaxon(effort="high", department_effort="low")
    client.script(resp(text("memo")), system_prefix=CTO)
    delegate_turn(client, [{"department": "cto", "brief": "Build or buy?"}])
    j.chat("x")
    assert client.calls_for(CTO)[0]["output_config"] == {"effort": "low"}
    assert all(c["output_config"] == {"effort": "high"} for c in client.calls_for("You are Jaxon Vale"))


def test_default_department_effort_is_medium(jaxon, client):
    client.script(resp(text("memo")), system_prefix=CTO)
    delegate_turn(client, [{"department": "cto", "brief": "Build or buy?"}])
    jaxon.chat("x")
    call = client.calls_for(CTO)[0]
    assert call["output_config"] == {"effort": "medium"}
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default"


def test_department_without_web_search(make_jaxon, client):
    j = make_jaxon(web_search=False)
    client.script(resp(text("memo")), system_prefix=CFO)
    delegate_turn(client, [{"department": "cfo", "brief": "x"}])
    j.chat("x")
    assert {t["name"] for t in client.calls_for(CFO)[0]["tools"]} == READ_ONLY


def test_one_department_failing_does_not_break_others(jaxon, client):
    client.script(RuntimeError("overloaded"), system_prefix=CFO)
    client.script(resp(text("CMO memo")), system_prefix=CMO)
    delegate_turn(client, [{"department": "cfo", "brief": "a"}, {"department": "cmo", "brief": "b"}])
    _, result = jaxon.chat("x")
    assert result.text == "Decision made."
    r = delegate_result(client)
    assert r["is_error"] is False
    memos = json.loads(r["content"])["memos"]
    assert memos["cmo"] == "CMO memo"
    assert "could not respond" in memos["cfo"] and "RuntimeError" in memos["cfo"] and "overloaded" in memos["cfo"]


def test_empty_memo_placeholder(jaxon, client):
    client.script(resp(stop="end_turn"), system_prefix=CFO)
    delegate_turn(client, [{"department": "cfo", "brief": "a"}])
    jaxon.chat("x")
    assert json.loads(delegate_result(client)["content"])["memos"]["cfo"] == "(no memo returned)"


@pytest.mark.parametrize("args, needle", [
    ({"briefs": [{"department": "ceo", "brief": "hi"}]}, "unknown department"),
    ({"briefs": [{"department": "cfo", "brief": "ok"}, {"department": "hr", "brief": "x"}]}, "unknown department"),
    ({"briefs": []}, "at least one"),
    ({}, "at least one"),
    ({"briefs": [{"department": "cfo", "brief": "   "}]}, "needs text"),
    ({"briefs": [{"department": "cfo"}]}, "needs text"),
    ({"briefs": ["cfo"]}, "delegate"),
])
def test_delegate_validation_errors(jaxon, client, args, needle):
    out, err = run_tool(jaxon.tools, "delegate", args)
    assert err is True
    assert needle in out
    assert client.calls == []  # validation happens before any sub-agent runs


def test_unknown_department_error_reaches_model(jaxon, client):
    delegate_turn(client, [{"department": "cio", "brief": "?"}])
    _, result = jaxon.chat("x")
    r = delegate_result(client)
    assert r["is_error"] is True and "unknown department" in r["content"]
    assert result.tool_calls == [{"name": "delegate", "input": {"briefs": [{"department": "cio", "brief": "?"}]},
                                  "error": True}]


def test_delegate_alongside_other_tool_uses(jaxon, client, memory):
    client.script(resp(text("memo")), system_prefix=CFO)
    client.script(
        resp(tool_use("r1", "remember", {"content": "Founder wants profitability"}),
             tool_use("del_1", "delegate", {"briefs": [{"department": "cfo", "brief": "path to profit?"}]}),
             stop="tool_use"),
        resp(text("Plan ready.")),
    )
    jaxon.chat("x")
    results = client.calls_for("You are Jaxon Vale")[-1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["r1", "del_1"]
    assert all(r["is_error"] is False for r in results)
    assert memory.recall("profitability")


def test_two_briefs_to_same_department_keep_both_memos(jaxon, client):
    def respond(kwargs):
        return resp(text("memo on " + kwargs["messages"][0]["content"]))
    client.script(respond, respond, system_prefix=CFO)
    delegate_turn(client, [{"department": "cfo", "brief": "pricing"}, {"department": "cfo", "brief": "runway"}])
    jaxon.chat("x")
    memos = json.loads(delegate_result(client)["content"])["memos"]
    assert memos == {"cfo": "memo on pricing", "cfo (2)": "memo on runway"}
