"""Tools Jaxon can use. Each tool is a JSON schema plus a Python handler."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from .memory import PRIORITIES, TASK_STATUSES, Memory


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    read_only: bool = False

    def definition(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.schema}


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


S = {"type": "string"}
N = {"type": "number"}
I = {"type": "integer"}


class ToolError(Exception):
    """Raised by handlers for errors the model should see and correct."""


def _need(args: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if args.get(k) in (None, "")]
    if missing:
        raise ToolError(f"missing required field(s): {', '.join(missing)}")


def build_tools(memory: Memory) -> dict[str, Tool]:
    m = memory
    tools: list[Tool] = []

    def add(name, description, schema, read_only=False):
        def deco(fn):
            tools.append(Tool(name, description, schema, fn, read_only))
            return fn
        return deco

    # -- memory --------------------------------------------------------
    @add("remember",
         "Save something to long-term memory so you know it in future conversations: facts about the "
         "company, customers, people, the founder's preferences, lessons learned. Save anything you would "
         "want to know next week. One clear, self-contained statement per call.",
         _obj({"content": S,
               "kind": {"type": "string", "enum": ["fact", "preference", "person", "customer", "insight", "lesson"]},
               "tags": {"type": "array", "items": S}}, ["content"]))
    def _remember(a):
        _need(a, "content")
        mid = m.remember(a["content"], a.get("kind") or "fact", a.get("tags") or [])
        return {"saved": True, "id": mid}

    @add("recall",
         "Search long-term memory by keywords. Use before answering anything that depends on past context. "
         "An empty query returns the most recent memories.",
         _obj({"query": S, "limit": I, "kind": S}, ["query"]), read_only=True)
    def _recall(a):
        return m.recall(a.get("query", ""), a.get("limit") or 10, a.get("kind") or None)

    @add("forget", "Delete a memory that is wrong or outdated, by id.", _obj({"id": I}, ["id"]))
    def _forget(a):
        return {"deleted": m.forget(int(a["id"]))}

    # -- company profile -------------------------------------------------
    @add("update_company_profile",
         "Set a field of the company profile (e.g. name, mission, product, stage, team_size, cash_balance, "
         "target_customer, founder_name). cash_balance is the opening cash in dollars; transactions add to it.",
         _obj({"field": S, "value": S}, ["field", "value"]))
    def _profile(a):
        _need(a, "field", "value")
        m.set_profile(a["field"], a["value"])
        return m.profile()

    @add("get_company_snapshot",
         "Get the current state of the company: profile, active tasks, goals, latest metrics, cash and runway.",
         _obj({}, []), read_only=True)
    def _snapshot(a):
        return m.snapshot()

    # -- tasks -----------------------------------------------------------
    @add("create_task",
         "Create a task or action item. owner is who does it: 'jaxon', 'founder', a department "
         "(cfo, cmo, cto, coo, cpo, counsel) or a person's name. due is YYYY-MM-DD.",
         _obj({"title": S, "details": S, "owner": S,
               "priority": {"type": "string", "enum": list(PRIORITIES)}, "due": S}, ["title"]))
    def _create_task(a):
        _need(a, "title")
        return m.create_task(a["title"], a.get("details") or "", a.get("owner") or "jaxon",
                             a.get("priority") or "medium", a.get("due") or None)

    @add("update_task", "Update a task's status, priority, due date or owner, and/or add a progress note.",
         _obj({"id": I, "status": {"type": "string", "enum": list(TASK_STATUSES)},
               "priority": {"type": "string", "enum": list(PRIORITIES)},
               "due": S, "owner": S, "note": S}, ["id"]))
    def _update_task(a):
        try:
            return m.update_task(int(a["id"]), a.get("status"), a.get("note"), a.get("priority"),
                                 a.get("due"), a.get("owner"))
        except (KeyError, ValueError) as e:
            raise ToolError(str(e))

    @add("list_tasks",
         "List tasks. status: 'active' (open, in progress or blocked; the default), 'all', or one exact status.",
         _obj({"status": S, "owner": S}, []), read_only=True)
    def _list_tasks(a):
        status = a.get("status") or "active"
        return m.list_tasks(None if status == "all" else status, a.get("owner") or None)

    # -- decisions -------------------------------------------------------
    @add("log_decision",
         "Record a significant decision with its rationale and the alternatives considered, so the "
         "reasoning can be revisited later.",
         _obj({"title": S, "decision": S, "rationale": S, "alternatives": S}, ["title", "decision"]))
    def _log_decision(a):
        _need(a, "title", "decision")
        return m.log_decision(a["title"], a["decision"], a.get("rationale") or "", a.get("alternatives") or "")

    @add("list_decisions", "List recent decisions, newest first.", _obj({"limit": I}, []), read_only=True)
    def _list_decisions(a):
        return m.list_decisions(a.get("limit") or 20)

    # -- goals -----------------------------------------------------------
    kr_schema = _obj({"description": S, "target": N, "current": N, "unit": S}, ["description", "target"])

    @add("set_goal", "Create an objective with measurable key results (OKR). period e.g. '2026-Q4'.",
         _obj({"objective": S, "period": S, "key_results": {"type": "array", "items": kr_schema}},
              ["objective", "key_results"]))
    def _set_goal(a):
        _need(a, "objective")
        if not a.get("key_results"):
            raise ToolError("at least one key result is required")
        return m.set_goal(a["objective"], a["key_results"], a.get("period") or "")

    @add("update_key_result", "Set the current value of a key result (kr_index is 0-based).",
         _obj({"goal_id": I, "kr_index": I, "current": N}, ["goal_id", "kr_index", "current"]))
    def _update_kr(a):
        try:
            return m.update_key_result(int(a["goal_id"]), int(a["kr_index"]), float(a["current"]))
        except (KeyError, IndexError) as e:
            raise ToolError(str(e))

    @add("set_goal_status", "Mark a goal active, achieved, missed or dropped.",
         _obj({"goal_id": I, "status": {"type": "string", "enum": ["active", "achieved", "missed", "dropped"]}},
              ["goal_id", "status"]))
    def _goal_status(a):
        try:
            return m.set_goal_status(int(a["goal_id"]), a["status"])
        except (KeyError, ValueError) as e:
            raise ToolError(str(e))

    @add("list_goals", "List goals with progress. status defaults to 'active'; 'all' lists every goal.",
         _obj({"status": S}, []), read_only=True)
    def _list_goals(a):
        status = a.get("status") or "active"
        return m.list_goals(None if status == "all" else status)

    # -- metrics ---------------------------------------------------------
    @add("record_metric",
         "Record a KPI reading (e.g. mrr, active_users, churn_rate, cac, nps). recorded_on is YYYY-MM-DD, "
         "default today.",
         _obj({"name": S, "value": N, "unit": S, "recorded_on": S, "note": S}, ["name", "value"]))
    def _record_metric(a):
        _need(a, "name")
        return m.record_metric(a["name"], a["value"], a.get("unit") or "", a.get("recorded_on") or None,
                               a.get("note") or "")

    @add("get_metrics", "Get the history of one KPI, or the latest value of every KPI if name is empty.",
         _obj({"name": S, "limit": I}, []), read_only=True)
    def _get_metrics(a):
        if a.get("name"):
            hist = m.metric_history(a["name"], a.get("limit") or 30)
            if len(hist) >= 2 and hist[0]["value"]:
                change = (hist[-1]["value"] - hist[0]["value"]) / abs(hist[0]["value"]) * 100
                return {"history": hist, "change_pct": round(change, 1)}
            return {"history": hist}
        return m.latest_metrics()

    # -- finance ---------------------------------------------------------
    @add("record_transaction",
         "Record money in or out. amount is positive for income and negative for spending. "
         "category e.g. revenue, payroll, marketing, infrastructure, rent, tools, fundraising.",
         _obj({"amount": N, "category": S, "description": S, "occurred_on": S}, ["amount", "category"]))
    def _record_tx(a):
        _need(a, "category")
        return m.record_transaction(a["amount"], a["category"], a.get("description") or "",
                                    a.get("occurred_on") or None)

    @add("financial_summary",
         "Cash on hand, average monthly burn, runway in months, monthly revenue/expenses and spend by category.",
         _obj({"months": I}, []), read_only=True)
    def _fin(a):
        return m.financial_summary(a.get("months") or 3)

    # -- misc ------------------------------------------------------------
    @add("get_current_time", "Current local date and time.", _obj({}, []), read_only=True)
    def _time(a):
        return datetime.now().astimezone().isoformat(timespec="minutes")

    return {t.name: t for t in tools}


def run_tool(tools: dict[str, Tool], name: str, args: Any) -> tuple[str, bool]:
    """Run a tool; returns (result_text, is_error). Never raises."""
    tool = tools.get(name)
    if tool is None:
        return f"Unknown tool: {name}", True
    if not isinstance(args, dict):
        return "Tool input must be a JSON object", True
    try:
        result = tool.handler(args)
    except ToolError as e:
        return str(e), True
    except (TypeError, ValueError, KeyError) as e:
        return f"Invalid input for {name}: {e}", True
    except Exception as e:  # keep the agent alive on unexpected handler errors
        return f"{name} failed: {type(e).__name__}: {e}", True
    return result if isinstance(result, str) else json.dumps(result, default=str), False
