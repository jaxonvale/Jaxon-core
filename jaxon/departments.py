"""Department heads Jaxon can delegate to. Each is a focused sub-agent with read-only access to company data."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from .tools import Tool, ToolError, _obj

if TYPE_CHECKING:
    from .agent import Jaxon

DEPARTMENTS: dict[str, dict[str, str]] = {
    "cfo": {
        "title": "Chief Financial Officer",
        "focus": "cash, burn, runway, unit economics, pricing, budgeting, fundraising and financial risk",
    },
    "cmo": {
        "title": "Chief Marketing Officer",
        "focus": "positioning, messaging, channels, growth experiments, brand, content and customer acquisition cost",
    },
    "cto": {
        "title": "Chief Technology Officer",
        "focus": "architecture, build-vs-buy, engineering estimates, technical debt, security and reliability",
    },
    "coo": {
        "title": "Chief Operating Officer",
        "focus": "operations, processes, hiring plans, vendors, execution cadence and accountability",
    },
    "cpo": {
        "title": "Chief Product Officer",
        "focus": "user problems, product strategy, roadmap prioritisation, discovery and product metrics",
    },
    "counsel": {
        "title": "General Counsel",
        "focus": "legal, contractual, IP, privacy and compliance risk (flag when a licensed lawyer is needed)",
    },
}

DEPT_SYSTEM = """You are the {title} of the company, reporting to Jaxon Vale, the CEO. Your remit: {focus}.

Jaxon will brief you on a question. Use the company data tools to ground your answer in real numbers and history, and web search for current external facts. You can read company data but not change it — recommend actions and Jaxon will decide.

Reply with a short memo: your recommendation first, then the key reasoning and numbers, risks, and concrete next steps with suggested owners. Say plainly what data is missing. Never invent figures."""


def build_delegate_tool(jaxon: "Jaxon") -> Tool:
    from .agent import run_loop

    def consult(dept: str, brief: str) -> str:
        info = DEPARTMENTS[dept]
        read_only = {n: t for n, t in jaxon.tools.items() if t.read_only}
        messages: list[dict[str, Any]] = [{"role": "user", "content": brief}]
        result = run_loop(
            jaxon.llm,
            system=DEPT_SYSTEM.format(**info),
            messages=messages,
            tools=read_only,
            extra_tools=jaxon.server_tools,
            effort=jaxon.config.department_effort,
            max_rounds=12,
        )
        return result.text or "(no memo returned)"

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        briefs = args.get("briefs") or []
        if not briefs:
            raise ToolError("briefs must contain at least one {department, brief}")
        for b in briefs:
            if b.get("department") not in DEPARTMENTS:
                raise ToolError(f"unknown department {b.get('department')!r}; choose from {list(DEPARTMENTS)}")
            if not (b.get("brief") or "").strip():
                raise ToolError("each brief needs text")
        with ThreadPoolExecutor(max_workers=min(6, len(briefs))) as pool:
            futures = [(b["department"], pool.submit(consult, b["department"], b["brief"])) for b in briefs]
            memos = {}
            for dept, fut in futures:
                try:
                    memos[dept] = fut.result()
                except Exception as e:  # one failed department should not sink the others
                    memos[dept] = f"(the {dept} could not respond: {type(e).__name__}: {e})"
        return {"memos": memos}

    return Tool(
        name="delegate",
        description=(
            "Brief one or more department heads and get their memos back. They run in parallel. "
            "Departments: " + ", ".join(f"{k} ({v['title']}: {v['focus']})" for k, v in DEPARTMENTS.items())
            + ". Give each a self-contained brief with the question, relevant context and what you need back."
        ),
        schema=_obj({"briefs": {"type": "array", "items": _obj(
            {"department": {"type": "string", "enum": list(DEPARTMENTS)}, "brief": {"type": "string"}},
            ["department", "brief"])}}, ["briefs"]),
        handler=handler,
    )
