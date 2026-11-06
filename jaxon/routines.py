"""Recurring CEO routines: daily briefing and weekly review."""

from __future__ import annotations

from .agent import Jaxon

DAILY_BRIEFING = """Run my daily briefing. Check the snapshot, overdue and blocked tasks, goal progress, KPI trends and cash/runway. Give me:
1. The one thing that matters most today and why.
2. Fires: anything overdue, blocked, or trending the wrong way.
3. Today's top 3 priorities, with owners.
4. Decisions you need from me (if any).
Update task statuses or create tasks where it's obvious. Keep it under 250 words."""

WEEKLY_REVIEW = """Run the weekly review. Brief the CFO on cash and runway, and whichever other department heads are relevant to our current goals. Then give me:
1. Scorecard: each goal's progress and whether it's on track.
2. What went well and what didn't this week (from tasks, decisions and metrics).
3. Risks to the company in the next 90 days.
4. The plan for next week: priorities, owners, due dates — and create those tasks.
Log any decisions you make."""


def daily_briefing(jaxon: Jaxon, session_id: str | None = None):
    return jaxon.chat(DAILY_BRIEFING, session_id=session_id)


def weekly_review(jaxon: Jaxon, session_id: str | None = None):
    return jaxon.chat(WEEKLY_REVIEW, session_id=session_id)
