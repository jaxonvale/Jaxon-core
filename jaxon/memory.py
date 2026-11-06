"""Persistent memory for Jaxon, backed by SQLite.

Holds long-term memories (searchable), the company profile, tasks, decisions,
goals/OKRs, metrics, finances, and conversation sessions.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, tags, content='memories', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags) VALUES ('delete', old.id, old.content, old.tags);
END;

CREATE TABLE IF NOT EXISTS profile (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    details TEXT NOT NULL DEFAULT '',
    owner TEXT NOT NULL DEFAULT 'jaxon',
    priority TEXT NOT NULL DEFAULT 'medium',
    status TEXT NOT NULL DEFAULT 'open',
    due TEXT,
    notes TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    decision TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    alternatives TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    objective TEXT NOT NULL,
    period TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    key_results TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL DEFAULT '',
    recorded_on TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS metrics_name ON metrics(name, recorded_on);

CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    amount REAL NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    occurred_on TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    messages TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

TASK_STATUSES = ("open", "in_progress", "blocked", "done", "cancelled")
PRIORITIES = ("low", "medium", "high", "critical")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def today() -> str:
    return date.today().isoformat()


class Memory:
    """Thread-safe SQLite store. One instance can be shared by the web server."""

    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            path = str(Path(path).expanduser())
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL" if path != ":memory:" else "PRAGMA journal_mode=MEMORY")
        self._db.executescript(SCHEMA)
        self._db.commit()

    # -- helpers ---------------------------------------------------------
    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, params)
            self._db.commit()
            return cur

    def _all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, params).fetchall()]

    def _one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        rows = self._all(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- long-term memories ---------------------------------------------
    def remember(self, content: str, kind: str = "fact", tags: list[str] | None = None) -> int:
        if not content.strip():
            raise ValueError("content must not be empty")
        cur = self._exec(
            "INSERT INTO memories(kind, content, tags, created_at) VALUES (?,?,?,?)",
            (kind, content.strip(), " ".join(tags or []), now()),
        )
        return cur.lastrowid

    def recall(self, query: str = "", limit: int = 10, kind: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 50))
        terms = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if t]
        if terms:
            fts = " OR ".join(f'"{t}"' for t in terms)
            sql = (
                "SELECT m.* FROM memories_fts f JOIN memories m ON m.id = f.rowid "
                "WHERE memories_fts MATCH ?" + (" AND m.kind = ?" if kind else "") +
                " ORDER BY bm25(memories_fts) LIMIT ?"
            )
            params: tuple = (fts, kind, limit) if kind else (fts, limit)
            return self._all(sql, params)
        sql = "SELECT * FROM memories" + (" WHERE kind = ?" if kind else "") + " ORDER BY id DESC LIMIT ?"
        return self._all(sql, (kind, limit) if kind else (limit,))

    def forget(self, memory_id: int) -> bool:
        return self._exec("DELETE FROM memories WHERE id = ?", (memory_id,)).rowcount > 0

    # -- company profile -------------------------------------------------
    def set_profile(self, key: str, value: str) -> None:
        self._exec(
            "INSERT INTO profile(key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key.strip().lower(), value.strip(), now()),
        )

    def profile(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self._all("SELECT key, value FROM profile ORDER BY key")}

    # -- tasks -----------------------------------------------------------
    def create_task(self, title: str, details: str = "", owner: str = "jaxon",
                    priority: str = "medium", due: str | None = None) -> dict[str, Any]:
        if priority not in PRIORITIES:
            raise ValueError(f"priority must be one of {PRIORITIES}")
        ts = now()
        cur = self._exec(
            "INSERT INTO tasks(title, details, owner, priority, due, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (title.strip(), details, owner, priority, due or None, ts, ts),
        )
        return self.get_task(cur.lastrowid)

    def get_task(self, task_id: int) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM tasks WHERE id = ?", (task_id,))
        if row:
            row["notes"] = json.loads(row["notes"])
        return row

    def update_task(self, task_id: int, status: str | None = None, note: str | None = None,
                    priority: str | None = None, due: str | None = None, owner: str | None = None) -> dict[str, Any]:
        task = self.get_task(task_id)
        if not task:
            raise KeyError(f"task {task_id} not found")
        if status and status not in TASK_STATUSES:
            raise ValueError(f"status must be one of {TASK_STATUSES}")
        if priority and priority not in PRIORITIES:
            raise ValueError(f"priority must be one of {PRIORITIES}")
        notes = task["notes"]
        if note:
            notes.append({"at": now(), "note": note})
        self._exec(
            "UPDATE tasks SET status=?, priority=?, due=?, owner=?, notes=?, updated_at=? WHERE id=?",
            (status or task["status"], priority or task["priority"], due or task["due"],
             owner or task["owner"], json.dumps(notes), now(), task_id),
        )
        return self.get_task(task_id)

    def list_tasks(self, status: str | None = None, owner: str | None = None) -> list[dict[str, Any]]:
        clauses, params = [], []
        if status == "active":
            clauses.append("status IN ('open','in_progress','blocked')")
        elif status:
            clauses.append("status = ?")
            params.append(status)
        if owner:
            clauses.append("owner = ?")
            params.append(owner)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        order = (" ORDER BY CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,"
                 " COALESCE(due, '9999'), id")
        rows = self._all("SELECT * FROM tasks" + where + order, tuple(params))
        for r in rows:
            r["notes"] = json.loads(r["notes"])
        return rows

    # -- decisions -------------------------------------------------------
    def log_decision(self, title: str, decision: str, rationale: str = "", alternatives: str = "") -> dict[str, Any]:
        cur = self._exec(
            "INSERT INTO decisions(title, decision, rationale, alternatives, created_at) VALUES (?,?,?,?,?)",
            (title, decision, rationale, alternatives, now()),
        )
        return self._one("SELECT * FROM decisions WHERE id = ?", (cur.lastrowid,))

    def list_decisions(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))

    # -- goals / OKRs ----------------------------------------------------
    def set_goal(self, objective: str, key_results: list[dict[str, Any]], period: str = "") -> dict[str, Any]:
        krs = []
        for kr in key_results:
            krs.append({
                "description": str(kr["description"]),
                "target": float(kr.get("target", 100)),
                "current": float(kr.get("current", 0)),
                "unit": str(kr.get("unit", "")),
            })
        ts = now()
        cur = self._exec(
            "INSERT INTO goals(objective, period, key_results, created_at, updated_at) VALUES (?,?,?,?,?)",
            (objective, period, json.dumps(krs), ts, ts),
        )
        return self.get_goal(cur.lastrowid)

    def get_goal(self, goal_id: int) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM goals WHERE id = ?", (goal_id,))
        if row:
            row["key_results"] = json.loads(row["key_results"])
            row["progress"] = goal_progress(row["key_results"])
        return row

    def update_key_result(self, goal_id: int, kr_index: int, current: float) -> dict[str, Any]:
        goal = self.get_goal(goal_id)
        if not goal:
            raise KeyError(f"goal {goal_id} not found")
        krs = goal["key_results"]
        if not 0 <= kr_index < len(krs):
            raise IndexError(f"goal {goal_id} has {len(krs)} key results; index {kr_index} is out of range")
        krs[kr_index]["current"] = float(current)
        self._exec("UPDATE goals SET key_results=?, updated_at=? WHERE id=?", (json.dumps(krs), now(), goal_id))
        return self.get_goal(goal_id)

    def set_goal_status(self, goal_id: int, status: str) -> dict[str, Any]:
        if status not in ("active", "achieved", "missed", "dropped"):
            raise ValueError("status must be active, achieved, missed or dropped")
        if not self.get_goal(goal_id):
            raise KeyError(f"goal {goal_id} not found")
        self._exec("UPDATE goals SET status=?, updated_at=? WHERE id=?", (status, now(), goal_id))
        return self.get_goal(goal_id)

    def list_goals(self, status: str | None = "active") -> list[dict[str, Any]]:
        rows = self._all("SELECT id FROM goals" + (" WHERE status = ?" if status else "") + " ORDER BY id",
                         (status,) if status else ())
        return [self.get_goal(r["id"]) for r in rows]

    # -- metrics ---------------------------------------------------------
    def record_metric(self, name: str, value: float, unit: str = "", recorded_on: str | None = None,
                      note: str = "") -> dict[str, Any]:
        cur = self._exec(
            "INSERT INTO metrics(name, value, unit, recorded_on, note) VALUES (?,?,?,?,?)",
            (name.strip().lower(), float(value), unit, recorded_on or today(), note),
        )
        return self._one("SELECT * FROM metrics WHERE id = ?", (cur.lastrowid,))

    def metric_history(self, name: str, limit: int = 30) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM metrics WHERE name = ? ORDER BY recorded_on DESC, id DESC LIMIT ?",
                         (name.strip().lower(), limit))
        return list(reversed(rows))

    def latest_metrics(self) -> list[dict[str, Any]]:
        return self._all(
            "SELECT m.* FROM metrics m JOIN (SELECT name, MAX(id) AS id FROM metrics GROUP BY name) l "
            "ON m.id = l.id ORDER BY m.name"
        )

    # -- finances --------------------------------------------------------
    def record_transaction(self, amount: float, category: str, description: str = "",
                           occurred_on: str | None = None) -> dict[str, Any]:
        cur = self._exec(
            "INSERT INTO transactions(amount, category, description, occurred_on) VALUES (?,?,?,?)",
            (float(amount), category.strip().lower(), description, occurred_on or today()),
        )
        return self._one("SELECT * FROM transactions WHERE id = ?", (cur.lastrowid,))

    def financial_summary(self, months: int = 3) -> dict[str, Any]:
        """Cash position, monthly burn and runway.

        Cash = profile 'cash_balance' (the starting balance) + all transactions.
        Burn = average net outflow over the last `months` calendar months that have data.
        """
        months = max(1, min(int(months), 24))
        prof = self.profile()
        try:
            opening = float(prof.get("cash_balance", "0").replace(",", "").replace("$", ""))
        except ValueError:
            opening = 0.0
        total = self._one("SELECT COALESCE(SUM(amount),0) AS s FROM transactions")["s"]
        by_month = self._all(
            "SELECT substr(occurred_on,1,7) AS month, "
            "SUM(CASE WHEN amount>0 THEN amount ELSE 0 END) AS revenue, "
            "SUM(CASE WHEN amount<0 THEN -amount ELSE 0 END) AS expenses, "
            "SUM(amount) AS net FROM transactions GROUP BY month ORDER BY month DESC LIMIT ?",
            (months,),
        )
        by_category = self._all(
            "SELECT category, SUM(amount) AS total, COUNT(*) AS n FROM transactions "
            "GROUP BY category ORDER BY total"
        )
        cash = opening + total
        avg_net = sum(m["net"] for m in by_month) / len(by_month) if by_month else 0.0
        burn = -avg_net if avg_net < 0 else 0.0
        runway = round(cash / burn, 1) if burn > 0 else None
        return {
            "cash": round(cash, 2),
            "opening_balance": opening,
            "monthly_burn": round(burn, 2),
            "avg_monthly_net": round(avg_net, 2),
            "runway_months": runway,
            "months": list(reversed(by_month)),
            "by_category": by_category,
        }

    # -- sessions --------------------------------------------------------
    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        row = self._one("SELECT messages FROM sessions WHERE id = ?", (session_id,))
        return json.loads(row["messages"]) if row else []

    def save_session(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        ts = now()
        self._exec(
            "INSERT INTO sessions(id, messages, created_at, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET messages=excluded.messages, updated_at=excluded.updated_at",
            (session_id, json.dumps(messages), ts, ts),
        )

    def list_sessions(self) -> list[dict[str, Any]]:
        return self._all("SELECT id, created_at, updated_at FROM sessions ORDER BY updated_at DESC")

    def delete_session(self, session_id: str) -> bool:
        return self._exec("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount > 0

    # -- snapshot --------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Compact state of the company, injected into each CEO turn."""
        active = self.list_tasks(status="active")
        return {
            "date": today(),
            "profile": self.profile(),
            "active_tasks": [
                {k: t[k] for k in ("id", "title", "owner", "priority", "status", "due")} for t in active[:15]
            ],
            "active_task_count": len(active),
            "goals": [
                {"id": g["id"], "objective": g["objective"], "period": g["period"], "progress": g["progress"]}
                for g in self.list_goals()
            ],
            "latest_metrics": [
                {"name": m["name"], "value": m["value"], "unit": m["unit"], "on": m["recorded_on"]}
                for m in self.latest_metrics()
            ],
            "finance": {k: v for k, v in self.financial_summary().items() if k in ("cash", "monthly_burn", "runway_months")},
            "recent_decisions": [{"id": d["id"], "title": d["title"]} for d in self.list_decisions(5)],
        }


def goal_progress(krs: list[dict[str, Any]]) -> float:
    """Average key-result completion, 0-100."""
    if not krs:
        return 0.0
    pcts = []
    for kr in krs:
        target = kr.get("target") or 0
        pcts.append(100.0 if target == 0 else max(0.0, min(100.0, 100.0 * kr.get("current", 0) / target)))
    return round(sum(pcts) / len(pcts), 1)
