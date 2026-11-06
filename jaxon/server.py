"""Web app: a JSON API over Jaxon plus a single-page UI (jaxon/web/index.html).

Run with `python -m jaxon serve`. Set JAXON_TOKEN to require `Authorization: Bearer <token>`
on every /api/* route except /api/health.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from flask import Flask, abort, jsonify, request, send_from_directory

from .config import Config
from .memory import Memory

log = logging.getLogger("jaxon.server")

WEB_DIR = Path(__file__).parent / "web"
MAX_MESSAGE_CHARS = 100_000


class _Error(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _body() -> dict[str, Any]:
    data = request.get_json(silent=True)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise _Error(400, "request body must be a JSON object")
    return data


def _opt_str(data: dict[str, Any], key: str) -> str | None:
    v = data.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise _Error(400, f"'{key}' must be a string")
    return v.strip() or None


def _int_arg(name: str, default: int, lo: int = 1, hi: int = 200) -> int:
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    try:
        return max(lo, min(hi, int(raw)))
    except ValueError:
        raise _Error(400, f"'{name}' must be an integer")


def _turn_json(session_id: str, result: Any) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "reply": result.text,
        "tool_calls": [{"name": c.get("name"), "error": bool(c.get("error"))} for c in result.tool_calls],
        "usage": result.usage,
    }


def create_app(db_path: str | None = None, jaxon: Any = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.json.sort_keys = False

    state: dict[str, Any] = {"jaxon": jaxon, "memory": jaxon.memory if jaxon is not None else None}
    init_lock = threading.Lock()
    session_locks: dict[str, threading.Lock] = {}
    session_locks_guard = threading.Lock()

    def config() -> Config:
        cfg = Config()
        if db_path:
            cfg.db_path = db_path
        return cfg

    def get_memory() -> Memory:
        # Dashboard routes only need the store, so they work without an API key.
        if state["memory"] is None:
            with init_lock:
                if state["memory"] is None:
                    state["memory"] = Memory(config().db_path)
        return state["memory"]

    def get_jaxon() -> Any:
        if state["jaxon"] is None:
            memory = get_memory()
            with init_lock:
                if state["jaxon"] is None:
                    from .agent import Jaxon

                    state["jaxon"] = Jaxon(config(), memory=memory)
        return state["jaxon"]

    def session_lock(session_id: str) -> threading.Lock:
        with session_locks_guard:
            lock = session_locks.get(session_id)
            if lock is None:
                lock = session_locks[session_id] = threading.Lock()
            return lock

    def run_turn(fn: Callable[[Any, str], tuple[str, Any]], session_id: str | None):
        session_id = session_id or uuid.uuid4().hex[:12]
        if len(session_id) > 128:
            raise _Error(400, "'session_id' is too long")
        try:
            j = get_jaxon()
            with session_lock(session_id):
                sid, result = fn(j, session_id)
        except _Error:
            raise
        except Exception as e:  # anthropic.APIError and anything else from the agent loop
            try:
                import anthropic

                kind = "Anthropic API error" if isinstance(e, anthropic.APIError) else "Agent error"
            except ImportError:  # pragma: no cover
                kind = "Agent error"
            log.exception("chat turn failed (session %s)", session_id)
            return jsonify({"error": f"{kind}: {type(e).__name__}: {e}", "session_id": session_id}), 502
        return jsonify(_turn_json(sid, result))

    # -- auth & errors ---------------------------------------------------
    @app.before_request
    def _auth():
        token = os.environ.get("JAXON_TOKEN", "")
        if not token or not request.path.startswith("/api/") or request.path == "/api/health":
            return None
        header = request.headers.get("Authorization", "")
        supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if not supplied or not hmac.compare_digest(supplied.encode(), token.encode()):
            return jsonify({"error": "unauthorized"}), 401
        return None

    @app.errorhandler(_Error)
    def _handle_error(e: _Error):
        return jsonify({"error": e.message}), e.status

    @app.errorhandler(404)
    def _not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "not found"}), 404
        return e

    @app.errorhandler(405)
    def _bad_method(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "method not allowed"}), 405
        return e

    # -- UI --------------------------------------------------------------
    @app.get("/")
    def index():
        if not (WEB_DIR / "index.html").is_file():
            abort(404)
        resp = send_from_directory(WEB_DIR, "index.html")
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    # -- chat ------------------------------------------------------------
    @app.get("/api/health")
    def health():
        return jsonify({"ok": True})

    @app.post("/api/chat")
    def chat():
        data = _body()
        message = data.get("message")
        if not isinstance(message, str) or not message.strip():
            raise _Error(400, "'message' is required")
        if len(message) > MAX_MESSAGE_CHARS:
            raise _Error(400, "'message' is too long")
        msg = message.strip()
        return run_turn(lambda j, sid: j.chat(msg, session_id=sid), _opt_str(data, "session_id"))

    @app.post("/api/briefing")
    def briefing():
        from . import routines

        return run_turn(lambda j, sid: routines.daily_briefing(j, session_id=sid),
                        _opt_str(_body(), "session_id"))

    @app.post("/api/review")
    def review():
        from . import routines

        return run_turn(lambda j, sid: routines.weekly_review(j, session_id=sid),
                        _opt_str(_body(), "session_id"))

    # -- sessions --------------------------------------------------------
    @app.get("/api/sessions")
    def sessions():
        return jsonify({"sessions": get_memory().list_sessions()})

    @app.get("/api/sessions/<session_id>")
    def session_history(session_id: str):
        memory = get_memory()
        if not memory.load_session(session_id):
            raise _Error(404, f"session '{session_id}' not found")
        j = state["jaxon"]
        if j is not None:
            messages = j.history(session_id)
        else:
            from .agent import Jaxon

            messages = Jaxon.history(_HistoryShim(memory), session_id)
        return jsonify({"session_id": session_id, "messages": messages})

    @app.delete("/api/sessions/<session_id>")
    def delete_session(session_id: str):
        with session_lock(session_id):
            deleted = get_memory().delete_session(session_id)
        if not deleted:
            raise _Error(404, f"session '{session_id}' not found")
        return jsonify({"deleted": True, "session_id": session_id})

    # -- company OS ------------------------------------------------------
    @app.get("/api/snapshot")
    def snapshot():
        return jsonify(get_memory().snapshot())

    @app.get("/api/tasks")
    def tasks():
        status = request.args.get("status", "active").strip() or "active"
        owner = request.args.get("owner") or None
        return jsonify({"tasks": get_memory().list_tasks(status=None if status == "all" else status, owner=owner)})

    @app.post("/api/tasks")
    def create_task():
        data = _body()
        title = _opt_str(data, "title")
        if not title:
            raise _Error(400, "'title' is required")
        kwargs = {k: v for k in ("details", "owner", "priority", "due") if (v := _opt_str(data, k)) is not None}
        try:
            task = get_memory().create_task(title, **kwargs)
        except ValueError as e:
            raise _Error(400, str(e))
        return jsonify(task), 201

    @app.patch("/api/tasks/<int:task_id>")
    def update_task(task_id: int):
        data = _body()
        kwargs = {k: _opt_str(data, k) for k in ("status", "note", "priority", "due", "owner")}
        try:
            task = get_memory().update_task(task_id, **kwargs)
        except KeyError:
            raise _Error(404, f"task {task_id} not found")
        except ValueError as e:
            raise _Error(400, str(e))
        return jsonify(task)

    @app.get("/api/goals")
    def goals():
        status = request.args.get("status", "active")
        return jsonify({"goals": get_memory().list_goals(None if status == "all" else status)})

    @app.get("/api/decisions")
    def decisions():
        return jsonify({"decisions": get_memory().list_decisions(_int_arg("limit", 20))})

    @app.get("/api/metrics")
    def metrics():
        return jsonify({"metrics": get_memory().latest_metrics()})

    @app.get("/api/metrics/<name>")
    def metric(name: str):
        return jsonify({"name": name.strip().lower(), "history": get_memory().metric_history(name, _int_arg("limit", 30))})

    @app.get("/api/finance")
    def finance():
        return jsonify(get_memory().financial_summary(_int_arg("months", 3, hi=24)))

    @app.get("/api/memories")
    def memories():
        q = request.args.get("q", "")
        kind = request.args.get("kind") or None
        return jsonify({"memories": get_memory().recall(q, limit=_int_arg("limit", 20, hi=50), kind=kind)})

    @app.get("/api/profile")
    def profile():
        return jsonify(get_memory().profile())

    return app


class _HistoryShim:
    """Lets Jaxon.history run against a bare Memory, without building an API client."""

    def __init__(self, memory: Memory):
        self.memory = memory
