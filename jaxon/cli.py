"""Command line interface: `python -m jaxon <command>`."""

from __future__ import annotations

import argparse
import json
import sys

from .config import Config


def _print_event(kind: str, data: dict) -> None:
    if kind == "tool":
        print(f"  · {data['name']}", file=sys.stderr)
    elif kind == "search":
        print(f"  · web search: {data['query']}", file=sys.stderr)


def _jaxon(args):
    from .agent import Jaxon

    cfg = Config()
    if args.db:
        cfg.db_path = args.db
    return Jaxon(cfg)


def cmd_chat(args) -> int:
    jaxon = _jaxon(args)
    session = args.session or "main"
    print(f"Jaxon Vale — CEO. Session '{session}'. Type 'exit' to quit.\n")
    while True:
        try:
            msg = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if msg.lower() in ("exit", "quit"):
            return 0
        if not msg:
            continue
        try:
            _, result = jaxon.chat(msg, session_id=session, on_event=_print_event)
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        print(f"\njaxon › {result.text}\n")


def cmd_ask(args) -> int:
    jaxon = _jaxon(args)
    _, result = jaxon.chat(" ".join(args.message), session_id=args.session, on_event=_print_event)
    print(result.text)
    return 0


def cmd_routine(args) -> int:
    from . import routines

    jaxon = _jaxon(args)
    fn = routines.daily_briefing if args.command == "briefing" else routines.weekly_review
    _, result = fn(jaxon, session_id=args.session)
    print(result.text)
    return 0


def cmd_status(args) -> int:
    from .memory import Memory

    print(json.dumps(Memory(args.db or Config().db_path).snapshot(), indent=2, default=str))
    return 0


def cmd_serve(args) -> int:
    from .server import create_app

    app = create_app(db_path=args.db)
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="jaxon", description="Jaxon Vale — AI CEO")
    p.add_argument("--db", help="path to the SQLite database (default ~/.jaxon/jaxon.db or $JAXON_DB)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("chat", help="interactive conversation")
    s.add_argument("--session", default="main")
    s.set_defaults(fn=cmd_chat)

    s = sub.add_parser("ask", help="one-shot question")
    s.add_argument("message", nargs="+")
    s.add_argument("--session", default="main")
    s.set_defaults(fn=cmd_ask)

    for name, help_ in (("briefing", "daily briefing"), ("review", "weekly review")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--session", default="main")
        s.set_defaults(fn=cmd_routine)

    s = sub.add_parser("status", help="print the company snapshot (no API call)")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5000)
    s.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    return args.fn(args)
