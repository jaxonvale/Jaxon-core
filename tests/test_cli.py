"""CLI commands, with the Anthropic client replaced by the fake."""

from __future__ import annotations

import json

import anthropic
import pytest
from conftest import FakeClient, resp, text, tool_use

from jaxon import cli, routines
from jaxon.memory import Memory


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "cli.db")


@pytest.fixture
def fake(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: client)
    return client


def test_status_prints_snapshot(db, capsys, fake):
    m = Memory(db)
    m.set_profile("name", "Acme")
    m.set_profile("cash_balance", "$12,000")
    m.record_transaction(-3000, "payroll", occurred_on="2026-01-01")
    m.create_task("Raise seed", priority="high")
    m.close()

    assert cli.main(["--db", db, "status"]) == 0
    snap = json.loads(capsys.readouterr().out)
    assert snap["profile"]["name"] == "Acme"
    assert snap["finance"] == {"cash": 9000.0, "monthly_burn": 3000.0, "runway_months": 3.0}
    assert snap["active_tasks"][0]["title"] == "Raise seed"
    assert fake.calls == []  # no API call


def test_status_uses_env_db_by_default(monkeypatch, tmp_path, capsys):
    path = tmp_path / "env.db"
    monkeypatch.setenv("JAXON_DB", str(path))
    Memory(str(path)).set_profile("name", "FromEnv")
    assert cli.main(["status"]) == 0
    assert json.loads(capsys.readouterr().out)["profile"] == {"name": "FromEnv"}


def test_ask_one_shot(db, capsys, fake):
    fake.script(resp(tool_use("t1", "create_task", {"title": "Call investor"}), stop="tool_use"),
                resp(text("Created the task.")))
    assert cli.main(["--db", db, "ask", "Remind", "me", "to", "call", "the", "investor"]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == "Created the task."
    assert "· create_task" in out.err  # tool events printed to stderr
    assert fake.calls[0]["messages"][-1]["content"][1]["text"] == "Remind me to call the investor"

    m = Memory(db)
    assert [t["title"] for t in m.list_tasks()] == ["Call investor"]
    assert len(m.load_session("main")) == 4  # default session "main"


def test_ask_custom_session_and_continuity(db, capsys, fake):
    fake.script(resp(text("one")), resp(text("two")))
    cli.main(["--db", db, "ask", "first", "--session", "s1"])
    cli.main(["--db", db, "ask", "second", "--session", "s1"])
    assert capsys.readouterr().out.split() == ["one", "two"]
    assert len(fake.calls[1]["messages"]) == 3
    assert Memory(db).load_session("main") == []


def test_ask_prints_web_search_events(db, capsys, fake):
    fake.script(resp({"type": "server_tool_use", "id": "s", "name": "web_search", "input": {"query": "tam"}},
                     text("Big."), stop="end_turn"))
    cli.main(["--db", db, "ask", "market size?"])
    assert "web search: tam" in capsys.readouterr().err


@pytest.mark.parametrize("command, prompt", [("briefing", routines.DAILY_BRIEFING),
                                             ("review", routines.WEEKLY_REVIEW)])
def test_routine_commands(db, capsys, fake, command, prompt):
    fake.script(resp(text("Here is your report.")))
    assert cli.main(["--db", db, command]) == 0
    assert capsys.readouterr().out.strip() == "Here is your report."
    assert fake.calls[0]["messages"][-1]["content"][1]["text"] == prompt


def test_chat_repl(db, capsys, fake, monkeypatch):
    inputs = iter(["hello", "", "break please", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    fake.script(resp(text("Hi there.")), RuntimeError("API down"))
    assert cli.main(["--db", db, "chat", "--session", "repl"]) == 0
    out = capsys.readouterr()
    assert "jaxon › Hi there." in out.out
    assert "error: RuntimeError: API down" in out.err  # errors don't kill the REPL
    assert len(fake.calls) == 2  # the empty line was skipped
    assert len(Memory(db).load_session("repl")) == 2


def test_chat_repl_eof(db, fake, monkeypatch):
    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    assert cli.main(["--db", db, "chat"]) == 0


def test_requires_a_command(capsys):
    with pytest.raises(SystemExit):
        cli.main([])
