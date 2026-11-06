"""The agent loop, the request shape, and session persistence."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from conftest import FakeClient, last_user_message, resp, text, tool_use

from jaxon import routines
from jaxon.agent import CEO_SYSTEM, Jaxon, TurnResult, run_loop
from jaxon.config import Config
from jaxon.llm import FALLBACK_BETA, LLM, WEB_SEARCH_TOOL
from jaxon.memory import Memory
from jaxon.tools import build_tools


def assert_valid_history(messages):
    """The invariants the Messages API needs for a history we will extend with a new user turn."""
    if not messages:
        return
    assert messages[0]["role"] == "user"
    last = messages[-1]
    ends_on_results = last["role"] == "user" and all(b.get("type") == "tool_result" for b in last["content"])
    assert last["role"] == "assistant" or ends_on_results, \
        "a saved session must end on a complete assistant turn or answered tool results"
    for i, msg in enumerate(messages):
        assert msg["role"] in ("user", "assistant")
        assert msg["content"], f"message {i} has empty content"
        if msg["role"] == "assistant":
            ids = [b["id"] for b in msg["content"] if b.get("type") == "tool_use"]
            if ids:
                assert i + 1 < len(messages), "dangling tool_use at end of history"
                nxt = messages[i + 1]
                assert nxt["role"] == "user"
                answered = [b["tool_use_id"] for b in nxt["content"] if b.get("type") == "tool_result"]
                assert answered == ids
        else:
            for b in msg["content"] if isinstance(msg["content"], list) else []:
                if b.get("type") == "tool_result":
                    prev = messages[i - 1]
                    assert prev["role"] == "assistant"
                    assert b["tool_use_id"] in {x.get("id") for x in prev["content"]}


# -- plain replies & request shape ----------------------------------------
def test_plain_text_reply(jaxon, client):
    client.script(resp(text("Hello, founder."), input_tokens=100, output_tokens=7, cache_read=40))
    sid, result = jaxon.chat("hi")
    assert isinstance(result, TurnResult)
    assert result.text == "Hello, founder."
    assert result.stop_reason == "end_turn"
    assert result.tool_calls == []
    assert result.usage == {"input_tokens": 100, "output_tokens": 7, "cache_read_input_tokens": 40}
    assert len(client.calls) == 1
    assert len(sid) == 12 and int(sid, 16) >= 0


def test_multiple_text_blocks_are_joined(jaxon, client):
    client.script(resp({"type": "thinking", "thinking": "", "signature": "sig"}, text("one"), text("two")))
    _, result = jaxon.chat("hi")
    assert result.text == "one\ntwo"


def test_request_shape_defaults(jaxon, client):
    client.script(resp(text("ok")))
    jaxon.chat("What should I focus on?")
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == 64000
    assert call["system"] == CEO_SYSTEM
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "high"}
    assert call["cache_control"] == {"type": "ephemeral"}
    assert FALLBACK_BETA in call["betas"]
    assert FALLBACK_BETA == "server-side-fallback-2026-07-01"
    assert call["fallbacks"] == "default"
    names = [t["name"] for t in call["tools"]]
    assert WEB_SEARCH_TOOL in call["tools"]
    assert {"remember", "recall", "create_task", "delegate", "web_search"} <= set(names)
    assert len(names) == len(set(names))
    for t in call["tools"]:
        if t["name"] != "web_search":
            assert set(t) == {"name", "description", "input_schema"}
    assert "temperature" not in call and "top_p" not in call


def test_request_without_fallbacks(make_jaxon, client):
    j = make_jaxon(fallbacks=False)
    client.script(resp(text("ok")))
    j.chat("hi")
    call = client.calls[0]
    assert "betas" not in call
    assert "fallbacks" not in call


def test_request_without_web_search(make_jaxon, client):
    j = make_jaxon(web_search=False)
    client.script(resp(text("ok")))
    j.chat("hi")
    names = [t["name"] for t in client.calls[0]["tools"]]
    assert "web_search" not in names
    assert all(t.get("type") is None for t in client.calls[0]["tools"])
    assert "delegate" in names


def test_effort_and_model_from_config(make_jaxon, client):
    j = make_jaxon(effort="max", model="claude-test-model", max_tokens=1234)
    client.script(resp(text("ok")))
    j.chat("hi")
    call = client.calls[0]
    assert call["output_config"] == {"effort": "max"}
    assert call["model"] == "claude-test-model"
    assert call["max_tokens"] == 1234


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("JAXON_MODEL", "m")
    monkeypatch.setenv("JAXON_EFFORT", "low")
    monkeypatch.setenv("JAXON_DEPT_EFFORT", "xhigh")
    monkeypatch.setenv("JAXON_WEB_SEARCH", "0")
    monkeypatch.setenv("JAXON_FALLBACKS", "0")
    monkeypatch.setenv("JAXON_DB", "/tmp/x.db")
    c = Config()
    assert (c.model, c.effort, c.department_effort, c.web_search, c.fallbacks, c.db_path) == (
        "m", "low", "xhigh", False, False, "/tmp/x.db")


def test_config_defaults():
    c = Config()
    assert (c.model, c.effort, c.department_effort, c.web_search, c.fallbacks) == (
        "claude-opus-5-5", "high", "medium", True, True)
    assert c.max_tool_rounds == 25


def test_llm_call_effort_override(client):
    llm = LLM(Config(), client)
    client.script(resp(text("x")))
    llm.call(system="s", messages=[{"role": "user", "content": "hi"}], tools=[], effort="low")
    assert client.calls[0]["output_config"] == {"effort": "low"}


def test_snapshot_block_prepended_to_user_turn(jaxon, client, memory):
    memory.set_profile("name", "Acme")
    memory.create_task("Close the round", priority="critical")
    client.script(resp(text("ok")))
    jaxon.chat("How are we doing?")
    msg = last_user_message(client.calls[0])
    assert len(msg["content"]) == 2
    snap_block, user_block = msg["content"]
    assert snap_block["type"] == "text"
    assert snap_block["text"].startswith("[Company snapshot]\n")
    snap = json.loads(snap_block["text"].split("\n", 1)[1])
    assert snap["profile"] == {"name": "Acme"}
    assert snap["active_tasks"][0]["title"] == "Close the round"
    assert user_block == {"type": "text", "text": "How are we doing?"}


def test_snapshot_is_fresh_each_turn(jaxon, client, memory):
    client.script(resp(text("a")), resp(text("b")))
    jaxon.chat("one", session_id="s")
    memory.set_profile("stage", "series-a")
    jaxon.chat("two", session_id="s")
    first = json.loads(client.calls[1]["messages"][0]["content"][0]["text"].split("\n", 1)[1])
    latest = json.loads(last_user_message(client.calls[1])["content"][0]["text"].split("\n", 1)[1])
    assert first["profile"] == {}  # history is not rewritten
    assert latest["profile"] == {"stage": "series-a"}


# -- tool use -------------------------------------------------------------
def test_tool_use_round_trip(jaxon, client, memory):
    client.script(
        resp(text("Saving that."), tool_use("tu_1", "remember", {"content": "We sell to dentists", "kind": "fact"}),
             stop="tool_use"),
        resp(text("Noted.")),
    )
    events = []
    _, result = jaxon.chat("We sell to dentists", on_event=lambda k, d: events.append((k, d)))
    assert result.text == "Noted."
    assert result.tool_calls == [{"name": "remember", "input": {"content": "We sell to dentists", "kind": "fact"},
                                  "error": False}]
    assert memory.recall("dentists")[0]["content"] == "We sell to dentists"
    assert events == [("tool", {"name": "remember", "input": {"content": "We sell to dentists", "kind": "fact"}})]

    second = client.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[1]["content"][1]["id"] == "tu_1"
    results = second[2]["content"]
    assert len(results) == 1
    r = results[0]
    assert r["type"] == "tool_result" and r["tool_use_id"] == "tu_1" and r["is_error"] is False
    assert json.loads(r["content"])["saved"] is True
    # The tool result is the whole user message: no extra text blocks.
    assert all(b["type"] == "tool_result" for b in results)


def test_parallel_tool_uses_answered_in_one_message(jaxon, client):
    client.script(
        resp(tool_use("a", "create_task", {"title": "Hire CTO"}),
             tool_use("b", "no_such_tool", {}),
             tool_use("c", "update_task", {"id": 999, "status": "done"}),
             tool_use("d", "get_current_time", {}),
             stop="tool_use"),
        resp(text("Done.")),
    )
    _, result = jaxon.chat("go")
    msgs = client.calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    results = msgs[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b", "c", "d"]
    assert [r["is_error"] for r in results] == [False, True, True, False]
    assert "Unknown tool" in results[1]["content"]
    assert [c["error"] for c in result.tool_calls] == [False, True, True, False]


def test_multiple_tool_rounds_and_usage_sum(jaxon, client):
    client.script(
        resp(tool_use("t1", "list_tasks", {}), stop="tool_use", input_tokens=10, output_tokens=1),
        resp(tool_use("t2", "list_goals", {}), stop="tool_use", input_tokens=20, output_tokens=2, cache_read=5),
        resp(text("Summary"), input_tokens=30, output_tokens=3),
    )
    _, result = jaxon.chat("status?")
    assert result.text == "Summary"
    assert result.usage == {"input_tokens": 60, "output_tokens": 6, "cache_read_input_tokens": 5}
    assert len(client.calls) == 3
    roles = [m["role"] for m in client.calls[2]["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]


def test_thinking_blocks_round_tripped(jaxon, client):
    thinking = {"type": "thinking", "thinking": "", "signature": "abc"}
    client.script(resp(thinking, tool_use("t1", "list_tasks"), stop="tool_use"), resp(text("ok")))
    jaxon.chat("x")
    assert client.calls[1]["messages"][1]["content"][0] == thinking


def test_sdk_objects_converted_to_dicts(memory):
    from anthropic.types import TextBlock

    class Dumpable:
        def model_dump_json(self):
            return json.dumps({"type": "text", "text": "dumped"})

    client = FakeClient()
    client.script(SimpleNamespace(content=[TextBlock(type="text", text="sdk"), Dumpable()],
                                  stop_reason="end_turn", usage=None))
    j = Jaxon(Config(db_path=":memory:"), memory=memory, client=client)
    sid, result = j.chat("hi")
    assert result.text == "sdk\ndumped"
    assert result.usage == {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    saved = memory.load_session(sid)
    assert saved[-1]["content"][0]["text"] == "sdk"
    assert saved[-1]["content"][1] == {"type": "text", "text": "dumped"}


# -- stop reasons ---------------------------------------------------------
def test_pause_turn_continues_without_new_user_message(jaxon, client):
    search = {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "competitor pricing"}}
    client.script(
        resp(search, stop="pause_turn"),
        resp(text("Competitors charge $99."), stop="end_turn"),
    )
    events = []
    _, result = jaxon.chat("What do competitors charge?", on_event=lambda k, d: events.append((k, d)))
    assert result.text == "Competitors charge $99."
    assert ("search", {"query": "competitor pricing"}) in events
    second = client.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant"]
    assert second[-1]["content"] == [search]


def test_refusal(jaxon, client):
    client.script(resp(stop="refusal"))
    _, result = jaxon.chat("something bad")
    assert result.stop_reason == "refusal"
    assert result.text == "I can't help with that request."


def test_refused_turn_not_kept_in_session(jaxon, client, memory):
    client.script(resp(text("Hi.")), resp(stop="refusal"), resp(text("Sure.")))
    jaxon.chat("hello", session_id="s")
    after_first = memory.load_session("s")
    jaxon.chat("bad request", session_id="s")
    assert memory.load_session("s") == after_first
    assert_valid_history(memory.load_session("s"))
    jaxon.chat("good request", session_id="s")
    msgs = client.calls[2]["messages"]
    assert msgs[:2] == after_first
    assert_valid_history(msgs[:-1])
    assert msgs[-1]["content"][1]["text"] == "good request"


def test_refusal_mid_stream_partial_output_dropped(jaxon, client, memory):
    client.script(resp(text("Here is how to"), stop="refusal"))
    sid, result = jaxon.chat("x")
    assert result.text == "I can't help with that request."
    assert memory.load_session(sid) == []


def test_max_tokens(jaxon, client):
    client.script(resp(text("Partial answer"), stop="max_tokens"))
    _, result = jaxon.chat("long question")
    assert result.stop_reason == "max_tokens"
    assert result.text.startswith("Partial answer")
    assert "cut off" in result.text


def test_max_rounds_stop(make_jaxon, client, memory):
    j = make_jaxon(max_tool_rounds=3)
    client.script(*[resp(tool_use(f"t{i}", "list_tasks"), stop="tool_use") for i in range(5)])
    sid, result = j.chat("loop forever")
    assert result.stop_reason == "max_rounds"
    assert result.text == "[Stopped: too many tool rounds.]"
    assert len(client.calls) == 3
    assert len(result.tool_calls) == 3
    assert_valid_history(memory.load_session(sid))


def test_other_stop_reason_returns_text(jaxon, client):
    client.script(resp(text("stopped"), stop="stop_sequence"))
    assert jaxon.chat("x")[1].text == "stopped"


def test_run_loop_directly(memory, client):
    llm = LLM(Config(), client)
    tools = build_tools(memory)
    messages = [{"role": "user", "content": "hi"}]
    client.script(resp(tool_use("x", "remember", {"content": "fact"}), stop="tool_use"), resp(text("ok")))
    result = run_loop(llm, system="SYS", messages=messages, tools=tools, effort="low", max_rounds=5)
    assert result.text == "ok"
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert all(c["system"] == "SYS" and c["output_config"] == {"effort": "low"} for c in client.calls)
    assert client.calls[0]["tools"] == [t.definition() for t in tools.values()]


# -- sessions -------------------------------------------------------------
def test_session_append_only_across_turns(jaxon, client, memory):
    client.script(
        resp(text("Let me check."), tool_use("t1", "list_tasks"), stop="tool_use"),
        resp(text("Nothing open.")),
        resp(text("Second answer.")),
    )
    sid, _ = jaxon.chat("first", session_id="sess")
    assert sid == "sess"
    saved_first = memory.load_session("sess")
    assert len(saved_first) == 4
    assert_valid_history(saved_first)

    jaxon.chat("second", session_id="sess")
    second_req = client.calls[2]["messages"]
    # The first turn is replayed byte-for-byte: same user turn (incl. its snapshot), same assistant turns.
    assert second_req[:4] == saved_first
    assert second_req[0] == client.calls[0]["messages"][0]
    assert second_req[1]["content"] == [text("Let me check."), tool_use("t1", "list_tasks")]
    assert second_req[4]["role"] == "user" and second_req[4]["content"][1]["text"] == "second"
    assert len(second_req) == 5
    full = memory.load_session("sess")
    assert full[:4] == saved_first and len(full) == 6
    assert_valid_history(full)


def test_separate_sessions_are_isolated(jaxon, client):
    client.script(resp(text("a")), resp(text("b")))
    s1, _ = jaxon.chat("one")
    s2, _ = jaxon.chat("two")
    assert s1 != s2
    assert len(client.calls[1]["messages"]) == 1


def test_api_error_on_first_call_leaves_valid_session(jaxon, client, memory):
    client.script(resp(text("hello")), RuntimeError("overloaded"))
    jaxon.chat("hi", session_id="s")
    before = memory.load_session("s")
    with pytest.raises(RuntimeError, match="overloaded"):
        jaxon.chat("are you there?", session_id="s")
    assert memory.load_session("s") == before
    assert_valid_history(memory.load_session("s"))


def test_api_error_on_brand_new_session(jaxon, client, memory):
    client.script(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        jaxon.chat("hi", session_id="fresh")
    assert memory.load_session("fresh") == []


def test_api_error_mid_turn_trims_dangling_tool_use(jaxon, client, memory):
    client.script(
        resp(text("Hi.")),
        resp(tool_use("t1", "create_task", {"title": "Do the thing"}), stop="tool_use"),
        ConnectionError("network down"),
        resp(text("Back.")),
    )
    jaxon.chat("hi", session_id="s")
    good = memory.load_session("s")
    with pytest.raises(ConnectionError):
        jaxon.chat("make a task", session_id="s")
    saved = memory.load_session("s")
    assert saved == good
    assert_valid_history(saved)
    # Side effects that already happened are kept in the DB.
    assert [t["title"] for t in memory.list_tasks()] == ["Do the thing"]
    # The session is usable afterwards.
    _, result = jaxon.chat("still there?", session_id="s")
    assert result.text == "Back."
    req = client.calls[-1]["messages"]
    assert req[:2] == good and len(req) == 3
    assert_valid_history(memory.load_session("s"))


def test_api_error_after_several_rounds(jaxon, client, memory):
    client.script(
        resp(tool_use("a", "list_tasks"), stop="tool_use"),
        resp(tool_use("b", "list_goals"), stop="tool_use"),
        RuntimeError("500"),
    )
    with pytest.raises(RuntimeError):
        jaxon.chat("x", session_id="s")
    assert memory.load_session("s") == []


def test_on_event_exception_leaves_valid_session(jaxon, client, memory):
    client.script(resp(text("Hi.")), resp(tool_use("t1", "list_tasks"), stop="tool_use"))
    jaxon.chat("hi", session_id="s")
    good = memory.load_session("s")

    def bad_event(kind, data):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        jaxon.chat("go", session_id="s", on_event=bad_event)
    assert memory.load_session("s") == good


def test_empty_final_assistant_message_not_left_in_session(jaxon, client, memory):
    # After tool results the model sometimes ends its turn with no content. An empty assistant
    # message mid-history is rejected by the API, so it must not be persisted.
    client.script(resp(tool_use("t1", "list_tasks"), stop="tool_use"), resp(stop="end_turn"), resp(text("ok")))
    sid, result = jaxon.chat("x")
    assert result.text == ""
    assert_valid_history(memory.load_session(sid))
    jaxon.chat("y", session_id=sid)
    assert_valid_history(client.calls[-1]["messages"][:-1])


# -- history --------------------------------------------------------------
def test_history_hides_snapshot_and_tool_results(jaxon, client):
    client.script(
        resp(tool_use("t1", "remember", {"content": "x"}), stop="tool_use"),
        resp(text("Saved it.")),
        resp(text("Second.")),
    )
    jaxon.chat("remember x", session_id="h")
    jaxon.chat("and then?", session_id="h")
    assert jaxon.history("h") == [
        {"role": "user", "text": "remember x"},
        {"role": "assistant", "text": "Saved it."},
        {"role": "user", "text": "and then?"},
        {"role": "assistant", "text": "Second."},
    ]
    assert jaxon.history("missing") == []
    for entry in jaxon.history("h"):
        assert "[Company snapshot]" not in entry["text"]


# -- construction ---------------------------------------------------------
def test_jaxon_builds_memory_from_config(tmp_path, client):
    j = Jaxon(Config(db_path=str(tmp_path / "a" / "j.db")), client=client)
    assert (tmp_path / "a" / "j.db").exists()
    assert "delegate" in j.tools and j.tools["delegate"].read_only is False


# -- routines -------------------------------------------------------------
@pytest.mark.parametrize("fn, prompt", [(routines.daily_briefing, routines.DAILY_BRIEFING),
                                        (routines.weekly_review, routines.WEEKLY_REVIEW)])
def test_routines_send_their_prompt(jaxon, client, fn, prompt):
    client.script(resp(text("brief")))
    sid, result = fn(jaxon, session_id="r")
    assert sid == "r" and result.text == "brief"
    assert last_user_message(client.calls[0])["content"][1]["text"] == prompt


def test_max_rounds_turn_kept_and_resumable(make_jaxon, client, memory):
    j = make_jaxon(max_tool_rounds=2)
    client.script(*[resp(tool_use(f"t{i}", "list_tasks"), stop="tool_use") for i in range(2)])
    sid, result = j.chat("loop")
    assert result.stop_reason == "max_rounds"
    saved = memory.load_session(sid)
    assert len(saved) == 5  # user, (assistant tool_use, tool_results) x2
    assert_valid_history(saved)
    client.script(resp(text("done"), stop="end_turn"))
    _, result2 = j.chat("continue", session_id=sid)
    assert result2.text == "done"
    assert client.calls[-1]["messages"][:5] == saved
