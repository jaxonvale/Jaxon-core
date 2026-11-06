"""Shared fixtures: an in-memory store and a scripted, thread-safe fake Anthropic client."""

from __future__ import annotations

import copy
import threading
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from jaxon.agent import Jaxon
from jaxon.config import Config
from jaxon.memory import Memory


# -- response builders ----------------------------------------------------
def text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


def tool_use(id_: str, name: str, input_: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "tool_use", "id": id_, "name": name, "input": input_ or {}}


def resp(*blocks: dict[str, Any], stop: str = "end_turn", input_tokens: int = 10,
         output_tokens: int = 5, cache_read: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                              cache_read_input_tokens=cache_read),
    )


# -- fake client ----------------------------------------------------------
class _Stream:
    def __init__(self, item: Any, kwargs: dict[str, Any]):
        self._item = item
        self._kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        item = self._item
        if callable(item) and not isinstance(item, SimpleNamespace):
            item = item(self._kwargs)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    """Mimics `client.beta.messages.stream(**kwargs)`.

    Responses are popped from a queue chosen by the request's system prompt: the first
    registered route whose prefix the system prompt starts with wins; otherwise the default
    queue is used. A queued item may be a response, an exception (raised from
    get_final_message) or a callable taking the request kwargs and returning either.
    Every call's kwargs are deep-copied into `calls` (messages are mutated in place by the loop).
    """

    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self._default: list[Any] = []
        self._routes: list[tuple[str, list[Any]]] = []
        self._lock = threading.Lock()
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def script(self, *items: Any, system_prefix: str | None = None) -> "FakeClient":
        with self._lock:
            if system_prefix is None:
                self._default.extend(items)
            else:
                for prefix, q in self._routes:
                    if prefix == system_prefix:
                        q.extend(items)
                        break
                else:
                    self._routes.append((system_prefix, list(items)))
        return self

    def calls_for(self, system_prefix: str) -> list[dict[str, Any]]:
        with self._lock:
            return [c for c in self.calls if str(c.get("system", "")).startswith(system_prefix)]

    def _stream(self, **kwargs: Any) -> _Stream:
        with self._lock:
            self.calls.append(copy.deepcopy(kwargs))
            system = str(kwargs.get("system", ""))
            queue = self._default
            for prefix, q in self._routes:
                if system.startswith(prefix):
                    queue = q
                    break
            if not queue:
                raise AssertionError(f"FakeClient: no scripted response left for system {system[:60]!r}")
            item = queue.pop(0)
        return _Stream(item, kwargs)

    def remaining(self) -> int:
        with self._lock:
            return len(self._default) + sum(len(q) for _, q in self._routes)


# -- fixtures -------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for var in ("JAXON_MODEL", "JAXON_EFFORT", "JAXON_DEPT_EFFORT", "JAXON_WEB_SEARCH", "JAXON_FALLBACKS"):
        monkeypatch.delenv(var, raising=False)
    # Never touch the real ~/.jaxon database.
    monkeypatch.setenv("JAXON_DB", str(tmp_path / "default.db"))


@pytest.fixture
def memory():
    m = Memory(":memory:")
    yield m
    m.close()


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(db_path=str(tmp_path / "jaxon.db"))


@pytest.fixture
def make_jaxon(memory, client):
    def make(**overrides: Any) -> Jaxon:
        cfg = Config(db_path=":memory:", **overrides)
        return Jaxon(cfg, memory=memory, client=client)
    return make


@pytest.fixture
def jaxon(make_jaxon) -> Jaxon:
    return make_jaxon()


def last_user_message(call: dict[str, Any]) -> dict[str, Any]:
    msgs = call["messages"]
    assert msgs[-1]["role"] == "user"
    return msgs[-1]


Responder = Callable[[dict[str, Any]], Any]
