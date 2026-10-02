"""ToolFilter sessions: one conversation's tool lists (keep used tools, sticky, append-only order), isolation
between conversations, deferral, threads, and a served head (RemoteHead)."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

import pytest

from tabagent.integrations.openai_tools import ToolFilter, called_tools

CATALOG = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
TOOLS = [
    {"type": "function", "function": {"name": n, "description": f"the {n} tool", "parameters": {}}}
    for n in CATALOG
]


class Scripted:
    """A head whose ranking is written in the last user message ("need: delta bravo"; "defer")."""

    adapter = None

    def live_prefix(self, run, adapter=None):
        return adapter.live_prefix(run)

    def decide(self, prefix, candidates, k=None, defer_below=None, *, run=None):
        users = [s["content"] for s in prefix["steps"] if s["name"] == "user"]
        text = users[-1] if users else ""
        need = re.findall(r"[a-z]+", text.partition("need:")[2])
        ranked = need + [c.id for c in candidates if c.id not in need]
        return {
            "candidates": ranked[:k],
            "scores": {},
            "confidence": 1.0,
            "deferred": "defer" in text,
            "error": None,
        }


def user(text):
    return {"role": "user", "content": text}


def call(name, i=0):
    tc = {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": "{}"}}
    return [
        {"role": "assistant", "content": None, "tool_calls": [tc]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": "ok"},
    ]


def names(tools):
    return [t["function"]["name"] for t in tools]


def converse(session, turns):
    """Run ``turns`` (user text, tool the agent then calls or None); the tool list sent at each call."""
    messages, sent = [{"role": "system", "content": "be helpful"}], []
    for i, (text, called) in enumerate(turns):
        messages.append(user(text))
        sent.append(names(session(messages, TOOLS)))
        if called:
            messages += call(called, i)
    return sent


CONVERSATION = [("need: delta bravo", "delta"), ("need: foxtrot alpha", None), ("need: echo", "echo")]


def test_a_session_keeps_used_tools_and_only_appends():
    s = ToolFilter(Scripted(), k=2).session()
    sent = converse(s, CONVERSATION)
    assert sent[0] == ["bravo", "delta"]  # the first list in catalog order
    assert sent[1] == ["bravo", "delta", "alpha", "foxtrot"]  # the new tools appended, in catalog order
    assert sent[2] == ["bravo", "delta", "alpha", "foxtrot", "echo"]
    assert all(
        later[: len(earlier)] == earlier for earlier, later in zip(sent, sent[1:], strict=False)
    )  # only grows
    assert s.used == ["delta"] and s.shown == sent[-1] and s.last["sent"] == sent[-1]


def test_without_sticky_the_list_is_the_top_k_and_the_used_tools_in_catalog_order():
    s = ToolFilter(Scripted(), k=2).session(sticky=False)
    sent = converse(s, CONVERSATION)
    assert sent[1] == ["alpha", "delta", "foxtrot"] and s.last["kept_as_used"] == ["delta"]
    assert sent[2] == ["alpha", "delta", "echo"]  # "need: echo" ranks echo, then alpha; delta was used
    plain = ToolFilter(Scripted(), k=2).session(keep_used=False, sticky=False)
    assert converse(plain, CONVERSATION)[1] == ["alpha", "foxtrot"]  # the stateless filter's answer


def test_sessions_of_one_filter_share_nothing():
    f = ToolFilter(Scripted(), k=2)
    a, b = f.session(), f.session()
    a_turns = [("need: alpha bravo", "alpha"), ("need: charlie delta", None)]
    assert converse(a, a_turns[:1]) == [["alpha", "bravo"]]
    assert converse(b, [("need: echo foxtrot", None)]) == [["echo", "foxtrot"]]  # nothing of a's
    assert names(f([user("need: charlie delta")], TOOLS)) == ["charlie", "delta"]  # the filter has no memory
    assert converse(a, a_turns)[-1] == ["alpha", "bravo", "charlie", "delta"]
    assert a.used == ["alpha"] and b.used == [] and b.shown == ["echo", "foxtrot"]
    assert f.last["candidates"] == ["charlie", "delta"]  # sessions keep their own answers, not the filter's


def test_a_deferred_call_sends_every_tool_and_sticky_keeps_them():
    turns = [("need: alpha", None), ("please defer", None), ("need: bravo", None)]
    sticky = converse(ToolFilter(Scripted(), k=1).session(), turns)
    assert sticky == [["alpha"], CATALOG, CATALOG]
    plain = converse(ToolFilter(Scripted(), k=1).session(sticky=False), turns)
    assert plain == [["alpha"], CATALOG, ["bravo"]]
    kept = converse(ToolFilter(Scripted(), k=1, defer_to_all=False).session(sticky=False), turns)
    assert kept[1] == ["alpha"]  # the head's ranking, not every tool


def test_the_task_fields_reach_the_prefix_and_used_tools_come_from_any_dialect():
    seen = []

    class Recording(Scripted):
        def decide(self, prefix, candidates, **kw):
            seen.append(prefix.get("domain"))
            return super().decide(prefix, candidates, **kw)

    s = ToolFilter(Recording(), k=1, task={"domain": "airline"}).session(task={"domain": "retail"})
    s([user("need: alpha")], TOOLS)
    assert seen == ["retail"]
    anthropic = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "echo", "input": {}}]}
    ]
    tau2 = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "name": "delta", "arguments": {}}]}
    ]
    assert called_tools(anthropic) == ["echo"] and called_tools(tau2) == ["delta"]
    assert called_tools([{"role": "user", "content": "hi", "tool_calls": [{"id": "u", "name": "x"}]}]) == []


def test_concurrent_conversations_and_calls():
    f = ToolFilter(Scripted(), k=2)
    convs = [
        [(f"need: {CATALOG[i % 6]} {CATALOG[(i + 1) % 6]}", CATALOG[i % 6]), ("need: echo", None)]
        for i in range(24)
    ]
    expected = [converse(ToolFilter(Scripted(), k=2).session(), c) for c in convs]
    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(lambda c: converse(f.session(), c), convs))
    assert got == expected
    s = f.session()  # one conversation, its calls from several threads
    msgs = [[user(f"need: {n}")] for n in CATALOG] * 4
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda m: s(m, TOOLS), msgs))
    assert sorted(s.shown) == CATALOG and len(s.shown) == len(set(s.shown))


def test_a_served_head_gives_the_same_lists_and_defers_when_unreachable():
    pytest.importorskip("fastapi")
    from tabagent.serve import RemoteHead

    from conftest import served

    class Head(Scripted):  # what the service needs from a head: decide(run=...) and its card's name
        target, card = "test", type("C", (), {"name": "scripted", "to_dict": lambda self: {}})()

        def decide(self, prefix=None, candidates=None, k=None, defer_below=None, *, run=None):
            from tabagent.adapters import OpenAIMessagesAdapter, parse_catalog

            if not candidates:
                raise ValueError("a ranking head needs the decision's candidates")
            prefix = OpenAIMessagesAdapter().live_prefix(run)
            return super().decide(prefix, list(parse_catalog(candidates).values()), k=k)

    local = converse(ToolFilter(Scripted(), k=2).session(), CONVERSATION)
    with served(Head()) as url:
        remote = ToolFilter(url, k=2)
        assert remote.remote and converse(remote.session(), CONVERSATION) == local
        with pytest.raises(ValueError, match="rejected"):
            RemoteHead(url).decide(run={"messages": []})  # no candidates: 400
    gone = ToolFilter(url, k=2).session()  # the service has stopped
    assert names(gone([user("need: alpha")], TOOLS)) == CATALOG and gone.last["deferred"]
    assert "could not answer" in gone.last["error"]
    assert json.loads(json.dumps(gone.last))  # the record stays JSON
