"""OpenAIMessagesAdapter on fabricated runs (OpenAI, Anthropic Messages API format, LangChain)."""

import json

import pytest
from fake_logs import (
    SYSTEM,
    TOOLS,
    langchain_run,
    messages_api_run,
    messages_api_tools,
    openai_messages,
    openai_run,
)

from tabagent.adapters import (
    NEXT_TOOL_CALL,
    TEXT_REPLY,
    TOOL_SHORTLIST,
    OpenAIMessagesAdapter,
    normalize_messages,
)
from tabagent.tabschema.trajectory import Trajectory


def _load(tmp_path, runs, **kw):
    for name, run in runs.items():
        (tmp_path / name).write_text(run if isinstance(run, str) else json.dumps(run))
    adapter = OpenAIMessagesAdapter(**kw)
    return adapter, {t.id: t for t in adapter.load(tmp_path)}


def test_task_decision_prefix_holds_only_pre_decision_information(tmp_path):
    run = openai_run(
        0,
        instruction="book travel",
        task={"expected_tools": ["search_flights"]},  # nested task fields never reach a prefix
        duration_s=12.5,
        final_answer="FINAL ANSWER leaked",
    )
    adapter, trajs = _load(tmp_path, {"run0.json": run})
    (dp,) = adapter.decision_points(trajs["run0"], TOOL_SHORTLIST)

    assert dp.step_index == 4 and dp.group == "task0" and dp.id == "run0#4"
    assert set(dp.prefix) == {"instruction", "steps"}  # the task id names the group, it is not a feature
    assert [s["name"] for s in dp.prefix["steps"]] == ["system", "user", "user"]  # no assistant chatter
    assert all(set(s) == {"name", "content", "tool_calls", "tool_name"} for s in dp.prefix["steps"])
    dumped = json.dumps(dp.prefix)
    assert "FINAL ANSWER" not in dumped and "result of" not in dumped and "search_flights" not in dumped
    assert not set(dp.prefix) & set(adapter.outcome_keys)
    assert dp.target_step == {"name": TOOL_SHORTLIST, "tool_calls": ["search_flights", "book_flight"]}
    assert adapter.labels(dp) == {"search_flights", "book_flight"}


def test_step_decisions_see_earlier_tool_results_but_not_the_turn(tmp_path):
    adapter, trajs = _load(tmp_path, {"run0.json": openai_run(0)}, granularity="step")
    first, second = adapter.decision_points(trajs["run0"], NEXT_TOOL_CALL)

    assert adapter.target == NEXT_TOOL_CALL and first.step_index == 4 and second.step_index == 6
    assert adapter.labels(first) == {"search_flights"} and adapter.labels(second) == {"book_flight"}
    assert [s["name"] for s in second.prefix["steps"]][-2:] == ["assistant", "tool"]
    assert second.prefix["steps"][-1]["tool_name"] == "search_flights"
    assert "book_flight" not in json.dumps(second.prefix) and "FINAL" not in json.dumps(second.prefix)
    # either target can be asked for, whatever the default granularity
    assert len(adapter.decision_points(trajs["run0"], TOOL_SHORTLIST)) == 1
    with pytest.raises(ValueError, match="builds the decisions"):
        adapter.decision_points(trajs["run0"], "ShortlisterAgent")


def test_candidates_come_from_the_run_then_the_given_catalog_then_the_tools_seen(tmp_path):
    runs = {"a.json": openai_run(0), "b.json": openai_run(1, tools=False)}
    adapter, trajs = _load(tmp_path, runs)
    (dp_a,) = adapter.decision_points(trajs["a"], TOOL_SHORTLIST)
    cands = {c.id: c for c in adapter.candidates(dp_a)}
    assert set(cands) == set(TOOLS)
    assert cands["search_flights"].description == "search flights between two airports"
    assert cands["search_flights"].arg_hints == "origin, destination, date"

    given, trajs = _load(tmp_path, runs, tools=messages_api_tools())
    (dp_b,) = given.decision_points(trajs["b"], TOOL_SHORTLIST)
    assert {c.id for c in given.candidates(dp_b)} == set(TOOLS)

    for f in tmp_path.iterdir():
        f.unlink()
    seen, trajs = _load(
        tmp_path, {"a.json": openai_run(0, tools=False), "b.json": openai_run(1, tools=False)}
    )
    (dp_b,) = seen.decision_points(trajs["b"], TOOL_SHORTLIST)
    cands = {c.id: c for c in seen.candidates(dp_b)}  # tools called in any loaded run, with observed args
    assert set(cands) == {"search_flights", "book_flight", "get_weather"}
    assert cands["search_flights"].arg_hints == "date, destination, origin"


def test_outcomes_failed_runs_and_missing_outcomes(tmp_path):
    lines = [
        openai_run(0, {"success": False}),
        openai_run(1, {"reward": 0}),
        openai_run(2, {"score": 0.5}),
        openai_run(3, {"success": "passed"}),
        openai_run(4, {}),
    ]
    text = "\n".join(json.dumps(r) for r in lines) + "\n{broken\n"
    adapter, trajs = _load(tmp_path, {"runs.jsonl": text})
    assert [trajs[f"runs:{i}"].success for i in range(1, 6)] == [False, False, True, True, True]
    assert any("1 of 5 runs have no outcome" in w for w in adapter.warnings)
    assert any("runs.jsonl:6" in w for w in adapter.warnings)
    assert adapter.components(list(trajs.values())) == {TOOL_SHORTLIST: 3}

    custom, trajs = _load(tmp_path, {"runs.jsonl": text}, success="solved")
    assert "solved" in custom.outcome_keys and not any(t.success is False for t in trajs.values())
    by_fn, trajs = _load(tmp_path, {"runs.jsonl": text}, success=lambda run: "score" in run)
    assert [t.success for t in trajs.values()] == [False, False, True, False, False]


def test_dialects_normalize_to_the_same_steps(tmp_path):
    openai_steps = normalize_messages(openai_messages(0))
    assert normalize_messages(messages_api_run(0)["messages"], system=SYSTEM) == openai_steps
    assert normalize_messages(langchain_run(0)["messages"]) == openai_steps
    assert [s["name"] for s in openai_steps] == [
        "system", "user", "assistant", "user", "assistant", "tool", "assistant", "tool", "assistant",
    ]  # fmt: skip
    assert openai_steps[5] == {
        "name": "tool",
        "content": "result of search_flights: ok",
        "tool_calls": [],
        "tool_name": "search_flights",
    }

    adapter, trajs = _load(
        tmp_path, {"o.json": openai_run(0), "a.json": messages_api_run(0), "l.json": langchain_run(0)}
    )
    dps = {k: adapter.decision_points(t, TOOL_SHORTLIST)[0] for k, t in trajs.items()}
    assert dps["a"].prefix["steps"] == dps["l"].prefix["steps"] == dps["o"].prefix["steps"]
    assert {frozenset(adapter.labels(d)) for d in dps.values()} == {
        frozenset({"search_flights", "book_flight"})
    }
    assert {c.id for c in adapter.candidates(dps["a"])} == set(
        TOOLS
    )  # Anthropic Messages API format tool schemas


def test_langchain_serializations_legacy_calls_and_responses_items():
    dumped = [  # messages_to_dict and dumpd
        {"type": "human", "data": {"content": "hi", "type": "human"}},
        {"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "AIMessage"],
         "kwargs": {"content": "", "tool_calls": [{"name": "get_weather", "args": {}, "id": "1"}]}},
        {"role": "function", "name": "get_weather", "content": "sunny"},
        {"role": "assistant", "content": None, "function_call": {"name": "play_song", "arguments": "{}"}},
    ]  # fmt: skip
    steps = normalize_messages(dumped)
    assert [(s["name"], s["tool_calls"], s["tool_name"]) for s in steps] == [
        ("user", [], ""),
        ("assistant", ["get_weather"], ""),
        ("tool", [], "get_weather"),
        ("assistant", ["play_song"], ""),
    ]
    responses = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
        {"type": "reasoning", "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]},
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{}",
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
    ]
    steps = normalize_messages(responses)
    assert [(s["name"], s["content"], s["tool_calls"], s["tool_name"]) for s in steps] == [
        ("user", "weather?", [], ""),
        ("assistant", "Checking.", ["get_weather"], ""),
        ("tool", "sunny", [], "get_weather"),
    ]


def test_llm_calls_are_the_assistant_turns_in_every_dialect():
    def with_usage(run, key, usage):
        for m in run["messages"]:
            if m.get("role") == "assistant" or m.get("type") == "ai":
                m[key] = usage
        return run

    runs = {
        "openai": with_usage(openai_run(0), "usage", {"prompt_tokens": 50, "completion_tokens": 5}),
        "anthropic": with_usage(messages_api_run(0), "usage", {"input_tokens": 50, "output_tokens": 5}),
        "langchain": with_usage(langchain_run(0), "usage_metadata", {"input_tokens": 50, "output_tokens": 5}),
    }
    adapter = OpenAIMessagesAdapter()
    for name, run in runs.items():
        calls = adapter.llm_calls(Trajectory(name, run))
        assert [(c.component, c.choices) for c in calls] == [
            (TEXT_REPLY, None),
            (NEXT_TOOL_CALL, ("search_flights",)),
            (NEXT_TOOL_CALL, ("book_flight",)),
            (TEXT_REPLY, None),
        ], name
        assert all(c.tokens == 55 for c in calls), name
        assert calls[1].decisions == (TOOL_SHORTLIST, NEXT_TOOL_CALL)
    assert set(adapter.llm_calls(Trajectory("o", runs["openai"]))[1].catalog) == set(TOOLS)
    assert adapter.llm_calls(Trajectory("l", runs["langchain"]))[1].catalog is None  # no tools recorded

    responses = [  # the output items of one response (reasoning, message, call) are one LLM call
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
        {"type": "reasoning", "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Sunny."}]},
    ]
    calls = adapter.llm_calls(Trajectory("r", {"messages": responses}))
    assert [(c.component, c.choices, c.text_chars) for c in calls] == [
        (NEXT_TOOL_CALL, ("get_weather",), len("Checking.") + len("{}")),
        (TEXT_REPLY, None, len("Sunny.")),
    ]


def test_media_and_long_text_are_reduced():
    payload = "data:image/png;base64," + "A" * 5000
    msgs = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {}}],
        },
        {"role": "user", "content": f"inline {payload} end"},
        {"role": "user", "content": "x" * 50},
    ]
    a, b, c = normalize_messages(msgs, max_chars=30)
    assert a["content"] == "look\n[image_url]" and b["content"] == "inline [binary] end"
    assert c["content"] == "x" * 30 + " ...[20 more characters]"


def test_live_prefix_matches_the_training_prefix(tmp_path):
    run = openai_run(2)
    adapter, trajs = _load(tmp_path, {"r.json": run})
    (dp,) = adapter.decision_points(trajs["r"], TOOL_SHORTLIST)
    live = {**run, "messages": run["messages"][: dp.step_index]}
    del live["reward"]
    assert adapter.live_prefix(live) == dp.prefix

    step = OpenAIMessagesAdapter(granularity="step")
    last = step.decision_points(trajs["r"], NEXT_TOOL_CALL)[-1]
    assert step.live_prefix({**run, "messages": run["messages"][: last.step_index]}) == last.prefix


def test_lists_of_runs_and_bare_message_lists(tmp_path):
    runs = json.dumps([openai_run(0), openai_run(1, {"reward": 0})])
    adapter, trajs = _load(
        tmp_path, {"many.json": runs, "bare.json": json.dumps(openai_messages(2)), "x.txt": "-"}
    )
    assert set(trajs) == {"many:0", "many:1", "bare"} and not trajs["many:1"].success
    assert trajs["bare"].group == "bare" and adapter.warnings  # no outcome in the bare list
