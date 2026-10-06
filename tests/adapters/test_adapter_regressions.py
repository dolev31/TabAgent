"""Regression tests for the log-adapter bugs found by the adversarial suite, one per bug (fabricated logs)."""

from __future__ import annotations

import json
import math

import pytest
from fake_logs import cuga_trajectory, openai_run, openai_tools, tau2_results

import tabagent
from tabagent.adapters import (
    NEXT_TOOL_CALL,
    SHORTLISTER,
    TOOL_SHORTLIST,
    CugaAdapter,
    GenericJsonAdapter,
    OpenAIMessagesAdapter,
    Tau2Adapter,
    detect_adapter,
    live_prefix,
    normalize_messages,
    parse_catalog,
)
from tabagent.adapters.cuga import available_apps, called_apis, chosen_apps, code_api_calls, decision_app
from tabagent.adapters.logfiles import MixedFormatsError
from tabagent.tabschema.trajectory import DecisionPoint, Trajectory, read_outcome


def _write(d, files):
    d.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        (d / name).write_text(value if isinstance(value, str) else json.dumps(value))
    return d


# --------------------------------------------------------------------------- outcomes
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True), ("true", True), ("PASS", True), ("1.0", True), ("yes", True), (0.5, True),
        (False, False), (0, False), ("no", False), ("FAIL", False), ("0", False), (-1, False),
        ("n/a", None), ("", None), (math.nan, None), (math.inf, None), ("inf", None), ({"reward": 1}, None),
        ([1], None), (None, None),
    ],
)  # fmt: skip
def test_outcome_values(value, expected):
    assert read_outcome(value) is expected


def test_an_outcome_object_is_a_reported_failure_not_a_success(tmp_path):
    """Before: a chat run whose reward was an object ({"reward": 0.0}) counted as successful (bool(dict))."""
    runs = {
        "a.json": openai_run(0, {"reward": {"reward": 0.0}}),
        "b.json": openai_run(1, {"success": math.inf}),
    }
    adapter = OpenAIMessagesAdapter()
    trajs = adapter.load(_write(tmp_path, runs))
    assert not any(t.success for t in trajs)
    (note,) = [w for w in adapter.warnings if "not a boolean" in w]
    assert note.startswith("2 of 2 runs have an outcome") and "a.json: reward=" in note and "success=" in note


# --------------------------------------------------------------------------- chat messages
def test_a_tools_field_is_never_read_as_a_file_path(tmp_path):
    """Before: a run's ``tools`` string was opened as a catalog file (a crash, or a read of a local file)."""
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps([{"name": "leaked_tool"}]))
    runs = {
        "path.json": {**openai_run(0, tools=False), "tools": str(secret)},
        "text.json": {**openai_run(1, tools=False), "tools": json.dumps(openai_tools())},
        "junk.json": {**openai_run(2, tools=False), "tools": 42},
    }
    adapter = OpenAIMessagesAdapter()
    trajs = {t.id: t for t in adapter.load(_write(tmp_path / "logs", runs))}
    dps = {k: adapter.decision_points(t, TOOL_SHORTLIST)[0] for k, t in trajs.items()}
    assert "leaked_tool" not in {c.id for dp in dps.values() for c in adapter.candidates(dp)}
    assert len(adapter.candidates(dps["text"])) == 6  # JSON text of a tool list is read
    assert any("`tools` field that is not a tool catalog" in w and "path" in w for w in adapter.warnings)


def test_a_list_of_runs_with_one_broken_run_keeps_the_others(tmp_path):
    """Before: one run without ``messages`` turned the file into a 'bare message list': every run was lost."""
    runs = [openai_run(0), {"task_id": "crashed", "error": "timeout"}, openai_run(1), {"messages": []}]
    adapter = OpenAIMessagesAdapter()
    trajs = adapter.load(_write(tmp_path, {"runs.json": runs}))
    assert [t.id for t in trajs] == ["runs:0", "runs:2"]
    text = "\n".join(adapter.warnings)
    assert "not a run of this log format, skipped: 1 record (first: runs.json[1])" in text
    assert "runs without messages, skipped: 1 run (first: runs.json[3])" in text


def test_a_conversation_written_one_message_per_line_is_one_run(tmp_path):
    lines = "\n".join(json.dumps(m) for m in openai_run(0)["messages"])
    adapter = detect_adapter(_write(tmp_path, {"conversation.jsonl": lines}))
    (traj,) = adapter.load(tmp_path)
    assert traj.id == "conversation" and adapter.decision_points(traj, TOOL_SHORTLIST)


def test_odd_tool_call_records_are_read_not_crashed_on():
    calls = [{"id": "1", "type": "function", "function": {"name": " search_flights ", "arguments": "{"}}]
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "additional_kwargs": "not a dict",
            "tool_calls": json.dumps(calls),
        },
        {"role": "assistant", "content": "", "tool_calls": {"name": "book_flight", "arguments": "{}"}},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": {"bad": 1}}}]},
        {"role": "user", "content": "now patch it"},
        {"type": "custom_tool_call", "call_id": "c", "name": "apply_patch", "input": "*** patch"},
        {"type": "custom_tool_call_output", "call_id": "c", "output": "done"},
        {"type": "mcp_call", "name": "srv.lookup", "arguments": "{}", "output": "found"},
    ]
    steps = normalize_messages(msgs)
    assert [s["tool_calls"] for s in steps if s["name"] == "assistant"] == [
        ["search_flights"],
        ["book_flight"],
        [],
        ["apply_patch"],
        ["srv.lookup"],
    ]
    assert [s["tool_name"] for s in steps if s["name"] == "tool"] == ["apply_patch", "srv.lookup"]


def test_binary_tool_results_become_placeholders():
    b64 = ("QUJDZGVm0123" * 400)[:4800]
    out = normalize_messages(
        [
            {"role": "tool", "content": "\x00\x01\x02\x03" * 30},
            {"role": "tool", "content": f"pdf: {b64}"},
            {"role": "tool", "content": "ACGT" * 2000},  # one case: a sequence, not base64
        ]
    )
    assert out[0]["content"] == "[binary: 120 characters]" and out[1]["content"] == "pdf: [binary]"
    assert out[2]["content"].startswith("ACGTACGT")


def test_candidates_do_not_depend_on_the_order_the_runs_were_read(tmp_path):
    """Before: when runs disagreed on a tool's description, the last file read won."""

    def tools(desc):
        return [{"name": "search_flights", "description": desc}]

    files = {
        "a.json": {**openai_run(0, tools=False), "task_id": "x", "tools": tools("short")},
        "b.json": {**openai_run(0, tools=False), "task_id": "y", "tools": tools("the longer description")},
        "c.json": {**openai_run(0, tools=False), "task_id": "z"},
    }
    descriptions = set()
    for order in (("a", "b"), ("b", "a")):
        renamed = {f"{order.index(k[0]) if k[0] in order else 9}{k}": v for k, v in files.items()}
        adapter = OpenAIMessagesAdapter()
        trajs = {t.group: t for t in adapter.load(_write(tmp_path / "".join(order), renamed))}
        (dp,) = adapter.decision_points(trajs["z"], TOOL_SHORTLIST)
        descriptions.add(next(c.description for c in adapter.candidates(dp) if c.id == "search_flights"))
    assert descriptions == {"the longer description"}


def test_loading_again_forgets_the_earlier_logs(tmp_path):
    adapter = OpenAIMessagesAdapter()
    adapter.load(_write(tmp_path / "a", {"a.json": openai_run(0, tools=False)}))
    (traj,) = adapter.load(_write(tmp_path / "b", {"b.json": openai_run(3, tools=False)}))
    (dp,) = adapter.decision_points(traj, TOOL_SHORTLIST)
    assert {c.id for c in adapter.candidates(dp)} == {"play_song"}  # not the tools of the first folder


def test_a_long_run_shares_its_prefix_steps(tmp_path):
    msgs = [{"role": "user", "content": "go"}]
    for k in range(300):
        call = {"id": f"c{k}", "function": {"name": "get_weather", "arguments": "{}"}}
        msgs += [
            {"role": "assistant", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": f"c{k}", "content": "x"},
        ]
    adapter = OpenAIMessagesAdapter("step")
    dps = adapter.decision_points(Trajectory("r", {"messages": msgs, "reward": 1}), NEXT_TOOL_CALL)
    assert len(dps) == 300 and dps[-1].prefix["steps"][0] is dps[0].prefix["steps"][0]


# --------------------------------------------------------------------------- tau2-bench
def test_tau2_files_with_odd_entries_are_read(tmp_path):
    res = tau2_results("airline", [("0", ["get_user_details"], 1.0), ("1", ["get_user_details"], 1.0)])
    res["simulations"][1]["reward_info"] = 1.0  # a bare number
    res["simulations"].append("not a simulation")
    res["simulations"].append({**res["simulations"][0], "task_id": "9"})  # a task the file does not list
    res["tasks"].append("not a task")
    copy = json.loads(json.dumps(res))
    adapter = Tau2Adapter()
    trajs = adapter.load(_write(tmp_path, {"a.json": res, "b.json": copy}))
    assert len(trajs) == 6 and all(t.success for t in trajs)
    text = "\n".join(adapter.warnings)
    assert "simulations that are not objects, skipped: 2 simulations" in text
    assert "simulations of a task the file's tasks do not list: 2 simulations" in text
    assert "simulations whose id occurs earlier too" in text
    broken = {
        "info": "x",
        "tasks": None,
        "simulations": [{"task_id": "0", "messages": "x", "reward_info": "1"}],
    }
    (t,) = Tau2Adapter().load(_write(tmp_path / "broken", {"r.json": broken}))
    assert t.group == ":0" and t.success


# --------------------------------------------------------------------------- CUGA
def test_call_api_is_read_as_python():
    code = "\n".join(
        [
            "```python",
            'a = await call_api(api_name="phone_call", app_name="phone")',  # keywords, reversed
            "# b = await call_api('phone', 'commented_out')",
            "print(\"call_api('phone', 'in_a_string')\")",
            'c = await call_api(f"venmo", "pay")',
            "d = await call_api(app, 'unknown_app')",
            "```",
        ]
    )
    assert code_api_calls(code) == {("phone", "phone_call"), ("venmo", "pay")}
    cut = 'r = await call_api(api_name="x_api", app_name="x"\n# call_api("x", "comment")'  # does not parse
    assert code_api_calls(cut) == {("x", "x_api")}
    escaped = 'r = await call_api(\\"phone\\", \\"phone_call\\")'
    assert code_api_calls(escaped) == {("phone", "phone_call")}
    assert called_apis({"name": "CodeAgent", "data": {"code": code}}) == {
        ("phone", "phone_call"),
        ("venmo", "venmo_pay"),
    }


def test_cuga_steps_of_any_shape_never_crash(tmp_path):
    rounds = [("phone", ["phone_a", "phone_b"], ["phone_a"])]
    run = cuga_trajectory("t1", rounds)
    k = next(i for i, s in enumerate(run["steps"]) if s["name"] == SHORTLISTER)
    run["steps"].insert(k + 1, "a step that is not an object")
    run["steps"].insert(1, {"name": ["unhashable"], "prompts": 5})
    run["steps"].insert(1, {"name": "APIPlannerAgent", "prompts": "not a list"})
    adapter = CugaAdapter()
    (traj,) = adapter.load(_write(tmp_path, {"t1.json": run}))
    (dp,) = adapter.decision_points(traj, SHORTLISTER)
    assert adapter.labels(dp) == {"phone_a"} and adapter.candidates(dp)
    assert adapter.llm_calls(traj) and "['unhashable']" not in adapter.components([traj])
    stripped = DecisionPoint(
        "r", 1, {"steps": ["x", {"name": "APIPlannerAgent", "prompts": 3}]}, {"prompts": 5}
    )
    assert decision_app(stripped) is None
    assert available_apps({"prompts": 7}) == {} and chosen_apps({"data": [{"no": "name"}, "phone"]}) == {
        "phone"
    }


def test_cuga_scores_are_read_like_outcomes_and_reported(tmp_path):
    runs = {
        f"{k}.json": {**cuga_trajectory(k, [("phone", ["phone_a"], ["phone_a"])]), "score": v}
        for k, v in {"t1": "true", "t2": None, "t3": "n/a", "t4": 1}.items()
    }
    adapter = CugaAdapter()
    trajs = {t.group: t.success for t in adapter.load(_write(tmp_path, runs))}
    assert trajs == {"t1": True, "t2": False, "t3": False, "t4": True}
    text = "\n".join(adapter.warnings)
    assert "1 of 4 runs have no outcome field (score); they were treated as failed" in text
    assert 'score="n/a"' in text


def test_components_ignore_names_that_are_not_strings_and_are_ordered(tmp_path):
    runs = {
        "a.json": {"task_id": "a", "steps": [{"name": "b"}, {"name": "a"}, {"name": 5}, {"name": None}]},
        "b.json": {"task_id": "b", "steps": [{"name": "a"}, {"name": "b"}, {"name": ["x"]}]},
    }
    adapter = GenericJsonAdapter()
    trajs = adapter.load(_write(tmp_path, runs))
    assert list(adapter.components(trajs)) == ["a", "b"]  # a tie, ordered by name, whatever the hash seed


def test_generic_logs_follow_links_and_read_jsonl_and_lists(tmp_path):
    real = _write(tmp_path / "real", {"r.json": {"task_id": "a", "steps": [{"name": "x"}], "score": 1}})
    logs = tmp_path / "logs"
    _write(
        logs, {"runs.jsonl": "\n".join(json.dumps({"task_id": t, "steps": [{"name": "x"}]}) for t in "bc")}
    )
    _write(logs, {"many.json": [{"task_id": "d", "steps": []}, {"task_id": "e", "steps": [{"name": "x"}]}]})
    try:
        import os

        os.symlink(real, logs / "linked")
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")
    ids = sorted(t.id for t in GenericJsonAdapter().load(logs))
    assert ids == ["linked/r", "many:0", "many:1", "runs:1", "runs:2"]


def test_runs_of_another_format_are_skipped_with_a_warning_or_refused_on_request(tmp_path):
    cuga = cuga_trajectory("t", [("phone", ["p"], ["p"])])
    logs = _write(tmp_path, {"a.json": openai_run(0), "b.json": cuga, "c.json": openai_run(1), "x.json": [1]})
    adapter = detect_adapter(logs)  # the format most files hold
    assert type(adapter) is OpenAIMessagesAdapter and len(adapter.load(logs)) == 2
    (note,) = [w for w in adapter.warnings if "another log format" in w]
    assert note.startswith("runs of another log format (CUGA trajectories, CugaAdapter), skipped: 1 record")
    assert "(first: b.json)" in note
    with pytest.raises(MixedFormatsError, match=r"more than one log format.*CugaAdapter.*b\.json"):
        detect_adapter(logs, on_other_formats="error")
    strict = OpenAIMessagesAdapter()
    strict.on_other_formats = "error"
    with pytest.raises(MixedFormatsError, match=r"2 runs of chat-message runs.*CUGA trajectories"):
        strict.load(logs)
    explicit = CugaAdapter()  # an explicit adapter reads its own runs and never raises
    assert [t.id for t in explicit.load(logs)] == ["b"]
    assert any(
        "chat-message runs, OpenAIMessagesAdapter), skipped: 2 records" in w for w in explicit.warnings
    )
    assert tabagent.onboard(logs, dry_run=True).n_trajectories == 2
    with pytest.raises(MixedFormatsError, match="point at a folder or file with one format"):
        tabagent.onboard(logs, dry_run=True, on_other_formats="error")
    (logs / "d.json").write_text(json.dumps(tau2_results("airline", [("0", ["t"], 1.0)])))
    strict = OpenAIMessagesAdapter()
    strict.on_other_formats = "error"
    with pytest.raises(MixedFormatsError) as e:  # every other format is named, with its own count
        strict.load(logs)
    assert "1 record (first: b.json)" in str(e.value) and "τ²-bench results, Tau2Adapter" in str(e.value)


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("prompt_template.json", {"messages": [{"role": "system", "content": "You are CUGA"}]}),
        ("few_shot.jsonl", [{"role": "user", "content": "example"}]),
        ("tools.json", [{"type": "function", "function": {"name": "x", "parameters": {}}}]),
        ("README.json", {"title": "my runs", "description": "x"}),
        ("results.json", {"messages": [{"role": "user", "content": "summary"}]}),
        ("pipeline.json", {"steps": [{"name": "download"}, {"name": "train"}]}),
        (
            "finetune.jsonl",
            [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}],
        ),
    ],
)
def test_a_cuga_folder_with_a_side_file_still_loads(tmp_path, name, content):
    """Files next to the runs (prompt templates, few-shot and fine-tuning examples, catalogs, configurations)
    never stop a load: they are skipped, with a warning when they hold anything run-like."""
    files = {f"t{i}.json": cuga_trajectory(f"t{i}", [("phone", ["p"], ["p"])]) for i in range(3)}
    logs = _write(tmp_path, files)
    lines = "".join(json.dumps(x) + "\n" for x in content) if name.endswith(".jsonl") else None
    (logs / name).write_text(lines if lines is not None else json.dumps(content))
    for adapter in (detect_adapter(logs), CugaAdapter()):
        assert type(adapter) is CugaAdapter
        assert sorted(t.id for t in adapter.load(logs)) == ["t0", "t1", "t2"]


def test_a_cuga_folder_whose_first_run_has_no_agent_step_is_still_cuga(tmp_path):
    first = {"task_id": "a", "steps": [{"name": "EvaluationResult", "data": "{}"}], "score": 0}
    logs = _write(tmp_path, {"a.json": first, "b.json": cuga_trajectory("b", [("phone", ["p"], ["p"])])})
    assert type(detect_adapter(logs)) is CugaAdapter


def test_catalogs_of_every_style(tmp_path):
    schema_per_tool = {"get_weather": {"type": "object", "properties": {"city": {}}, "required": ["city"]}}
    assert {k: c.arg_hints for k, c in parse_catalog(schema_per_tool).items()} == {"get_weather": "city"}
    assert set(parse_catalog({"phone": {"phone_call": {"api_name": "phone_call"}}, "empty_app": {}})) == {
        "phone_call"
    }
    malformed = [
        {"type": "function"},
        5,
        None,
        {"name": "  "},
        {"name": "ok", "parameters": {"properties": [1]}},
    ]
    assert list(parse_catalog(malformed)) == ["ok"]
    dupes = [{"name": "a", "description": "first"}, {"name": "a", "description": "last"}]
    assert parse_catalog(dupes)["a"].description == "last"
    with pytest.raises(FileNotFoundError, match="tool catalog .*no such file"):
        parse_catalog(str(tmp_path / "missing.json"))
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(ValueError, match="tool catalog .*not valid JSON"):
        parse_catalog(tmp_path / "bad.json")


def test_errors_say_where(tmp_path):
    adapter = GenericJsonAdapter(success=lambda run: run["missing"])
    with pytest.raises(ValueError, match=r"r\.json: the success function raised KeyError"):
        adapter.load(_write(tmp_path, {"r.json": {"steps": []}}))
    with pytest.raises(KeyError, match="call decision_points"):
        GenericJsonAdapter(labels_fn=lambda dp, raw: set()).labels(DecisionPoint("r", 0, {}, {}))
    with pytest.raises(TypeError, match="a live run is"):
        live_prefix(OpenAIMessagesAdapter(), ["not", "a", "run"])
    found = r"no runs found .*\(1 file read by CugaAdapter.*first: x\.json\).*check-logs"
    with pytest.raises(ValueError, match=found):
        tabagent.scan(_write(tmp_path / "none", {"x.json": {"steps": "nothing"}}), adapter=CugaAdapter())


# --------------------------------------------------------------------------- review round: L1, L2, L3, L6, M1
@pytest.mark.parametrize("lazy", [False, True])
def test_user_hooks_get_copies_so_edits_never_persist(tmp_path, lazy):
    runs = {
        f"r{i}.json": {"task_id": f"t{i}", "steps": [{"name": "router", "data": "a"}], "score": 1}
        for i in range(2)
    }
    logs = _write(tmp_path, runs)

    def success(run):  # a function that edits the run it is given
        run["steps"][0]["data"] = "EDITED"
        return True

    def labels(dp, run):
        run["steps"].append({"name": "EDITED"})
        return {"a"}

    adapter = GenericJsonAdapter(success=success, labels_fn=labels)
    trajs = adapter.load(logs, lazy=lazy)
    (dp,) = adapter.decision_points(trajs[0], "router")
    adapter.labels(dp)
    adapter.labels(dp)
    assert trajs[0].raw == runs["r0.json"] and dp.target_step == {"name": "router", "data": "a"}
    redacted = trajs[0].with_raw(lambda r: r.update(task_id="X") or r)  # an in-place redaction
    assert redacted.raw["task_id"] == "X" and trajs[0].raw["task_id"] == "t0"
    assert redacted.is_lazy == lazy


def test_fences_are_removed_only_from_code_that_does_not_parse_with_them():
    code = 'fence = "```"\nr = await call_api("phone", "phone_call")'
    assert code_api_calls(code) == {("phone", "phone_call")}  # it parses: the fence is a string
    assert code_api_calls(f"```python\n{code}\n```") == {("phone", "phone_call")}  # fenced code


def test_names_assigned_a_literal_once_are_resolved_and_others_counted_as_dynamic(tmp_path):
    from tabagent.adapters.cuga import code_api_calls_counted

    code = "\n".join(
        [
            'app = "venmo"',
            'r = await call_api(app, "pay")',
            'api = "a"',
            'api = "b"',  # assigned twice: dynamic
            'r = await call_api("phone", api)',
            "for x in apps:",
            '    r = await call_api(x, "search")',  # a loop variable: dynamic
        ]
    )
    assert code_api_calls_counted(code) == ({("venmo", "pay")}, 2)
    run = cuga_trajectory("t", [("venmo", ["venmo_pay"], ["pay"])])
    run["steps"].append({"name": "CodeAgent", "data": json.dumps({"code": code})})
    adapter = CugaAdapter()
    adapter.load(_write(tmp_path, {"t.json": run}))
    assert any("2 calls" in w and "not a literal" in w for w in adapter.warnings)


def test_a_jsonl_file_that_starts_with_a_message_is_still_read_line_by_line(tmp_path):
    lines = [{"role": "system", "content": "a header message"}] + [openai_run(i) for i in range(3)]
    (tmp_path / "runs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    adapter = OpenAIMessagesAdapter()
    trajs = adapter.load(tmp_path)
    assert [t.id for t in trajs] == ["runs:2", "runs:3", "runs:4"]
    assert any("runs.jsonl:1" in w for w in adapter.warnings)


def test_redaction_keeps_lazy_runs_lazy_and_the_dry_run_shows_the_log_notes(tmp_path, monkeypatch):
    from tabagent.adapters import logfiles as lf
    from tabagent.tabschema.trajectory import Trajectory

    logs = tmp_path / "logs"
    logs.mkdir()
    for i in range(6):
        (logs / f"r{i}.json").write_text(json.dumps(openai_run(i)))
    (logs / "broken.json").write_text("{")
    monkeypatch.setattr(lf, "LAZY_INPUT_BYTES", 0)
    made = []
    original = Trajectory.with_raw
    monkeypatch.setattr(Trajectory, "with_raw", lambda self, fn: made.append(original(self, fn)) or made[-1])
    dry = tabagent.onboard(logs, "ToolShortlist", dry_run=True, redact=lambda r: {**r, "instruction": "x"})
    assert len(made) == 6 and all(t.is_lazy for t in made)
    assert any("broken.json" in w for w in dry.warnings) and "Logs: not valid JSON" in str(dry)


def test_cli_check_logs_and_reading_flags(tmp_path, capsys):
    from fake_logs import write_openai_logs

    from tabagent.cli import main

    logs = write_openai_logs(tmp_path / "logs", n=8)
    assert main(["check-logs", str(logs), "--no-lazy"]) == 0
    assert "Next: tabagent onboard" in capsys.readouterr().out
    assert main(["check-logs", str(logs), "--format", "json", "--decision", "NextToolCall"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    (logs / "zz.json").write_text(json.dumps(cuga_trajectory("t", [("phone", ["p"], ["p"])])))
    assert main(["check-logs", str(logs), "--on-other-formats", "error"]) == 1
    assert "more than one log format" in " ".join(capsys.readouterr().out.split())  # wrapped text
    assert main(["components", str(logs), "--on-other-formats", "error"]) == 2
    assert "more than one log format" in capsys.readouterr().err
    assert main(["components", str(logs)]) == 0
    assert "another log format" in capsys.readouterr().err
