"""Property-based invariants of every log adapter, end to end (detect -> load -> components -> decision points
-> candidates and labels -> live prefixes -> scan -> onboard dry run), on generated messy logs.

For every generated folder: no crash except a documented error with a message; prefixes never hold the
decision step, a later step, an outcome field or the task id; the live prefix rebuilt from the run before the
decision equals the training prefix; the output is the same when the files are read again, when they are
written under other names (file order), and when runs are read lazily from their files (large inputs).
``TABAGENT_FUZZ=2000 pytest tests/adapters/test_invariants.py`` explores new examples (see messy_logs).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from messy_logs import MUTATIONS, chat_run, check_pipeline, step_run, tau2_results, write

from tabagent.adapters import (
    CugaAdapter,
    GenericJsonAdapter,
    MixedFormatsError,
    OpenAIMessagesAdapter,
    Tau2Adapter,
    detect_adapter,
    detect_format,
)
from tabagent.adapters import logfiles as lf
from tabagent.adapters.detect import _family, _first_record
from tabagent.tabschema.trajectory import Candidate


def _generic():
    names = ["router", "check", "lookup"]
    return GenericJsonAdapter(
        candidates_fn=lambda dp: [Candidate(n, n) for n in names],
        labels_fn=lambda dp, raw: {str(dp.target_step.get("data") or "")[:1] or "router"},
    )


def _same_runs(a: dict, b: dict) -> None:
    """The same decisions whatever the run ids (file names): the decision points as a multiset, without their
    group (a run without a task id is its own group, named after its file)."""
    assert sorted(a.get("components", {}).items()) == sorted(b.get("components", {}).items())
    left, right = (sorted(json.dumps(v[1:]) for v in x.get("dps", {}).values()) for x in (a, b))
    assert left == right


def _check(
    files: dict, adapter_factory=None, onboard: bool = False, rename: bool = True, mutation: str = "none"
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        logs = write(root / "logs", files, mutation)
        first = check_pipeline(logs, adapter_factory() if adapter_factory else None, onboard=onboard)
        again = check_pipeline(logs, adapter_factory() if adapter_factory else None)
        assert first == again, "same input, same output"
        lazy_before = lf.LAZY_INPUT_BYTES
        try:  # every run read again from its file when needed: the same output
            lf.LAZY_INPUT_BYTES = 0
            assert check_pipeline(logs, adapter_factory() if adapter_factory else None) == first
        finally:
            lf.LAZY_INPUT_BYTES = lazy_before
        if rename and "error" not in first:  # other file names, in reverse order: the same decisions
            names = sorted(files)
            renamed = {f"z{len(names) - i:03d}_{n}": files[n] for i, n in enumerate(names)}
            fresh = adapter_factory() if adapter_factory else None
            other = check_pipeline(write(root / "renamed", renamed, mutation), fresh)
            _same_runs(first, other)
        return first


MUTATION = st.sampled_from(MUTATIONS)


@given(st.lists(chat_run(), min_size=1, max_size=4), st.sampled_from(["json", "jsonl", "list"]), MUTATION)
def test_chat_message_logs(runs, layout, mutation):
    for k, run in enumerate(runs):  # distinct task ids where the generator drew a sentinel
        if isinstance(run.get("task_id"), str) and run["task_id"].startswith("TASKID-"):
            run["task_id"] = f"TASKID-{k}"
    if layout == "json":
        files = {f"run{k}.json": r for k, r in enumerate(runs)}
    elif layout == "jsonl":
        files = {"runs.jsonl": runs}
    else:
        files = {"runs.json": runs}
    _check(files, OpenAIMessagesAdapter, rename=layout == "json", mutation=mutation)


@given(st.lists(chat_run(), min_size=1, max_size=3))
def test_chat_message_logs_detected_and_onboarded(runs):
    out = _check({f"r{k}.json": r for k, r in enumerate(runs)}, onboard=True)
    assert "error" not in out or out["error"] in ("ValueError",)


@given(st.lists(step_run(cuga=True), min_size=1, max_size=4), MUTATION)
def test_cuga_logs(runs, mutation):
    for k, run in enumerate(runs):
        if isinstance(run.get("task_id"), str) and run["task_id"].startswith("TASKID-"):
            run["task_id"] = f"TASKID-{k}"
    files = {f"t{k}.json": r for k, r in enumerate(runs)}
    files["results.json"] = {"summary": True}  # a run summary, never a run
    _check(files, CugaAdapter, onboard=True, mutation=mutation)


@given(
    st.lists(step_run(cuga=False), min_size=1, max_size=4),
    st.sampled_from(["json", "jsonl", "list"]),
    MUTATION,
)
def test_generic_step_logs(runs, layout, mutation):
    for k, run in enumerate(runs):
        if isinstance(run.get("task_id"), str) and run["task_id"].startswith("TASKID-"):
            run["task_id"] = f"TASKID-{k}"
    if layout == "json":
        files = {f"nested/run{k}.json": r for k, r in enumerate(runs)}
    elif layout == "jsonl":
        files = {"runs.jsonl": runs}
    else:
        files = {"runs.json": runs}
    _check(files, _generic, rename=layout == "json", mutation=mutation)


@given(st.lists(tau2_results(), min_size=1, max_size=3), MUTATION)
def test_tau2_results(results, mutation):
    _check({f"results_{k}.json": r for k, r in enumerate(results)}, Tau2Adapter, mutation=mutation)


@given(
    chat_run(),
    step_run(cuga=True),
    tau2_results(),
    st.sampled_from([("chat", "cuga"), ("chat", "tau2"), ("cuga", "tau2"), ("chat", "cuga", "tau2")]),
)
def test_mixed_formats_follow_the_documented_policy(chat, cuga, tau, kinds):
    """Detection reads the format most files hold (CUGA over generic steps; the first file on a tie) and
    skips the other formats with a warning that names them; ``on_other_formats="error"`` refuses a folder with
    two formats; an adapter given explicitly reads its own runs and never refuses."""
    cuga["steps"].append({"name": "CodeAgent", "data": ""})  # recognizably CUGA
    pick = {"chat": ("a_chat.json", chat), "cuga": ("b_cuga.json", cuga), "tau2": ("c_tau2.json", tau)}
    files = dict(pick[k] for k in kinds)
    with tempfile.TemporaryDirectory() as tmp:
        logs = write(Path(tmp) / "logs", files)
        found = {name: detect_format(_first_record(logs / name, logs)) for name in sorted(files)}
        families = {_family(c) for c in found.values() if c is not None}
        if len(families) > 1:
            with pytest.raises(MixedFormatsError, match="more than one log format"):
                detect_adapter(logs, on_other_formats="error")
        if not families:
            return
        adapter = detect_adapter(logs)
        counts = {fam: sum(_family(c) == fam for c in found.values() if c is not None) for fam in families}
        first_of = {
            fam: next(n for n, c in found.items() if c is not None and _family(c) == fam) for fam in families
        }
        best = max(sorted(families, key=lambda fam: first_of[fam]), key=lambda fam: counts[fam])
        assert _family(type(adapter)) == best
        out = check_pipeline(logs, adapter)
        if out.get("error") is None and len(families) > 1:
            assert any("another log format" in w for w in adapter.warnings), adapter.warnings
        for explicit in (OpenAIMessagesAdapter(), CugaAdapter(), Tau2Adapter(), GenericJsonAdapter()):
            try:
                explicit.load(logs)
            except MixedFormatsError:  # pragma: no cover - the documented policy says never
                raise AssertionError("an explicit adapter never refuses a folder") from None
            except ValueError as e:
                assert "no runs found" in str(e)


@given(st.binary(max_size=300), st.sampled_from([".json", ".jsonl"]))
def test_arbitrary_bytes_never_crash(data, suffix):
    with tempfile.TemporaryDirectory() as tmp:
        logs = write(Path(tmp) / "logs", {f"x{suffix}": data})
        for adapter in (None, OpenAIMessagesAdapter(), CugaAdapter(), Tau2Adapter(), GenericJsonAdapter()):
            check_pipeline(logs, adapter)


@given(
    st.recursive(
        st.one_of(st.none(), st.booleans(), st.integers(), st.floats(), st.text(max_size=8)),
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(
                st.sampled_from(
                    [
                        "messages",
                        "steps",
                        "name",
                        "role",
                        "content",
                        "tool_calls",
                        "simulations",
                        "task_id",
                        "score",
                        "tools",
                        "data",
                        "prompts",
                    ]
                ),
                children,
                max_size=5,
            ),
        ),  # fmt: skip
        max_leaves=30,
    )
)
def test_arbitrary_json_never_crashes(value):
    with tempfile.TemporaryDirectory() as tmp:
        logs = write(Path(tmp) / "logs", {"x.json": value, "y.jsonl": [value, value]})
        for adapter in (None, OpenAIMessagesAdapter(), CugaAdapter(), Tau2Adapter(), _generic()):
            check_pipeline(logs, adapter)
