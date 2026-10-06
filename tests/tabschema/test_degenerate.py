"""Degenerate and extreme data, and misbehaving extraction code: each case ends in a correct card or a clear
error, within a bounded time.

The data: one task, two runs; decisions with one candidate or no label; identical prefixes; prefixes with no
steps; labels that never vary; one label class; very long prefixes; many decision points (memory and time of
the compile step's batching and quality gates). The code: a loop that never ends on one prefix (in Python, and
in C, out of reach of its alarm), a memory blowup, megabytes of printing, NaN and infinities, an exception on
rare prefixes.
"""

from __future__ import annotations

import json
import re
import resource
import sys
import time

import pytest
from test_robustness import (
    PROPOSALS,
    TARGET,
    TASKS,
    FakeLLM,
    adapter,
    code_for,
    discover,
    raw_run,
    runs,
    schema_for,
    spec,
    tagged,
    waits,  # noqa: F401 - autouse: backoff is recorded, not slept
)

import tabagent
from tabagent.tabschema import Candidate, Trajectory
from tabagent.tabschema import compile as C
from tabagent.tabschema.agents import fit_for_prompt
from tabagent.tabschema.compile import ExtractionError
from tabagent.tabschema.sandbox import SubprocessSandbox

FAST = 60  # seconds: every case here ends well within this


def timed(fn):
    t0 = time.monotonic()
    out = fn()
    assert time.monotonic() - t0 < FAST
    return out


def code_llm(code: str, *names: str) -> FakeLLM:
    """A fake LLM proposing ``names`` (default: the usual three) whose CodeExtractor always writes
    ``code``."""
    proposals = [spec(n, "numeric") for n in names] if names else PROPOSALS
    return FakeLLM(
        analyzer=lambda p, n: json.dumps({"potential_features": proposals}),
        extractor=lambda p, n: tagged(code),
    )


# =========================================================================== data
def test_one_task_with_two_runs(tmp_path):
    trajs = [Trajectory(f"r{i}", raw_run(i), True, "the-task") for i in range(2)]
    result = timed(lambda: discover(tmp_path, FakeLLM(), trajs, usefulness=True))
    assert set(result.extractor.features) <= {"user_intent", "intent_words", "plan_note"}
    assert result.report.usefulness["skipped"].startswith(("fewer than 2 tasks", "ValueError"))
    with pytest.raises(ValueError, match="one task group"):  # a head needs a second task to hold out
        tabagent.onboard(trajs, TARGET, adapter=adapter(), llm=FakeLLM(), run_dir=tmp_path / "o")


def test_decisions_with_one_candidate(tmp_path):
    def one(dp):
        used = TASKS[int(dp.trajectory_id[3:]) % len(TASKS)][1][0]
        return [Candidate(used, used, "the only tool offered")]

    schema = schema_for(tmp_path, FakeLLM(), ad=adapter(candidates_fn=one), usefulness=True)
    result = timed(lambda: schema.discover(runs(), TARGET))
    table = schema.build_table(result, runs(), TARGET)
    assert (table.df.groupby("decision_id").size() == 1).all() and table.df["chosen"].eq(1).all()
    assert result.report.usefulness.get("skipped") or result.report.usefulness["features"]


def test_decisions_without_labels(tmp_path):
    none = adapter(labels_fn=lambda dp, raw: set())
    result = timed(lambda: discover(tmp_path, FakeLLM(), ad=none, usefulness=True))
    assert result.extractor.features  # discovery needs no labels; the estimate of usefulness does
    assert "skipped" in result.report.usefulness
    with pytest.raises(ValueError, match="no chosen candidate"):  # and onboarding stops before any call
        tabagent.onboard(runs(), TARGET, adapter=none, llm=FakeLLM(), run_dir=tmp_path / "o")


def test_labels_that_never_vary(tmp_path):
    same = adapter(labels_fn=lambda dp, raw: {"get_weather"})
    result = timed(lambda: discover(tmp_path, FakeLLM(), ad=same, usefulness=True))
    assert result.extractor.features
    u = result.report.usefulness
    assert u.get("skipped") or all(r["verdict"] == "neutral" for r in u["features"].values())


def test_a_single_label_class(tmp_path):
    schema = schema_for(tmp_path, FakeLLM(), ad=adapter(labels_fn=lambda dp, raw: {"only"}), usefulness=True)
    result = timed(lambda: schema.discover(runs(), TARGET, kind="single"))
    assert result.card.spec.kind == "single" and result.extractor.features
    assert "skipped" in result.report.usefulness  # one class: nothing to cross-validate
    table = schema.build_table(result, runs(), TARGET)
    assert set(table.df["label"]) == {"only"}


def test_identical_prefixes_end_in_a_clear_error(tmp_path):
    trajs = [Trajectory(f"r{i}", raw_run(0), True, f"task{i}") for i in range(8)]
    with pytest.raises(ExtractionError, match="every compiled feature failed the quality gates.*constant"):
        timed(lambda: discover(tmp_path, FakeLLM(), trajs))


def test_prefixes_with_no_steps(tmp_path):
    def first(i):  # on half the runs the decision is the run's first step
        raw = raw_run(i)
        if i % 2:
            raw["steps"] = [raw["steps"][2], *raw["steps"][3:]]
        return raw

    result = timed(lambda: discover(tmp_path, FakeLLM(), runs(16, first), n_analyze=6))
    assert set(result.extractor.features) == {"user_intent", "intent_words", "plan_note"}
    assert any(o["plan_note"] == "" for o in result.extractor.outputs)  # no plan before the decision

    def every(i):
        raw = raw_run(i)
        raw["steps"] = raw["steps"][2:]
        return raw

    with pytest.raises(RuntimeError, match="(?s)no feasible feature proposals.*unknown_field: steps"):
        discover(tmp_path, FakeLLM(), runs(16, every), run_id="all")  # no step field is known at all


def test_very_long_prefixes_fit_every_prompt(tmp_path):
    def long(i):
        raw = raw_run(i)
        noise = [{"name": "Tool", "data": f"call {k}"} for k in range(20_000)]  # many short steps
        raw["steps"][2:2] = noise
        return raw

    llm = FakeLLM()
    result = timed(lambda: discover(tmp_path, llm, runs(8, long), n_analyze=2, max_prompt_chars=50_000))
    assert set(result.extractor.features) == {"user_intent", "intent_words", "plan_note"}
    for role in ("analyzer", "auditor"):
        assert all(len(p) < 80_000 for p in llm.prompts(role)), role
    assert any("items omitted to fit the prompt" in p for p in llm.prompts("analyzer"))
    value = {"steps": [{"data": "x" * 10}] * 50_000}
    fitted = fit_for_prompt(value, 5_000)
    assert len(json.dumps(fitted)) <= 5_000 and fitted["steps"][0] == {"data": "x" * 10}


# =========================================================================== extraction code
def _with(condition: str, body: str) -> str:
    """Code for the usual features, running ``body`` first on the prefixes where ``condition`` holds."""
    return code_for([p["feature_name"] for p in PROPOSALS], f"    if {condition}:\n{body}")


LIMA = "trajectory_data['intent'] == 'play a song from lima'"  # one run of the sixteen


@pytest.mark.parametrize(
    "case, body",
    [
        ("python_loop", "        while True:\n            pass\n"),
        ("c_loop", "        re.match(r'(a+)+$', 'a' * 64 + 'b')\n"),
        ("raises", "        raise ValueError('rare prefix')\n"),
        ("prints", "        for _ in range(200):\n            print('x' * 100_000)\n"),
    ],
)
def test_code_that_misbehaves_on_one_prefix_loses_that_row_only(tmp_path, case, body):
    code = ("import re\n" if case == "c_loop" else "") + _with(LIMA, body)
    llm = FakeLLM(extractor=lambda p, n: tagged(code))
    schema = schema_for(tmp_path, llm, timeout_s=2, compile_min_success=0.9)
    schema.sandbox = SubprocessSandbox(timeout_s=2)
    t0 = time.monotonic()
    result = schema.discover(runs(), TARGET)
    assert time.monotonic() - t0 < FAST
    if case == "prints":  # printing is no failure: what it prints is capped
        assert all(o is not None for o in result.extractor.outputs)
    else:
        assert sum(o is None for o in result.extractor.outputs) == 1  # the lima run, alone
    assert set(result.extractor.features) == {"user_intent", "intent_words", "plan_note"}


@pytest.mark.parametrize("kind", ["runaway", "spike"])
def test_a_memory_blowup_on_one_prefix_loses_that_row_only(tmp_path, kind):
    """A trajectory whose code keeps allocating (runaway) or allocates past the limit and finishes (spike)
    fails alone; the others run in a fresh interpreter. Linux enforces the limit as an address-space limit,
    elsewhere the interpreter watches its own peak memory."""
    body = {
        "runaway": "        hog = []\n        while True:\n            hog.append(b'x' * (8 << 20))\n",
        "spike": "        hog = b'x' * (500 << 20)\n",
    }[kind]
    box = SubprocessSandbox(timeout_s=20, mem_mb=300, chunk=16)
    prefixes = [dp.prefix for t in runs() for dp in adapter().decision_points(t, TARGET)]
    t0 = time.monotonic()
    res = box.run_batch(_with(LIMA, body), prefixes)
    assert time.monotonic() - t0 < FAST
    failed = [i for i, r in enumerate(res) if not r.ok]
    assert failed == [11]  # the lima run alone
    assert "MemoryError" in res[11].error or "exited early" in res[11].error
    assert peak_mb() < 2_000  # the parent never held it


def peak_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 20) if sys.platform == "darwin" else r / 1024


def test_nan_and_infinity_are_missing_values(tmp_path):
    code = (
        "intent = trajectory['intent']\n"
        "features = {\n"
        "    'user_intent': intent,\n"
        "    'ratio': float('nan') if 'weather' in intent else len(intent) / 7,\n"
        "    'scale': float('inf') if 'song' in intent else\n"
        "        -float('inf') if 'book' in intent else len(intent),\n"
        "}\n"
    )
    result = timed(lambda: discover(tmp_path, code_llm(code, "user_intent", "ratio", "scale")))
    outputs = [o for o in result.extractor.outputs if o]
    assert {o["ratio"] is None for o in outputs} == {True, False}
    assert {o["scale"] is None for o in outputs} == {True, False}
    assert "_tabagent_flatten" in result.extractor.code  # the same mapping runs at serving


# =========================================================================== many decision points
def wide_code() -> str:
    """Word counts: one column per word of the intent, which carries a word of its own on every run."""
    return (
        "words = {}\n"
        "for w in trajectory['intent'].split():\n"
        "    words[w] = words[w] + 1 if w in words else 1\n"
        "features = {'user_intent': trajectory['intent'], 'intent_words': words}\n"
    )


def unique_runs(n: int) -> list[Trajectory]:
    return runs(n, lambda i: {**raw_run(i), "intent": raw_run(i)["intent"] + f" ref{i}"})


def test_wide_outputs_on_many_decision_points_are_narrowed_before_the_gates(tmp_path, monkeypatch):
    """Columns x decision points past the memory budget: the columns are narrowed on a sample first, so the
    gates never hold the full table (here a budget of 100,000 cells stands in for the default 20 million)."""
    monkeypatch.setattr(C, "CELL_BUDGET", 100_000)
    llm = code_llm(wide_code(), "user_intent", "intent_words")
    result = timed(lambda: discover(tmp_path, llm, unique_runs(600), workers=4))
    ext = result.extractor
    assert ext.features["user_intent"] == "text" and len(ext.features) <= 1 + 64
    assert any("quality gates ran on statistics over every decision point" in n for n in ext.notes)
    # each run-specific word is a presence flag, decided from statistics; the per-feature cap keeps 64
    narrowed = [r for r in ext.dropped.values() if "max_columns_per_feature" in r or "(memory" in r]
    assert len(narrowed) > 400  # most of the 600 run-specific words
    assert all(set(o) <= set(ext.features) for o in ext.outputs if o)  # only kept columns are held


@pytest.mark.slow
def test_one_hundred_thousand_decision_points(tmp_path):
    """Bounded memory and time on 100,000 decision points with 100,000 distinct columns (word counts with a
    word of each run's own). Measured on a laptop: about 65 s and 1 GB of peak memory."""
    llm = code_llm(wide_code(), "user_intent", "intent_words")
    t0 = time.monotonic()
    result = discover(tmp_path, llm, unique_runs(100_000), workers=4)
    assert time.monotonic() - t0 < 600 and peak_mb() < 4_000
    assert len(result.extractor.features) <= 1 + 64 and result.report.n_decision_points == 100_000
    assert re.search(r"produced 1000\d\d distinct columns", " ".join(result.extractor.notes))


def test_on_linux_the_memory_budget_is_measured_over_the_loaded_payload(monkeypatch):
    """The address-space limit set before the payload is parsed only bounds the parse; the interpreter then
    tightens it to what it holds plus ``mem_mb``, so ``mem_mb`` is the extractor's own budget (as the macOS
    watch measures growth over the loaded payload)."""
    import json as _json

    box = SubprocessSandbox(timeout_s=2, mem_mb=300)
    monkeypatch.setattr(sys, "platform", "linux")
    linux = _json.loads(box._payload("features = {}", [{"steps": []}]))
    assert linux["mem_budget"] == 300 << 20 and linux["mem_watch"] is None
    monkeypatch.setattr(sys, "platform", "darwin")
    mac = _json.loads(box._payload("features = {}", [{"steps": []}]))
    assert mac["mem_budget"] is None and mac["mem_watch"] == 300 << 20
    unbounded = SubprocessSandbox(timeout_s=2, mem_mb=None)
    assert _json.loads(unbounded._payload("x = 1", []))["mem_budget"] is None
