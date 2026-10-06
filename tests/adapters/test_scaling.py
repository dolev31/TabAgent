"""Scaling: long runs take memory linear in their length (decision points share their prefix steps), and a
large input is not held in memory (runs are read again from their files) while giving the same results."""

from __future__ import annotations

import json
import time
import tracemalloc

from fake_logs import cuga_trajectory

from tabagent.adapters import NEXT_TOOL_CALL, SHORTLISTER, CugaAdapter, OpenAIMessagesAdapter, live_prefix
from tabagent.adapters import logfiles as lf
from tabagent.adapters.live import run_before
from tabagent.tabschema.trajectory import GenericJsonAdapter, Trajectory


def _peak_mb(fn):
    tracemalloc.start()
    try:
        t0 = time.perf_counter()
        out = fn()
        seconds = time.perf_counter() - t0
        return out, tracemalloc.get_traced_memory()[1] / 1e6, seconds
    finally:
        tracemalloc.stop()


def _long_chat(turns: int) -> dict:
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "go " * 50}]
    for k in range(turns):
        call = {"id": f"c{k}", "type": "function", "function": {"name": f"tool_{k % 7}", "arguments": "{}"}}
        msgs += [
            {"role": "assistant", "content": f"step {k}", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": f"c{k}", "content": "result " * 20},
        ]
    return {"task_id": "t", "messages": msgs, "reward": 1}


def test_a_long_chat_run_is_linear_in_memory():
    """1,500 tool-calling turns: 1,500 prefixes of up to 3,000 steps. Deep copies took about 1 GB here; shared
    steps take the prefixes' lists of references (tens of MB)."""
    adapter = OpenAIMessagesAdapter("step")
    traj = Trajectory("long", _long_chat(1500))

    def build():
        dps = adapter.decision_points(traj, NEXT_TOOL_CALL)
        return dps, [adapter.labels(dp) for dp in dps], adapter.candidates(dps[-1])

    (dps, labels, cands), peak, seconds = _peak_mb(build)
    assert len(dps) == 1500 and labels[-1] == {"tool_1"} and len(cands) == 7
    assert peak < 80, f"{peak:.0f} MB"
    assert seconds < 20
    last = dps[-1]
    assert live_prefix(adapter, run_before(adapter, traj, last), NEXT_TOOL_CALL) == last.prefix


def test_a_long_step_run_is_linear_in_memory():
    steps = [
        {"name": "router", "data": "x" * 100, "prompts": [{"role": "user", "value": "p"}]}
        for _ in range(1500)
    ]
    adapter = GenericJsonAdapter(labels_fn=lambda dp, raw: {"x"})
    traj = Trajectory("long", {"task_id": "t", "steps": steps, "score": 1})
    dps, peak, seconds = _peak_mb(lambda: adapter.decision_points(traj, "router"))
    assert len(dps) == 1500 and len(dps[-1].prefix["steps"]) == 1499
    assert peak < 80, f"{peak:.0f} MB"
    assert seconds < 20, f"{seconds:.1f} s"


def test_a_long_cuga_run_labels_each_shortlisting_once():
    rounds = [("phone", [f"phone_api{k}"], [f"phone_api{k}"]) for k in range(150)]
    traj = Trajectory("long", cuga_trajectory("t", rounds))
    adapter = CugaAdapter()

    def build():
        dps = adapter.decision_points(traj, SHORTLISTER)
        return [adapter.labels(dp) for dp in dps], [adapter.candidates(dp) for dp in dps]

    (labels, cands), peak, seconds = _peak_mb(build)
    assert labels[42] == {"phone_api42"} and len(cands[0]) == 3
    assert seconds < 20


def _big_cuga_logs(folder, n=12, prompt_chars=200_000):
    folder.mkdir()
    for i in range(n):
        run = cuga_trajectory(f"t{i}", [("phone", ["phone_search_contacts"], ["phone_search_contacts"])])
        for s in run["steps"]:
            s["prompts"] = [
                *s.get("prompts", []),
                {"role": "system", "value": f"run {i} " + "lorem " * prompt_chars},
            ]
        (folder / f"t{i}.json").write_text(json.dumps(run))
    return folder


def _onboarding_inputs(adapter, logs):
    trajs = adapter.load(logs)
    good = [t for t in trajs if t.success]
    dps = [dp for t in good for dp in adapter.decision_points(t, "TaskAnalyzerAgent")]
    return trajs, [
        (dp.id, dp.prefix, sorted(adapter.labels(dp)), [c.id for c in adapter.candidates(dp)]) for dp in dps
    ]


def test_a_large_input_is_not_held_in_memory_and_gives_the_same_results(tmp_path, monkeypatch):
    """The memory benchmark in small: 12 runs of about 14 MB of text. Read lazily, an onboarding's inputs
    (the trajectories and the decision points of an early decision) hold a fraction of the logs; the decision
    points, labels and candidates are the same as when every run is held in memory."""
    logs = _big_cuga_logs(tmp_path / "logs")
    (eager_trajs, eager), eager_mb, _ = _peak_mb(lambda: _onboarding_inputs(CugaAdapter(), logs))
    monkeypatch.setattr(lf, "LAZY_INPUT_BYTES", 0)
    monkeypatch.setattr(lf, "RUN_CACHE_BYTES", 0)  # the few parsed runs a load keeps are not held here
    tracemalloc.start()
    try:
        lazy_trajs, lazy = _onboarding_inputs(CugaAdapter(), logs)
        held_mb = tracemalloc.get_traced_memory()[0] / 1e6
    finally:
        tracemalloc.stop()
    assert lazy == eager and [t.raw for t in lazy_trajs] == [t.raw for t in eager_trajs]
    assert all(t.is_lazy for t in lazy_trajs) and not any(t.is_lazy for t in eager_trajs)
    assert eager_mb > 100 and held_mb < eager_mb / 20, (eager_mb, held_mb)


def test_a_huge_single_line_is_read(tmp_path):
    run = _long_chat(10)
    run["messages"][1]["content"] = "x" * 30_000_000
    (tmp_path / "big.jsonl").write_text(json.dumps(run) + "\n" + json.dumps(_long_chat(2)) + "\n")
    trajs, peak, _ = _peak_mb(lambda: OpenAIMessagesAdapter().load(tmp_path))
    assert len(trajs) == 2


def test_one_large_jsonl_file_is_read_one_line_at_a_time(tmp_path, monkeypatch):
    """A single large JSON Lines file (many chat runs) is streamed: loading it lazily never holds it whole."""
    run = _long_chat(20)
    run["messages"][1]["content"] = "words " * 50_000  # about 300 KB per run
    with (tmp_path / "runs.jsonl").open("w") as fh:
        for k in range(60):
            fh.write(json.dumps({**run, "task_id": f"t{k}"}) + "\n")
    monkeypatch.setattr(lf, "LAZY_INPUT_BYTES", 0)
    trajs, peak, _ = _peak_mb(lambda: OpenAIMessagesAdapter().load(tmp_path))
    assert len(trajs) == 60 and all(t.is_lazy for t in trajs)
    assert peak < 8, f"{peak:.1f} MB to load an 18 MB file"
    assert trajs[59].raw["task_id"] == "t59" and trajs[0].raw == {**run, "task_id": "t0"}
