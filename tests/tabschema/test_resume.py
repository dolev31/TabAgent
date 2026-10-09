"""Crash and resume at every stage boundary of TabSchema discovery.

A run is stopped in the middle of every stage: by an interruption raised from inside an LLM call (Ctrl+C), by
``SIGKILL`` of the whole process, by the budget (``max_cost_usd``), and by a provider that stays down.
Rerunning with the same run directory must reuse every paid answer, never pay for one twice, and produce
exactly the card and extractor of an uninterrupted run. Two processes on one run directory at once must not
corrupt it, nor pay twice for one answer.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from collections import Counter
from pathlib import Path

import pytest
from test_robustness import (
    TARGET,
    FakeLLM,
    adapter,
    config,
    runs,
    waits,  # noqa: F401 - autouse: backoff is recorded, not slept
)

from tabagent.tabschema import FeatureCard, TabSchema, agents, runstore
from tabagent.tabschema.llm import BudgetExceeded, LLMUnavailable
from tabagent.tabschema.runstore import RunStore, atomic_write, read_jsonl

HERE = Path(__file__).parent
PRICES = {"fake": (1.0, 2.0)}  # USD per 1M tokens: every call has a cost
#: Prices for the budget tests: the fake's answers are short, so a reservation that assumes a full answer
#: would stop a run early; pricing input only keeps each stop inside the targeted stage.
INPUT_ONLY = {"fake": (1.0, 0.0)}


class Recording(FakeLLM):
    """A FakeLLM that can stop the process after ``n`` calls of ``role``: ``mode`` is ``interrupt`` (raise
    KeyboardInterrupt, as Ctrl+C would), ``down`` (a transient provider error from then on) or ``kill``
    (SIGKILL the process)."""

    def __init__(self, role: str | None = None, n: int = 0, mode: str = "interrupt", **kw):
        super().__init__(**kw)
        self.stop_role, self.stop_n, self.mode = role, n, mode

    def complete(self, system, *, role=""):
        with self._lock:
            seen = sum(r == role for r, _ in self.calls)
        if role == self.stop_role and seen >= self.stop_n:
            if self.mode == "kill":
                os.kill(os.getpid(), signal.SIGKILL)
            if self.mode == "down":
                raise ConnectionResetError("connection reset by peer")
            raise KeyboardInterrupt
        return super().complete(system, role=role)

    def paid(self) -> Counter:
        return Counter(self.calls)


def schema(tmp_path: Path, llm, run_id: str = "run", **kw) -> TabSchema:
    cfg = config(**{"prices": PRICES, **kw})
    return TabSchema(llm, cfg, adapter(), store=RunStore(tmp_path / "runs", run_id))


def outputs(tmp_path: Path, run_id: str) -> tuple[dict, str]:
    run = tmp_path / "runs" / run_id
    return FeatureCard.load(run / "card.yaml").to_dict(), (run / "extract.py").read_text()


def without_cost(card: dict) -> dict:
    return {**card, "provenance": {k: v for k, v in card["provenance"].items() if k != "cost_usd"}}


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """An uninterrupted run: its paid calls, card and extractor, and the cost at each stage boundary."""
    base = tmp_path_factory.mktemp("reference")
    llm = Recording()
    result = schema(base, llm).discover(runs(), TARGET)
    summary = json.loads((base / "runs" / "run" / "run.json").read_text())
    return {
        "paid": llm.paid(),
        "roles": llm.roles(),
        "card": result.card.to_dict(),
        "code": result.extractor.code,
        "stages": summary["stages"],
    }


#: Where each kill lands: (role, calls of that role before it), one or more per stage. The three analyzed
#: points ask the CodeExtractor the same prompt (one paid answer); each point's CodeValidator prompt is its
#: own.
KILL_POINTS = [
    ("analyzer", 0),
    ("analyzer", 2),
    ("extractor", 0),
    ("validator", 1),
    ("aggregator", 0),
    ("judge_relevance", 0),
    ("judge_impact", 0),
    ("meta_judge", 0),
    ("extractor", 1),  # the compile step's code
    ("validator", 3),  # the compile step's review
    ("auditor", 0),
    ("auditor", 6),
]


def _join_new_threads(before: set, timeout: float = 60.0) -> None:
    """Wait for the threads started since ``before``, as the interpreter does at exit. A thread that was
    started but is not running yet cannot be joined (RuntimeError); it is retried until it runs or the time is
    up."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        new = [t for t in threading.enumerate() if t not in before and t is not threading.current_thread()]
        if not new:
            return
        for t in new:
            try:
                t.join(max(0.0, min(1.0, deadline - time.monotonic())))
            except RuntimeError:  # started, not yet running
                time.sleep(0.01)


@pytest.mark.parametrize("role, n", KILL_POINTS)
def test_an_interrupted_run_resumes_to_the_same_card_without_paying_twice(tmp_path, reference, role, n):
    if reference["roles"][role] <= n:
        pytest.skip(f"the reference run makes only {reference['roles'][role]} {role} calls")
    first = Recording(role, n)
    before = set(threading.enumerate())
    with pytest.raises(KeyboardInterrupt):
        schema(tmp_path, first).discover(runs(), TARGET)
    # The run returns without waiting for a thread whose LLM call was under way at the interruption (it stores
    # that answer, then stops); the interpreter waits for such threads at exit, and so does the test.
    _join_new_threads(before)
    run = tmp_path / "runs" / "run"
    summary = json.loads((run / "run.json").read_text())
    assert summary["status"] == "stopped" and summary["stopped_by"] == "KeyboardInterrupt"
    assert any(r.get("status") == "stopped" for r in summary["stages"].values())
    assert (run / "report.md").exists() and not list(run.rglob("*.tmp"))

    second = Recording()
    result = schema(tmp_path, second).discover(runs(), TARGET)
    assert result.card.to_dict() == reference["card"] and result.extractor.code == reference["code"]
    assert first.paid() + second.paid() == reference["paid"]  # every answer paid once, none twice
    assert not first.paid() & second.paid()
    sessions = json.loads((run / "run.json").read_text())["sessions"]
    assert [s["status"] for s in sessions] == ["stopped", "completed"]


def test_an_interrupt_waits_for_the_files_being_written_not_for_the_llm_calls(tmp_path, monkeypatch):
    """Ctrl+C in a parallel stage returns once the answers being stored are written (no temporary file is left
    behind), without waiting for a thread whose LLM call has not answered yet."""
    writing, answered, tmp = threading.Event(), threading.Event(), []
    replace = os.replace

    def slow_replace(src, dst):  # a slow disk: the rename lands 0.3 s into the write
        if Path(dst).name == "answer.json":
            tmp.append(Path(src).name)
            writing.set()
            time.sleep(0.3)
        replace(src, dst)

    monkeypatch.setattr(runstore.os, "replace", slow_replace)

    def work(i):
        if i == 0:  # Ctrl+C, while the next thread is storing its answer
            writing.wait(10)
            raise KeyboardInterrupt
        if i == 1:
            atomic_write(tmp_path / "answer.json", "{}")
        else:
            answered.wait(10)  # an LLM call still waiting for its answer

    t0 = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            agents.in_parallel(work, [0, 1, 2], workers=3)
        assert (tmp_path / "answer.json").exists() and not list(tmp_path.glob("*.tmp"))
        assert time.monotonic() - t0 < 1.5
        assert tmp[0].startswith(f".answer.json.{os.getpid()}.")  # names its writer (stale-file cleanup)
    finally:
        answered.set()
        agents.STOP.clear()


def test_an_interrupted_run_starts_no_new_llm_call():
    """After Ctrl+C, a thread the run left running stops at its next paid call instead of making it."""
    llm = FakeLLM()
    agents.STOP.set()
    try:
        with pytest.raises(KeyboardInterrupt):
            agents.FeatureAggregator(llm).call("a prompt")
    finally:
        agents.STOP.clear()
    assert not llm.calls


@pytest.mark.parametrize("stage", ["analyze", "aggregate", "judge", "meta_judge", "compile", "audit"])
def test_a_budget_stop_inside_any_stage_is_clean_and_resumable(tmp_path, reference, stage):
    """The budget stops the run before a call that could cross it; the run directory holds every paid answer,
    and the rerun with a larger budget pays only for the rest."""
    order = list(reference["stages"])

    def usd(name: str) -> float:  # the stage's cost at INPUT_ONLY prices
        return (reference["stages"][name].get("llm") or {}).get("input_tokens", 0) / 1e6

    before = sum(usd(name) for name in order[: order.index(stage)])
    inside = usd(stage)
    assert inside > 0
    first = Recording()
    with pytest.raises(BudgetExceeded, match="max_cost_usd.*resume"):
        schema(tmp_path, first, max_cost_usd=before + inside / 2, workers=1, prices=INPUT_ONLY).discover(
            runs(), TARGET
        )
    summary = json.loads((tmp_path / "runs" / "run" / "run.json").read_text())
    assert summary["status"] == "stopped" and summary["stopped_by"] == "BudgetExceeded"
    assert summary["stages"][stage]["status"] == "stopped"
    second = Recording()
    result = schema(tmp_path, second, max_cost_usd=100.0, prices=INPUT_ONLY).discover(runs(), TARGET)
    assert without_cost(result.card.to_dict()) == without_cost(reference["card"])  # priced differently
    assert result.extractor.code == reference["code"]
    assert first.paid() + second.paid() == reference["paid"]


@pytest.mark.parametrize("role", ["analyzer", "aggregator", "judge_generality", "meta_judge", "auditor"])
def test_a_provider_that_stays_down_stops_the_run_resumably(tmp_path, reference, role):
    first = Recording(role, 1 if role in ("analyzer", "auditor") else 0, mode="down")
    with pytest.raises(LLMUnavailable, match="rerun with that run directory to resume"):
        schema(tmp_path, first).discover(runs(), TARGET)
    run = tmp_path / "runs" / "run"
    assert json.loads((run / "run.json").read_text())["status"] == "stopped"
    events = [r for r in read_jsonl(run / "llm_calls.jsonl") if r.get("event")]
    assert {e["event"] for e in events} == {"retry", "error"} and all(e["stage"] == role for e in events)
    second = Recording()
    result = schema(tmp_path, second).discover(runs(), TARGET)
    assert result.card.to_dict() == reference["card"]
    assert first.paid() + second.paid() == reference["paid"]


# --------------------------------------------------------------------------- real processes
CHILD = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {here!r})
    import test_robustness as R
    from test_resume import Recording, schema
    from tabagent.tabschema import agents
    from pathlib import Path

    agents._sleep = lambda s: None
    root, role, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    mode, delay = sys.argv[4], float(sys.argv[5])


    class Slow(Recording):
        def complete(self, system, *, role=""):
            time.sleep(delay)
            return super().complete(system, role=role)


    # one call at a time when killing: a call that another thread completed in the instant of the kill, paid
    # but not yet written, would be paid again (the one window a kill leaves; see docs/onboarding.md)
    workers = 1 if mode == "kill" else 2
    schema(root, Slow(role or None, n, mode), workers=workers).discover(R.runs(), R.TARGET)
    print("done")
    """
)


def child(
    tmp_path: Path, role: str = "", n: int = 0, mode: str = "kill", delay: float = 0.0
) -> subprocess.Popen:
    script = tmp_path / "child.py"
    script.write_text(CHILD.format(here=str(HERE)))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")])}
    return subprocess.Popen(
        [sys.executable, str(script), str(tmp_path), role, str(n), mode, str(delay)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def paid_by_role(run: Path) -> Counter:
    return Counter(r["stage"] for r in read_jsonl(run / "llm_calls.jsonl") if not r.get("event"))


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL is POSIX")
@pytest.mark.parametrize("role, n", [("analyzer", 1), ("aggregator", 0), ("meta_judge", 0), ("extractor", 1)])
def test_a_killed_process_leaves_a_run_directory_that_resumes(tmp_path, reference, role, n):
    proc = child(tmp_path, role, n)
    out, err = proc.communicate(timeout=300)
    assert proc.returncode == -signal.SIGKILL, err[-2000:]
    run = tmp_path / "runs" / "run"
    for f in run.rglob("*.json"):  # every file it left is complete
        json.loads(f.read_text())
    assert all(f.name.startswith(".") for f in run.rglob("*.tmp"))  # a write cut short: hidden, ignored
    killed_paid = paid_by_role(run)
    second = Recording()
    result = schema(tmp_path, second).discover(runs(), TARGET)
    assert result.card.to_dict() == reference["card"] and result.extractor.code == reference["code"]
    assert not list(run.rglob("*.tmp"))  # the killed writer's were removed when the run directory was opened
    assert killed_paid + second.roles() == reference["roles"]  # nothing paid twice
    sessions = json.loads((run / "run.json").read_text())["sessions"]
    assert [s["status"] for s in sessions] == ["killed", "completed"]


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl locks are POSIX")
def test_two_processes_on_one_run_directory_share_answers_and_corrupt_nothing(tmp_path, reference):
    a, b = child(tmp_path, mode="none", delay=0.02), child(tmp_path, mode="none", delay=0.02)
    for p in (a, b):
        out, err = p.communicate(timeout=600)
        assert p.returncode == 0 and "done" in out, err[-2000:]
    run = tmp_path / "runs" / "run"
    assert paid_by_role(run) == reference["roles"]  # each answer paid by one process, read by the other
    for f in run.rglob("*.json"):
        json.loads(f.read_text())
    assert FeatureCard.load(run / "card.yaml").to_dict() == reference["card"]
    sessions = json.loads((run / "run.json").read_text())["sessions"]
    assert len(sessions) == 2 and all(s["status"] == "completed" for s in sessions)  # merged, not lost


def test_a_kill_between_two_runs_marks_the_dead_session(tmp_path):
    """A session that ended without closing (killed) is marked ``killed`` by the next one."""
    store = RunStore(tmp_path / "runs", "run")
    (store.dir / "run.json").write_text(
        json.dumps(
            {"sessions": [{"id": "x", "pid": 2**22 + 7, "host": socket.gethostname(), "status": "running"}]}
        )
    )
    second = Recording()
    TabSchema(second, config(), adapter(), store=store).discover(runs(), TARGET)
    sessions = json.loads((store.dir / "run.json").read_text())["sessions"]
    assert [s["status"] for s in sessions] == ["killed", "completed"]


def test_progress_is_logged_with_a_time_estimate(tmp_path, caplog):
    caplog.set_level("INFO", logger="tabagent")
    schema(tmp_path, Recording(), n_analyze=4).discover(runs(), TARGET)
    lines = [r.getMessage() for r in caplog.records if "analyzed decision points" in r.getMessage()]
    assert lines[-1].endswith("4/4") and any("left" in line for line in lines[:-1])
    stages = json.loads((tmp_path / "runs" / "run" / "run.json").read_text())["stages"]
    assert stages["analyze"]["progress"] == {"done": 4, "total": 4, "failed": 0}
