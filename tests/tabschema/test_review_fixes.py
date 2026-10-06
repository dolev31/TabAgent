"""Regression tests for the defects an independent review of the TabSchema hardening found: a fallback model
added after a filtered stop, replay costs, raised output limits, MetaJudge batches, the memory narrowing of
wide outputs, file modes, the macOS memory watch, platforms without ``resource`` or ``os.waitid``, auditor
provider errors, mixed failures, interruptions during backoff, slow chunks, config types and fold budgets."""

from __future__ import annotations

import json
import logging
import os
import random
import signal
import sys
import threading
import time

import pytest
from test_chaos import ProviderError
from test_robustness import (
    DEFAULT,
    EXPRESSIONS,
    TARGET,
    FakeLiteLLMModule,
    FakeLLM,
    adapter,
    config,
    discover,
    names_in,
    raw_run,
    runs,
    waits,  # noqa: F401 - autouse: backoff is recorded, not slept
)

from tabagent import cli
from tabagent.tabschema import TabSchema, TabSchemaConfig, agents, judges
from tabagent.tabschema import compile as C
from tabagent.tabschema.agents import FeatureSpec
from tabagent.tabschema.llm import BudgetExceeded, LiteLLM, LLMContentFiltered, LLMUnavailable
from tabagent.tabschema.runstore import UMASK, RunStore
from tabagent.tabschema.sandbox import _RUNNER, SubprocessSandbox

TRIVIAL = "features = {'n': len(trajectory['steps'])}\n"


def make(tmp_path, llm, run_id="r1", stage_llms=None, **kw) -> TabSchema:
    return TabSchema(
        llm, config(**kw), adapter(), store=RunStore(tmp_path / "runs", run_id), stage_llms=stage_llms
    )


def filtered_aggregator() -> FakeLLM:
    return FakeLLM(aggregator=lambda p, n: ("", "content_filter"))


# --------------------------------------------------------------------------- H2, L5: fallback and replay
def test_a_fallback_added_after_a_filtered_stop_is_asked_on_the_rerun(tmp_path):
    with pytest.raises(LLMContentFiltered, match="fallback_model"):
        make(tmp_path, filtered_aggregator()).discover(runs(), TARGET)
    fallback = FakeLLM(model="fallback")
    rerun = filtered_aggregator()
    result = make(tmp_path, rerun, stage_llms={"fallback": fallback}, fallback_model="fallback").discover(
        runs(), TARGET
    )
    assert set(result.extractor.features) == set(EXPRESSIONS)
    assert fallback.roles()["aggregator"] == 1 and "aggregator" not in rerun.roles()  # filter replayed free
    again = FakeLLM(model="fallback")
    make(tmp_path, filtered_aggregator(), stage_llms={"fallback": again}, fallback_model="fallback").discover(
        runs(), TARGET
    )
    assert again.calls == []  # the fallback's answer is stored under its own key


def test_a_full_replay_counts_the_same_cost_and_writes_the_same_card(tmp_path):
    prices = {"fake": (1000.0, 1000.0), "fallback": (1000.0, 1000.0)}

    def schema() -> TabSchema:
        primary = FakeLLM(
            analyzer=lambda p, n: ("", "content_filter") if "weather" in p else DEFAULT["analyzer"](p, n)
        )
        return make(
            tmp_path,
            primary,
            prices=prices,
            fallback_model="fallback",
            stage_llms={"fallback": FakeLLM(model="fallback")},
        )

    first = schema().discover(runs(), TARGET)
    card1 = (tmp_path / "runs" / "r1" / "card.yaml").read_bytes()
    replay = schema()
    second = replay.discover(runs(), TARGET)
    assert replay.meter.tokens()[0] == 0  # nothing paid now
    assert first.card.provenance["cost_usd"] == second.card.provenance["cost_usd"] > 0
    assert (tmp_path / "runs" / "r1" / "card.yaml").read_bytes() == card1


# --------------------------------------------------------------------------- M1: a raised output limit
def test_a_raised_output_limit_asks_again_only_the_truncated_answers(tmp_path):
    def analyzer(p, n):
        if "weather" in p:
            return '{"potential_features": [', "length"
        return DEFAULT["analyzer"](p, n)

    small = FakeLLM(analyzer=analyzer)
    small.max_tokens = 1_000
    first = make(tmp_path, small, n_analyze=8).discover(runs(), TARGET)
    lost = [a for a in first.report.analyzed if a["error"]]
    assert lost and all("LLMTruncated" in a["error"] for a in lost)
    same = FakeLLM()
    same.max_tokens = 1_000
    make(tmp_path, same, n_analyze=8).discover(runs(), TARGET)
    assert "analyzer" not in same.roles()  # the same limit: the truncation is replayed, not paid again
    bigger = FakeLLM()
    bigger.max_tokens = 4_000
    second = make(tmp_path, bigger, n_analyze=8).discover(runs(), TARGET)
    assert bigger.roles()["analyzer"] == len(lost)  # only the truncated ones are asked again
    assert all("weather" in p for p in bigger.prompts("analyzer"))
    assert not [a for a in second.report.analyzed if a["error"]]


# --------------------------------------------------------------------------- M2: MetaJudge batches
def test_a_metajudge_decision_on_another_batch_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(judges.MetaJudge, "batch", 1)

    def meta(p, n):
        names = names_in(p)
        out = [{"feature_name": x, "final_decision": "accept"} for x in names]
        if n == 0:  # the first batch also decides on every other feature
            out += [{"feature_name": x, "final_decision": "reject"} for x in EXPRESSIONS if x not in names]
        return json.dumps(out)

    result = make(tmp_path, FakeLLM(meta_judge=meta), workers=1).discover(runs(), TARGET)
    assert {j["final_decision"] for j in result.report.judged} == {"accept"}
    events = json.loads((tmp_path / "runs" / "r1" / "judges.json").read_text())["events"]
    assert any(e["event"] == "ignored" and e["stage"] == "meta_judge" for e in events)


# --------------------------------------------------------------------------- M3: wide outputs, many points
def test_rare_flags_survive_the_memory_narrowing(monkeypatch):
    """6,000 decision points (30 per run) and 362 presence-flag columns over a budget that allows the table
    333 of them: flags are decided from statistics over every point, so a 0.4% flag and a flag set only on
    each run's last decision are kept, as the full gates keep them."""
    monkeypatch.setattr(C, "CELL_BUDGET", 2_000_000)
    rng = random.Random(0)
    runs_, per = 200, 30
    by_key = {f"flag_group_{g}": FeatureSpec(f"flag_group_{g}", "boolean") for g in range(6)}
    freq = {f"flag_group_{g}_{j}": 0.01 + 0.2 * rng.random() for g in range(6) for j in range(60)}
    outputs: list[dict] = []
    for _ in range(runs_):
        for k in range(per):
            row = {c: True for c, p in freq.items() if rng.random() < p}
            if rng.random() < 0.004:
                row["flag_group_0_refund_requested"] = True
            if k == per - 1:
                row["flag_group_1_final_confirmation"] = True
            outputs.append(row)
    outputs[5]["flag_group_2_constant"] = False  # only ever false: constant
    keys = list(dict.fromkeys(k for o in outputs for k in o))
    result = C.CompiledExtractor("", {})
    kept, flags, _ = C._precap(result, keys, outputs, by_key, set(), {}, 0.5, 64, set(), seed=0)
    assert "flag_group_0_refund_requested" in flags and "flag_group_1_final_confirmation" in flags
    assert result.dropped["flag_group_2_constant"].startswith("constant")
    assert len(flags) == len(keys) - 1 and not kept


def test_the_narrowing_ranks_on_a_seeded_random_sample_and_says_when_memory_binds(monkeypatch):
    monkeypatch.setattr(C, "CELL_BUDGET", 50_000)
    by_key = {"word": FeatureSpec("word", "numeric")}
    outputs = [{f"word_{j}": (i * (j + 1)) % 7 for j in range(300)} for i in range(1000)]
    keys = list(outputs[0])
    result = C.CompiledExtractor("", {})
    kept, flags, _ = C._precap(result, keys, outputs, by_key, set(), {}, 0.5, 64, set(), seed=0)
    assert not flags and len(kept) == 64  # max(64, 50,000 / 1,000): the memory cap
    assert any("memory cap bound" in n for n in result.notes)
    again = C.CompiledExtractor("", {})
    assert C._precap(again, keys, outputs, by_key, set(), {}, 0.5, 64, set(), seed=0)[0] == kept


# --------------------------------------------------------------------------- M5: file modes
def test_written_files_get_the_ordinary_mode(tmp_path):
    result = discover(tmp_path, FakeLLM())
    want = 0o666 & ~UMASK
    for name in ("card.yaml", "run.json", "report.md", "extract.py"):
        assert (tmp_path / "runs" / "r1" / name).stat().st_mode & 0o777 == want, name
    out = result.save(tmp_path / "saved")
    for name in ("extract.py", "card.yaml", "discovery.md", "discovery.json"):
        assert (out / name).stat().st_mode & 0o777 == want, name


# --------------------------------------------------------------------------- M6, H1, L11: the sandbox
def big_prefix(mb: float) -> dict:
    step = {
        "role": "tool",
        "name": "search",
        "content": "lorem ipsum dolor sit amet " * 40,
        "args": {"q": "x"},
    }
    per = len(json.dumps(step))
    return {"steps": [dict(step, i=i) for i in range(int(mb * (1 << 20) / per))]}


@pytest.mark.skipif(sys.platform == "win32", reason="memory limits are POSIX")
def test_a_large_payload_does_not_count_against_the_extractors_memory():
    """20 prefixes of 4 MB (80 MB of JSON) with a 200 MB memory limit: the payload is split by bytes, and the
    limit measures the extractor's growth over the loaded payload, so every trivial trajectory passes."""
    items = [big_prefix(4.0) for _ in range(20)]
    rs = SubprocessSandbox(timeout_s=60, mem_mb=200, chunk=50).run_batch(TRIVIAL, items)
    assert [r.ok for r in rs] == [True] * 20, rs[0].error


class NoResource(SubprocessSandbox):
    """A platform without the ``resource`` module (Windows), simulated."""

    def _command(self):
        return [self.python, "-I", "-S", "-c", "import sys; sys.modules['resource'] = None\n" + _RUNNER]


def test_the_sandbox_runs_without_the_resource_module():
    items = [{"steps": [1, 2]} for _ in range(3)]
    assert [r.ok for r in NoResource(timeout_s=10).run_batch(TRIVIAL, items)] == [True] * 3
    assert [r.ok for r in NoResource(timeout_s=10, mem_mb=None).run_batch(TRIVIAL, items)] == [True] * 3


def test_slow_but_legal_trajectories_finish_across_interpreters():
    """Each trajectory takes 0.3 s (its limit is 1 s); the chunk's wall-clock limit (2 s) stops the
    interpreter several times. Restarts that made progress do not count against ``max_restarts``."""
    slow = (
        "import time\n"
        "t = time.monotonic()\n"
        "while time.monotonic() - t < 0.3:\n"
        "    pass\n"
        "features = {'i': trajectory['i']}\n"
    )
    items = [{"i": i} for i in range(16)]
    box = SubprocessSandbox(timeout_s=1, chunk=16, max_restarts=1)
    box._wall = lambda n: 1.2  # a short wall-clock budget per interpreter: about 4 trajectories each
    rs = box.run_batch(slow, items)
    assert [r.ok for r in rs] == [True] * 16 and [r.features["i"] for r in rs] == list(range(16))


def test_the_session_check_never_signals_a_process_on_windows(monkeypatch):
    from tabagent.tabschema import runlog

    monkeypatch.setattr(runlog.sys, "platform", "win32")
    monkeypatch.setattr(runlog.os, "kill", lambda *a: pytest.fail("os.kill would send Ctrl+C on Windows"))
    assert runlog._pid_alive(os.getpid()) is True  # unknown there without ctypes.windll: kept as alive


# --------------------------------------------------------------------------- M7, L7: the auditor
def test_an_auditor_provider_error_stops_the_run(tmp_path):
    def bad_key(p, n):
        raise ProviderError(401, "Incorrect API key provided")

    with pytest.raises(ProviderError, match="Incorrect API key"):
        discover(tmp_path, FakeLLM(auditor=bad_key))
    assert json.loads((tmp_path / "runs" / "r1" / "run.json").read_text())["status"] == "failed"


def test_a_filtered_audit_prompt_is_clipped_not_split(tmp_path, monkeypatch):
    """A filter on an audit prompt is about the prefix: it is sent again clipped to half the budget (here the
    clipping removes the word the filter reacts to); the point's features are never split for it."""
    from tabagent.tabschema import audit

    monkeypatch.setattr(audit.OutputAuditor, "max_chars", 3_000)

    def auditor(p, n):
        return ("", "content_filter") if "FILTERME" in p else DEFAULT["auditor"](p, n)

    def long_run(i):
        raw = raw_run(i)
        raw["steps"][0]["data"] += " " + "x" * 1800 + " FILTERME " + "y" * 1000
        return raw

    llm = FakeLLM(auditor=auditor)
    result = discover(tmp_path, llm, runs(16, long_run))
    assert result.report.audit["answered"] == result.report.audit["samples"] > 0
    audit_events = json.loads((tmp_path / "runs" / "r1" / "run.json").read_text())["stages"]["audit"][
        "events"
    ]
    assert audit_events.get("auditor: clipped") and "auditor: split" not in audit_events


# --------------------------------------------------------------------------- L6: mixed failures
def test_each_failed_part_forgets_by_its_own_error(tmp_path):
    def aggregator(p, n):
        names = names_in(p)
        if names == ["plan_note"] or len(names) == 1 and "plan_note" in names:
            return "", "content_filter"
        return DEFAULT["aggregator"](p, n)[:30], "length"

    llm = FakeLLM(aggregator=aggregator)
    with pytest.raises(Exception, match="aggregator"):
        discover(tmp_path, llm)
    store = RunStore(tmp_path / "runs", "r1")
    stage = agents.FeatureAggregator(llm, store=store)
    kept = {p: store.peek("llm", stage._key(p)) for p in llm.prompts("aggregator")}
    filtered = [p for p in kept if names_in(p) == ["plan_note"]]
    truncated = [p for p in kept if p not in filtered]
    assert filtered and all(kept[p] is not None for p in filtered)  # a filter is kept: it would recur
    assert truncated and all(kept[p] is None for p in truncated)  # truncated answers are asked again


# --------------------------------------------------------------------------- L9: Ctrl+C during an outage
@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="POSIX signals")
def test_an_interrupt_during_backoff_returns_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(agents, "_sleep", agents._wait)  # real, interruptible backoff
    monkeypatch.setattr(agents.Stage, "retry", agents.RetryPolicy(base_s=30, max_s=60))

    def down(p, n):
        raise ConnectionResetError("connection reset by peer")

    main = threading.main_thread().ident
    timer = threading.Timer(0.5, lambda: signal.pthread_kill(main, signal.SIGINT))
    t0 = time.monotonic()
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            discover(tmp_path, FakeLLM(analyzer=down), workers=2)
    finally:
        timer.cancel()
    assert time.monotonic() - t0 < 3
    assert json.loads((tmp_path / "runs" / "r1" / "run.json").read_text())["status"] == "stopped"
    agents.STOP.clear()


# --------------------------------------------------------------------------- L2, L3, L4, L10, L12
def test_config_accepts_numpy_numbers_and_makes_them_python_ones():
    np = pytest.importorskip("numpy")
    cfg = TabSchemaConfig(n_analyze=np.int64(5), audit=np.bool_(False), max_null_rate=np.float32(0.25))
    assert (type(cfg.n_analyze), type(cfg.audit), type(cfg.max_null_rate)) == (int, bool, float)
    with pytest.raises(ValueError, match="request_timeout_s"):
        TabSchemaConfig(request_timeout_s=0)


def test_a_fold_without_budget_left_is_a_budget_stop(tmp_path):
    from tabagent.evaluation.discovery import cross_validate_discovery

    with pytest.raises(BudgetExceeded):
        cross_validate_discovery(
            runs(),
            TARGET,
            adapter=adapter(),
            llm=FakeLLM(),
            config=config(max_cost_usd=0.0, prices={"fake": (1.0, 1.0)}),
            run_dir=tmp_path / "cv",
            n_splits=2,
        )


def test_the_api_baseline_retries_its_transient_errors(monkeypatch):
    pytest.importorskip("openai")
    baselines = pytest.importorskip("experiments.baselines.run")  # needs the baselines' own extras

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backends = baselines.Backends.__new__(baselines.Backends)
    backends.model, backends.args = "gpt-x", type("A", (), {"dry_run": False, "device": "cpu"})()
    assert backends._make("api").max_retries == 4


def test_a_global_litellm_retry_setting_is_warned_about_once(monkeypatch, caplog):
    from tabagent.tabschema import llm as llm_module

    fake = FakeLiteLLMModule()
    fake.num_retries = 3
    monkeypatch.setitem(sys.modules, "litellm", fake)
    monkeypatch.setattr(llm_module, "_HIDDEN_RETRIES_WARNED", [])
    caplog.set_level(logging.WARNING)
    lite = LiteLLM("anthropic/claude-x")
    lite.complete("PROMPT", role="analyzer")
    lite.complete("PROMPT", role="analyzer")
    assert sum("litellm.num_retries=3" in r.getMessage() for r in caplog.records) == 1


def test_auto_llm_passes_the_request_timeout(monkeypatch):
    pytest.importorskip("openai")
    from tabagent.tabschema.llm import auto_llm

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    assert auto_llm("gpt-x", timeout=42).timeout == 42 and auto_llm("gpt-x").timeout is None


def test_a_resumable_stop_keeps_its_status_under_vv(monkeypatch, capsys):
    def stop(args):
        raise LLMUnavailable("provider down; rerun with the same run directory to resume")

    monkeypatch.setattr(cli, "_runs", stop)
    assert cli.main(["-vv", "runs", "list", "x"]) == 3
    err = capsys.readouterr().err
    assert "Traceback" in err and "stopped: provider down" in err


def test_dead_estimators_are_gone():
    import tabagent.api as api
    from tabagent.tabschema import audit

    assert not hasattr(api, "_estimate") and not hasattr(audit, "estimate")
