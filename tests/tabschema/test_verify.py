"""The release's checks of the values a compiled TabSchema extractor computes (not in the paper).

Compile-time verification (determinism, outcome and future invariance, robustness), the OutputAuditor, the
golden tests shipped with the extractor (``extractor_tests.json``, ``tabagent verify-head``) and the
usefulness estimate. Every test drives the real pipeline with scripted fake LLMs: no network, no credentials.
"""

from __future__ import annotations

import dataclasses
import json
import random
import re

import pytest
from test_robustness import (
    TARGET,
    TOOLS,
    FakeLLM,
    adapter,
    config,
    names_in,
    raw_run,
    runs,
    schema_for,
    spec,
    tagged,
)

import tabagent
from tabagent import cli
from tabagent.tabschema import TabSchemaConfig, Trajectory, prompts
from tabagent.tabschema.audit import OutputAuditor, normalize_audit_verdict, run_audit
from tabagent.tabschema.compile import context_texts
from tabagent.tabschema.llm import CostMeter
from tabagent.tabschema.runstore import RunStore
from tabagent.tabschema.trajectory import GenericJsonAdapter
from tabagent.tabschema.verify import (
    FUTURE_MARK,
    GOLDEN_STRING_CHARS,
    TESTS_FILE,
    PrefixLeakError,
    changed_futures,
    make_tests,
    mask_secrets,
    verify_head,
    verify_tests,
)


def proposing(*features: tuple[str, str]):
    """An analyzer answer proposing ``(name, type)`` features."""
    return lambda p, n: json.dumps({"potential_features": [spec(name, t) for name, t in features]})


def llm_for(code: str, *features: tuple[str, str], **handlers) -> FakeLLM:
    """A fake LLM whose analyzer proposes ``features`` and whose CodeExtractor always writes ``code``."""
    return FakeLLM(analyzer=proposing(*features), extractor=lambda p, n: tagged(code), **handlers)


def dps_of(trajs, ad=None):
    ad = ad or adapter()
    return [dp for t in trajs for dp in ad.decision_points(t, TARGET)]


# =========================================================================== compile-time verification
NONDETERMINISTIC = (
    "import string\n"
    "words = trajectory['intent'].split() + ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta']\n"
    "try:\n"
    "    string.tabagent_calls += 1\n"
    "except AttributeError:\n"
    "    string.tabagent_calls = 1\n"
    "features = {\n"
    "    'user_intent': trajectory['intent'],\n"
    "    'intent_word_set': ' '.join(set(words)),\n"
    "    'call_index': string.tabagent_calls,\n"
    "}\n"
)
NONDETERMINISTIC_FEATURES = (("user_intent", "text"), ("intent_word_set", "text"), ("call_index", "numeric"))


def test_nondeterministic_columns_are_dropped(tmp_path):
    llm = llm_for(NONDETERMINISTIC, *NONDETERMINISTIC_FEATURES)
    result = schema_for(tmp_path, llm).discover(runs(), TARGET)
    ext = result.extractor
    assert set(ext.features) == {"user_intent"}
    # set order follows the per-process hash seed; a counter kept across calls follows the batch order
    for col in ("intent_word_set", "call_index"):
        assert ext.dropped[col].startswith("nondeterministic: running it again (fresh interpreters, shuffled")
        assert col in ext.verification["determinism"]["unstable"]
    assert (
        "determinism" in result.report.to_markdown() and "2 column(s) differ" in result.report.to_markdown()
    )
    tests = json.loads((tmp_path / "runs" / "r1" / TESTS_FILE).read_text())
    assert tests["columns"] == ["user_intent"]  # the goldens hold only what reproduces
    assert verify_tests(ext.code, tests).passed


NOSY = (
    "import json\n"
    "features = {\n"
    "    'user_intent': trajectory['intent'],\n"
    "    'record_size': len(json.dumps(trajectory)),\n"
    "    'top_level_text': ' '.join(str(v) for k, v in trajectory.items() if k != 'steps'),\n"
    "}\n"
)


def test_columns_that_read_outcome_fields_are_dropped(tmp_path):
    llm = llm_for(NOSY, ("user_intent", "text"), ("record_size", "numeric"), ("top_level_text", "text"))
    result = schema_for(tmp_path, llm, context_text=True).discover(runs(), TARGET)
    ext = result.extractor
    assert set(ext.features) == {"user_intent", "context_text"}  # the appended context text skips outcomes
    for col in ("record_size", "top_level_text"):
        assert ext.dropped[col].startswith("reads the run's outcome fields: adding score, eval, evaluation")
    assert set(ext.verification["outcome"]["keys"]) == set(adapter().outcome_keys)
    outcome = {"intent": "book a flight", "outcome": "failure", "reward": 0.0, "steps": []}
    assert context_texts([outcome], skip=("outcome", "reward")) == ["book a flight"]


def varied_run(i: int) -> dict:
    """A toy run with 0 to 2 extra planning notes before the decision (so history features vary)."""
    raw = raw_run(i)
    raw["steps"][2:2] = [{"name": "PlanControllerAgent", "data": f"note {k} on {i}"} for k in range(i % 3)]
    return raw


HISTORY = (
    "steps = trajectory['steps']\n"
    "features = {\n"
    "    'user_intent': trajectory['intent'],\n"
    "    'n_steps': len(steps),\n"
    "    'last_data': steps[-1]['data'] if steps else '',\n"
    "}\n"
)
HISTORY_FEATURES = (("user_intent", "text"), ("n_steps", "numeric"), ("last_data", "text"))


class LateCut(GenericJsonAdapter):
    """A buggy custom adapter: its prefixes end one step too late, so they hold the decision step itself."""

    def decision_points(self, traj, target):
        cut = super().decision_points(traj, target)
        return [dataclasses.replace(dp, prefix=self._prefix(traj.raw, dp.step_index + 1)) for dp in cut]


def test_history_features_pass_the_future_check_and_a_leaking_adapter_fails_closed(tmp_path):
    trajs = runs(16, varied_run)
    ok = schema_for(tmp_path, llm_for(HISTORY, *HISTORY_FEATURES)).discover(trajs, TARGET)
    future = ok.extractor.verification["future"]
    assert future["checked"] == 16 and not future["leaks"] and not ok.extractor.dropped
    assert {"n_steps", "last_data"} <= set(ok.extractor.features)  # history is not mistaken for the future
    assert "are unchanged" in ok.report.to_markdown()

    late = LateCut(candidates_fn=adapter().candidates_fn, labels_fn=adapter().labels_fn)
    with pytest.raises(PrefixLeakError, match=r"adapter LateCut puts post-decision data.*at prefix\.steps"):
        schema_for(tmp_path, llm_for(HISTORY, *HISTORY_FEATURES), "r2", ad=late).discover(trajs, TARGET)
    report = (tmp_path / "runs" / "r2" / "report.md").read_text()  # the run's report says why it stopped
    assert "onboarding stopped before any LLM call: the adapter LateCut" in report
    assert "future invariance: **FAILED**" in report


def test_the_changed_futures_leave_everything_before_the_decision_alone():
    raw = varied_run(2)
    (dp,) = adapter().decision_points(Trajectory("r", raw, True, "t"), TARGET)
    cut = dp.step_index + 1
    variants = dict(changed_futures(raw, "steps", cut, ("score", "reward"), "name"))
    for label in ("future changed", "future changed (structural keys kept)"):
        changed = variants[label]
        assert changed["steps"][: dp.step_index] == raw["steps"][: dp.step_index]
        decision = changed["steps"][dp.step_index]
        assert decision["name"] == TARGET and decision["data"].endswith(FUTURE_MARK)  # still found, changed
        assert len(changed["steps"]) == cut + 2 * (len(raw["steps"]) - cut) + 1 and changed["score"] == 0.0
    later = variants["future changed"]["steps"][cut]
    assert all(c.endswith(FUTURE_MARK) for c in later["calls"])  # tool choices are perturbed too
    truncated = variants["run cut after the decision"]
    assert truncated["steps"] == raw["steps"][:cut] and "score" not in truncated
    assert raw["score"] == 1.0 and not raw["steps"][dp.step_index]["data"].endswith(FUTURE_MARK)


FRAGILE = (
    "features = {'user_intent': trajectory['intent']}\n"
    "if trajectory['steps']:\n"
    "    features['last_data'] = trajectory['steps'][-1]['data']\n"
)


def test_robustness_is_reported_and_never_drops(tmp_path):
    llm = llm_for(FRAGILE, ("user_intent", "text"), ("last_data", "text"))
    result = schema_for(tmp_path, llm).discover(runs(), TARGET)
    assert set(result.extractor.features) == {"user_intent", "last_data"} and not result.extractor.dropped
    cases = {c["case"]: c for c in result.extractor.verification["robustness"]["cases"]}
    assert cases["no steps"]["raised"] is None and cases["no steps"]["missing"] == ["last_data"]
    assert cases["without the task field 'intent'"]["raised"] == "KeyError: 'intent'"
    md = result.report.to_markdown()
    assert "robustness (reported, not dropped)" in md and "KeyError: 'intent'" in md


def test_a_dropped_co_usage_column_leaves_no_co_usage_statistic():
    from tabagent.tabschema.compile import CompiledExtractor

    ext = CompiledExtractor(
        "features = {}", {"tools_so_far": "text", "n": "num"}, co_usage_context="tools_so_far"
    )
    assert ext.without(["tools_so_far"]).co_usage_context is None and ext.co_usage_context == "tools_so_far"
    ext.drop("tools_so_far", "OutputAuditor: ...")
    assert ext.co_usage_context is None and ext.dropped == {"tools_so_far": "OutputAuditor: ..."}
    assert ext.features == {"n": "num"} and ext.llm_columns == ["n"]


def test_verification_can_be_turned_off(tmp_path):
    llm = llm_for(NONDETERMINISTIC, *NONDETERMINISTIC_FEATURES)
    result = schema_for(tmp_path, llm, verify=False, audit=False).discover(runs(), TARGET)
    assert {"intent_word_set", "call_index"} <= set(result.extractor.features)
    assert result.extractor.verification == {} and "Verification" not in result.report.to_markdown()


# =========================================================================== golden tests
#: A made-up home directory, assembled at run time (a literal one would trip the repository's privacy hook).
FAKE_HOME = "/".join(["", "Users", "jane"])


def secret_run(i: int) -> dict:
    raw = raw_run(i)
    note = f"key sk-{'A' * 24} in {FAKE_HOME}/notes.txt; " + "x" * 30_000
    raw["steps"][0]["data"] = f"{raw['steps'][0]['data']} {note}"
    return raw


SNIPPET = (
    "features = {'user_intent': trajectory['intent'], 'analysis_head': trajectory['steps'][0]['data'][:90]}\n"
)


def test_goldens_are_clipped_masked_and_reproduce(tmp_path):
    llm = llm_for(SNIPPET, ("user_intent", "text"), ("analysis_head", "text"))
    result = schema_for(tmp_path, llm, goldens=5).discover(runs(16, secret_run), TARGET)
    text = (tmp_path / "runs" / "r1" / TESTS_FILE).read_text()
    tests = json.loads(text)
    assert len(tests["goldens"]) == 5 and tests["columns"] == ["user_intent", "analysis_head"]
    assert "sk-AAAA" not in text and FAKE_HOME not in text and str(tmp_path) not in text
    for g in tests["goldens"]:
        data = g["prefix"]["steps"][0]["data"]
        assert len(data) <= GOLDEN_STRING_CHARS + 40 and "more characters]" in data  # clipped
        assert "[masked key]" in g["features"]["analysis_head"]  # computed on the masked snapshot
    assert tests["extractor_sha256"] == result.extractor.sha256
    assert verify_tests(result.extractor.code, tests).passed  # the snapshots reproduce exactly
    saved = result.save(tmp_path / "card")
    assert json.loads((saved / TESTS_FILE).read_text()) == tests
    assert mask_secrets({"a": ["Bearer abcdefghijklmnopqrstuvwxyz0123"]}) == {"a": ["Bearer [masked token]"]}


def test_verify_tests_catch_nondeterminism_and_outcome_reads():
    dps = dps_of(runs(8))
    clock = "import datetime\nfeatures = {'clock': datetime.datetime.now().microsecond}\n"
    rep = verify_tests(clock, make_tests(clock, ["clock"], dps, outcome_keys=("reward",)))
    assert not rep.passed and {c["name"]: c["passed"] for c in rep.checks}["determinism"] is False
    nosy = "features = {'n_fields': len(trajectory)}\n"
    rep = verify_tests(nosy, make_tests(nosy, ["n_fields"], dps, outcome_keys=("reward",)))
    status = {c["name"]: c["passed"] for c in rep.checks}
    assert status == {
        "extractor hash": True,
        "goldens": True,
        "determinism": True,
        "outcome invariance": False,
    }
    assert "FAILED" in str(rep) and "[FAIL] outcome invariance" in str(rep)


def test_the_head_ships_its_goldens_and_verify_head_gates_deployment(tmp_path, capsys):
    out = tmp_path / "head"
    head = tabagent.onboard(
        runs(), TARGET, adapter=adapter(), llm=FakeLLM(), config=config(), out=out, n_splits=2, iterations=20
    )
    assert head.extractor_tests and (out / TESTS_FILE).exists()
    again = tabagent.load(out)  # sandboxed
    assert again.extractor_tests == head.extractor_tests
    prefix = {"intent": "what is the weather in rome", "steps": raw_run(1)["steps"][:2]}
    assert again.decide(prefix, list(TOOLS), k=1) == head.decide(prefix, list(TOOLS), k=1)
    report = verify_head(out)
    assert report.passed, str(report)
    assert cli.main(["verify-head", str(out)]) == 0 and "PASSED" in capsys.readouterr().out

    tests = json.loads((out / TESTS_FILE).read_text())
    tests["goldens"][0]["features"]["intent_words"] += 1  # a recorded value the code no longer gives
    (out / TESTS_FILE).write_text(json.dumps(tests))
    assert cli.main(["verify-head", str(out)]) == 1
    printed = capsys.readouterr().out
    assert "[FAIL] goldens" in printed and "golden 0, intent_words: expected" in printed

    (out / TESTS_FILE).unlink()  # no tests: the hashes and the artifact are still checked, with a warning
    assert verify_head(out).passed and "no extractor_tests.json" in str(verify_head(out))
    assert not verify_head(out, require_goldens=True).passed
    assert cli.main(["verify-head", str(out), "--require-goldens"]) == 1
    (out / "extract.py").write_text("features = {}\n")
    assert {c["name"]: c["passed"] for c in verify_head(out).checks}["card hash"] is False
    assert cli.main(["verify-head", str(tmp_path / "nothing")]) == 2  # not a head: a user error


def old_head(tmp_path, head):
    """``head`` saved the way earlier versions could: a card that pins no hash for extract.py."""
    out = tmp_path / "old_head"
    head.agent.card_.extractor = None
    head.agent.save(out / "model")
    (out / "extract.py").write_text(head.extractor_code)
    (out / "head.json").write_text(json.dumps({"format_version": 1, "target": TARGET, "has_extractor": True}))
    return out


def test_a_head_whose_card_pins_no_hash_is_refused_unless_allowed(tmp_path):
    head = tabagent.onboard(
        runs(), TARGET, adapter=adapter(), llm=FakeLLM(), config=config(), n_splits=2, iterations=20
    )
    out = old_head(tmp_path, head)
    with pytest.raises(
        ValueError, match="pins no hash for extract.py.*allow_unpinned=True and save it again"
    ):
        tabagent.load(out)
    status = {c["name"]: c for c in verify_head(out).checks}
    assert not status["card hash"]["passed"] and "pins no hash" in status["card hash"]["detail"]
    with pytest.warns(UserWarning, match="allow_unpinned=True: running an extract.py whose hash"):
        allowed = tabagent.load(out, allow_unpinned=True)
    prefix = {"intent": "play a song from lima", "steps": raw_run(3)["steps"][:2]}
    assert allowed.decide(prefix, list(TOOLS), k=1)["candidates"]
    allowed.save(tmp_path / "resaved")  # saving again pins the hash
    assert (
        tabagent.load(tmp_path / "resaved").card.extractor["sha256"]
        == allowed.agent.card_.extractor["sha256"]
    )
    (tmp_path / "resaved" / "extract.py").write_text("features = {}\n")
    with pytest.raises(ValueError, match="does not match the card's extractor hash"):
        tabagent.load(tmp_path / "resaved", allow_unpinned=True)  # the opt-in never excuses a mismatch


def test_a_recipe_without_the_prior_keeps_the_extractor_hash():
    # a head whose recipe turned the candidate prior off used to lose the card's extractor block, so load()
    # skipped the extract.py hash check (and verify-head reported the mismatch)
    from tabagent import DecisionSpec, Feature, FeatureCard
    from tabagent.auto import Recipe, build_agent

    spec_ = DecisionSpec("ranking", "task", "chosen", "candidate", decision=("decision_id",))
    prior = Feature("candidate_prior", "dependency", "num", "fitted:candidate_prior", {"key": "candidate"})
    feats = [
        Feature("n_words", "state", "num", "extractor"),
        Feature("candidate_name", "schema", "cat"),
        prior,
    ]
    card = FeatureCard("c", spec_, feats, extractor={"path": "extract.py", "sha256": "ab" * 32})
    agent = build_agent(card, Recipe(0, False, "paper"))
    assert "candidate_prior" not in agent.card.feature_names and agent.card.extractor == card.extractor


# =========================================================================== the OutputAuditor
def auditor(verdict_for):
    """An auditor answer: ``verdict_for(feature, call_index)`` per feature named in the prompt."""

    def answer(p, n):
        rows = [
            {"feature_name": x, "verdict": verdict_for(x, n), "reason": "seen in the trace"}
            for x in names_in(p)
        ]
        return json.dumps({"features": rows})

    return answer


def test_features_the_auditor_judges_incorrect_are_dropped(tmp_path):
    wrong = auditor(lambda x, n: "incorrect" if x == "plan_note" else "correct")
    result = discover_with(tmp_path, FakeLLM(auditor=wrong))
    ext = result.extractor
    assert "plan_note" not in ext.features and {"user_intent", "intent_words"} <= set(ext.features)
    assert ext.dropped["plan_note"].startswith("OutputAuditor: incorrect on 12 of 12 judged decision points")
    assert "plan_note" not in {f.name for f in result.card.features}
    run = tmp_path / "runs" / "r1"
    audit = json.loads((run / "audit.json").read_text())
    assert audit["features"]["plan_note"]["incorrect"] == 12 and len(audit["points"]) == 12
    assert audit["features"]["plan_note"]["examples"][0]["reason"] == "seen in the trace"
    assert "plan_note" not in json.loads((run / TESTS_FILE).read_text())["columns"]
    assert "| plan_note | 0 | 12 | 0 | 100% | no |" in (run / "report.md").read_text()
    sample = set(json.loads((run / "sample.json").read_text()))
    assert not sample & {p["decision_point"] for p in audit["points"]}  # points the analyzer did not see


def discover_with(tmp_path, llm, run_id="r1", trajs=None, **kw):
    return schema_for(tmp_path, llm, run_id, **kw).discover(runs() if trajs is None else trajs, TARGET)


def unique_run(i: int) -> dict:
    """A toy run whose prefix no other run shares (identical prefixes share one cached auditor answer)."""
    return {**raw_run(i), "intent": f"{raw_run(i)['intent']} (request {i})"}


@pytest.mark.parametrize(
    "verdicts, dropped",
    [
        (["incorrect"] * 2 + ["cannot_tell"] * 10, False),  # judged on 2 points only: too few to drop
        (["incorrect"] + ["correct"] * 2 + ["cannot_tell"] * 9, False),  # one verdict never drops
        (["incorrect"] * 3 + ["correct"] * 9, False),  # 25%: not above the threshold
        (["incorrect"] * 4 + ["correct"] * 8, False),  # 33%, but its 95% lower bound is 16%
        (["incorrect"] * 7 + ["correct"] * 5, True),  # 58%, lower bound 36%
        (["Incorrect."] * 7 + ["CORRECT"] * 5, True),  # verdicts are read tolerantly
        (["correct/incorrect/cannot_tell"] * 12, False),  # an echoed template is no verdict
    ],
)
def test_the_audit_threshold_counts_only_judged_points(tmp_path, verdicts, dropped):
    llm = FakeLLM(auditor=auditor(lambda x, n: verdicts[n] if x == "plan_note" else "correct"))
    result = discover_with(tmp_path, llm, trajs=runs(16, unique_run), workers=1)
    assert ("plan_note" in result.extractor.dropped) is dropped
    assert (
        normalize_audit_verdict("cannot tell") == "cannot_tell"
        and normalize_audit_verdict("") == "cannot_tell"
    )


@pytest.mark.parametrize(
    "answer",
    [
        lambda p, n: ("", "content_filter"),
        lambda p, n: "I cannot judge this.",
        lambda p, n: '{"features": []}',
    ],
    ids=["content-filter", "unusable", "empty"],
)
def test_an_auditor_without_usable_answers_skips_the_audit(tmp_path, answer):
    result = discover_with(tmp_path, FakeLLM(auditor=answer))
    assert {"user_intent", "plan_note", "intent_words"} == set(result.extractor.features)
    assert result.report.audit["skipped"] and not result.report.audit["dropped"]
    assert any(n.startswith("output audit skipped") for n in result.report.notes)
    assert "skipped:" in result.report.to_markdown().split("## Output audit")[1]


def test_the_auditor_is_a_routable_cached_stage(tmp_path):
    main, judge = FakeLLM("main"), FakeLLM("judge-model")
    kw = dict(config=config(stage_models={"auditor": "judge-model"}), stage_llms={"judge-model": judge})
    schema = tabagent.TabSchema(main, adapter=adapter(), store=RunStore(tmp_path / "runs", "r1"), **kw)
    result = schema.discover(runs(), TARGET)
    assert set(judge.roles()) == {"auditor"} and "auditor" not in main.roles()
    assert result.report.audit["model"] == "judge-model" and schema.models()["auditor"] == "judge-model"
    meter = schema.meter  # metered like any stage; prefixes that repeat reuse one cached answer
    assert meter.by_stage["auditor"]["calls"] == len(judge.calls) and meter.by_stage["auditor"]["calls"] > 0
    assert meter.by_stage["auditor"]["calls"] + meter.cached["auditor"]["calls"] == 12
    again = FakeLLM("judge-model")
    kw["stage_llms"] = {"judge-model": again}
    tabagent.TabSchema(
        FakeLLM("main"), adapter=adapter(), store=RunStore(tmp_path / "runs", "r1"), **kw
    ).discover(runs(), TARGET)
    assert again.calls == []  # a resumed run reuses the audit's answers
    assert "output_auditor" in result.card.provenance["prompts_sha256"]
    assert prompts.sha256("output_auditor") == result.card.provenance["prompts_sha256"]["output_auditor"]


def test_a_spent_budget_skips_the_audit_instead_of_failing_the_run():
    dps = dps_of(runs())
    meter = CostMeter(prices={"fake": (1e6, 1e6)}, max_cost_usd=0.01)  # no call fits
    stage = OutputAuditor(FakeLLM(), meter)
    columns = {"user_intent": spec("user_intent")}
    outputs = [{"user_intent": dp.prefix["intent"]} for dp in dps]
    rec = run_audit(stage, columns, dps, outputs, TARGET, samples=4, max_incorrect=0.25, seed=0)
    assert rec["skipped"] == "the LLM budget ran out before the audit" and not rec["dropped"]


def test_dry_runs_count_the_audit(tmp_path):
    with_audit = tabagent.onboard(
        runs(), TARGET, adapter=adapter(), dry_run=True, prices={"gpt-5.4": (1.0, 1.0)}
    )
    without = tabagent.onboard(
        runs(),
        TARGET,
        adapter=adapter(),
        dry_run=True,
        config=TabSchemaConfig(audit=False, prices={"gpt-5.4": (1.0, 1.0)}),
    )
    assert with_audit.est_input_tokens > without.est_input_tokens
    assert with_audit.est_cost_usd > without.est_cost_usd


# =========================================================================== usefulness
def coded_run(i: int) -> dict:
    """A run whose tools depend only on a task code, next to a field of pure noise."""
    code = ["alpha", "beta", "gamma", "delta"][i % 4]
    used = {"alpha": ["search_flights", "book_flight"], "beta": ["get_weather"]}.get(
        code, ["read_inbox", "send_email"]
    )
    if code == "delta":
        used = ["play_song"]
    return {
        "task_id": f"task{i}",
        "intent": f"request number {i}",
        "code": code,
        "noise": round(random.Random(i).random(), 4),
        "score": 1.0,
        "steps": [
            {"name": "TaskAnalyzerAgent", "data": f"request number {i}"},
            {"name": TARGET, "data": ", ".join(used)},
            {"name": "CodeAgent", "calls": used},
        ],
    }


CODED = "features = {'task_code': trajectory['code'], 'noise_value': trajectory['noise']}\n"


def test_usefulness_tells_an_informative_feature_from_noise(tmp_path):
    ad = adapter(task_keys=("intent", "code", "noise"))
    llm = llm_for(CODED, ("task_code", "categorical"), ("noise_value", "numeric"))
    result = schema_for(tmp_path, llm, ad=ad, usefulness=True).discover(runs(48, coded_run), TARGET)
    use = result.report.usefulness
    assert use["metric"] == "p@r" and use["decisions"] == 48 and use["folds"] == 3
    assert use["features"]["task_code"]["verdict"] == "useful" and use["features"]["task_code"]["delta"] > 0.2
    assert abs(use["features"]["noise_value"]["delta"]) < 0.1
    assert set(result.extractor.features) == {"task_code", "noise_value"}  # a report only, by default
    saved = json.loads((tmp_path / "runs" / "r1" / "usefulness.json").read_text())
    assert saved["features"] == use["features"]
    assert "| task_code |" in result.report.to_markdown().split("## Usefulness")[1]


def test_prune_harmful_drops_only_clearly_harmful_llm_features(tmp_path, monkeypatch):
    from tabagent.tabschema import orchestrator

    seeds = []

    def verdicts(table, card_for, columns, **kw):
        seeds.append(kw.get("seeds"))
        harmful = {"verdict": "harmful", "delta": -0.2, "ci": [-0.3, -0.1], "p_holm": 0.001, "without": 0.9}
        neutral = {"verdict": "neutral", "delta": 0.0, "ci": [-0.1, 0.1], "p_holm": 1.0, "without": 0.7}
        feats = {c: harmful if c in ("plan_note", "context_text") else neutral for c in columns}
        return {
            "metric": "p@r",
            "folds": 3,
            "iterations": 50,
            "seeds": kw.get("seeds", 1),
            "decisions": 16,
            "rows": 96,
            "tasks": 16,
            "base": 0.7,
            "features": feats,
        }

    monkeypatch.setattr(orchestrator, "feature_usefulness", verdicts)
    kept = discover_with(tmp_path, FakeLLM(), "keep", usefulness=True, context_text=True)
    assert "plan_note" in kept.extractor.features  # report only
    pruned = discover_with(
        tmp_path, FakeLLM(), "prune", usefulness=True, context_text=True, prune_harmful=True
    )
    assert "plan_note" not in pruned.extractor.features
    assert pruned.extractor.dropped["plan_note"].startswith("harmful (prune_harmful): removing it raised p@r")
    assert "context_text" in pruned.extractor.features  # the library's own column is never pruned
    assert pruned.report.usefulness["pruned"] == ["plan_note"]
    assert seeds == [1, 2]  # pruning needs the verdict of two seeds; a report needs one
    assert any("onboarding report) is optimistic" in n for n in pruned.report.notes)

    monkeypatch.setattr(
        orchestrator,
        "feature_usefulness",
        lambda table, card_for, columns, **kw: {
            **verdicts(table, card_for, columns),
            "features": {
                c: {"verdict": "harmful", "delta": -0.2, "ci": [-0.3, -0.1], "p_holm": 0.0, "without": 0.9}
                for c in columns
            },
        },
    )
    every = discover_with(tmp_path, FakeLLM(), "all", usefulness=True, prune_harmful=True)
    assert len(every.extractor.llm_columns) == 3 and any("none was pruned" in n for n in every.report.notes)


def test_usefulness_uses_only_the_given_decisions_and_degrades_without_labels(tmp_path):
    capped = discover_with(tmp_path, FakeLLM(), "cap", usefulness=True, usefulness_max_decisions=10)
    assert capped.report.usefulness["decisions"] == 10  # a sample of the onboarding decisions, never more
    bare = GenericJsonAdapter()  # no candidates or labels: nothing to cross-validate against
    result = schema_for(tmp_path, FakeLLM(), "bare", ad=bare, usefulness=True).discover(runs(), TARGET)
    assert result.report.usefulness["skipped"].startswith("NotImplementedError")
    assert set(result.extractor.features) == {"user_intent", "plan_note", "intent_words"}


# =========================================================================== the CLI
CHAT_CODE = (
    "users = [s['content'] for s in trajectory['steps'] if s['name'] == 'user']\n"
    "features = {\n"
    "    'first_request': users[0] if users else '',\n"
    "    'mentions_flight': int('flight' in ' '.join(users)),\n"
    "}\n"
)


def test_cli_onboard_flags_and_a_verified_round_trip(tmp_path, monkeypatch, capsys):
    """CLI onboarding with --no-audit --no-verify-extractor --prune-harmful on chat logs, then verify-head,
    load and decide: the flags reach TabSchemaConfig, and the saved head still carries checkable goldens."""
    from adapters.fake_logs import write_openai_logs

    from tabagent.tabschema import llm as llm_module

    logs = tmp_path / "logs"
    logs.mkdir()
    write_openai_logs(logs, n=16)
    llms: list[FakeLLM] = []

    def fake_llm(model=None):
        llms.append(llm_for(CHAT_CODE, ("first_request", "text"), ("mentions_flight", "numeric")))
        return llms[-1]

    monkeypatch.setattr(llm_module, "auto_llm", fake_llm)
    seen = []
    onboard = tabagent.onboard
    monkeypatch.setattr(tabagent, "onboard", lambda *a, **kw: seen.append(kw["config"]) or onboard(*a, **kw))
    head = tmp_path / "head"
    flags = ["--no-audit", "--no-verify-extractor", "--prune-harmful"]
    args = ["onboard", str(logs), "--decision", "ToolShortlist", "--n-analyze", "3", "--out", str(head)]
    assert cli.main([*args, "--run-dir", str(tmp_path / "run"), *flags]) == 0
    (cfg,) = seen
    assert (cfg.audit, cfg.verify, cfg.prune_harmful, cfg.n_analyze) == (False, False, True, 3)
    assert "auditor" not in llms[0].roles()  # --no-audit: no auditor call
    run = tmp_path / "run"
    assert not (run / "audit.json").exists() and json.loads((run / "usefulness.json").read_text())["features"]
    assert "Verification of the extracted values" not in (run / "report.md").read_text()

    assert cli.main(["verify-head", str(head)]) == 0  # the goldens are recorded with or without the checks
    assert "PASSED" in capsys.readouterr().out
    loaded = tabagent.load(head)
    prefix = loaded.live_prefix(
        {"messages": [{"role": "user", "content": "find a flight to rome and book it"}]}
    )
    out = loaded.decide(prefix, ["search_flights", "book_flight", "get_weather", "play_song"], k=2)
    assert len(out["candidates"]) == 2 and out["error"] is None


def test_cli_onboard_flags_default_to_the_checks_and_change_the_dry_run(tmp_path, capsys):
    from adapters.fake_logs import write_openai_logs

    write_openai_logs(tmp_path, n=12)
    base = ["onboard", str(tmp_path), "--dry-run", "--price-in", "1", "--price-out", "8"]
    assert cli.main(base) == 0
    with_audit = capsys.readouterr().out
    assert cli.main([*base, "--no-audit"]) == 0
    without = capsys.readouterr().out
    tokens = [int(re.search(r"~([\d,]+) input", o).group(1).replace(",", "")) for o in (with_audit, without)]
    assert tokens[0] > tokens[1]  # the dry run counts the auditor's calls unless --no-audit
    with pytest.raises(SystemExit):
        cli.main(["onboard", "--help"])
    help_text = capsys.readouterr().out
    assert (
        "--no-audit" in help_text and "--no-verify-extractor" in help_text and "--prune-harmful" in help_text
    )
