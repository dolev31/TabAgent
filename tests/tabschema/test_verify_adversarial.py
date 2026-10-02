"""Adversarial regressions for the release's value checks, from an independent review of the first version.

Leaking adapters the future check must catch (and built-in adapters it must never flag), nondeterminism at
every decision position, masking of everything that ships, the audit's drop rule, artifact integrity,
``verify-head``'s edge cases, the usefulness estimate's statistics and runtime bound, and the plumbing of
``allow_unpinned``. Scripted fake LLMs only: no network, no credentials.
"""

from __future__ import annotations

import dataclasses
import json
import random
import sys
import types

import pytest
import yaml
from adapters.fake_logs import (
    cuga_trajectory,
    langchain_run,
    messages_api_run,
    openai_run,
    tau2_results,
)
from test_robustness import TARGET, TOOLS, FakeLLM, adapter, config, names_in, raw_run, runs, schema_for, spec
from test_verify import CODED, HISTORY, HISTORY_FEATURES, LateCut, coded_run, llm_for, old_head, varied_run

import tabagent
from tabagent import cli
from tabagent.adapters import CugaAdapter, OpenAIMessagesAdapter, Tau2Adapter
from tabagent.tabschema import TabSchemaConfig, Trajectory, checks
from tabagent.tabschema.audit import OutputAuditor, run_audit
from tabagent.tabschema.orchestrator import _verification_md
from tabagent.tabschema.sampling import sample_decision_points
from tabagent.tabschema.trajectory import GenericJsonAdapter
from tabagent.tabschema.usefulness import MIN_DECISIONS
from tabagent.tabschema.verify import (
    GOLDEN_MAX_CHARS,
    TESTS_FILE,
    FutureCheck,
    PrefixLeakError,
    isolated_adapter,
    mask_secrets,
    snapshot,
    verify_extractor,
    verify_head,
)


def kw() -> dict:
    a = adapter()
    return {"candidates_fn": a.candidates_fn, "labels_fn": a.labels_fn}


def judge_all(verdict: str):
    """An auditor answer giving ``verdict`` to every feature named in the prompt."""

    def answer(p, n):
        rows = [{"feature_name": x, "verdict": verdict, "reason": "."} for x in names_in(p)]
        return json.dumps({"features": rows})

    return answer


def future_results(ad, trajs, target=TARGET) -> list[dict]:
    dps = [dp for t in trajs for dp in ad.decision_points(t, target)]
    check = FutureCheck(ad, trajs, target, getattr(ad, "outcome_keys", ()))
    return [check(dp) for dp in dps]


# =========================================================================== H1, H2: leaking adapters
def tool_step_run(i: int) -> dict:
    """The decision step records its choice only under structural keys (arguments, tool_name)."""
    raw = varied_run(i)
    for s in raw["steps"]:
        if s["name"] == TARGET:
            used = s.pop("data").split(", ")
            s["arguments"] = {"tools": used}
            s["tool_name"] = used[0]
        if s["name"] == "CodeAgent":
            s["tool_name"] = ",".join(s["calls"])
    return raw


class TwoLate(GenericJsonAdapter):
    """Prefixes hold the decision step and the step right after it (the tool execution)."""

    def decision_points(self, traj, target):
        cut = super().decision_points(traj, target)
        return [dataclasses.replace(dp, prefix=self._prefix(traj.raw, dp.step_index + 2)) for dp in cut]


class WholeRun(GenericJsonAdapter):
    """Prefixes hold the whole run."""

    def decision_points(self, traj, target):
        n = len(traj.raw["steps"])
        return [
            dataclasses.replace(dp, prefix=self._prefix(traj.raw, n))
            for dp in super().decision_points(traj, target)
        ]


def renamed(raw: dict) -> dict:
    """The same run with the step name under ``agent`` (a custom log format)."""
    raw = json.loads(json.dumps(raw))
    for s in raw["steps"]:
        s["agent"] = s.pop("name")
    return raw


@pytest.mark.parametrize(
    "make_adapter, make_run",
    [
        (lambda: LateCut(**kw()), tool_step_run),  # the label sits under tool_name/arguments only
        (lambda: TwoLate(**kw()), varied_run),  # the tool execution right after the decision leaks
        (lambda: WholeRun(**kw()), tool_step_run),  # set-valued leaks of every later step
        (lambda: LateCut(step_name_key="agent", **kw()), lambda i: renamed(tool_step_run(i))),  # H2
    ],
    ids=["late-cut-structural", "two-late", "whole-run", "late-cut-custom-name-key"],
)
def test_the_future_check_catches_adapters_that_cut_late(make_adapter, make_run):
    trajs = runs(16, make_run)
    results = future_results(make_adapter(), trajs)
    assert all("leak" in r for r in results), results[:2]
    assert all("prefix.steps" in r["leak"]["path"] for r in results)


def test_a_leaking_adapter_stops_onboarding_with_a_clear_error_in_the_cli(tmp_path, monkeypatch, capsys):
    late = LateCut(**kw())
    with pytest.raises(PrefixLeakError, match=r"adapter LateCut puts post-decision data.*onboard again"):
        schema_for(tmp_path, llm_for(HISTORY, *HISTORY_FEATURES), ad=late).discover(
            runs(16, varied_run), TARGET
        )
    assert isinstance(PrefixLeakError("x"), ValueError)  # a user error: the CLI prints it, exit status 2
    monkeypatch.setattr(
        tabagent, "onboard", lambda *a, **k: (_ for _ in ()).throw(PrefixLeakError("the adapter X"))
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run.json").write_text(json.dumps(openai_run(0)))
    assert cli.main(["onboard", str(logs), "--decision", "ToolShortlist", "--dry-run"]) == 2
    assert "the adapter X" in capsys.readouterr().err


def write_tau2(tmp_path):
    sims = [(str(i), [TOOLS_LIST[i % 3], TOOLS_LIST[(i + 1) % 3]], 1.0) for i in range(6)]
    path = tmp_path / "tau2"
    path.mkdir()
    (path / "airline.json").write_text(json.dumps(tau2_results("airline", sims)))
    return path


TOOLS_LIST = ["get_reservation_details", "update_reservation_flights", "get_user_details"]


@pytest.mark.parametrize(
    "build",
    [
        lambda tmp: (GenericJsonAdapter(**kw()), runs(12, varied_run), TARGET),
        lambda tmp: (OpenAIMessagesAdapter("task"), chat(openai_run), "ToolShortlist"),
        lambda tmp: (OpenAIMessagesAdapter("step"), chat(openai_run), "NextToolCall"),
        lambda tmp: (OpenAIMessagesAdapter("step"), chat(messages_api_run), "NextToolCall"),
        lambda tmp: (OpenAIMessagesAdapter("step"), chat(langchain_run), "NextToolCall"),
        lambda tmp: (Tau2Adapter("step"), Tau2Adapter("step").load(write_tau2(tmp)), "NextToolCall"),
        lambda tmp: (CugaAdapter(), cuga(), "ShortlisterAgent"),
    ],
    ids=["generic", "chat-shortlist", "chat-next-call", "anthropic", "langchain", "tau2", "cuga"],
)
def test_the_built_in_adapters_never_trigger_the_future_check(tmp_path, build):
    ad, trajs, target = build(tmp_path)
    results = future_results(ad, trajs, target)
    assert results and all("ok" in r for r in results), [r for r in results if "ok" not in r][:2]


def chat(make) -> list[Trajectory]:
    return [Trajectory(f"r{i}", make(i), True, f"t{i}") for i in range(8)]


def cuga() -> list[Trajectory]:
    offered = ["venmo_send", "venmo_pay", "venmo_request"]
    raws = [cuga_trajectory(f"t{i}", [("venmo", offered, [offered[i % 3]])]) for i in range(6)]
    return [Trajectory(f"c{i}", r, True, f"t{i}") for i, r in enumerate(raws)]


def test_the_future_check_never_touches_the_adapters_own_state():
    ad = OpenAIMessagesAdapter("step")
    trajs = chat(openai_run)
    for t in trajs:
        ad.decision_points(t, "NextToolCall")
    before = json.dumps(
        [ad._seen, {k: {t: sorted(a) for t, a in v.items()} for k, v in ad._called.items()}],
        sort_keys=True,
        default=str,
    )
    future_results(ad, trajs, "NextToolCall")
    after = json.dumps(
        [ad._seen, {k: {t: sorted(a) for t, a in v.items()} for k, v in ad._called.items()}],
        sort_keys=True,
        default=str,
    )
    assert before == after  # no perturbed tool name reached the candidate pool
    fresh = isolated_adapter(ad)
    assert (
        type(fresh) is OpenAIMessagesAdapter and fresh is not ad and not fresh._own
    )  # built from its description
    late = LateCut(**kw())
    copied = isolated_adapter(late)
    assert copied._raw is not late._raw and copied._left_out is not late._left_out  # containers, not objects


def test_a_future_check_that_checks_nothing_is_reported_as_skipped():
    ad = adapter()
    trajs = runs(8, varied_run)
    dps = [dp for t in trajs for dp in ad.decision_points(t, TARGET)]
    check = FutureCheck(ad, [], TARGET, ad.outcome_keys)  # no run to rebuild from
    rec = verify_extractor(
        HISTORY, [n for n, _ in HISTORY_FEATURES], dps, future=check, outcome_keys=ad.outcome_keys
    )
    assert rec["future"]["checked"] == 0 and rec["future"]["skipped"] == 8
    assert any("future-invariance check was skipped" in n for n in rec["notes"])
    assert "future invariance: **skipped**, not passed" in "\n".join(_verification_md(rec))


# ======================================================================= H3, L3: every position, every order
def multi_decision_run(i: int) -> dict:
    """A run with 4 decisions; the tools called earlier accumulate (a next-tool-call decision)."""
    tools = list(TOOLS)
    steps = [{"name": "TaskAnalyzerAgent", "data": f"request {i}"}]
    for k in range(4):
        steps.append({"name": TARGET, "data": f"decision {k}"})
        steps.append({"name": "CodeAgent", "calls": [tools[(i + k) % 6], tools[(i + 2 * k + 1) % 6]]})
    return {"task_id": f"task{i}", "intent": f"request {i}", "score": 1.0, "steps": steps}


SET_SO_FAR = (
    "called = set()\n"
    "for s in trajectory['steps']:\n"
    "    if s['name'] == 'CodeAgent':\n"
    "        called.update(s['calls'])\n"
    "features = {'intent': trajectory['intent'], 'tools_so_far': ','.join(called)}\n"
)


def test_nondeterminism_after_a_runs_first_decision_is_caught(tmp_path):
    llm = llm_for(SET_SO_FAR, ("intent", "text"), ("tools_so_far", "text"))
    result = schema_for(tmp_path, llm).discover(runs(24, multi_decision_run), TARGET)
    assert result.report.n_decision_points == 96 and "tools_so_far" not in result.extractor.features
    assert result.extractor.dropped["tools_so_far"].startswith("nondeterministic")
    assert result.extractor.verification["determinism"]["prefixes"] == 96  # every point, not a sample


def test_the_audit_sample_covers_positions_beyond_a_runs_first_decision():
    ad = adapter()
    dps = [dp for t in runs(24, multi_decision_run) for dp in ad.decision_points(t, TARGET)]
    outputs = [{"n_prior_calls": sum(s["name"] == "CodeAgent" for s in dp.prefix["steps"])} for dp in dps]
    answer = judge_all("correct")
    rec = run_audit(
        OutputAuditor(FakeLLM(auditor=answer)),
        {"n_prior_calls": spec("n_prior_calls")},
        dps,
        outputs,
        TARGET,
        samples=12,
        max_incorrect=0.25,
        seed=0,
        label_of=ad.labels,
    )
    audited = {p["decision_point"] for p in rec["points"]}
    values = {o["n_prior_calls"] for dp, o in zip(dps, outputs, strict=True) if dp.id in audited}
    assert len(values) > 1  # not only the trivial value of every run's first decision
    assert len({dp.step_index for dp in sample_decision_points(dps, 12, 0, ad.labels, "coverage")}) > 1


@pytest.mark.parametrize(
    "line",
    [
        "x = datetime.datetime.now().microsecond",
        "x = datetime.date.today().isoformat()",
        "x = datetime.datetime.utcnow().hour",
        "x = hash(trajectory['intent']) % 13",
        "x = id(trajectory) % 97",
    ],
)
def test_reading_the_clock_or_a_per_process_value_is_refused_statically(line):
    issues = checks.check_code(f"import datetime\n{line}\nfeatures = {{'x': x}}\n")
    assert [i.kind for i in issues] == ["clock"]


def test_an_extractor_that_reads_the_clock_is_rewritten(tmp_path):
    clock = (
        "import datetime\n"
        "features = {'user_intent': trajectory['intent'], 'day': datetime.date.today().day}\n"
    )
    good = "features = {'user_intent': trajectory['intent'], 'n_words': len(trajectory['intent'].split())}\n"
    llm = FakeLLM(
        analyzer=lambda p, n: json.dumps(
            {"potential_features": [spec("user_intent"), spec("n_words", "numeric")]}
        ),
        extractor=lambda p, n: f"<code>{good if 'PREVIOUS ATTEMPT FAILED' in p else clock}</code>",
    )
    result = schema_for(tmp_path, llm).discover(runs(), TARGET)
    assert (
        set(result.extractor.features) == {"user_intent", "n_words"}
        and "datetime" not in result.extractor.code
    )
    assert any("reads the clock" in p for p in llm.prompts("extractor"))  # the retry said why


# =========================================================================== M1, M2: masking
KEY = "sk-" + "Q" * 30


def secret_intent_run(i: int) -> dict:
    raw = raw_run(i)
    raw["intent"] = f"{raw['intent']} using key {KEY} and words alpha beta gamma delta"
    return raw


def test_secrets_reach_nothing_that_ships_or_is_reported(tmp_path):
    code = (
        "words = trajectory['intent'].split()\n"
        "features = {'user_intent': trajectory['intent'][:20], 'word_bag': ' '.join(set(words)),"
        " 'intent_tail': trajectory['intent'][-60:]}\n"
    )

    def auditor(p, n):
        rows = [
            {
                "feature_name": x,
                "verdict": "incorrect" if x == "intent_tail" else "correct",
                "reason": f"saw {KEY}",
            }
            for x in names_in(p)
        ]
        return json.dumps({"features": rows})

    llm = llm_for(
        code, ("user_intent", "text"), ("word_bag", "text"), ("intent_tail", "text"), auditor=auditor
    )
    out = tmp_path / "head"
    head = tabagent.onboard(
        runs(16, secret_intent_run),
        TARGET,
        adapter=adapter(),
        llm=llm,
        config=config(),
        out=out,
        run_dir=tmp_path / "run",
        n_splits=2,
        iterations=20,
    )
    assert {"word_bag", "intent_tail"} <= set(head.discovery.dropped)  # nondeterministic; judged incorrect
    for f in (
        out / TESTS_FILE,
        out / "discovery.md",
        tmp_path / "run" / "report.md",
        tmp_path / "run" / "audit.json",
        tmp_path / "run" / TESTS_FILE,
    ):
        assert KEY not in f.read_text(), f


HOME = "/".join(["", "Users", "jane"])
#: Made-up credentials, assembled at run time (literal ones would trip the repository's privacy hook).
SECRETS = {
    "openai": "sk" + "-proj-AbCdEfGhIjKlMnOpQrStUvWx1234",
    "anthropic": "sk" + "-ant-api03-AbCdEfGhIjKlMnOpQrStUvWx",
    "stripe": "sk_" + "live_" + "51HabcdefGhijklmnopqrstu",
    "hf": "hf" + "_AbCdEfGhIjKlMnOpQrStUvWxYz012345",
    "github_pat": "github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz",
    "ghp": "ghp_" + "a" * 36,
    "hex32": "api-key: " + "0123456789abcdef" * 2,  # an Azure-style key, named
    "aws_secret": "aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "password_kv": "password=hunter2hunter2",
    "json_pair": '{"api_key": "' + "abcd1234" + 'efgh"}',
    "basic_auth_url": "https://alice:s3cr3tpass" + "@" + "db.internal.example/x",
    "email": "jane.doe" + "@" + "example.org",
    "mac_home": f"{HOME}/projects/x.txt",
    "linux_home": "/".join(["", "home", "jane", ".ssh", "id_rsa"]),
    "win_home": "\\".join(["C:", "Users", "jane", "Desktop", "a.txt"]),
    "win_fwd": "/".join(["C:", "Users", "jane", "Desktop", "a.txt"]),
    "unc_home": "\\\\" + "\\".join(["server", "Users", "jane", "a.txt"]),
    "tilde": "~" + "jane/x",
    "root": "/".join(["", "root", ".aws", "credentials"]),
    "slack": "xox" + "b-1234567890-abcdefghij",
    "jwt": ".".join(
        [
            "eyJ" + "hbGciOiJIUzI1NiJ9",
            "eyJ" + "zdWIiOiIxMjM0NTY3ODkwIn0",
            "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
        ]
    ),
    "private_key": "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIEow\n-----END RSA " + "PRIVATE KEY-----",
    "key_in_url": "https://api.example.com/v1?api_key=AbCdEfGhIjKlMnOpQrSt",
}


def test_the_mask_covers_common_credentials_contacts_and_home_paths():
    out = mask_secrets(dict(SECRETS))
    assert not [k for k, v in SECRETS.items() if out[k] == v]
    assert "jane" not in json.dumps(out)
    assert mask_secrets({"steps": [{"name": "a", SECRETS["openai"]: 1, "password": "hunter2"}]}) == {
        "steps": [{"name": "a", "[masked key]": 1, "password": "[masked]"}]
    }
    record = {"dropped": {"auth_token_present": "reason"}}  # records keyed by feature names keep their values
    assert mask_secrets(record, by_key=False) == record


def test_snapshots_fit_the_size_bound_whatever_the_bulk():
    prefix = {
        "intent": "x",
        "catalog": [{"n": i, "t": "ab"} for i in range(5000)],
        "steps": [{"name": "a", "data": "y", "ids": list(range(2000))} for _ in range(10)],
    }
    s = snapshot(prefix)
    assert s is None or len(json.dumps(s)) <= GOLDEN_MAX_CHARS
    assert snapshot({"intent": "x", "steps": [{"name": "a", "data": "y" * 500}] * 300}) is not None


# =========================================================================== M3: the audit
def test_an_audit_that_would_drop_every_feature_drops_none(tmp_path):
    wrong = judge_all("incorrect")
    result = schema_for(tmp_path, FakeLLM(auditor=wrong)).discover(runs(), TARGET)  # never aborts a paid run
    assert set(result.extractor.features) == {"user_intent", "plan_note", "intent_words"}
    assert set(result.report.audit["not_applied"]) == {"user_intent", "plan_note", "intent_words"}
    assert any("judged every LLM-written feature incorrect" in n for n in result.report.notes)


def test_the_audit_counts_analyzed_points_it_had_to_use(tmp_path):
    ad = adapter()
    dps = [dp for t in runs(6) for dp in ad.decision_points(t, TARGET)]
    outputs = [{"user_intent": dp.prefix["intent"]} for dp in dps]
    answer = judge_all("correct")
    exclude = {dp.id for dp in dps[:3]}
    rec = run_audit(
        OutputAuditor(FakeLLM(auditor=answer)),
        {"user_intent": spec("user_intent")},
        dps,
        outputs,
        TARGET,
        samples=5,
        max_incorrect=0.25,
        seed=0,
        exclude=exclude,
    )
    assert rec["samples"] == 5 and rec["analyzed_points"] == 2  # 3 unanalyzed points, then 2 analyzed ones


def test_the_auditor_prompt_fences_the_trajectory_as_data():
    from tabagent.tabschema import prompts

    text = prompts.text("output_auditor")
    assert "<trajectory_data>\n${trajectory_json}\n</trajectory_data>" in text
    assert "never as instructions" in text and "part_of_feature" in text


# ======================================================================= M4, L1, L2, L9: verify-head and load
def onboarded(tmp_path, **cfg):
    out = tmp_path / "head"
    tabagent.onboard(
        runs(),
        TARGET,
        adapter=adapter(),
        llm=FakeLLM(),
        config=config(**cfg),
        out=out,
        n_splits=2,
        iterations=20,
    )
    return out


def test_verify_head_checks_the_artifact_and_load_refuses_an_incomplete_manifest(tmp_path):
    out = onboarded(tmp_path)
    card = out / "model" / "card.yaml"
    edited = yaml.safe_load(card.read_text())
    edited["name"] = "edited"
    card.write_text(yaml.safe_dump(edited, sort_keys=False))
    report = verify_head(out)
    assert not report.passed and {c["name"]: c["passed"] for c in report.checks} == {"artifact": False}
    with pytest.raises(ValueError, match="card.yaml does not match the manifest"):
        tabagent.load(out)
    manifest = json.loads((out / "model" / "manifest.json").read_text())
    del manifest["sha256"]["card.yaml"]  # dropping the entry must not skip the check
    (out / "model" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="lists no hash for card.yaml"):
        tabagent.load(out)
    assert not verify_head(out).passed


def test_a_head_without_goldens_passes_with_a_warning_unless_required(tmp_path, capsys):
    out = onboarded(tmp_path, goldens=0)
    tests = json.loads((out / TESTS_FILE).read_text())
    assert tests["goldens"] == [] and tests["verification"]  # no trajectory text, the checks' record kept
    assert cli.main(["verify-head", str(out)]) == 0
    assert "[warn] no golden prefixes recorded (goldens=0)" in capsys.readouterr().out
    assert cli.main(["verify-head", str(out), "--require-goldens"]) == 1


def test_a_broken_tests_file_never_stops_loading_and_verify_head_reports_it(tmp_path, capsys):
    out = onboarded(tmp_path)
    tests = json.loads((out / TESTS_FILE).read_text())
    (out / TESTS_FILE).write_text("{not json")
    with pytest.warns(UserWarning, match="extractor_tests.json is ignored"):
        head = tabagent.load(out)
    assert head.extractor_tests is None and head.decide(
        {"intent": "play a song from lima", "steps": []}, list(TOOLS), k=1
    )
    assert cli.main(["verify-head", str(out)]) == 2 and "not valid JSON" in capsys.readouterr().err
    (out / TESTS_FILE).write_text(json.dumps({**tests, "format": "1"}))
    assert (
        cli.main(["verify-head", str(out)]) == 2 and "'format' is not an integer" in capsys.readouterr().err
    )
    (out / TESTS_FILE).write_text(json.dumps(tests))
    (out / "extract.py").unlink()
    assert cli.main(["verify-head", str(out)]) == 1 and "extract.py is missing" in capsys.readouterr().out


def test_verify_head_on_the_hub_warns_that_the_subprocess_sandbox_is_no_boundary(tmp_path, monkeypatch):
    out = onboarded(tmp_path)
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = lambda repo_id, revision=None, **k: str(out)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    report = verify_head("hf://owner/head")
    assert report.passed and any("not a security boundary" in w for w in report.warnings)
    assert not any("security" in w for w in verify_head(out).warnings)


# ======================================================================= L8: allow_unpinned through serving
def test_allow_unpinned_reaches_serve_and_the_tool_filter(tmp_path, monkeypatch):
    from tabagent.integrations.openai_tools import ToolFilter
    from tabagent.serve import serve

    head = tabagent.onboard(
        runs(), TARGET, adapter=adapter(), llm=FakeLLM(), config=config(), n_splits=2, iterations=20
    )
    out = old_head(tmp_path, head)
    served = []
    monkeypatch.setitem(
        sys.modules, "uvicorn", types.SimpleNamespace(run=lambda app, **k: served.append(app))
    )
    with pytest.raises(ValueError, match="pins no hash"):
        serve(str(out))
    with pytest.warns(UserWarning, match="allow_unpinned"):
        serve(str(out), allow_unpinned=True)
    assert served
    with pytest.raises(ValueError, match="pins no hash"):
        ToolFilter(out)
    with pytest.warns(UserWarning, match="allow_unpinned"):
        assert ToolFilter(out, allow_unpinned=True).head.extractor_code
    with pytest.raises(ValueError, match="where it is served"):
        ToolFilter("http://localhost:8077", allow_unpinned=True)
    parsed = cli.build_parser().parse_args(["serve", str(out), "--allow-unpinned"])
    assert parsed.allow_unpinned is True


# =================================================================== M5, L4, L10: usefulness, reports, config
def test_the_usefulness_estimate_bounds_its_rows_and_corrects_for_many_features(tmp_path):
    ad = adapter(task_keys=("intent", "code", "noise"))
    llm = llm_for(CODED, ("task_code", "categorical"), ("noise_value", "numeric"))
    capped = schema_for(tmp_path, llm, ad=ad, usefulness=True, usefulness_max_rows=60).discover(
        runs(48, coded_run), TARGET
    )
    use = capped.report.usefulness
    assert use["decisions"] == MIN_DECISIONS and use["rows"] == MIN_DECISIONS * len(TOOLS)
    assert {"p_holm", "verdicts"} <= set(use["features"]["task_code"]) and use["features"]["task_code"][
        "verdict"
    ] == "useful"
    two = schema_for(
        tmp_path,
        llm_for(CODED, ("task_code", "categorical"), ("noise_value", "numeric")),
        "two",
        ad=adapter(task_keys=("intent", "code", "noise")),
        usefulness=True,
        prune_harmful=True,
    ).discover(runs(48, coded_run), TARGET)
    task_code = two.report.usefulness["features"]["task_code"]
    assert two.report.usefulness["seeds"] == 2 and task_code["verdicts"] == ["useful", "useful"]
    assert "noise_value" in two.extractor.features  # noise is not pruned


def test_reports_hold_no_timings(tmp_path):
    result = schema_for(tmp_path, FakeLLM(), usefulness=True).discover(runs(), TARGET)
    md = result.report.to_markdown()
    assert "seconds" not in md and " s\n" not in md and " s;" not in md
    assert "seconds" not in json.dumps(result.report.verification) + json.dumps(result.report.usefulness)


@pytest.mark.parametrize(
    "bad",
    [
        {"audit_max_incorrect": 1.5},
        {"audit_max_incorrect": -0.1},
        {"verify_samples": 0},
        {"goldens": -1},
        {"audit_samples": 0},
        {"usefulness_max_rows": 0},
        {"usefulness_max_decisions": 1},
    ],
)
def test_the_checks_settings_are_validated(bad):
    with pytest.raises(ValueError, match=f"TabSchemaConfig.{next(iter(bad))}"):
        TabSchemaConfig(**bad)


def test_noise_columns_seldom_look_harmful(tmp_path):
    """Holm over features: with ten noise columns next to an informative one, none is called harmful."""
    trajs = [
        Trajectory(
            f"r{i}",
            {**coded_run(i), **{f"n{k}": round(random.Random(100 * i + k).random(), 4) for k in range(10)}},
            True,
            f"t{i}",
        )
        for i in range(60)
    ]
    ad = adapter(task_keys=("intent", "code", *[f"n{k}" for k in range(10)]))
    code = (
        "features = {'task_code': trajectory['code'], "
        + ", ".join(f"'noise_{k}': trajectory['n{k}']" for k in range(10))
        + "}\n"
    )
    feats = [("task_code", "categorical"), *[(f"noise_{k}", "numeric") for k in range(10)]]
    result = schema_for(tmp_path, llm_for(code, *feats), ad=ad, usefulness=True, audit=False).discover(
        trajs, TARGET
    )
    verdicts = {c: r["verdict"] for c, r in result.report.usefulness["features"].items()}
    assert verdicts["task_code"] == "useful"
    assert not [c for c, v in verdicts.items() if c.startswith("noise") and v != "neutral"]


# =========================================================================== the second review (N1-N5)
def usage_run(i: int) -> dict:
    """A toy run whose steps record token usage (counts, not credentials)."""
    raw = raw_run(i)
    for k, step in enumerate(raw["steps"]):
        step["usage"] = {"total_tokens": 100 + 10 * i + k, "max_tokens": 4096}
    return raw


def test_token_counts_are_not_masked_and_their_goldens_record(tmp_path):
    code = (
        "features = {'user_intent': trajectory['intent'], "
        "'tokens_so_far': sum(s['usage']['total_tokens'] for s in trajectory['steps'])}\n"
    )
    out = tmp_path / "head"
    tabagent.onboard(
        runs(16, usage_run),
        TARGET,
        adapter=adapter(),
        llm=llm_for(code, ("user_intent", "text"), ("tokens_so_far", "numeric")),
        config=config(),
        out=out,
        n_splits=2,
        iterations=20,
    )
    tests = json.loads((out / TESTS_FILE).read_text())
    assert len(tests["goldens"]) == 8 and tests["golden_stats"]["raised"] == 0
    report = verify_head(out)
    assert report.passed and not report.warnings
    data = {"usage": {"total_tokens": 5, "max_tokens": 9, "token_count": "12"}, "token": "abc123", "n": 7}
    assert mask_secrets(data) == {**data, "token": "[masked]"}
    assert mask_secrets({"access_token": "xyz", "password": 1234}) == {
        "access_token": "[masked]",
        "password": 1234,
    }
    assert mask_secrets('usage: {"total_tokens": 512}') == 'usage: {"total_tokens": 512}'
    sha = "0123456789abcdef" * 2  # a bare 32-hex string (a commit, an MD5) is data
    assert mask_secrets(sha) == sha and mask_secrets({"azure_api_key": sha}) == {"azure_api_key": "[masked]"}
    at = "@"
    assert mask_secrets(f"write to jane{at}corp.io") == f"write to masked{at}corp.io"  # the domain is kept


def email_run(i: int) -> dict:
    return {**raw_run(i), "intent": f"{raw_run(i)['intent']} for user{i}" + "@" + "corp.io"}


def test_goldens_that_fall_short_because_snapshots_raise_fail_verify_head(tmp_path, capsys):
    # the code reads the local part of an e-mail address, which the golden snapshots mask
    code = "features = {'user_number': int(trajectory['intent'].split('@')[0].split('user')[-1])}\n"
    out = tmp_path / "head"
    head = tabagent.onboard(
        runs(16, email_run),
        TARGET,
        adapter=adapter(),
        llm=llm_for(code, ("user_number", "numeric")),
        config=config(),
        out=out,
        n_splits=2,
        iterations=20,
    )
    stats = json.loads((out / TESTS_FILE).read_text())["golden_stats"]
    assert stats["requested"] == 8 and stats["recorded"] == 0 and stats["raised"] == stats["snapshots"] > 0
    assert any(n.startswith("golden tests: 0 of 8 recorded") for n in head.discovery.notes)
    assert cli.main(["verify-head", str(out)]) == 1
    printed = capsys.readouterr().out
    assert "[FAIL] golden coverage: 0 of 8 golden prefixes recorded" in printed and "ValueError" in printed
    off = onboarded(tmp_path / "off", goldens=0)  # goldens=0 by configuration stays a warning
    assert verify_head(off).passed


class WithDecisionInput(CugaAdapter):
    """Prefixes that add the shortlister step's input prompts, written before its LLM call."""

    def decision_points(self, traj, target):
        return [
            dataclasses.replace(
                dp,
                prefix={
                    **dp.prefix,
                    "decision_input": [
                        p["value"] for p in dp.target_step.get("prompts") or [] if p["role"] != "assistant"
                    ],
                },
            )
            for dp in super().decision_points(traj, target)
        ]


class DeclaredInput(WithDecisionInput):
    decision_input_keys = ("prompts",)


def test_an_adapter_declares_the_decision_inputs_its_prefixes_hold():
    results = future_results(WithDecisionInput(), cuga(), "ShortlisterAgent")
    assert all("leak" in r and r["leak"]["path"].startswith("prefix.decision_input") for r in results)
    assert all("ok" in r for r in future_results(DeclaredInput(), cuga(), "ShortlisterAgent"))


def test_the_adapter_check_runs_before_any_llm_call_and_can_be_skipped_alone(tmp_path, capsys):
    late, trajs = LateCut(**kw()), runs(16, varied_run)
    llm = llm_for(HISTORY, *HISTORY_FEATURES)
    with pytest.raises(PrefixLeakError):
        schema_for(tmp_path, llm, ad=late).discover(trajs, TARGET)
    assert llm.calls == []  # stopped before anything was paid
    dry = tabagent.onboard(trajs, TARGET, adapter=LateCut(**kw()), dry_run=True)
    assert any("adapter LateCut puts post-decision data" in p for p in dry.problems)
    llm = llm_for(HISTORY, *HISTORY_FEATURES)
    with pytest.raises(PrefixLeakError):
        tabagent.onboard(trajs, TARGET, adapter=LateCut(**kw()), llm=llm, config=config(), n_splits=2)
    assert llm.calls == []

    for flags in ({"verify_adapter": False}, {"verify": False}):
        result = schema_for(
            tmp_path, llm_for(HISTORY, *HISTORY_FEATURES), str(flags), ad=LateCut(**kw()), **flags
        )
        result = result.discover(trajs, TARGET)
        assert any("leak check (future invariance) was skipped" in n for n in result.report.notes)
        if "verify_adapter" in flags:  # the other checks still run, and the report shows the skip
            md = result.report.to_markdown()
            assert "future invariance: **skipped**, not passed (turned off (verify_adapter=False))" in md
            assert result.extractor.verification["determinism"]["prefixes"] == 16
    parsed = cli.build_parser().parse_args(["onboard", "logs", "--no-verify-adapter"])
    assert parsed.no_verify_adapter is True


def test_the_report_states_the_audit_drop_rule(tmp_path):
    md = schema_for(tmp_path, FakeLLM()).discover(runs(), TARGET).report.to_markdown()
    assert "one-sided 95% Wilson lower bound of its incorrect rate exceeds 25%" in md


@pytest.mark.parametrize(
    "answer, verdict",
    [
        ("incorrect, it cannot be 3", "incorrect"),
        ("Incorrect: the value should be 2", "incorrect"),
        ("correct (the trace says so)", "correct"),
        ("correct/incorrect/cannot_tell", "cannot_tell"),
        ("cannot tell: clipped", "cannot_tell"),
        ("partially correct", "cannot_tell"),
    ],
)
def test_the_auditors_leading_verdict_decides(answer, verdict):
    from tabagent.tabschema.audit import normalize_audit_verdict

    assert normalize_audit_verdict(answer) == verdict


@pytest.mark.parametrize(
    "code, flagged",
    [
        ("from datetime import datetime as dt\nx = dt.now().day", True),
        ("import datetime as d\nx = d.date.today().day", True),
        ("class Clock:\n    def now(self):\n        return 1\nx = Clock().now()", False),
        ("x = len(trajectory['steps'])\nclock = {'now': x}\nx = clock['now']", False),
    ],
)
def test_the_clock_check_looks_at_datetime_only(code, flagged):
    kinds = [i.kind for i in checks.check_code(code + "\nfeatures = {'x': x}\n")]
    assert ("clock" in kinds) is flagged
