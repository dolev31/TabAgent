"""`tabagent scan` on fabricated logs: ranking, the closed-set test, missing usage, prices, CLI, JSON."""

import json

import pytest
from adapters.fake_logs import (
    cuga_trajectory,
    openai_run,
    tau2_results,
    tau2_with_usage,
    write_generic_logs,
    write_openai_logs,
)

import tabagent
from tabagent.adapters import (
    NEXT_TOOL_CALL,
    SHORTLISTER,
    TEXT_REPLY,
    TOOL_SHORTLIST,
    GenericJsonAdapter,
    LLMCall,
    OpenAIMessagesAdapter,
    Tau2Adapter,
    output_fields,
    read_usage,
)
from tabagent.cli import main
from tabagent.scanner import PAPER_REFERENCE
from tabagent.tabschema.trajectory import Trajectory


def _by_name(report):
    return {c.name: c for c in report.components}


# --------------------------------------------------------------------------- ranking and cost
def test_closed_set_components_rank_first_by_their_share_of_the_cost(tmp_path):
    report = tabagent.scan(write_generic_logs(tmp_path / "logs", n=10))
    assert report.adapter == "GenericJsonAdapter" and report.n_runs == 10 and report.n_successful == 8
    assert [c.name for c in report.components] == ["router", "check", "writer"]  # lookup calls no LLM
    assert [c.verdict for c in report.components] == ["closed-set", "closed-set", "free text"]
    assert report.ranked_by == "cost" and report.cost_source == "recorded"

    router, check, writer = report.components
    assert writer.cost_usd.share == pytest.approx(0.1 / 0.144)  # the costliest, but free text: last
    assert router.cost_usd.share == pytest.approx(0.04 / 0.144) and router.cost_usd.per_call == pytest.approx(
        0.004
    )
    assert router.tokens.per_call == 1005 and router.tokens.share == pytest.approx(10050 / 35060)
    assert router.latency_s.per_call == pytest.approx(0.8) and writer.latency_s.per_call == pytest.approx(5.0)
    assert router.calls == 10 and router.calls_per_run == 1.0 and router.successful_runs == 8
    assert router.replaceable is True and writer.replaceable is False

    ev = router.evidence
    assert (ev.field, ev.values, ev.distinct, ev.recurring_share, ev.catalog_size) == (
        "output",
        10,
        3,
        1.0,
        None,
    )
    assert dict(ev.top) == {"billing": 4, "shipping": 3, "returns": 3}
    assert "3 distinct values" in router.reason and "no catalog recorded" in router.reason
    assert "free text" in writer.reason and "characters per call" in writer.reason


def test_prices_price_the_recorded_tokens(tmp_path):
    logs = write_generic_logs(tmp_path / "logs", n=10, cost=False, model="m1")
    assert tabagent.scan(logs).ranked_by == "tokens"  # no cost recorded: the next measure ranks

    report = tabagent.scan(logs, prices=(1.0, 2.0))
    router = _by_name(report)["router"]
    assert report.ranked_by == "cost" and report.cost_source == "your prices"
    assert router.cost_usd.per_call == pytest.approx((1000 * 1.0 + 5 * 2.0) / 1e6)
    assert any("your prices" in n for n in report.notes)

    by_model = tabagent.scan(logs, prices={"m1": (1.0, 2.0)})
    assert _by_name(by_model)["router"].cost_usd.per_call == router.cost_usd.per_call
    unpriced = tabagent.scan(logs, prices={"other-model": (1.0, 2.0)})
    assert unpriced.ranked_by == "tokens" and _by_name(unpriced)["router"].cost_usd.total is None
    with pytest.raises(ValueError, match="USD per 1M input tokens"):
        tabagent.scan(logs, prices=(1.0,))


# --------------------------------------------------------------------------- the closed-set test
def test_tool_calls_are_closed_set_against_the_recorded_catalog(tmp_path):
    logs = write_openai_logs(tmp_path / "logs", n=12)
    report = tabagent.scan(logs)
    comps = _by_name(report)
    assert set(comps) == {NEXT_TOOL_CALL, TEXT_REPLY}
    tool = comps[NEXT_TOOL_CALL]
    assert tool.verdict == "closed-set" and tool.evidence.catalog_size == 6
    assert tool.evidence.in_catalog_share == 1.0 and "in the catalog offered to the call" in tool.reason
    assert tool.free_text_per_call > 0  # the arguments are generated text
    assert comps[TEXT_REPLY].verdict == "free text"
    assert tool.decisions == [TOOL_SHORTLIST, NEXT_TOOL_CALL] and tool.onboard_ready
    assert tool.next_steps == [
        f"tabagent onboard {logs} --decision {TOOL_SHORTLIST}",
        f"tabagent onboard {logs} --decision {NEXT_TOOL_CALL}",
    ]
    # no usage in these logs: ranked by calls, every measure "not recorded"
    assert report.ranked_by == "calls" and tool.tokens.total is None and tool.cost_usd.share is None
    assert "Tokens, latency and cost: not recorded in these logs." in report.summary()


def test_choices_outside_the_catalog_are_not_closed_set():
    trajs = []
    for i in range(6):
        run = openai_run(i)
        for m in run["messages"]:
            for tc in m.get("tool_calls") or []:
                tc["function"]["name"] = f"made_up_tool_{i}"  # never in the run's tools
        trajs.append(Trajectory(f"r{i}", run, True, f"t{i}"))
    report = tabagent.scan(trajs, adapter=OpenAIMessagesAdapter())
    tool = _by_name(report)[NEXT_TOOL_CALL]
    assert tool.verdict == "open values" and tool.evidence.in_catalog_share == 0.0
    assert "only 0% of its choices are in the offered catalog" in tool.reason


def test_unique_short_outputs_constant_outputs_and_too_few_calls(tmp_path):
    adapter = GenericJsonAdapter()
    steps = [
        [
            {"name": "namer", "output": f"name-{i}", "model": "m"},
            {"name": "ack", "output": "ok", "model": "m"},
        ]
        for i in range(6)
    ]
    trajs = [Trajectory(f"r{i}", {"steps": s}) for i, s in enumerate(steps)]
    comps = _by_name(tabagent.scan(trajs, adapter=adapter))
    assert comps["namer"].verdict == "open values" and comps["namer"].evidence.distinct == 6
    assert "only 0% of them recur" in comps["namer"].reason
    assert comps["ack"].verdict == "unknown" and "always take the same value" in comps["ack"].reason
    few = _by_name(tabagent.scan(trajs[:3], adapter=adapter))
    assert few["namer"].verdict == "unknown" and "too few calls" in few["namer"].reason


def test_output_fields_and_usage_shapes():
    out = {
        "thoughts": ["a long chain of reasoning that is not an output"],
        "action": "CoderAgent",
        "done": False,
        "apps": ["phone", "venmo"],
        "result": [{"name": "phone_search_contacts", "relevance_score": 0.9}],
        "plan": "x" * 100,
        "detail": {"app_name": "phone", "score": 3},
    }
    fields, text = output_fields(out)
    assert fields == {
        "action": ("CoderAgent",),
        "done": ("false",),
        "apps": ("phone", "venmo"),
        "result": ("phone_search_contacts",),
        "detail.app_name": ("phone",),
    }
    assert text == 100  # the plan; reasoning fields and numbers count as neither

    assert read_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 2}, "cost": 0.5}) == {
        "input_tokens": 10,
        "output_tokens": 2,
        "cost_usd": 0.5,
    }
    anthropic = {"usage": {"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 100}}
    assert read_usage(anthropic) == {"input_tokens": 105, "output_tokens": 1}  # cache reads are input
    langchain = {
        "usage_metadata": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        "response_metadata": {"model_name": "gpt-x", "token_usage": {"prompt_tokens": 7}},
    }
    assert read_usage(langchain) == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
        "model": "gpt-x",
    }
    assert read_usage({"latency_ms": 250, "tokens": 40}) == {"latency_s": 0.25, "total_tokens": 40}
    assert read_usage({"usage": None, "cost": True}) == {}  # a boolean is not a number
    assert LLMCall("c", total_tokens=40).tokens == 40 and LLMCall("c").tokens is None


# --------------------------------------------------------------------------- what each format records
def test_tau2_agent_calls_usage_and_latency_from_timestamps(tmp_path):
    sims = [(str(i), ["get_user_details", "cancel_reservation"], 1.0 if i < 3 else 0.0) for i in range(4)]
    results = tau2_with_usage(tau2_results("telecom", sims))  # telecom: the user simulator also calls tools
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "agent_telecom.json").write_text(json.dumps(results))
    report = tabagent.scan(tmp_path / "logs")
    assert report.adapter == "Tau2Adapter" and report.n_runs == 4 and report.n_successful == 3
    comps = _by_name(report)
    assert set(comps) == {NEXT_TOOL_CALL, TEXT_REPLY}  # no greeting, no user-simulator turn
    tool, reply = comps[NEXT_TOOL_CALL], comps[TEXT_REPLY]
    assert (tool.calls, reply.calls) == (8, 4)
    assert tool.tokens.per_call == 1020 and tool.cost_usd.per_call == pytest.approx(0.002)
    assert tool.latency_s.per_call == pytest.approx(1.5)  # each agent message 1.5 s after the one before
    assert tool.cost_usd.share == pytest.approx(8 / 12) and report.cost_source == "recorded"
    assert any("timestamp per message" in n for n in report.notes)

    adapter = Tau2Adapter()
    calls = adapter.llm_calls(adapter.load(tmp_path / "logs")[0])
    assert {c.model for c in calls} == {"agent-model"}
    assert [c.choices for c in calls if c.component == NEXT_TOOL_CALL] == [
        ("get_user_details",),
        ("cancel_reservation",),
    ]


def _cuga_logs(tmp_path, n=6, **kw):
    exp = tmp_path / "logs" / "exp_1"
    exp.mkdir(parents=True)
    phone = ["phone_search_contacts", "phone_show_text_messages"]
    for i in range(n):
        rounds = [("phone", phone, phone)] + (
            [("venmo", ["venmo_send_money"], ["send_money"])] if i % 2 else []
        )
        traj = cuga_trajectory(f"t{i}", rounds, score=0.0 if i == n - 1 else 1.0, **kw)
        traj["steps"][-2]["data"] = json.dumps({"final_answer": f"done with task {i}"})
        (exp / f"t{i}.json").write_text(json.dumps(traj))
    return tmp_path / "logs"


@pytest.mark.parametrize("merged_prompts", [False, True])
def test_cuga_records_no_usage_and_the_shortlister_is_closed_set(tmp_path, merged_prompts):
    logs = _cuga_logs(tmp_path, merged_prompts=merged_prompts)
    report = tabagent.scan(logs)
    assert report.adapter == "CugaAdapter" and report.ranked_by == "calls"
    comps = _by_name(report)
    short = comps[SHORTLISTER]
    assert short.verdict == "closed-set" and short.evidence.in_catalog_share == 1.0
    assert short.evidence.catalog_size == 5  # the phone and venmo APIs offered across the runs
    assert short.next_steps == [f"tabagent onboard {logs} --decision {SHORTLISTER}"]
    assert comps["CodeAgent"].verdict == "free text" and comps["FinalAnswerAgent"].verdict == "free text"
    assert comps["TaskAnalyzerAgent"].verdict == "closed-set"  # the apps it picks recur
    for c in report.components:
        assert c.tokens.total is None and c.latency_s.total is None and c.cost_usd.total is None
    rank = {"closed-set": 0, "unknown": 1, "open values": 2, "free text": 2}
    order = [rank[c.verdict] for c in report.components]
    assert order == sorted(order) and comps["PlanControllerAgent"].verdict == "unknown"  # a constant output

    text = report.summary()
    assert "Tokens, latency and cost: not recorded in these logs." in text
    assert "tokens, latency and cost not recorded" in text  # in each candidate's cost line
    assert any("results.json holds per-task totals" in n for n in report.notes)

    planner = comps["APIPlannerAgent"]  # a closed-set route the adapter builds no candidates for
    assert planner.verdict == "closed-set" and planner.evidence.field == "action"
    assert not planner.onboard_ready and "builds no candidates or labels" in planner.next_steps[0]
    assert planner.next_steps[1].startswith("tabagent.onboard(") and "candidates_fn=" in planner.next_steps[1]
    assert "```python" in report.to_markdown()


def test_an_adapter_without_the_hook_counts_decision_points(tmp_path):
    class Minimal:  # the adapter protocol without llm_calls
        def __init__(self):
            self.inner = GenericJsonAdapter(
                candidates_fn=lambda dp: [],
                labels_fn=lambda dp, raw: {dp.target_step["output"]},
            )

        def load(self, path):
            return self.inner.load(path)

        def components(self, trajs):
            return self.inner.components(trajs)

        def decision_points(self, traj, target):
            return self.inner.decision_points(traj, target)

        def candidates(self, dp):
            return self.inner.candidates(dp)

        def labels(self, dp):
            return self.inner.labels(dp)

    logs = write_generic_logs(tmp_path / "logs", n=10)
    report = tabagent.scan(logs, adapter=Minimal())
    assert any("no llm_calls hook" in n for n in report.notes)
    router = _by_name(report)["router"]
    assert router.calls == 10 and router.inferred_calls == 10  # one per decision point
    assert router.verdict == "closed-set" and router.tokens.total is None and report.ranked_by == "calls"


def test_steps_without_llm_marks_are_counted_and_flagged(tmp_path):
    logs = write_generic_logs(tmp_path / "logs", n=10, usage=False, cost=False)
    report = tabagent.scan(logs)
    assert {c.name for c in report.components} == {"router", "check", "writer", "lookup"}
    assert all(c.inferred_calls == c.calls for c in report.components)
    assert any("do not mark which steps call an LLM" in n for n in report.notes)


# --------------------------------------------------------------------------- output, CLI
def test_json_markdown_and_the_paper_reference_points(tmp_path):
    report = tabagent.scan(write_generic_logs(tmp_path / "logs", n=10))
    d = json.loads(report.to_json())
    assert [c["name"] for c in d["components"]] == ["router", "check", "writer"]
    first = d["components"][0]
    assert first["rank"] == 1 and first["replaceable"] is True and first["verdict"] == "closed-set"
    assert first["evidence"]["top"][0] == ["billing", 4]
    assert set(first["cost_usd"]) == {"total", "per_call", "share", "recorded_calls"}
    assert d["paper_reference"]["text"] == PAPER_REFERENCE
    assert d["paper_reference"]["appworld_successful_trajectories"] == 118
    assert d["paper_reference"]["tau2_bench_agents_successful_trajectories"] == [37, 51]
    assert PAPER_REFERENCE == (
        "TabHead reaches 90% of its full-pool P@R from 118 successful trajectories on AppWorld and from 37 "
        "to 51 on the four τ²-bench agents"
    )
    md = report.to_markdown()
    assert md.startswith("# TabAgent scan") and "| 1 | router | closed-set |" in md
    for text in (md, report.summary()):
        assert "not a prediction for your agent" in text
        assert "accuracy" not in text.lower()


def test_cli_scan(tmp_path, capsys):
    logs = write_openai_logs(tmp_path / "logs", n=12)
    assert main(["scan", str(logs)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Scanned 12 runs (11 successful) with OpenAIMessagesAdapter")
    assert f"tabagent onboard {logs} --decision ToolShortlist" in out

    assert main(["scan", str(logs), "--format", "json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["components"][0]["name"] == NEXT_TOOL_CALL and d["runs"] == 12

    assert main(["scan", str(logs), "--format", "markdown", "--top", "1"]) == 0
    assert capsys.readouterr().out.startswith("# TabAgent scan")

    assert main(["scan", str(logs), "--price-in", "1"]) == 2
    assert "go together" in capsys.readouterr().err
    assert main(["scan", str(tmp_path / "missing")]) == 2
    assert "no such file or directory" in capsys.readouterr().err


def test_cli_scan_help(capsys):
    with pytest.raises(SystemExit):
        main(["scan", "--help"])
    out = capsys.readouterr().out
    assert "--price-in" in out and "--format" in out and "closed set" in out
