"""The live-validation harness (experiments/live) on fabricated τ²-bench results: the task split, the
scaffold's state at each decision, the replay through every serving path, and the paired loop report."""

from __future__ import annotations

import json

import pytest

from experiments.live import loop_report
from experiments.live.common import scaffold_states, split_tasks
from tabagent import TabAgent, TabHead
from tabagent.adapters import NEXT_TOOL_CALL, TOOL_SHORTLIST, Tau2Adapter
from tabagent.api import TabAgentHead
from tabagent.table import DecisionSpec
from tabagent.tabschema import Feature, FeatureCard
from tabagent.tabschema.compile import ExtractorRunner
from tabagent.tabschema.orchestrator import CANDIDATE, DECISION, GROUP, LABEL, rows_to_table

TOOLS = {
    "get_user_details": "Get the details of a user.",
    "cancel_reservation": "Cancel the whole reservation.",
    "search_direct_flight": "Search for direct flights between two cities.",
    "update_reservation_baggages": "Update the baggage information of a reservation.",
}
CODE = """
def extract_features_for_classification(trajectory_data):
    steps = trajectory_data['steps']
    users = [s['content'] for s in steps if s['name'] == 'user']
    called = [t for s in steps for t in s['tool_calls']]
    last = called[-1] if called else ""
    return {"domain": trajectory_data['domain'], "n_calls": len(called), "last_call": last,
            "user_text": " ".join(users)}
features = extract_features_for_classification(trajectory)
"""


def _messages(i, tools):
    msgs = [
        {"role": "assistant", "content": "Hi! How can I help you today?", "tool_calls": None},
        {"role": "user", "content": f"Request {i}: please help with {tools[-1].replace('_', ' ')}."},
    ]
    for k, t in enumerate(tools):
        call = {"id": f"c{k}", "name": t, "arguments": {"user_id": "u1"}, "requestor": "assistant"}
        msgs += [
            {"role": "assistant", "content": None, "tool_calls": [call], "raw_data": {"x": 1}},
            {"id": f"c{k}", "role": "tool", "content": f"{t} ok", "requestor": "assistant"},
            {"role": "user", "content": "Thanks, go on."},
        ]
    return msgs + [{"role": "assistant", "content": "Done.", "tool_calls": None}]


def _results(domain="airline", n_tasks=12, reward=1.0, usage=False):
    defs = [
        {
            "type": "function",
            "function": {"name": n, "description": d, "parameters": {"properties": {"user_id": {}}}},
        }
        for n, d in TOOLS.items()
    ]
    plans = [
        ["get_user_details", "cancel_reservation"],
        ["search_direct_flight"],
        ["update_reservation_baggages"],
    ]
    sims = []
    for t in range(n_tasks):
        msgs = _messages(t, plans[t % 3])
        if usage:
            for m in msgs[2:]:
                if m["role"] == "assistant":
                    m["usage"] = {"prompt_tokens": 1000, "completion_tokens": 10}
                    m["raw_data"] = {
                        "usage": {"prompt_tokens_details": {"cached_tokens": 800 if reward else 200}}
                    }
        sims.append(
            {
                "task_id": str(t),
                "trial": 0,
                "messages": msgs,
                "reward_info": {"reward": reward if t % 2 else 1.0},
            }
        )
    return {
        "info": {"agent_info": {"llm": "m"}, "environment_info": {"domain_name": domain, "tool_defs": defs}},
        "tasks": [{"id": str(t)} for t in range(n_tasks)],
        "simulations": sims,
    }


@pytest.fixture(scope="module")
def logs(tmp_path_factory):
    d = tmp_path_factory.mktemp("tau2")
    (d / "airline.json").write_text(json.dumps(_results()))
    return d


def test_split_is_seeded_stratified_and_by_task(logs):
    trajs = Tau2Adapter().load(logs)
    a, b = split_tasks(trajs, 0.25, seed=3), split_tasks(trajs, 0.25, seed=3)
    assert a == b and len(a["test"]) == 3 and not set(a["train"]) & set(a["test"])
    assert set(a["train"]) | set(a["test"]) == {t.group for t in trajs}


@pytest.mark.parametrize("target", [NEXT_TOOL_CALL, TOOL_SHORTLIST])
def test_scaffold_states_rebuild_the_training_prefixes(logs, target):
    adapter = Tau2Adapter(granularity="step" if target == NEXT_TOOL_CALL else "task")
    head = TabAgentHead(None, target, adapter=adapter)
    for traj in adapter.load(logs):
        states = scaffold_states(adapter, traj, target)
        assert len(states) == ((len(traj.raw["messages"]) - 3) // 3 if target == NEXT_TOOL_CALL else 1)
        for dp, live in states:
            assert "reward" not in live and live["domain"] == "airline" and live["tools"]
            assert head.live_prefix(live) == dp.prefix


@pytest.fixture(scope="module")
def harness_head(logs):
    """A NextToolCall head fitted on the fabricated runs, with CODE as its extractor."""
    adapter = Tau2Adapter(granularity="step")
    trajs = adapter.load(logs)
    dps = [dp for t in trajs for dp in adapter.decision_points(t, NEXT_TOOL_CALL)]
    spec = DecisionSpec("ranking", GROUP, LABEL, CANDIDATE, decision=(DECISION,))
    card = FeatureCard(
        "live_harness",
        spec,
        [
            Feature("domain", "state", "cat", "extractor"),
            Feature("n_calls", "state", "num", "extractor"),
            Feature("last_call", "state", "cat", "extractor"),
            Feature("user_text", "state", "text", "extractor"),
            Feature("candidate_name", "schema", "cat"),
            Feature("candidate_description", "schema", "text"),
            Feature("candidate_args", "schema", "text"),
        ],
    )
    table = rows_to_table(
        card, dps, ExtractorRunner(CODE, trust_extractor=True).run([d.prefix for d in dps]), adapter
    )
    agent = TabAgent(card, head=TabHead("ranking", iterations=20)).fit(table)
    return agent, adapter, trajs


@pytest.mark.filterwarnings("ignore:the TabSchema extractor failed:RuntimeWarning")
def test_replay_finds_full_parity_and_scores_quality(harness_head):
    pytest.importorskip("fastapi")
    from experiments.live.replay import quality, replay, robustness, summarize

    agent, adapter, trajs = harness_head
    head = TabAgentHead(agent, NEXT_TOOL_CALL, CODE, adapter=adapter, trust_extractor=True)
    sandboxed = TabAgentHead(agent, NEXT_TOOL_CALL, CODE, adapter=adapter)

    records = replay(head, adapter, trajs[:4], NEXT_TOOL_CALL, k=2, sandboxed=sandboxed, sandbox_every=3)
    summary = summarize(records, [1, 2, 4])
    assert summary["n_decisions"] == len(records) > 0
    assert set(summary["parity_pct"].values()) == {100.0} and summary["max_score_diff"] < 1e-9
    assert summary["quality"]["k=4"]["recall"] == 1.0  # every tool is in the top 4 of 4
    assert quality(records, [1])["k=1"]["recall_with_used"] >= quality(records, [1])["k=1"]["recall"]

    live = scaffold_states(adapter, trajs[0], NEXT_TOOL_CALL)[0][1]
    rob = robustness({"in_process": head}, [live], k=2)
    outcomes = {p: {o for c in v["outcomes"].values() for o in c} for p, v in rob.items()}
    assert outcomes["missing_run_level_field"] == {"deferred"}
    assert all("crashed" not in o for o in outcomes.values()), rob


def test_the_loop_sessions_on_a_served_head_match_the_local_head(harness_head):
    """tau2_loop.py's code path without τ²-bench: a ToolSession per conversation on the served head (the run
    holds the domain, the τ²-bench messages and the tool schemas), as the in-loop arms ran it."""
    pytest.importorskip("fastapi")
    from tabagent.integrations.openai_tools import ToolFilter

    from conftest import served

    agent, adapter, trajs = harness_head
    head = TabAgentHead(agent, NEXT_TOOL_CALL, CODE, adapter=adapter, trust_extractor=True)
    with served(head) as url:
        for sticky in (False, True):
            remote, local = ToolFilter(url, k=2, task={"domain": "airline"}), ToolFilter(head, k=2)
            for traj in trajs[:4]:
                a, b = remote.session(sticky=sticky), local.session(sticky=sticky)
                prev: list[str] = []
                for _, live in scaffold_states(adapter, traj, NEXT_TOOL_CALL):
                    sent = a.select(live["messages"], live["tools"])
                    task = {"domain": live["domain"]}
                    assert sent == b.select(live["messages"], live["tools"], task) and not a.last["error"]
                    assert set(a.used) <= set(sent) and set(a.last["candidates"]) <= set(sent)
                    assert not sticky or sent[: len(prev)] == prev  # sticky lists only grow
                    prev = sent


def test_loop_report_pairs_arms_by_task(tmp_path):
    base, filt = tmp_path / "a.json", tmp_path / "b.json"
    base.write_text(json.dumps(_results(usage=True)))
    filt.write_text(json.dumps(_results(reward=0.0, usage=True)))  # odd tasks fail, cache share drops
    rep = loop_report.report({"base": [str(base)], "filtered": [str(filt)]})
    assert rep["n_tasks"] == 12 and rep["arms"]["base"]["pass_rate"] == 1.0
    assert rep["arms"]["filtered"]["pass_rate"] == 0.5
    pair = rep["paired_vs_base"]["filtered"]
    assert pair["pass_rate_diff"]["mean"] == pytest.approx(-0.5) and pair["tasks_worse"] == 6
    assert rep["arms"]["base"]["agent_cached_share"] == pytest.approx(0.8)
    assert pair["agent_cost_per_sim_diff"]["mean"] > 0  # the uncached share is billed at the full price
