"""Tau2Adapter on fabricated tau2-bench results files."""

import json

from fake_logs import openai_tools, tau2_results

from tabagent.adapters import NEXT_TOOL_CALL, TOOL_SHORTLIST, Tau2Adapter


def _write(d, name, results):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(results))


def _logs(tmp_path):
    airline = [("0", ["get_user_details", "cancel_reservation"], 1.0), ("0", ["get_user_details"], 0.0)]
    airline.append(("1", ["search_direct_flight"], None))  # no reward_info: counted as failed
    _write(tmp_path / "final", "agentA_airline.json", tau2_results("airline", airline))
    _write(tmp_path / "final", "agentA_telecom.json", tau2_results("telecom", [("0", ["get_customer"], 1.0)]))
    return tmp_path / "final"


def test_simulations_become_runs_with_domain_groups(tmp_path):
    adapter = Tau2Adapter()
    trajs = {t.id: t for t in adapter.load(_logs(tmp_path))}
    assert set(trajs) == {"agentA_airline:0", "agentA_airline:1", "agentA_airline:2", "agentA_telecom:0"}
    assert [trajs[f"agentA_airline:{i}"].success for i in range(3)] == [True, False, False]
    assert trajs["agentA_airline:0"].group == "airline:0" and trajs["agentA_telecom:0"].group == "telecom:0"
    assert adapter.components(list(trajs.values())) == {TOOL_SHORTLIST: 2}


def test_prefix_hides_scenario_gold_actions_and_reward(tmp_path):
    adapter = Tau2Adapter()
    trajs = {t.id: t for t in adapter.load(_logs(tmp_path))}
    (dp,) = adapter.decision_points(trajs["agentA_airline:0"], TOOL_SHORTLIST)

    assert dp.prefix["task_id"] == "0" and dp.prefix["domain"] == "airline" and dp.prefix["ticket"] == ""
    assert [s["name"] for s in dp.prefix["steps"]] == ["user"]  # the greeting is an assistant turn
    dumped = json.dumps(dp.prefix)
    for secret in ("SECRET SCENARIO", "gold_only_tool", "RAW PAYLOAD", "reward", "get_user_details"):
        assert secret not in dumped
    assert adapter.labels(dp) == {"get_user_details", "cancel_reservation"}


def test_candidates_are_the_agent_tools_of_the_domain(tmp_path):
    logs = _logs(tmp_path)
    solo = tau2_results("telecom", [("0", ["get_customer", "toggle_data"], 1.0)], "llm_agent_solo")
    _write(logs, "agentA_telecom_solo.json", solo)
    adapter = Tau2Adapter()
    trajs = {t.id: t for t in adapter.load(logs)}
    (dp,) = adapter.decision_points(trajs["agentA_telecom_solo:0"], TOOL_SHORTLIST)
    assert {c.id for c in adapter.candidates(dp)} == {"get_customer", "toggle_data"}  # solo: device tools too

    (dp,) = adapter.decision_points(trajs["agentA_airline:0"], TOOL_SHORTLIST)
    cands = {c.id: c for c in adapter.candidates(dp)}
    # tools called by any airline run, failed ones included; nothing from telecom
    assert set(cands) == {"get_user_details", "cancel_reservation", "search_direct_flight"}
    assert cands["cancel_reservation"].arg_hints == "user_id"

    (tel,) = adapter.decision_points(trajs["agentA_telecom:0"], TOOL_SHORTLIST)
    assert {c.id for c in adapter.candidates(tel)} == {"get_customer"}  # the user's device tool is not one
    steps = adapter.decision_points(trajs["agentA_telecom:0"], NEXT_TOOL_CALL)
    assert len(steps) == 1 and adapter.labels(steps[0]) == {"get_customer"}

    given = Tau2Adapter(tools=openai_tools())
    trajs = {t.id: t for t in given.load(tmp_path / "final")}
    (dp,) = given.decision_points(trajs["agentA_airline:0"], TOOL_SHORTLIST)
    assert len(given.candidates(dp)) == len(openai_tools())


def test_tool_defs_solo_tickets_and_policy(tmp_path):
    sims = [("7", ["get_customer"], 1.0)]
    results = tau2_results("telecom", sims, implementation="llm_agent_solo", tool_defs=openai_tools())
    results["simulations"][0]["messages"] = results["simulations"][0]["messages"][2:]  # solo: no user turns
    _write(tmp_path, "solo.json", results)

    adapter = Tau2Adapter(include_policy=True, granularity="step")
    (traj,) = adapter.load(tmp_path)
    (dp,) = adapter.decision_points(traj, adapter.target)
    assert dp.prefix["ticket"] == "TICKET for task 7"
    assert dp.prefix["steps"][0] == {
        "name": "system",
        "content": "# telecom policy",
        "tool_calls": [],
        "tool_name": "",
    }
    assert len(adapter.candidates(dp)) == len(openai_tools())  # the file's tool_defs


def test_calls_to_tools_the_agent_lacks_are_not_candidates(tmp_path):
    results = tau2_results("telecom", [("0", ["get_customer", "check_status_bar"], 1.0)])
    for m in results["simulations"][0]["messages"]:
        if m.get("content") == "check_status_bar output":
            m["content"], m["error"] = "Error: Tool 'check_status_bar' not found.", True
    _write(tmp_path, "r.json", results)
    adapter = Tau2Adapter()
    (traj,) = adapter.load(tmp_path)
    (dp,) = adapter.decision_points(traj, TOOL_SHORTLIST)
    assert {c.id for c in adapter.candidates(dp)} == {"get_customer"}
