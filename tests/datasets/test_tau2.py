"""τ²-bench loaders on fabricated inputs written in the upstream formats (no benchmark data)."""

import json

import pandas as pd
import pytest

from tabagent import datasets
from tabagent.datasets import tau2
from tabagent.evaluation import stable_uid
from tabagent.table import DecisionTable

AGENT = "claude-3-7-sonnet-20250219"
SCENARIO = {"persona": None, "instructions": {"reason_for_call": "Cancel my trip.", "domain": "airline"}}


def _sim(sid, task_id, reward, *calls):
    msgs = [
        {"role": "assistant", "content": "Hi", "tool_calls": None},
        {"role": "user", "content": f"customer request for task {task_id}", "tool_calls": None},
    ]
    for i, (name, requestor) in enumerate(calls):
        call = {"id": f"{sid}-{i}", "name": name, "arguments": {}, "requestor": requestor}
        msgs.append({"role": requestor, "content": None, "tool_calls": [call]})
        msgs.append({"role": "tool", "id": call["id"], "content": "ok"})
    return {"id": sid, "task_id": task_id, "trial": 0, "reward_info": {"reward": reward}, "messages": msgs}


def _result_file(domain, agent=AGENT, commit="c0ffee", sims=()):
    tasks = [
        {"id": "0", "description": None, "user_scenario": SCENARIO},
        {
            "id": "1",
            "description": None,
            "user_scenario": {"persona": "calm", "instructions": "Pay my bill."},
        },
        {"id": "2", "description": None, "user_scenario": None},
    ]
    return {
        "info": {
            "git_commit": commit,
            "agent_info": {"llm": agent},
            "environment_info": {"domain_name": domain},
        },
        "tasks": tasks,
        "simulations": list(sims),
    }


def _task_uid(commit, domain, task):
    return stable_uid(
        "task", "tau2_official_final", f"{commit}:{domain}:{task['id']}", tau2.canonical_hash(task)
    )


# --------------------------------------------------------------------------- official release
def test_scenario_text_is_the_repr_of_the_key_sorted_dict():
    text = tau2.scenario_text(SCENARIO)
    assert (
        text
        == "{'instructions': {'domain': 'airline', 'reason_for_call': 'Cancel my trip.'}, 'persona': None}"
    )
    assert tau2.scenario_text(None) == "" and tau2.scenario_text("plain") == "plain"


def test_agent_labels_union_successful_runs_only():
    airline = _result_file(
        "airline",
        sims=[
            _sim("a", "0", 1.0, ("get_user_details", "assistant"), ("cancel_reservation", "assistant")),
            _sim("b", "0", 1.0, ("calculate", "assistant")),
            _sim("c", "0", 0.0, ("book_reservation", "assistant")),  # failed run: ignored
            _sim("d", "1", 0.0, ("calculate", "assistant")),  # task 1 never succeeds: dropped
            _sim("e", "2", 1.0),  # success without a tool call: dropped
        ],
    )
    telecom = _result_file("telecom", sims=[_sim("f", "1", 1, ("toggle_data", "user"))])
    other = _result_file(
        "retail", agent="o4-mini-2025-04-16", sims=[_sim("g", "0", 1.0, ("calculate", "assistant"))]
    )
    df = tau2.agent_decisions([("x.json", airline), ("y.json", telecom), ("z.json", other)], AGENT)

    assert list(df.columns) == tau2.AGENT_COLUMNS
    assert len(tau2.AGENT_CATALOG) == 67 and list(tau2.AGENT_CATALOG) == sorted(tau2.AGENT_CATALOG)
    t0 = _task_uid("c0ffee", "airline", airline["tasks"][0])
    t1 = _task_uid("c0ffee", "telecom", telecom["tasks"][1])
    assert list(pd.unique(df.task_uid)) == [t0, t1] and len(df) == 2 * 67
    pos = df[df.used == 1].groupby("task_uid").tool.apply(set).to_dict()
    assert pos == {t0: {"get_user_details", "cancel_reservation", "calculate"}, t1: {"toggle_data"}}
    row = df[(df.task_uid == t1) & (df.tool == "toggle_data")].iloc[0]
    assert row.tool_name == "toggle data" and row.domain == "telecom"
    assert row.instruction == "customer request for task 1"  # what the agent sees, not the hidden scenario
    payloads = [("x.json", airline), ("y.json", telecom)]
    scenario = tau2.agent_decisions(payloads, AGENT, context="user_scenario")
    assert (
        scenario[scenario.task_uid == t1].instruction.iloc[0]
        == "{'instructions': 'Pay my bill.', 'persona': 'calm'}"
    )
    with pytest.raises(ValueError, match="context"):
        tau2.agent_decisions([("x.json", airline)], AGENT, context="oracle")


def test_agent_duplicate_runs_and_shared_tasks():
    # the same simulation id twice in one file counts once; an identical task in two files is one group
    first = _result_file("airline", sims=[_sim("a", "0", 1.0, ("calculate", "assistant"))] * 2)
    second = _result_file("airline", sims=[_sim("a", "0", 1.0, ("list_all_airports", "assistant"))])
    df = tau2.agent_decisions([("x.json", first), ("y.json", second)], AGENT)
    assert df.task_uid.nunique() == 1
    assert set(df[df.used == 1].tool) == {"calculate", "list_all_airports"}


def test_agent_rejects_unknown_tools_and_orphan_simulations():
    bad = _result_file("airline", sims=[_sim("a", "0", 1.0, ("teleport", "assistant"))])
    with pytest.raises(ValueError, match="teleport"):
        tau2.agent_decisions([("x.json", bad)], AGENT)
    orphan = _result_file("airline", sims=[_sim("a", "9", 1.0, ("calculate", "assistant"))])
    with pytest.raises(ValueError, match="not embedded"):
        tau2.agent_decisions([("x.json", orphan)], AGENT)


def _mirror(tmp_path, monkeypatch, source, files):
    root = tmp_path / "mirror"
    for rel, content in files.items():
        p = root / source.id / source.release_id / "files" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content if isinstance(content, str) else json.dumps(content))
    monkeypatch.setenv("TABAGENT_SOURCE_MIRROR", str(root))
    monkeypatch.setenv("TABAGENT_CACHE", str(tmp_path / "cache"))


def test_load_agent_through_mirror(tmp_path, monkeypatch):
    files = {}
    for rel in tau2.OFFICIAL.files(tau2.RESULTS_DIR):
        name = rel.rsplit("/", 1)[1]
        if name.startswith(AGENT + "_"):
            domain = name[len(AGENT) + 1 :].split("_")[0]
            files[rel] = _result_file(domain, sims=[_sim("a", "0", 1.0, ("calculate", "assistant"))])
    assert len(files) == 3  # airline, retail and telecom
    _mirror(tmp_path, monkeypatch, tau2.OFFICIAL, files)
    table = datasets.load(f"tau2/{AGENT}")
    assert isinstance(table, DecisionTable) and table.df.task_uid.nunique() == 3
    assert table.spec == datasets.card(f"tau2/{AGENT}").spec
    with pytest.raises(ValueError, match="no result files"):
        tau2.build_agent(tau2.OFFICIAL, agent="gpt-5")


# --------------------------------------------------------------------------- tau2-sft-final
TOOLKIT = '''
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class DemoTools(ToolKitBase):
    def _helper(self, x):
        """Not a tool."""

    @is_tool(ToolType.READ)
    def get_user_details(self, user_id: str) -> dict:
        """
        Get the details of a user.

        Args:
            user_id: The user id.
        """

    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """Calculate the result of a mathematical expression."""

    @is_tool(ToolType.WRITE)
    def {write}(self, order_id: str, *, reason: str) -> str:
        """
        {doc}
        """

    # @is_tool(ToolType.THINK)
    # def think(self, thought: str) -> str:
    #     return ""
'''
CATALOGS = {
    "airline": tau2.parse_toolkit(TOOLKIT.format(write="cancel_reservation", doc="Cancel a reservation.")),
    "retail": tau2.parse_toolkit(TOOLKIT.format(write="return_items", doc="Return delivered items.")),
}


def test_parse_toolkit_reads_decorated_methods_only():
    tools = {t["tool"]: t for t in CATALOGS["airline"]}
    assert list(tools) == ["calculate", "cancel_reservation", "get_user_details"]
    assert tools["get_user_details"] == {
        "tool": "get_user_details",
        "description": "Get the details of a user.",
        "args": ["user_id"],
    }
    assert tools["cancel_reservation"]["args"] == ["order_id", "reason"]


def test_called_tools_reads_both_trace_formats():
    msgs = [
        {"role": "assistant", "content": "[ACTION]\nget_user_details(user_id='u1')\n[/ACTION]"},
        {"role": "user", "content": "[TOOL_RESULT]\nget_user_details(user_id='u1')\n{}\n[/TOOL_RESULT]"},
        {"role": "assistant", "content": '{"name": "calculate", "arguments": {}}'},
        {"role": "user", "content": 'my "name": "return_items" is not a call'},
    ]
    assert tau2.called_tools(msgs) == {"get_user_details", "calculate"}


def _record(i, domain, task, tools, reward=None, source="gpt52"):
    msgs = [{"role": "system", "content": "policy"}, {"role": "user", "content": f"request {i} for {domain}"}]
    for t in tools:
        msgs.append({"role": "assistant", "content": f"[ACTION]\n{t}(x=1)\n[/ACTION]"})
        msgs.append({"role": "user", "content": f"[TOOL_RESULT]\n{t}(x=1)\nok\n[/TOOL_RESULT]"})
    meta = {"original_task_id": f"[{domain}]{task}"} | ({"reward": reward} if reward is not None else {})
    return {
        "task_id": f"[{domain}]{task}[sample_{i}]",
        "tau2_task_id": task,
        "domain": domain,
        "source": source,
        "messages": msgs,
        "metadata": meta,
    }


RECORDS = [
    _record(0, "airline", "7", ["get_user_details", "cancel_reservation"], reward=1.0),
    _record(1, "airline", "7", ["calculate"], reward=0.0),
    _record(2, "airline", "8", ["search_flights"]),  # no catalog tool: dropped
    _record(3, "retail", "7", ["return_items", "get_user_details"], source="hf_converted"),
    _record(4, "telecom", "1", ["calculate"]),  # domain without a catalog: dropped
]


def test_domain_rows_labels_and_groups():
    df = tau2.domain_decisions(RECORDS, CATALOGS, "airline")
    assert list(df.columns) == tau2.DOMAIN_COLUMNS
    assert df.traj_uid.nunique() == 2 and len(df) == 2 * 3
    assert set(df.task_id) == {"airline|7"}  # both trajectories of one task share a group
    first = df[df.traj_uid == df.traj_uid.iloc[0]]
    assert first.set_index("tool").used.to_dict() == {
        "calculate": 0,
        "cancel_reservation": 1,
        "get_user_details": 1,
    }
    assert first.instruction.iloc[0] == "request 0 for airline" and first.reward.iloc[0] == 1.0
    msgs = RECORDS[0]["messages"]
    assert first.traj_uid.iloc[0] == stable_uid(
        "traj", "tau2_sft_final", RECORDS[0]["task_id"], tau2.canonical_hash(msgs)
    )
    row = first.set_index("tool").loc["cancel_reservation"]
    assert row.tool_desc == "Cancel a reservation." and row.tool_args == "order id reason"
    with pytest.raises(ValueError, match="unknown domain"):
        tau2.domain_decisions(RECORDS, CATALOGS, "banking")


def test_all_domains_merges_shared_tools():
    df = tau2.domain_decisions(RECORDS, CATALOGS, None)
    assert df.traj_uid.nunique() == 3 and set(df.task_id) == {"airline|7", "retail|7"}
    assert sorted(df.tool.unique()) == ["calculate", "cancel_reservation", "get_user_details", "return_items"]
    retail = df[df.domain == "retail"].set_index("tool")
    assert retail.loc["cancel_reservation", "tool_domain"] == "airline"  # another domain's tool
    assert retail.loc["get_user_details", "tool_domain"] == "retail"  # shared: described by own domain
    assert retail.loc["return_items", "used"] == 1 and pd.isna(retail.loc["return_items", "reward"])


def test_load_domain_through_mirror_and_card(tmp_path, monkeypatch):
    official = {
        f"src/tau2/domains/{d}/tools.py": TOOLKIT.format(write=w, doc="A write tool.")
        for d, w in zip(
            tau2.DOMAINS, ("cancel_reservation", "return_items", "cancel_reservation"), strict=True
        )
    }
    _mirror(tmp_path, monkeypatch, tau2.OFFICIAL, official)
    root = tmp_path / "mirror" / tau2.SFT.id / tau2.SFT.release_id / "files"
    root.mkdir(parents=True)
    (root / "tau2_sft_final.jsonl").write_text("".join(json.dumps(r) + "\n" for r in RECORDS))

    table = datasets.load("tau2/all-domains")
    assert table.df.traj_uid.nunique() == 4  # the telecom record calls calculate, now in its catalog
    card = datasets.card("tau2/all-domains")
    for f in card.features:  # the dependency-free encoder keeps the test offline and fast
        if f.source == "fitted:embedding_cosine":
            f.args = {**f.args, "encoder": "hashing"}
    X = card.fit(table).transform(table.drop_labels())
    assert list(X.columns) == card.feature_names
    assert sorted(set(X.domain_match.round(6))) == [0.0, 1.0]  # e.g. return_items on an airline request
    assert "reward" in table.spec.outcome_cols and "reward" not in X.columns


def test_registry_entries():
    names = set(datasets.available().index)
    assert {f"tau2/{a}" for a in tau2.AGENTS} | {"tau2/airline", "tau2/all-domains"} <= names
    for name in ("tau2/o4-mini-2025-04-16", "tau2/telecom", "tau2/all-domains"):
        info = datasets.info(name)
        assert datasets.card(name).spec.kind == "ranking"
        folds = datasets.folds(name)
        assert set(folds.values()) == set(range(5))
        assert info.source.manifest  # every source file has a pinned URL and checksum
    assert len(datasets.folds("tau2/airline")) == 24 and len(datasets.folds("tau2/gpt-4.1-2025-04-14")) == 594
