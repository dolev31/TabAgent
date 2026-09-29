"""τ²-bench tool shortlisting (Barres et al., 2026; paper Sec. 5, Table 1 τ² rows, Table 2, App. B.2).

The paper uses two public trace sources, each matched to the question it answers:

``tau2/<agent>`` (cross-backbone transfer, Table 2 / ``tab:transfer_new``; trajectory efficiency)
    The official τ²-bench trajectory release: four agents that share one ReAct scaffold and differ only in
    the LLM backbone, run four trials per task against a GPT-4.1 user simulator. One decision per task. The
    candidates are the 67-tool dual-control catalog (agent tools and the telecom user's device tools), and a
    tool is relevant if the agent called it in a run that the benchmark's evaluator scored reward 1 (union
    over trials and result files). Tasks with no successful run that calls a tool are dropped.

``tau2/airline``, ``tau2/retail``, ``tau2/telecom``, ``tau2/all-domains`` (Table 1, ``tab:taubench_new``)
    The public tau2-sft-final release (416 trajectories, mostly from a GPT-5.2 agent). One decision per
    trajectory that calls at least one agent tool of its domain (airline 91, retail 153, telecom 82). The
    candidates are the agent tools of the domain's toolkit (``src/tau2/domains/<domain>/tools.py`` of the
    official repository: 14 / 16 / 13 tools, 39 distinct tools for all domains once the shared
    ``calculate``, ``transfer_to_human_agents`` and ``get_user_details`` are merged). A tool is relevant if
    the trajectory called it. These labels are outcome-agnostic (App. B.2). The trajectory's reward, when the
    release records one, is kept as an evaluation-only column. Groups are τ² tasks, so all trajectories of
    one task stay in one fold.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from tabagent.datasets._core import DatasetInfo, Source
from tabagent.evaluation.cv import stable_uid

OFFICIAL = Source(
    id="tau2_official_final",
    benchmark="tau2-bench",
    url="https://github.com/sierra-research/tau2-bench",
    revision="cf71a8070269883e38a365ffa85f78f46844c1f4",
    release_id="git-cf71a8070269883e38a365ffa85f78f46844c1f4",
    license="MIT",
    redistribution="redistributable",
    citation="barres2026taubench",
    notes="Official trajectory release (data/tau2/results/final, 26 result files, 4 trials per task).",
)
SFT = Source(
    id="tau2_sft_final",
    benchmark="tau2-bench",
    url="https://huggingface.co/datasets/Jarrodbarnes/tau2-sft-final",
    revision="b6a74d4c62296113ae183d2f8c5ec92be1f59bab",
    release_id="hf-b6a74d4c62296113ae183d2f8c5ec92be1f59bab",
    license="Apache-2.0",
    redistribution="redistributable",
    citation="barres2026taubench",
    notes="Tool catalogs are read from the official tau2-bench toolkits (MIT) at the pinned revision.",
)

AGENTS = (
    "claude-3-7-sonnet-20250219",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "o4-mini-2025-04-16",
)
DOMAINS = ("airline", "retail", "telecom")
RESULTS_DIR = "data/tau2/results/final/"

#: The dual-control catalog of the official release: every tool that any of the four agents called. These
#: are the agent tools of airline, retail and telecom plus the telecom user's device tools, except for
#: ``connect_vpn`` and ``get_item_details``, which no agent ever calls (69 - 2 = 67).
_CATALOG = """
        book_reservation calculate can_send_mms cancel_pending_order cancel_reservation
        check_apn_settings check_app_permissions check_app_status check_data_restriction_status
        check_installed_apps check_network_mode_preference check_network_status
        check_payment_request check_sim_status check_status_bar check_vpn_status
        check_wifi_calling_status check_wifi_status disable_roaming disconnect_vpn enable_roaming
        exchange_delivered_order_items find_user_id_by_email find_user_id_by_name_zip
        get_bills_for_customer get_customer_by_id get_customer_by_name get_customer_by_phone
        get_data_usage get_details_by_id get_flight_status get_order_details get_product_details
        get_reservation_details get_user_details grant_app_permission list_all_airports
        list_all_product_types make_payment modify_pending_order_address modify_pending_order_items
        modify_pending_order_payment modify_user_address reboot_device refuel_data reseat_sim_card
        reset_apn_settings resume_line return_delivered_order_items run_speed_test
        search_direct_flight search_onestop_flight send_certificate send_payment_request
        set_apn_settings set_network_mode_preference suspend_line toggle_airplane_mode toggle_data
        toggle_data_saver_mode toggle_roaming toggle_wifi toggle_wifi_calling
        transfer_to_human_agents update_reservation_baggages update_reservation_flights
        update_reservation_passengers
        """
AGENT_CATALOG: tuple[str, ...] = tuple(sorted(_CATALOG.split()))
AGENT_COLUMNS = ["task_uid", "domain", "instruction", "tool", "tool_name", "used"]
DOMAIN_COLUMNS = [
    "task_id",
    "traj_uid",
    "domain",
    "origin",
    "instruction",
    "tool",
    "tool_name",
    "tool_domain",
    "tool_desc",
    "tool_args",
    "used",
    "reward",
]


def canonical_hash(value: Any) -> str:
    """SHA-256 of the canonical JSON of ``value`` (identical to the trajectory-corpus builder)."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scenario_text(user_scenario: Any) -> str:
    """The task's user scenario rendered as in the pre-release tables: the repr of the key-sorted dict."""
    if user_scenario is None:
        return ""
    if isinstance(user_scenario, str):
        return user_scenario
    return str(json.loads(json.dumps(user_scenario, sort_keys=True)))


# --------------------------------------------------------------------------- official release (per agent)
def agent_decisions(payloads: Iterable[tuple[str, Mapping]], agent: str) -> pd.DataFrame:
    """Decision rows of one agent from ``(file name, parsed result file)`` pairs, in file order.

    Task ids follow the trajectory-corpus builder: ``stable_uid("task", source, "<git commit>:<domain>:<id>",
    canonical_hash(task))``, so a task embedded identically in several result files is one group.
    """
    tasks: dict[str, tuple[str, str]] = {}  # task uid -> (domain, instruction)
    used: dict[str, set[str]] = {}  # task uid -> tools of its successful runs, in order of first success
    seen: set[str] = set()
    for name, payload in payloads:
        info = payload.get("info") or {}
        if (info.get("agent_info") or {}).get("llm") != agent:
            continue
        domain = (info.get("environment_info") or {}).get("domain_name") or "unknown"
        revision = info.get("git_commit") or "unknown"
        lookup = {}
        for i, task in enumerate(payload.get("tasks") or []):
            native = str(task.get("id", i))
            uid = stable_uid("task", OFFICIAL.id, f"{revision}:{domain}:{native}", canonical_hash(task))
            lookup[native] = uid
            tasks.setdefault(uid, (domain, scenario_text(task.get("user_scenario"))))
        for j, sim in enumerate(payload.get("simulations") or []):
            traj = stable_uid("traj", OFFICIAL.id, name, sim.get("id") or f"{name}:{j}")
            if traj in seen:
                continue
            seen.add(traj)
            if (sim.get("reward_info") or {}).get("reward") != 1:
                continue
            called = {
                call["name"]
                for msg in sim.get("messages") or []
                for call in msg.get("tool_calls") or []
                if call.get("name")
            }
            if called:
                task_id = str(sim.get("task_id"))
                if task_id not in lookup:
                    raise ValueError(
                        f"{name}: simulation {j} refers to task {task_id!r}, not embedded in the file"
                    )
                used.setdefault(lookup[task_id], set()).update(called)
    unknown = set().union(*used.values()) - set(AGENT_CATALOG)
    if unknown:
        raise ValueError(f"tools outside the 67-tool catalog: {sorted(unknown)}")
    rows = [
        (t, *tasks[t], c, c.replace("_", " "), int(c in tools))
        for t, tools in used.items()
        for c in AGENT_CATALOG
    ]
    return pd.DataFrame(rows, columns=AGENT_COLUMNS)


def build_agent(source: Source, download: bool = False, agent: str = AGENTS[0]) -> pd.DataFrame:
    """Rebuild ``tau2/<agent>`` from the agent's result files of the pinned official release."""
    files = [p for p in source.files(RESULTS_DIR) if Path(p).name.startswith(f"{agent}_")]
    if not files:
        raise ValueError(f"no result files for agent {agent!r}; choose one of {AGENTS}")

    def payloads() -> Iterator[tuple[str, dict]]:
        for rel in files:  # one file in memory at a time (the largest is 51 MB)
            yield Path(rel).name, json.loads(source.path(rel, download).read_text(encoding="utf-8"))

    return agent_decisions(payloads(), agent)


# --------------------------------------------------------------------------- tau2-sft-final (per domain)
_RESULT_CALL = re.compile(r"\[TOOL_RESULT\]\s*\n?\s*([a-z_][a-z0-9_]*)\s*\(")
_JSON_CALL = re.compile(r'"name"\s*:\s*"([a-z_][a-z0-9_]*)"')
_DOC_SECTION = re.compile(r"\n\s*(?:Args|Arguments|Returns|Raises|Example|Examples)\s*:")


def _is_tool(decorator: ast.expr) -> bool:
    fn = decorator.func if isinstance(decorator, ast.Call) else decorator
    return (isinstance(fn, ast.Name) and fn.id == "is_tool") or (
        isinstance(fn, ast.Attribute) and fn.attr == "is_tool"
    )


def parse_toolkit(code: str) -> list[dict[str, Any]]:
    """The agent-callable tools of a τ²-bench toolkit module: every method decorated with ``@is_tool``.

    Returns ``[{"tool", "description", "args"}]`` sorted by name; the description is the docstring up to
    its ``Args:`` section, with whitespace collapsed.
    """
    out = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            _is_tool(d) for d in node.decorator_list
        ):
            doc = _DOC_SECTION.split(ast.get_docstring(node) or "")[0]
            a = node.args
            args = [x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs) if x.arg not in ("self", "cls")]
            out[node.name] = {"tool": node.name, "description": " ".join(doc.split()), "args": args}
    return [out[k] for k in sorted(out)]


def called_tools(messages: Iterable[Mapping]) -> set[str]:
    """Tools a trajectory called: ``[TOOL_RESULT] name(`` in any message, or ``"name": "..."`` in an
    assistant message (the release mixes action-delimited and JSON tool-call formats)."""
    out: set[str] = set()
    for m in messages:
        text = str(m.get("content") or "")
        out.update(_RESULT_CALL.findall(text))
        if m.get("role") == "assistant":
            out.update(_JSON_CALL.findall(text))
    return out


def _first_user(messages: Iterable[Mapping]) -> str:
    return next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "")


def domain_decisions(
    records: Iterable[Mapping], catalogs: Mapping[str, list[dict]], domain: str | None = None
) -> pd.DataFrame:
    """Decision rows from tau2-sft-final records and per-domain tool catalogs (``parse_toolkit``).

    ``domain=None`` pools every domain over the merged catalog; a tool shared by several domains is one
    candidate, described by the context domain's toolkit when it has the tool (``tool_domain``).
    """
    if domain is not None and domain not in catalogs:
        raise ValueError(f"unknown domain {domain!r}; choose one of {sorted(catalogs)}")
    specs: dict[str, dict[str, dict]] = {}  # tool -> {domain: spec}, domains in catalog order
    for d, tools in catalogs.items():
        for t in tools:
            specs.setdefault(t["tool"], {})[d] = t
    rows = []
    for rec in records:
        d = rec.get("domain")
        if d not in catalogs or (domain is not None and d != domain):
            continue
        messages = rec.get("messages") or []
        own = {t["tool"] for t in catalogs[d]}
        used = called_tools(messages) & own
        if not used:
            continue
        task_id = f"{d}|{rec.get('tau2_task_id') or rec.get('task_id')}"
        traj_uid = stable_uid("traj", SFT.id, rec.get("task_id"), canonical_hash(messages))
        reward = (rec.get("metadata") or {}).get("reward")
        head = (task_id, traj_uid, d, rec.get("source"), _first_user(messages))
        for c in sorted(own) if domain is not None else sorted(specs):
            home = d if d in specs[c] else next(iter(specs[c]))
            s = specs[c][home]
            args = " ".join(a.replace("_", " ") for a in s["args"])
            rows.append((*head, c, c.replace("_", " "), home, s["description"], args, int(c in used), reward))
    return pd.DataFrame(rows, columns=DOMAIN_COLUMNS)


def build_domain(source: Source, download: bool = False, domain: str | None = None) -> pd.DataFrame:
    """Rebuild ``tau2/<domain>`` (or ``tau2/all-domains`` for ``domain=None``) from the pinned releases."""
    catalogs = {
        d: parse_toolkit(
            OFFICIAL.path(f"src/tau2/domains/{d}/tools.py", download).read_text(encoding="utf-8")
        )
        for d in DOMAINS
    }
    with open(source.path("tau2_sft_final.jsonl", download), encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return domain_decisions(records, catalogs, domain)


# --------------------------------------------------------------------------- registry
_AGENT_PAPER = "Table 2 (tab:transfer_new); App. B.2; trajectory efficiency (tab:traj_per_tool)"
_DOMAIN_PAPER = "Table 1 (τ²-bench rows); App. B.2 (tab:taubench_new)"

DATASETS: list[DatasetInfo] = [
    DatasetInfo(
        name=f"tau2/{agent}",
        source=OFFICIAL,
        build=build_agent,
        card="tau2_tool_shortlisting",
        folds="tau2_agents",
        decision="tool shortlisting (ranking, 67-tool dual-control catalog, one decision per task)",
        paper=_AGENT_PAPER,
        description=(
            f"Tools the {agent} agent called in its successful official τ²-bench runs. As in the paper's "
            "Table 2, the context text is the task's user scenario (the simulator's instructions), which the "
            "agent itself never sees; use the tau2/<domain> tables for a deployment-realistic context."
        ),
        args={"agent": agent},
    )
    for agent in AGENTS
] + [
    DatasetInfo(
        name=f"tau2/{d or 'all-domains'}",
        source=SFT,
        build=build_domain,
        card="tau2_domain_shortlisting",
        folds=f"tau2_{d or 'all_domains'}",
        decision=(
            "tool shortlisting (ranking over the "
            + (f"{d} agent toolkit" if d else "merged agent toolkits of airline, retail and telecom")
            + ", one decision per trajectory)"
        ),
        paper=_DOMAIN_PAPER,
        description="Tools each tau2-sft-final trajectory called (outcome-agnostic; reward kept for eval).",
        args={"domain": d},
    )
    for d in (*DOMAINS, None)
]
