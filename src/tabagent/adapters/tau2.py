"""τ²-bench results files (github.com/sierra-research/tau2-bench, ``data/tau2/results/**/*.json``).

A results file is ``{"timestamp", "info", "tasks", "simulations"}``. Each simulation is one run: ``task_id``,
``trial``, ``messages`` (roles assistant/user/tool; ``tool_calls`` carry a ``requestor``) and
``reward_info.reward``. Simulations are mapped to the chat-message steps of ``OpenAIMessagesAdapter`` and
reuse its decisions (``ToolShortlist`` by default, ``NextToolCall`` with ``granularity="step"``).

What the agent could see is all a prefix holds: the domain and, for solo agents (``no-user`` modes, which
receive the task as a ticket), the ticket; never the user simulator's scenario, the evaluation criteria (gold
actions), the reward or the oracle plan given to ``*_gt`` agents. The task id names the run's group, not a
prefix field. The domain policy (the agent's system prompt, identical for every task of a domain) is added
as a system step with ``include_policy=True``.

A run succeeds when ``reward_info.reward == 1``; the group is ``<domain>:<task_id>``, so repeated trials
and agents of one task never straddle CV folds. Candidates are the domain's tools: ``tool_defs`` when the
file has them, else a ``tools`` catalog given here, else the tools the agents called across the loaded
files, with the argument names observed in the calls as hints. User-side device tools are excluded, except
for solo agents, which operate the device themselves and so get their own per-domain pool.

For ``tabagent.scan``, the agent's messages record their tokens (``usage``) and cost; the latency of a call
is the time since the previous message, from the per-message timestamps.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tabagent.adapters.common import iter_files, run_id
from tabagent.adapters.openai_messages import OUTCOME_KEYS, OpenAIMessagesAdapter
from tabagent.tabschema.trajectory import Trajectory


def _seconds(ts: Any) -> float | None:
    """A τ²-bench message timestamp (ISO 8601, or epoch seconds) in epoch seconds."""
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        return float(ts)
    if isinstance(ts, str) and ts:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _solved(run: dict) -> bool:
    r = run.get("reward")
    try:
        return r is not None and abs(float(r) - 1.0) < 1e-9
    except (TypeError, ValueError):
        return False


class Tau2Adapter(OpenAIMessagesAdapter):
    """τ²-bench simulations -> trajectories and tool-selection decision points (see the module docstring)."""

    scan_notes = (
        "only the agent's LLM calls are counted: the user simulator's turns and the scripted greeting "
        "(no usage, zero cost) are not",
        "latency is the time from the previous message to the agent's message: τ²-bench records a "
        "timestamp per message, not a duration per call",
    )

    def __init__(
        self,
        granularity: str = "task",
        tools: Any = None,
        include_policy: bool = False,
        max_chars: int | None = 20_000,
    ):
        super().__init__(
            granularity,
            tools,
            success=_solved,
            task_keys=("task_id", "domain", "ticket"),
            outcome_keys=(*OUTCOME_KEYS, "reward_info", "evaluation_criteria", "user_scenario"),
            max_chars=max_chars,
        )
        self.include_policy = include_policy

    def _pool(self, run: dict) -> str:  # solo agents also hold the user's device tools
        return f"{run.get('domain', '')}{'/solo' if run.get('solo') else ''}"

    def _is_llm_call(self, msg: dict) -> bool:  # the scripted greeting records no usage and costs nothing
        return not ("usage" in msg and msg["usage"] is None and not msg.get("cost"))

    def _latency(self, msgs: list[dict], i: int) -> float | None:
        if i == 0:
            return None
        now, before = _seconds(msgs[i].get("timestamp")), _seconds(msgs[i - 1].get("timestamp"))
        return now - before if now is not None and before is not None and now >= before else None

    def _model(self, run: dict) -> str | None:
        return run.get("agent") or None

    def load(self, path: str | Path) -> list[Trajectory]:
        root = Path(path)
        self.warnings = []
        out: list[Trajectory] = []
        for f in iter_files(root, (".json",)):
            try:
                data = json.loads(f.read_text())
            except ValueError:
                self.warnings.append(f"{f.name}: not valid JSON, skipped")
                continue
            if not (isinstance(data, dict) and isinstance(data.get("simulations"), list)):
                continue
            info = data.get("info") or {}
            env = info.get("environment_info") or {}
            agent = info.get("agent_info") or {}
            domain = str(env.get("domain_name") or "")
            solo = "solo" in str(agent.get("implementation", ""))
            tickets = {str(t.get("id")): t.get("ticket") or "" for t in data.get("tasks") or []}
            base = run_id(root, f)
            for i, sim in enumerate(data["simulations"]):
                tid = str(sim.get("task_id"))
                run: dict[str, Any] = {
                    "task_id": tid,
                    "domain": domain,
                    "ticket": tickets.get(tid, "") if solo else "",
                    "agent": str(agent.get("llm") or ""),
                    "solo": solo,
                    "trial": sim.get("trial"),
                    "messages": [
                        {k: v for k, v in m.items() if k != "raw_data"}  # the raw LLM payload duplicates it
                        for m in sim.get("messages") or []
                        if isinstance(m, dict)
                    ],
                    "reward": (sim.get("reward_info") or {}).get("reward"),
                }
                if self.include_policy and env.get("policy"):
                    run["system"] = env["policy"]
                if env.get("tool_defs"):
                    run["tools"] = env["tool_defs"]
                traj = Trajectory(f"{base}:{i}", run, _solved(run), f"{domain}:{tid}", f.name)
                self._note(traj)
                out.append(traj)
        return out
