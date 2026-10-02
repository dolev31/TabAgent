"""TabAgent in the τ²-bench agent loop: before every agent LLM call, a served head ranks the domain's tools
from the conversation so far and the agent sees only the top k.

Runs in τ²-bench's own environment (github.com/sierra-research/tau2-bench, which is not a TabAgent
dependency), with TabAgent's core installed there too (``pip install tabagent``, no extras); the head is
served from TabAgent's environment and reached over HTTP, as a deployed agent would:

    tabagent serve ./head --port 8077                                  # TabAgent's environment
    python experiments/live/tau2_loop.py --domain airline --task-ids 3 7 12 --arm tabagent --k 7 \
        --head-url http://127.0.0.1:8077 --save-to tabagent-airline     # τ²-bench's environment

The ``tabagent`` arm subclasses τ²-bench's ``LLMAgent`` and changes one thing: the tools passed to the LLM.
They come from the library's ``ToolFilter`` on the served head (``RemoteHead``), one ``ToolSession`` per
conversation, before every call: the run (``{"domain", "messages", "tools"}``, the shape of the τ²-bench logs
the head was onboarded on) goes to ``/decide``, and the session keeps

* the head's top ``k`` tools,
* every tool the agent already called in the conversation (the safety rule, ``keep_used``),
* every tool when the head defers (confidence gate or extractor failure) or the service cannot be reached,
* with ``--sticky``, every tool once shown, newly shown tools appended in catalog order (``sticky``).

The tool definitions are part of the prompt prefix that providers cache (OpenAI, Azure OpenAI), so a tool
list that changes between calls invalidates the cached prefix; a list that only grows by appending keeps it.

The environment still executes any tool the agent calls; the filter only changes what the LLM is shown.
Each assistant message records the decision in ``raw_data["tabagent_filter"]`` (tools shown, deferral,
error, service latency), so ``experiments.live.loop_report`` can pair it with the call's tokens. The
``baseline`` arm is τ²-bench's own ``llm_agent`` with every tool. Both arms take the same tasks, trials and
seed. LLMs are reached through LiteLLM (``openai/<model>`` with ``OPENAI_BASE_URL`` for an
OpenAI-compatible proxy).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from tau2 import TextRunConfig
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import MultiToolMessage, UserMessage
from tau2.evaluator import evaluator_nl_assertions
from tau2.registry import registry
from tau2.runner import run_domain
from tau2.utils.llm_utils import generate
from tau2.utils.utils import DATA_DIR

from tabagent.integrations.openai_tools import ToolFilter

AGENT_NAME = "tabagent_filter_agent"


class TabAgentFilterAgent(LLMAgent):
    """τ²-bench's ``LLMAgent`` with the tool list shortlisted before each call by a ``ToolSession``: one per
    agent instance, which τ²-bench builds for each simulation, so one per conversation."""

    def __init__(self, tools, domain_policy, llm, llm_args=None, *, tool_filter: ToolFilter, sticky: bool):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.schemas = [t.openai_schema for t in tools]
        self.session = tool_filter.session(keep_used=True, sticky=sticky)

    def select_tools(self, messages: list) -> tuple[list, dict[str, Any]]:
        """The tools to show the LLM for the conversation ``messages``, in the order to send them, and the
        decision's record."""
        dumps = [m.model_dump(mode="json", exclude={"raw_data"}) for m in messages]
        t0 = time.perf_counter()
        sent = self.session.select(dumps, self.schemas)
        ms = 1e3 * (time.perf_counter() - t0)
        last = self.session.last
        record = {
            "shown": sent,
            "top_k": last["candidates"],
            "kept_as_used": last["kept_as_used"],
            "deferred": last["deferred"],
            "error": last.get("error"),
            "confidence": last.get("confidence"),
            "ms": round(ms, 2),
            "n_tools": len(self.tools),
        }
        by_name = {t.name: t for t in self.tools}
        return [by_name[n] for n in sent], record

    def _generate_next_message(self, message, state):
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        tools, record = self.select_tools(state.messages)
        reply = generate(
            model=self.llm,
            tools=tools,
            messages=state.system_messages + state.messages,
            call_name="agent_response",
            **self.llm_args,
        )
        called = [tc.name for tc in reply.tool_calls or []]
        record["called_unshown"] = [n for n in called if n not in record["shown"]]
        reply.raw_data = {**(reply.raw_data or {}), "tabagent_filter": record}
        return reply


def drop_infra_errors(path) -> None:
    """Remove the simulations that ended in an infrastructure error from a saved run, so resuming it runs
    them again (τ²-bench resumes every (trial, task, seed) the file holds)."""
    data = json.loads(path.read_text())
    keep = [s for s in data["simulations"] if s.get("termination_reason") != "infrastructure_error"]
    print(f"resuming {path.parent.name}: re-running {len(data['simulations']) - len(keep)} simulations")
    data["simulations"] = keep
    path.write_text(json.dumps(data))


def register(domain: str, head_url: str, k: int, sticky: bool = False) -> None:
    """Register the ``tabagent`` arm's agent: ``ToolFilter`` on the head served at ``head_url`` (stateless,
    shared), with a ``ToolSession`` per agent."""
    tool_filter = ToolFilter(head_url, k=k, task={"domain": domain})

    def factory(tools, domain_policy, llm=None, llm_args=None, **_):
        return TabAgentFilterAgent(
            tools, domain_policy, llm, llm_args, tool_filter=tool_filter, sticky=sticky
        )

    registry.register_agent_factory(factory, AGENT_NAME)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", required=True, choices=["airline", "retail", "telecom"])
    ap.add_argument("--task-ids", nargs="+", help="τ²-bench task ids (default: every task)")
    ap.add_argument("--split", help="a split JSON: run its held-out tasks of --domain")
    ap.add_argument("--arm", required=True, choices=["baseline", "tabagent"])
    ap.add_argument("--head-url", default="http://127.0.0.1:8077")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--sticky", action="store_true", help="never remove a tool once shown (keeps the prompt cache warm)"
    )
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-retries", type=int, default=1, help="re-runs of a simulation after an error")
    ap.add_argument("--agent-llm", default="openai/Azure/gpt-4.1")
    ap.add_argument("--user-llm", default="openai/Azure/gpt-4.1")
    ap.add_argument("--save-to", required=True, help="run name (under τ²-bench's data/simulations/)")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue the run saved under --save-to, re-running its simulations that ended in an "
        "infrastructure error",
    )
    args = ap.parse_args(argv)

    ids = args.task_ids
    if args.split:
        with open(args.split) as f:
            held = json.load(f)["test"]
        ids = sorted((g.split(":", 1)[1] for g in held if g.startswith(f"{args.domain}:")), key=int)
    if args.arm == "tabagent":
        register(args.domain, args.head_url, args.k, args.sticky)
    # τ²-bench judges natural-language assertions (some retail tasks) with a hard-coded default model; use
    # the user simulator's, so every LLM call goes through the same endpoint
    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = args.user_llm
    if args.resume:
        drop_infra_errors(DATA_DIR / "simulations" / args.save_to / "results.json")
    config = TextRunConfig(
        domain=args.domain,
        agent=AGENT_NAME if args.arm == "tabagent" else "llm_agent",
        llm_agent=args.agent_llm,
        llm_args_agent={"temperature": 0.0},
        llm_user=args.user_llm,
        llm_args_user={"temperature": 0.0},
        task_ids=ids,
        num_trials=args.trials,
        seed=args.seed,
        max_concurrency=args.concurrency,
        max_retries=args.max_retries,
        auto_resume=args.resume,
        save_to=args.save_to,
    )
    t0 = time.time()
    run_domain(config)
    print(f"wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
