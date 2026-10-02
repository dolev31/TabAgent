"""Filter an agent's tool list with a TabAgent head before each LLM call.

The cheapest way to use TabAgent in an existing tool-calling agent: keep the agent, keep the LLM, and send the
LLM only the tools the head shortlists for this request (the paper's tool-shortlisting decision). Uncertain
decisions can be deferred to the full tool list (the confidence-gated fallback, App. N.3).

    from tabagent.integrations.openai_tools import ToolFilter
    tool_filter = ToolFilter("my_head", k=9, task={"domain": "airline"})
    response = client.chat.completions.create(model=..., messages=messages,
                                              tools=tool_filter(messages, tools, task={"instruction": goal}))

Works with OpenAI-style tool schemas, the Anthropic Messages API format (``name`` / ``input_schema``) and MCP
tool listings. The prefix is built exactly as the adapter built it for training: from the messages and the
run-level task fields the training logs carried next to them (``instruction``, ``domain``, ...), passed as
``task``. A field the training runs had and ``task`` lacks is missing from the live prefix; when the extractor
needs it, the decision is deferred (the full tool list is sent) and ``last["error"]`` says which field.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tabagent.adapters import OpenAIMessagesAdapter, parse_catalog


def tool_name(tool: dict[str, Any]) -> str:
    """The name of a tool schema (OpenAI ``function.name`` or a top-level ``name``)."""
    return str(tool.get("function", {}).get("name") or tool.get("name"))


class ToolFilter:
    """Shortlist an agent's tools with a saved head before each LLM call.

    Parameters
    ----------
    head: a ``TabAgentHead``, the directory it was saved to, or ``hf://OWNER/REPO[@REVISION]``.
    k: tools to keep.
    always: tool names that are always kept.
    defer_to_all: when the head defers the decision (its confidence gate, or an extractor failure), keep
        every tool.
    adapter: builds the live prefix from the messages; default the adapter recorded with the head, else
        ``OpenAIMessagesAdapter()``.
    trust_extractor: when loading ``head`` from a directory, run its extractor in-process.
    task: run-level task fields sent with every call (``{"domain": "airline"}``, say); ``__call__`` and
        ``select`` take per-call fields too (``{"instruction": ...}``), which take precedence.
    sandbox, redact: when loading ``head`` from a directory, see ``TabAgentHead.load``.

    ``last`` holds the head's answer for the latest call.
    """

    def __init__(
        self,
        head,
        k: int = 9,
        always: Iterable[str] = (),
        defer_to_all: bool = True,
        adapter: OpenAIMessagesAdapter | None = None,
        trust_extractor: bool = False,
        *,
        task: dict[str, Any] | None = None,
        sandbox: Any = None,
        redact: Callable[[dict], dict] | str | None = None,
    ):
        if isinstance(head, (str, Path)):
            from tabagent.api import TabAgentHead

            head = TabAgentHead.load(head, trust_extractor=trust_extractor, sandbox=sandbox, redact=redact)
        elif sandbox is not None or redact is not None:
            raise ValueError("sandbox= and redact= apply when loading a head; set them on the head instead")
        self.head, self.k, self.always, self.defer_to_all = head, k, set(always), defer_to_all
        self.adapter = adapter
        self.task = dict(task or {})
        self.last: dict[str, Any] | None = None

    def prefix(self, messages: list[dict], tools: list[dict], task: dict[str, Any] | None = None) -> dict:
        """The live prefix the head scores for this call (see the module docstring)."""
        run = {**self.task, **(task or {}), "messages": messages, "tools": tools}
        adapter = self.adapter
        if adapter is None and self.head.adapter is None:
            adapter = OpenAIMessagesAdapter()
        return self.head.live_prefix(run, adapter=adapter)

    def select(
        self, messages: list[dict], tools: list[dict], task: dict[str, Any] | None = None
    ) -> list[str]:
        """The names of the tools to send with ``messages``."""
        candidates = list(parse_catalog(tools).values())
        self.last = self.head.decide(self.prefix(messages, tools, task), candidates, k=self.k)
        if self.last["deferred"] and self.defer_to_all:
            return [tool_name(t) for t in tools]
        return list(dict.fromkeys([*self.last["candidates"], *self.always]))

    def __call__(
        self, messages: list[dict], tools: list[dict], task: dict[str, Any] | None = None
    ) -> list[dict]:
        """The subset of ``tools`` to send with ``messages``, in their original order."""
        keep = set(self.select(messages, tools, task))
        return [t for t in tools if tool_name(t) in keep]
