"""Filter an agent's tool list with a TabAgent head before each LLM call.

The cheapest way to use TabAgent in an existing tool-calling agent: keep the agent, keep the LLM, and send the
LLM only the tools the head shortlists for this request (the paper's tool-shortlisting decision). Uncertain
decisions can be deferred to the full tool list (the confidence-gated fallback, App. N.3).

    from tabagent.integrations.openai_tools import ToolFilter
    tool_filter = ToolFilter("my_head", k=9)
    response = client.chat.completions.create(model=..., messages=messages,
                                              tools=tool_filter(messages, tools))

Works with OpenAI-style tool schemas, the Anthropic Messages API format (``name`` / ``input_schema``) and MCP
tool listings; the prefix is built exactly as ``OpenAIMessagesAdapter`` built it for training.
"""

from __future__ import annotations

from collections.abc import Iterable
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
    head: a ``TabAgentHead`` or the directory it was saved to.
    k: tools to keep.
    always: tool names that are always kept.
    defer_to_all: when the head's gate defers the decision, keep every tool.
    adapter: builds the live prefix from the messages (default ``OpenAIMessagesAdapter``, as in training).
    trust_extractor: when loading ``head`` from a directory, run its extractor in-process.

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
    ):
        if isinstance(head, (str, Path)):
            from tabagent.api import TabAgentHead

            head = TabAgentHead.load(head, trust_extractor=trust_extractor)
        self.head, self.k, self.always, self.defer_to_all = head, k, set(always), defer_to_all
        self.adapter = adapter or OpenAIMessagesAdapter()
        self.last: dict[str, Any] | None = None

    def select(self, messages: list[dict], tools: list[dict]) -> list[str]:
        """The names of the tools to send with ``messages``."""
        prefix = self.adapter.live_prefix({"messages": messages, "tools": tools})
        candidates = list(parse_catalog(tools).values())
        self.last = self.head.decide(prefix, candidates, k=self.k)
        if self.last["deferred"] and self.defer_to_all:
            return [tool_name(t) for t in tools]
        return list(dict.fromkeys([*self.last["candidates"], *self.always]))

    def __call__(self, messages: list[dict], tools: list[dict]) -> list[dict]:
        """The subset of ``tools`` to send with ``messages``, in their original order."""
        keep = set(self.select(messages, tools))
        return [t for t in tools if tool_name(t) in keep]
