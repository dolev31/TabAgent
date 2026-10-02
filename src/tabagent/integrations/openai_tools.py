"""Shortlist an agent's tool list with a TabAgent head before each LLM call.

The agent and its LLM stay as they are; before each call, a head trained on the agent's own runs ranks the
catalog from the conversation so far, and the LLM is sent the top ``k`` tools. When the head defers (its
confidence gate, or an extractor failure), the LLM gets the full list.

    from tabagent.integrations.openai_tools import ToolFilter
    tool_filter = ToolFilter("my_head", k=7, task={"domain": "airline"})  # or a `tabagent serve` URL
    session = tool_filter.session()                                       # one per conversation
    response = client.chat.completions.create(model=..., messages=messages, tools=session(messages, tools))

``ToolFilter`` answers one call at a time and keeps no state between calls. A session (``ToolFilter.session``)
is one conversation, and by default it

* keeps every tool the agent already called in the conversation (``keep_used``),
* keeps every tool it once showed (``sticky``), and
* appends tools it newly shows at the end of the list, in catalog order, so the list only ever grows.

Measured in the loop (τ²-bench, GPT-4.1, ``experiments/live/README.md``): re-ranking the list at every call
without ``sticky`` cost 6.9 points of pass rate; with ``sticky`` the pass rate matched sending every tool.
Both cost more than sending every tool (+21% and +11% per run): a list that changes between calls defeats
the provider's prompt cache, and τ²-bench's tool schemas are a small share of its prompts. Whether filtering
pays depends on the catalog and the provider (see "Does filtering the tool list pay?" in the deployment
guide).

Works with OpenAI-style tool schemas, the Anthropic Messages API format (``name`` / ``input_schema``) and MCP
tool listings. The prefix is built exactly as the adapter built it for training: from the messages and the
run-level task fields the training logs carried next to them (``instruction``, ``domain``, ...), passed as
``task``. A field the training runs had and ``task`` lacks is missing from the live prefix; when the extractor
needs it, the decision is deferred (the full tool list is sent) and ``last["error"]`` says which field.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tabagent.adapters import OpenAIMessagesAdapter, normalize_messages, parse_catalog


def tool_name(tool: dict[str, Any]) -> str:
    """The name of a tool schema (OpenAI ``function.name`` or a top-level ``name``)."""
    f = tool.get("function")
    return str((f.get("name") if isinstance(f, dict) else None) or tool.get("name"))


def called_tools(messages: list[dict]) -> list[str]:
    """The tools the agent called in ``messages`` (any supported dialect), in order of first call."""
    steps = normalize_messages(messages, max_chars=None)
    return list(dict.fromkeys(t for s in steps if s["name"] == "assistant" for t in s["tool_calls"]))


class ToolFilter:
    """Shortlist an agent's tools with a head before each LLM call (see the module docstring).

    Parameters
    ----------
    head: a ``TabAgentHead``, the directory it was saved to, ``hf://OWNER/REPO[@REVISION]``, or the URL of a
        ``tabagent serve`` service (``http://host:8077``; the service builds the prefix with its recorded
        adapter, and an unreachable service defers, so the full list is sent).
    k: tools to keep.
    always: tool names that are always kept.
    defer_to_all: when the head defers the decision (its confidence gate, or an extractor failure), keep
        every tool.
    adapter: builds the live prefix from the messages; default the adapter recorded with the head, else
        ``OpenAIMessagesAdapter()``.
    trust_extractor: when loading ``head`` from a directory, run its extractor in-process.
    task: run-level task fields sent with every call (``{"domain": "airline"}``, say); ``__call__``,
        ``select`` and ``session`` take further fields (``{"instruction": ...}``), which take precedence.
    sandbox, redact, allow_unpinned: when loading ``head`` from a directory or the Hub, see
        ``TabAgentHead.load`` (a served head is loaded, and checked, by ``tabagent serve``).

    ``last`` holds the head's answer for the latest ``select`` on this filter; a session keeps its own.
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
        allow_unpinned: bool = False,
    ):
        if isinstance(head, str) and head.startswith(("http://", "https://")):
            if sandbox is not None or redact is not None or trust_extractor or allow_unpinned:
                raise ValueError("a served head runs its extractor where it is served; set these on `serve`")
            from tabagent.serve import RemoteHead

            head = RemoteHead(head)
        elif isinstance(head, (str, Path)):
            from tabagent.api import TabAgentHead

            head = TabAgentHead.load(
                head,
                trust_extractor=trust_extractor,
                sandbox=sandbox,
                redact=redact,
                allow_unpinned=allow_unpinned,
            )
        elif sandbox is not None or redact is not None or allow_unpinned:
            raise ValueError(
                "sandbox=, redact= and allow_unpinned= apply when loading a head; set them on the head "
                "instead"
            )
        self.head, self.k, self.always, self.defer_to_all = head, k, set(always), defer_to_all
        self.adapter = adapter
        self.task = dict(task or {})
        self.last: dict[str, Any] | None = None

    @property
    def remote(self) -> bool:
        """Whether the head is a served one (it builds the prefix itself)."""
        return not hasattr(self.head, "live_prefix")

    def _run(self, messages: list[dict], tools: list[dict], task: dict[str, Any] | None) -> dict:
        return {**self.task, **(task or {}), "messages": messages, "tools": tools}

    def prefix(self, messages: list[dict], tools: list[dict], task: dict[str, Any] | None = None) -> dict:
        """The live prefix the head scores for this call (see the module docstring)."""
        if self.remote:
            raise TypeError("a served head builds the prefix from the run itself; there is none to show here")
        adapter = self.adapter
        if adapter is None and self.head.adapter is None:
            adapter = OpenAIMessagesAdapter()
        return self.head.live_prefix(self._run(messages, tools, task), adapter=adapter)

    def answer(self, messages: list[dict], tools: list[dict], task: dict[str, Any] | None = None) -> dict:
        """The head's answer for this call (``TabAgentHead.decide``), without changing ``last``."""
        if self.remote:  # the service builds the prefix; tool schemas travel as JSON
            return self.head.decide(run=self._run(messages, tools, task), candidates=list(tools), k=self.k)
        candidates = list(parse_catalog(tools).values())
        return self.head.decide(self.prefix(messages, tools, task), candidates, k=self.k)

    def select(
        self, messages: list[dict], tools: list[dict], task: dict[str, Any] | None = None
    ) -> list[str]:
        """The names of the tools to send with ``messages``: the head's top ``k`` and ``always``, or every
        tool when the head defers."""
        self.last = self.answer(messages, tools, task)
        if self.last["deferred"] and self.defer_to_all:
            return [tool_name(t) for t in tools]
        return list(dict.fromkeys([*self.last["candidates"], *self.always]))

    def __call__(
        self, messages: list[dict], tools: list[dict], task: dict[str, Any] | None = None
    ) -> list[dict]:
        """The subset of ``tools`` to send with ``messages``, in their original order."""
        keep = set(self.select(messages, tools, task))
        return [t for t in tools if tool_name(t) in keep]

    def session(
        self, task: dict[str, Any] | None = None, *, keep_used: bool = True, sticky: bool = True
    ) -> ToolSession:
        """A new conversation (see ``ToolSession``); ``task`` adds run-level fields for all its calls."""
        return ToolSession(self, task, keep_used=keep_used, sticky=sticky)


class ToolSession:
    """The tool lists of one conversation, from a ``ToolFilter``.

    Each call sends the head's top ``k`` and the filter's ``always`` tools, plus

    * with ``keep_used``, every tool the agent already called in the conversation;
    * with ``sticky``, every tool shown at an earlier call, the earlier list first and the newly shown tools
      appended in catalog order, so the list only grows (a provider's prompt cache survives the calls that add
      nothing). Without ``sticky``, the tools go in catalog order.

    When the head defers, every tool is sent (with ``defer_to_all``); with ``sticky`` they then stay. A
    session holds the state of its conversation only: start one per conversation. Its calls may come from
    several threads.

    Attributes
    ----------
    shown: with ``sticky``, the tools shown so far, in the order they are sent.
    used: the tools the agent called so far.
    last: the head's answer for the latest call, with the tools sent (``"sent"``) and those kept only because
        the agent had used them (``"kept_as_used"``).
    """

    def __init__(
        self,
        tool_filter: ToolFilter,
        task: dict[str, Any] | None = None,
        *,
        keep_used: bool = True,
        sticky: bool = True,
    ):
        self.filter, self.task = tool_filter, dict(task or {})
        self.keep_used, self.sticky = keep_used, sticky
        self.shown: list[str] = []
        self.used: list[str] = []
        self.last: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def select(
        self, messages: list[dict], tools: list[dict], task: dict[str, Any] | None = None
    ) -> list[str]:
        """The names of the tools to send with ``messages``, in the order to send them."""
        names = list(dict.fromkeys(tool_name(t) for t in tools))
        answer = self.filter.answer(messages, tools, {**self.task, **(task or {})})
        calls = called_tools(messages)
        with self._lock:
            self.used += [t for t in calls if t not in self.used]
            deferred = answer["deferred"] and self.filter.defer_to_all
            top = set(names) if deferred else {*answer["candidates"], *self.filter.always}
            used = set(self.used) - top if self.keep_used and not deferred else set()
            keep = (top | used | (set(self.shown) if self.sticky else set())) & set(names)
            if self.sticky:
                old = [n for n in self.shown if n in keep]
                self.shown = old + [n for n in names if n in keep and n not in old]
                sent = list(self.shown)
            else:
                sent = [n for n in names if n in keep]
            self.last = {**answer, "sent": sent, "kept_as_used": sorted(used & keep)}
        return sent

    def __call__(
        self, messages: list[dict], tools: list[dict], task: dict[str, Any] | None = None
    ) -> list[dict]:
        """The tool schemas to send with ``messages``, in the order to send them."""
        by_name: dict[str, dict] = {}
        for t in tools:
            by_name.setdefault(tool_name(t), t)
        return [by_name[n] for n in self.select(messages, tools, task)]
