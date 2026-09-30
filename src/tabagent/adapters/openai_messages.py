"""Chat-message logs: OpenAI Chat Completions, the Anthropic Messages API format and LangChain dumps.

A run is one JSON object with a ``messages`` list: one per ``.json`` file (or a list of runs), or one per line
of a ``.jsonl`` file. Optional fields: ``tools`` (the tool catalog), an outcome (``success``, ``reward`` or
``score``) and a task id (``task_id`` or ``id``). The message dialects (OpenAI, including legacy
``function_call`` and Responses API items; the Anthropic Messages API format with ``tool_use`` and
``tool_result`` blocks and a top-level ``system``; LangChain ``type``/``tool_calls`` dumps) are normalized
to one step shape, which is what TabSchema's extraction code reads::

    {"task_id": "t1", "steps": [{"name": "user", "content": "...", "tool_calls": [], "tool_name": ""}, ...]}

Every step has the same four keys: ``name`` (system, user, assistant or tool), ``content`` (text; images and
base64 payloads become placeholders, very long text is clipped), ``tool_calls`` (the tools a turn called)
and ``tool_name`` (for a tool result, the tool that produced it). Only scalar task fields from a whitelist
(``task_keys``) are copied, so post-hoc run fields (durations, costs, final answers) never reach a prefix.

Two decision targets:

* ``ToolShortlist`` (``granularity="task"``, the paper's tool shortlisting): one decision per run, placed
  before the first assistant turn that calls a tool. The prefix holds the task fields and the system and
  user messages before it; the labels are all the tools the run called (G(t), read from the full run).
* ``NextToolCall`` (``granularity="step"``): one decision per assistant turn that calls tools. The prefix
  holds every message before that turn; the labels are the tools called in it.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tabagent.adapters.common import iter_files, parse_catalog, run_id
from tabagent.tabschema.trajectory import Candidate, DecisionPoint, Trajectory

TOOL_SHORTLIST, NEXT_TOOL_CALL = "ToolShortlist", "NextToolCall"
TASK_KEYS = ("task_id", "instruction", "task", "query", "question", "goal", "intent", "domain")
OUTCOME_KEYS = (
    "success",
    "reward",
    "score",
    "eval",
    "evaluation",
    "outcome",
    "reward_info",
    "passed",
    "resolved",
    "ground_truth",
    "expected",
    "answer",
)
ROLES = {
    "system": "system",
    "developer": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "model": "assistant",
    "tool": "tool",
    "function": "tool",
    "ipython": "tool",
}
MEDIA = {"image", "image_url", "input_image", "audio", "input_audio", "document", "file", "input_file"}
_BINARY = re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=\r\n]{64,}")
_NO_TOOL = re.compile(  # a call to a tool the agent does not have (tau2, LangGraph ToolNode)
    r"\s*error:?\s*(?:tool\s+['\"]?[\w.-]+['\"]?\s+(?:not found|does not exist)|[\w.-]+ is not a valid tool)",
    re.I,
)
_TRUE = {"true", "yes", "pass", "passed", "success", "succeeded", "solved", "resolved"}

Call = tuple[str, str, list[str]]  # (call id, tool name, argument names)


# --------------------------------------------------------------------------- message normalization
def _text(content: Any) -> str:
    """Text of a message or block content; media become ``[image]``-style placeholders, other blocks drop."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return str(content)
    parts = []
    for b in content:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict) and isinstance(b.get("text"), str):
            parts.append(b["text"])
        elif isinstance(b, dict) and b.get("type") in MEDIA:
            parts.append(f"[{b['type']}]")
    return "\n".join(p for p in parts if p)


def _clip(s: str, max_chars: int | None) -> str:
    s = _BINARY.sub("[binary]", s)
    if max_chars and len(s) > max_chars:
        s = f"{s[:max_chars]} ...[{len(s) - max_chars} more characters]"
    return s


def _call(tc: dict) -> Call | None:
    f = tc["function"] if isinstance(tc.get("function"), dict) else tc
    if not f.get("name"):
        return None
    args = f.get("arguments", f.get("args", f.get("input")))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = None
    keys = sorted(str(k) for k in args) if isinstance(args, dict) else []
    return str(tc.get("call_id") or tc.get("id") or ""), str(f["name"]), keys


def _calls(msg: dict) -> list[Call]:
    """Tool calls of one message, from the first dialect that has any (skips LangChain duplicates)."""
    blocks = msg["content"] if isinstance(msg.get("content"), list) else []
    sources = (
        msg.get("tool_calls"),  # OpenAI, tau2, LangChain
        (msg.get("additional_kwargs") or {}).get("tool_calls"),  # LangChain, raw OpenAI payload
        [msg["function_call"]] if isinstance(msg.get("function_call"), dict) else None,  # legacy OpenAI
        # Anthropic Messages API format
        [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"],
    )
    raw = next((s for s in sources if s), [])
    return [c for c in (_call(tc) for tc in raw if isinstance(tc, dict)) if c]


def message_role(msg: dict) -> str:
    """The normalized role of a message (``system``, ``user``, ``assistant``, ``tool``), from its ``role`` or,
    for LangChain dumps, its ``type`` or class name."""
    r = str(msg.get("role") or msg.get("type") or "").lower()
    r = r.removesuffix("messagechunk").removesuffix("message")  # LangChain class names (AIMessage, ...)
    return ROLES.get(r, r)


def unwrap_message(m: dict) -> dict:
    """A message as a flat dict, unwrapping LangChain ``messages_to_dict`` (``{"type", "data"}``) and
    ``dumpd`` (``{"lc", "id", "kwargs"}``) serializations; other messages are returned unchanged."""
    if isinstance(m.get("kwargs"), dict) and m.get("lc") and isinstance(m.get("id"), list):
        return {**m["kwargs"], "type": str(m["id"][-1])}
    if isinstance(m.get("data"), dict) and "content" in m["data"] and "type" in m:
        return {**m["data"], "type": m["type"]}
    return m


def _normalize(messages: Any, system: Any, max_chars: int | None) -> tuple[list[dict], list[Call]]:
    steps: list[dict] = []
    agent_calls: list[Call] = []
    names: dict[str, str] = {}  # call id -> tool name, to name tool results

    def add(name: str, content: Any = "", calls: Iterable[str] = (), tool_name: str = "") -> None:
        text = _clip(_text(content), max_chars)
        steps.append({"name": name, "content": text, "tool_calls": list(calls), "tool_name": tool_name})

    if system:
        add("system", system)
    for m in messages if isinstance(messages, list) else []:
        if not isinstance(m, dict):
            continue
        m = unwrap_message(m)
        kind = m.get("type")
        if kind == "function_call":  # OpenAI Responses API: a call item joins the assistant turn before it
            c = _call(m)
            if c:
                names[c[0]] = c[1]
                agent_calls.append(c)
                if steps and steps[-1]["name"] == "assistant":
                    steps[-1]["tool_calls"].append(c[1])
                else:
                    add("assistant", "", [c[1]])
            continue
        if kind == "function_call_output":
            add("tool", m.get("output"), (), names.get(str(m.get("call_id", "")), ""))
            continue
        role = message_role(m)
        if not role or kind == "reasoning":
            continue
        content = m.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):  # Anthropic Messages API format: tool results are user-message blocks
            rest = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    add("tool", b.get("content"), (), names.get(str(b.get("tool_use_id", "")), ""))
                else:
                    rest.append(b)
            if _text(rest):
                add(role, rest)
            continue
        calls = _calls(m)
        names.update((c[0], c[1]) for c in calls if c[0])
        if role == "assistant":
            agent_calls += calls
        tool_name = ""
        if role == "tool":
            tool_name = str(m.get("name") or names.get(str(m.get("tool_call_id") or m.get("id") or ""), ""))
        add(role, content, [c[1] for c in calls], tool_name)
    missing = {s["tool_name"] for s in steps if s["name"] == "tool" and _NO_TOOL.match(s["content"])}
    return steps, [c for c in agent_calls if c[1] not in missing]


def normalize_messages(messages: list, system: Any = None, max_chars: int | None = 20_000) -> list[dict]:
    """Chat messages in any supported dialect -> steps ``{"name", "content", "tool_calls", "tool_name"}``."""
    return _normalize(messages, system, max_chars)[0]


def _positive(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v > 0
    if isinstance(v, str):
        try:
            return float(v) > 0
        except ValueError:
            return v.strip().lower() in _TRUE
    return bool(v)


# --------------------------------------------------------------------------- adapter
class OpenAIMessagesAdapter:
    """Chat-message runs -> trajectories and tool-selection decision points (see the module docstring).

    Candidates come from the run's own ``tools``, else the ``tools`` catalog given here, else the union of
    the tools called across the loaded runs (argument hints then list the argument names observed in calls;
    calls answered with a tool-not-found error are left out).
    ``success`` is an outcome key or a function of the run; runs without an outcome count as successful and
    are reported in ``warnings``.
    """

    targets = (TOOL_SHORTLIST, NEXT_TOOL_CALL)

    def __init__(
        self,
        granularity: str = "task",
        tools: Any = None,
        success: str | Callable[[dict], bool] | None = None,
        task_id_key: str | None = None,
        task_keys: Iterable[str] = TASK_KEYS,
        outcome_keys: Iterable[str] = OUTCOME_KEYS,
        max_chars: int | None = 20_000,
    ):
        if granularity not in ("task", "step"):
            raise ValueError(f"granularity must be 'task' or 'step', not {granularity!r}")
        self.granularity = granularity
        self.target = TOOL_SHORTLIST if granularity == "task" else NEXT_TOOL_CALL
        self.catalog = parse_catalog(tools)
        self.success = success
        self.task_id_key = task_id_key
        extra = [success] if isinstance(success, str) else []
        self.outcome_keys = tuple(dict.fromkeys([*outcome_keys, *extra]))
        self.task_keys = tuple(k for k in task_keys if k not in self.outcome_keys)
        self.max_chars = max_chars
        self.warnings: list[str] = []
        self._own: dict[str, dict[str, Candidate]] = {}  # trajectory id -> the run's own catalog
        self._pool_of: dict[str, str] = {}  # trajectory id -> candidate pool (tau2: the domain)
        self._seen: dict[str, dict[str, Candidate]] = {}  # pool -> catalog entries seen in loaded runs
        self._called: dict[str, dict[str, set[str]]] = {}  # pool -> tool -> argument names seen in calls

    # ------------------------------------------------------------------ loading
    def load(self, path: str | Path) -> list[Trajectory]:
        root = Path(path)
        self.warnings = []
        out: list[Trajectory] = []
        missing = 0
        for f in iter_files(root):
            for rid, run in self._read(root, f):
                ok = self._outcome(run)
                missing += ok is None
                traj = Trajectory(rid, run, ok is not False, self._task_id(run, rid), f.name)
                self._note(traj)
                out.append(traj)
        if missing:
            keys = self.success if isinstance(self.success, str) else "success/reward/score"
            n = f"{missing} of {len(out)} runs"
            self.warnings.append(f"{n} have no outcome field ({keys}); they were treated as successful")
        return out

    def _read(self, root: Path, f: Path) -> list[tuple[str, dict]]:
        base = run_id(root, f)
        rows: list[tuple[str, Any]] = []
        if f.suffix == ".jsonl":
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if line.strip():
                    try:
                        rows.append((f"{base}:{i}", json.loads(line)))
                    except ValueError:
                        self.warnings.append(f"{f.name}:{i}: not valid JSON, skipped")
        else:
            try:
                data = json.loads(f.read_text())
            except ValueError:
                self.warnings.append(f"{f.name}: not valid JSON, skipped")
                return []
            if isinstance(data, list) and data and all(isinstance(x, dict) and "messages" in x for x in data):
                rows = [(f"{base}:{i}", x) for i, x in enumerate(data)]
            elif isinstance(data, list):  # a bare message list
                rows = [(base, {"messages": data})]
            else:
                rows = [(base, data)]
        return [
            (rid, r)
            for rid, r in rows
            if isinstance(r, dict) and isinstance(r.get("messages"), list) and r["messages"]
        ]

    def _outcome(self, run: dict) -> bool | None:
        if callable(self.success):
            return bool(self.success(run))
        for k in (self.success,) if self.success else ("success", "reward", "score"):
            if run.get(k) is not None:
                return _positive(run[k])
        return None

    def _task_id(self, run: dict, default: str) -> str:
        for k in (self.task_id_key,) if self.task_id_key else ("task_id", "id"):
            if run.get(k) not in (None, ""):
                return str(run[k])
        return default

    def _pool(self, run: dict) -> str:
        return ""

    def _steps(self, run: dict) -> tuple[list[dict], list[Call]]:
        return _normalize(run.get("messages"), run.get("system"), self.max_chars)

    def _note(self, traj: Trajectory) -> list[dict]:
        """Normalize a run, remembering its catalog and its tool calls (the fallback candidate pool)."""
        run = traj.raw
        pool = self._pool_of[traj.id] = self._pool(run)
        own = parse_catalog(run.get("tools"))
        self._own[traj.id] = own
        self._seen.setdefault(pool, {}).update(own)
        steps, calls = self._steps(run)
        called = self._called.setdefault(pool, {})
        for _, name, args in calls:
            called.setdefault(name, set()).update(args)
        return steps

    def _fields(self, run: dict, task_id: str) -> dict[str, Any]:
        out: dict[str, Any] = {"task_id": task_id}
        for k in self.task_keys:
            if k != "task_id" and isinstance(run.get(k), (str, int, float, bool)):
                out[k] = run[k]
        return out

    # ------------------------------------------------------------------ the adapter protocol
    def decision_points(self, traj: Trajectory, target: str) -> list[DecisionPoint]:
        if target not in self.targets:
            raise ValueError(f"{type(self).__name__} builds the decisions {self.targets}, not {target!r}")
        steps = self._note(traj)
        turns = [i for i, s in enumerate(steps) if s["name"] == "assistant" and s["tool_calls"]]
        if not turns:
            return []
        fields = self._fields(traj.raw, self._task_id(traj.raw, traj.id))

        def dp(i: int, prefix_steps: list[dict], calls: list[str]) -> DecisionPoint:
            prefix = {**fields, "steps": copy.deepcopy(prefix_steps)}
            chosen = {"name": target, "tool_calls": list(calls)}
            return DecisionPoint(traj.id, i, prefix, chosen, traj.group, traj.source)

        if target == TOOL_SHORTLIST:
            first = turns[0]
            used = list(dict.fromkeys(t for i in turns for t in steps[i]["tool_calls"]))
            return [dp(first, [s for s in steps[:first] if s["name"] in ("system", "user")], used)]
        return [dp(i, steps[:i], steps[i]["tool_calls"]) for i in turns]

    def candidates(self, dp: DecisionPoint) -> list[Candidate]:
        cat = self._own.get(dp.trajectory_id) or self.catalog or self._pool_candidates(dp.trajectory_id)
        return sorted(cat.values(), key=lambda c: c.id)

    def _pool_candidates(self, trajectory_id: str) -> dict[str, Candidate]:
        pool = self._pool_of.get(trajectory_id, "")
        out = {
            n: Candidate(n, n, "", ", ".join(sorted(args))) for n, args in self._called.get(pool, {}).items()
        }
        out.update(self._seen.get(pool, {}))
        return out

    def labels(self, dp: DecisionPoint) -> set[str]:
        """The tools chosen at the decision (for ``ToolShortlist``: every tool the run called)."""
        return set(dp.target_step["tool_calls"])

    def components(self, trajs: list[Trajectory]) -> dict[str, int]:
        """The decision this adapter builds -> number of successful runs with at least one tool call."""
        n = sum(
            1
            for t in trajs
            if t.success and any(s["name"] == "assistant" and s["tool_calls"] for s in self._steps(t.raw)[0])
        )
        return {self.target: n} if n else {}

    def live_prefix(self, run: dict, target: str | None = None) -> dict[str, Any]:
        """The prefix of a live decision, built exactly as in training; ``run["messages"]`` holds the
        conversation so far (for ``ToolShortlist``, the messages before the first tool call)."""
        steps, _ = self._steps(run)
        if (target or self.target) == TOOL_SHORTLIST:
            steps = [s for s in steps if s["name"] in ("system", "user")]
        return {**self._fields(run, self._task_id(run, "")), "steps": steps}
