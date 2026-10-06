"""Chat-message logs: OpenAI Chat Completions, the Anthropic Messages API format and LangChain dumps.

A run is one JSON object with a ``messages`` list: one per ``.json`` file (or a list of runs), or one per line
of a ``.jsonl`` file. Optional fields: ``tools`` (the tool catalog), an outcome (``success``, ``reward`` or
``score``) and a task id (``task_id`` or ``id``). The message dialects (OpenAI, including legacy
``function_call`` and Responses API items; the Anthropic Messages API format with ``tool_use`` and
``tool_result`` blocks and a top-level ``system``; LangChain ``type``/``tool_calls`` dumps) are normalized
to one step shape, which is what TabSchema's extraction code reads::

    {"instruction": "...", "steps": [{"name": "user", "content": "...", "tool_calls": [], "tool_name": ""}]}

Every step has the same four keys: ``name`` (system, user, assistant or tool), ``content`` (text; images and
base64 payloads become placeholders, very long text is clipped), ``tool_calls`` (the tools a turn called)
and ``tool_name`` (for a tool result, the tool that produced it). Only scalar task fields from a whitelist
(``task_keys``) are copied, so post-hoc run fields (durations, costs, final answers) never reach a prefix. The
task id is not copied either: it names the run's cross-validation group, and as a feature it could only
memorize training tasks (a new task's id never occurs in training).

Two decision targets:

* ``ToolShortlist`` (``granularity="task"``, the paper's tool shortlisting): one decision per run, placed
  before the first assistant turn that calls a tool. The prefix holds the task fields and the system and
  user messages before it; the labels are all the tools the run called (G(t), read from the full run).
* ``NextToolCall`` (``granularity="step"``): one decision per assistant turn that calls tools. The prefix
  holds every message before that turn; the labels are the tools called in it.

For ``tabagent.scan``, every assistant turn is one LLM call: a ``NextToolCall`` when it calls tools, else a
``TextReply`` (free text), with the tokens, cost and latency the message records, if any.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tabagent.adapters import logfiles as lf
from tabagent.adapters.common import parse_catalog, run_catalog
from tabagent.tabschema.trajectory import (
    Candidate,
    DecisionPoint,
    LLMCall,
    Trajectory,
    _example,
    _not_a_run,
    _refuse_mixed,
    json_copy,
    parse_json,
    read_outcome,
    read_usage,
)

TOOL_SHORTLIST, NEXT_TOOL_CALL = "ToolShortlist", "NextToolCall"
TEXT_REPLY = "TextReply"  # an assistant turn without a tool call (free text; reported by tabagent.scan)
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
_BINARY = re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=\r\n]{64,}|[A-Za-z0-9+/]{4096,}={0,2}")
_MIXED = (re.compile("[a-z]"), re.compile("[A-Z]"), re.compile("[0-9]"))
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]")
#: Responses API items that call a tool (their output is a separate item, or, for MCP calls, inside them).
CALL_ITEMS = ("function_call", "custom_tool_call", "mcp_call")
OUTPUT_ITEMS = ("function_call_output", "custom_tool_call_output")
_NO_TOOL = re.compile(  # a call to a tool the agent does not have (tau2, LangGraph ToolNode)
    r"\s*error:?\s*(?:tool\s+['\"]?[\w.-]+['\"]?\s+(?:not found|does not exist)|[\w.-]+ is not a valid tool)",
    re.I,
)
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


def _binary(m: re.Match) -> str:
    """A data URL, or a bare base64 run of 4,096 characters or more (lower and upper case and digits, unlike
    a long word, hex digest or DNA sequence), becomes ``[binary]``."""
    text = m.group()
    if text.startswith("data:") or all(p.search(text) for p in _MIXED):
        return "[binary]"
    return text


def _clip(s: str, max_chars: int | None) -> str:
    s = _BINARY.sub(_binary, s)
    if len(s) >= 64 and len(_CONTROL.findall(s[:4096])) > len(s[:4096]) // 10:  # bytes read as text
        return f"[binary: {len(s):,} characters]"
    if max_chars and len(s) > max_chars:
        s = f"{s[:max_chars]} ...[{len(s) - max_chars} more characters]"
    return s


def _tool_name(value: Any) -> str | None:
    """A tool name as logged: a non-empty string (surrounding spaces dropped), or None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _call(tc: dict) -> Call | None:
    f = tc["function"] if isinstance(tc.get("function"), dict) else tc
    name = _tool_name(f.get("name"))
    if name is None:
        return None
    args = f.get("arguments", f.get("args", f.get("input")))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = None
    keys = sorted(str(k) for k in args) if isinstance(args, dict) else []
    cid = tc.get("call_id") or tc.get("id") or ""
    return cid if isinstance(cid, str) else json.dumps(cid, default=str), name, keys


def _raw_calls(msg: dict) -> list[dict]:
    """Tool-call records of one message, from the first dialect that has any (skips LangChain duplicates)."""
    blocks = msg["content"] if isinstance(msg.get("content"), list) else []
    extra = msg.get("additional_kwargs")
    sources = (
        _as_list(msg.get("tool_calls")),  # OpenAI, tau2, LangChain
        _as_list(extra.get("tool_calls")) if isinstance(extra, dict) else [],  # LangChain, raw OpenAI payload
        _as_list(msg.get("function_call")),  # legacy OpenAI
        # Anthropic Messages API format
        [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"],
    )
    return next((s for s in sources if s), [])


def _as_list(value: Any) -> list[dict]:
    """Tool-call records: a list of them, one record, or JSON text of either (some loggers write a string)."""
    if isinstance(value, str):
        value = parse_json(value)
    if isinstance(value, dict):
        value = [value]
    return [tc for tc in value if isinstance(tc, dict)] if isinstance(value, list) else []


def _calls(msg: dict) -> list[Call]:
    """Tool calls of one message (see :func:`_raw_calls`)."""
    return [c for c in map(_call, _raw_calls(msg)) if c]


def _args_chars(tc: dict) -> int:
    """Length of a tool call's arguments as written (generated text)."""
    f = tc["function"] if isinstance(tc.get("function"), dict) else tc
    args = f.get("arguments", f.get("args", f.get("input")))
    if args is None:
        return 0
    return len(args) if isinstance(args, str) else len(json.dumps(args, default=str))


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
        if kind in CALL_ITEMS:  # OpenAI Responses API: a call item joins the assistant turn before it
            c = _call(m)
            if c:
                names[c[0]] = c[1]
                agent_calls.append(c)
                if steps and steps[-1]["name"] == "assistant":
                    steps[-1]["tool_calls"].append(c[1])
                else:
                    add("assistant", "", [c[1]])
                if kind == "mcp_call" and m.get("output") is not None:  # an MCP call carries its output
                    add("tool", m.get("output"), (), c[1])
            continue
        if kind in OUTPUT_ITEMS:
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


def assistant_turns(messages: list[dict]) -> list[list[int]]:
    """Indices of each assistant turn (one LLM call) in unwrapped messages: an assistant message, or the
    consecutive output items (reasoning, messages, function calls) of one OpenAI Responses API response."""
    turns: list[list[int]] = []
    in_response = False
    for i, m in enumerate(messages):
        kind = m.get("type")
        if kind in (*CALL_ITEMS, "reasoning") or (kind == "message" and message_role(m) == "assistant"):
            if not in_response:
                turns.append([])
            turns[-1].append(i)
            in_response = True
            continue
        if message_role(m) == "assistant":
            turns.append([i])
        in_response = False
    return turns


def normalize_messages(messages: list, system: Any = None, max_chars: int | None = 20_000) -> list[dict]:
    """Chat messages in any supported dialect -> steps ``{"name", "content", "tool_calls", "tool_name"}``."""
    return _normalize(messages, system, max_chars)[0]


# --------------------------------------------------------------------------- adapter
class OpenAIMessagesAdapter:
    """Chat-message runs -> trajectories and tool-selection decision points (see the module docstring).

    Candidates come from the run's own ``tools``, else the ``tools`` catalog given here, else the union of
    the tools called across the loaded runs (argument hints then list the argument names observed in calls;
    calls answered with a tool-not-found error are left out).
    ``success`` is an outcome key or a function of the run; runs without an outcome count as successful, and
    runs whose outcome cannot be read (see ``tabagent.tabschema.trajectory.read_outcome``) as failed; both
    are reported in ``warnings``, with every file, line or record ``load`` skipped or repaired.
    """

    targets = (TOOL_SHORTLIST, NEXT_TOOL_CALL)
    scan_notes: tuple[str, ...] = ()
    format_name = "chat-message runs"

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
        self.notes = lf.LoadNotes()
        self._reset()

    def _reset(self) -> None:
        """Forget the runs, catalogs and tool pools of an earlier ``load``."""
        self._own: dict[str, dict[str, Candidate]] = {}  # trajectory id -> the run's own catalog
        self._pool_of: dict[str, str] = {}  # trajectory id -> candidate pool (tau2: the domain)
        self._seen: dict[str, dict[str, Candidate]] = {}  # pool -> the catalog entries seen in loaded runs
        self._seen_counts: dict[str, dict[str, Counter]] = {}  # pool -> tool -> runs per entry (most wins)
        self._called: dict[str, dict[str, set[str]]] = {}  # pool -> tool -> argument names seen in calls
        self._n_bad, self._bad = 0, []  # runs with an unreadable outcome, and the first ones
        # trajectory id -> (the trajectory, whether it calls a tool), so components() need not read it again
        self._calls_tools: dict[str, tuple[Trajectory, bool]] = {}

    @property
    def warnings(self) -> list[str]:
        """Data notes for the onboarding report: files and records skipped or repaired, runs without an
        outcome or with an unreadable one."""
        return self.notes.warnings()

    # ------------------------------------------------------------------ loading
    #: What ``load`` does with runs of another log format: ``"skip"`` them with a warning (an adapter given
    #: explicitly always does), or ``"error"`` (``detect_adapter(..., on_other_formats="error")``).
    on_other_formats = "skip"

    def load(self, path: str | Path, *, lazy: bool | None = None) -> list[Trajectory]:
        """The runs in one file or every ``.json``/``.jsonl`` file below a directory (see the module
        docstring); what was skipped or repaired, runs of another log format (τ²-bench results, step logs)
        included, is reported in ``warnings``. ``lazy``: read runs again from their files when needed instead
        of holding them (default: from 256 MB of logs, ``logfiles.LAZY_INPUT_BYTES``)."""
        root = Path(path)
        self.notes = notes = lf.LoadNotes()
        self._reset()
        files = lf.iter_files(root)
        lazy = lf.use_lazy(files, lazy)
        cache = lf.RunCache()  # the parsed runs read last; it goes away with these trajectories
        ids = lf.RunIds()
        out: list[Trajectory] = []
        no_outcome = 0
        for f in files:
            base = lf.run_id(root, f)
            for index, run, where, pick, doc in self._file_runs(root, f, notes):
                rid = ids(base, f, index)
                ok = self._outcome(run, where)
                no_outcome += ok is None
                load = lf.lazy_loader(doc, pick, cache) if lazy and pick is not False else None
                success, group = ok is not False, self._task_id(run, rid)
                traj = (
                    Trajectory(rid, run, success, group, f.name)
                    if load is None
                    else Trajectory.lazy(rid, load, success, group, f.name)
                )
                self._note(traj, run)
                out.append(traj)
        keys = self.success if isinstance(self.success, str) else "success/reward/score"
        if no_outcome:
            notes.set_text(
                "no_outcome",
                f"{no_outcome} of {len(out)} runs have no outcome field ({keys}); they were treated as "
                "successful",
                n=no_outcome,
            )
        if self._n_bad:
            notes.set_text(
                "bad_outcome",
                f"{self._n_bad} of {len(out)} runs have an outcome ({keys}) that is not a boolean, a finite "
                f"number or a pass/fail word (first: {'; '.join(self._bad)}); they were treated as failed "
                "(pass success= to read them)",
                n=self._n_bad,
            )
        _refuse_mixed(self, root, notes, len(out), len(files))
        return out

    def _file_runs(self, root: Path, f: Path, notes: Any):
        """``(index, run, where, pick, doc)`` for each run of one file. A ``.jsonl`` file whose lines are all
        messages (no ``messages`` key) is one conversation: one run."""
        docs = iter(lf.read_docs(root, f, notes))  # one at a time: a large file is never held whole
        held: list[Any] = []  # the leading lines that are bare messages (a conversation, one per line?)
        for doc in docs:
            if doc.line is not None and _bare_message(doc.data):
                held.append(doc)
                continue
            docs = itertools.chain(held, [doc], docs)  # not a conversation: every line on its own
            held = []
            break
        if held:
            yield None, {"messages": [d.data for d in held]}, lf.relative(root, f), False, held[0]
            return
        for doc in docs:
            for index, run, where, pick in self._runs(doc, notes):
                yield index, run, where, pick, doc

    def _runs(self, doc: Any, notes: Any) -> list[tuple[Any, dict, str, Any]]:
        """``(index, run, where, pick)`` for each run in one JSON document: a run object, a list of runs, or a
        bare message list. ``pick`` rebuilds the run from the document read again (None: the document is the
        run; False: it cannot be read again on its own)."""
        data, where = doc.data, doc.where
        if isinstance(data, dict):
            return self._checked([(doc.line, data, where, None)], notes)
        if not isinstance(data, list) or not data:
            _not_a_run(self, data, where, notes)
            return []
        if any(isinstance(x, dict) and "messages" in x for x in data):  # a list of runs
            rows = [
                (i if doc.line is None else f"{doc.line}.{i}", x, f"{where}[{i}]", False)
                for i, x in enumerate(data)
            ]
            return self._checked(rows, notes)
        if sum(_looks_like_message(x) for x in data) * 2 >= len(data):  # a bare message list
            return [(doc.line, {"messages": data}, where, _bare)]
        _not_a_run(self, data[0] if isinstance(data[0], dict) else data, where, notes)
        return []

    def _checked(self, rows: list, notes: Any) -> list:
        out = []
        for index, run, where, pick in rows:
            msgs = run.get("messages") if isinstance(run, dict) else None
            if isinstance(msgs, list) and msgs:
                out.append((index, run, where, pick))
            elif isinstance(run, dict) and "messages" in run:
                notes.add("no_messages", where, "runs", found="runs without messages, skipped")
            else:
                _not_a_run(self, run, where, notes)
        return out

    def _outcome(self, run: dict, where: str = "") -> bool | None:
        """Whether the run succeeded; None when it records no outcome. A value that cannot be read counts as
        a failure, and is reported."""
        if callable(self.success):
            try:
                return bool(self.success(json_copy(run)))  # a copy: an edit never changes the loaded run
            except Exception as e:  # noqa: BLE001 - the user's function: say where it failed
                raise ValueError(
                    f"{where}: the success function raised {type(e).__name__}: {e}; make it return True or "
                    "False for every run (or pass success= with the name of the outcome field)"
                ) from e
        for k in (self.success,) if self.success else ("success", "reward", "score"):
            if run.get(k) is not None:
                ok = read_outcome(run[k])
                if ok is None:
                    self._n_bad += 1
                    if len(self._bad) < 3:
                        self._bad.append(f"{where}: {k}={_example(run[k])}")
                    return False
                return ok
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

    def _note(self, traj: Trajectory, run: dict | None = None) -> list[dict]:
        """Normalize a run, remembering its catalog and its tool calls (the fallback candidate pool)."""
        run = traj.raw if run is None else run
        pool = self._pool_of[traj.id] = self._pool(run)
        own, readable = run_catalog(run.get("tools"))
        if traj.id not in self._own:  # a run counts once, however often it is read
            if not readable:
                self.notes.add(
                    "tools_field",
                    traj.id,
                    "runs",
                    found="a `tools` field that is not a tool catalog (a list or mapping of tools), ignored",
                )
            counts, seen = self._seen_counts.setdefault(pool, {}), self._seen.setdefault(pool, {})
            for name, c in own.items():
                counts.setdefault(name, Counter())[c] += 1
                seen[name] = _most_common(counts[name])
        self._own[traj.id] = own
        steps, calls = self._steps(run)
        called = self._called.setdefault(pool, {})
        for _, name, args in calls:
            called.setdefault(name, set()).update(args)
        self._calls_tools[traj.id] = (traj, _calls_a_tool(steps))
        return steps

    def _fields(self, run: dict) -> dict[str, Any]:
        """The run-level task fields of a prefix: the scalar values of ``task_keys``, without the task id."""
        ids = {"task_id", "id", self.task_id_key}
        scalar = (str, int, float, bool)
        return {k: run[k] for k in self.task_keys if k not in ids and isinstance(run.get(k), scalar)}

    # ------------------------------------------------------------------ the adapter protocol
    def decision_points(self, traj: Trajectory, target: str) -> list[DecisionPoint]:
        """The decision points of a run (see the module docstring). The prefixes of one run share their step
        objects (see ``DecisionPoint``): treat them as read-only."""
        if target not in self.targets:
            raise ValueError(f"{type(self).__name__} builds the decisions {self.targets}, not {target!r}")
        steps = self._note(traj)
        turns = [i for i, s in enumerate(steps) if s["name"] == "assistant" and s["tool_calls"]]
        if not turns:
            return []
        fields = self._fields(traj.raw)

        def dp(i: int, prefix_steps: list[dict], calls: list[str]) -> DecisionPoint:
            prefix = {**fields, "steps": prefix_steps}
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
        n = 0
        for t in trajs:
            if t.success:
                seen = self._calls_tools.get(t.id)
                n += seen[1] if seen is not None and seen[0] is t else _calls_a_tool(self._steps(t.raw)[0])
        return {self.target: n} if n else {}

    # ------------------------------------------------------------------ tabagent.scan
    def llm_calls(self, traj: Trajectory) -> list[LLMCall]:
        """The agent's LLM calls in a run, for ``tabagent.scan``: one per assistant turn (see
        :func:`assistant_turns`). A turn that calls tools is a ``NextToolCall``: its choices are the tools it
        called, its catalog the run's ``tools`` (else the catalog given here), its free text the content and
        the arguments it wrote. A turn that only writes text is a ``TextReply``. Usage is read from the turn's
        own fields (``usage``, ``usage_metadata``, ``response_metadata``, ``cost``, ...) when recorded."""
        run = traj.raw
        catalog = tuple(sorted(run_catalog(run.get("tools"))[0] or self.catalog)) or None
        messages = run.get("messages")
        msgs = (
            [unwrap_message(m) for m in messages if isinstance(m, dict)] if isinstance(messages, list) else []
        )
        decisions = (self.target, *(t for t in self.targets if t != self.target))
        model = self._model(run)
        out: list[LLMCall] = []
        for turn in assistant_turns(msgs):
            items = [msgs[i] for i in turn]
            if not self._is_llm_call(items[0]):
                continue
            records = [r for m in items for r in ([m] if m.get("type") in CALL_ITEMS else _raw_calls(m))]
            tools = tuple(c[1] for c in map(_call, records) if c)
            text = sum(len(_text(m.get("content"))) for m in items) + sum(map(_args_chars, records))
            usage = next((u for u in map(read_usage, items) if u), {})
            if "latency_s" not in usage and (latency := self._latency(msgs, turn[0])) is not None:
                usage["latency_s"] = latency
            if model and "model" not in usage:
                usage["model"] = model
            if tools:
                kw = {"choices": tools, "catalog": catalog, "decisions": decisions}
                out.append(LLMCall(NEXT_TOOL_CALL, text_chars=text, **kw, **usage))
            else:
                out.append(LLMCall(TEXT_REPLY, text_chars=text, **usage))
        return out

    def _is_llm_call(self, msg: dict) -> bool:
        """Whether an assistant turn came from the LLM (adapters for logs with scripted turns override it)."""
        return True

    def _latency(self, msgs: list[dict], i: int) -> float | None:
        """Seconds the LLM call that wrote ``msgs[i]`` took, when no latency field records it."""
        return None

    def _model(self, run: dict) -> str | None:
        return run["model"] if isinstance(run.get("model"), str) and run["model"] else None

    def live_prefix(self, run: dict, target: str | None = None) -> dict[str, Any]:
        """The prefix of a live decision, built exactly as in training.

        ``run`` has the shape of a logged run: ``messages`` holds the conversation so far (for
        ``ToolShortlist``, the messages before the first tool call) and the run-level task fields
        (``instruction``, ``domain``, ... as listed in ``task_keys``) sit next to it, as in the logs the head
        was trained on. A field the training runs had and ``run`` lacks is missing from the prefix too.
        """
        if not isinstance(run, dict):
            raise TypeError(
                "a live run is the conversation so far, a dict like {'messages': [...], <task fields>}, not "
                f"{type(run).__name__}"
            )
        steps, _ = self._steps(run)
        if (target or self.target) == TOOL_SHORTLIST:
            steps = [s for s in steps if s["name"] in ("system", "user")]
        return {**self._fields(run), "steps": steps}


def _calls_a_tool(steps: list[dict]) -> bool:
    return any(s["name"] == "assistant" and s["tool_calls"] for s in steps)


def _bare(messages: Any) -> dict[str, Any]:
    return {"messages": messages}


def _bare_message(m: Any) -> bool:
    """A message on its own (a line of a conversation file), not a run."""
    return isinstance(m, dict) and "messages" not in m and _looks_like_message(m)


def _looks_like_message(m: Any) -> bool:
    if not isinstance(m, dict):
        return False
    m = unwrap_message(m)
    return (
        bool(m.get("role"))
        or m.get("type") in (*CALL_ITEMS, *OUTPUT_ITEMS, "message", "reasoning")
        or (isinstance(m.get("type"), str) and "content" in m)
    )


def _most_common(entries: Counter) -> Candidate:
    """The catalog entry of a tool most runs recorded (ties: the longest description, then the first in
    order), the same whatever order the runs were read in."""
    return max(
        entries.items(), key=lambda kv: (kv[1], len(kv[0].description), kv[0].description, kv[0].arg_hints)
    )[0]
