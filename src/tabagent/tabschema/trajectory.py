"""Trajectories, decision points and the JSON utilities TabSchema's prompts are built on (paper App. D).

TabSchema works on an agent's *native* logs: the extraction code it writes reads the agent's own logging
schema, which only has to be stable across the onboarding pool (Sec. 3.1). A trajectory is a JSON object with
a list of steps; each step names the component that produced it. A *decision point* is one occurrence of the
component being replaced (for example ``ShortlisterAgent``) inside a successful trajectory.

Features use only what precedes the decision (App. D.2, FeatureAnalyzer and CodeExtractor prompts), and this
is enforced structurally: the ``prefix`` a decision point exposes is a copy of the trajectory truncated
before the decision step, with post-hoc outcome fields (scores, evaluator results) removed. Extraction code
never receives the decision step, later steps, or outcomes.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Candidate:
    """One legal choice at a decision: a tool, skill, API, application, model or arm."""

    id: str
    name: str = ""
    description: str = ""
    arg_hints: str = ""


@dataclass(frozen=True)
class Trajectory:
    """One agent run: its raw log, whether it succeeded, its task (group) and source file."""

    id: str
    raw: dict[str, Any]
    success: bool = True
    group: str | None = None  # task id; decision points of one task never straddle CV folds
    source: str = ""  # file name, for provenance (the analyzer prompt's ``trajectory_file``)


@dataclass(frozen=True)
class DecisionPoint:
    """One occurrence of the replaced component: the redacted prefix before it and the step itself."""

    trajectory_id: str
    step_index: int
    prefix: dict[str, Any] = field(repr=False)
    target_step: dict[str, Any] = field(repr=False)
    group: str | None = None
    source: str = ""

    @property
    def id(self) -> str:
        """``<trajectory id>#<step index>``."""
        return f"{self.trajectory_id}#{self.step_index}"


@dataclass(frozen=True)
class LLMCall:
    """One LLM call of an agent run, as an adapter's optional ``llm_calls(trajectory)`` hook reports it to
    ``tabagent.scan``.

    ``choices`` is what the call chose when the adapter knows the component's decision (the tools a turn
    called, the APIs a shortlister returned), and ``catalog`` what it was offered, when the log records it.
    ``fields`` holds the short categorical values of a structured output the adapter has no decision for
    (``{"action": ("CoderAgent",)}``, see :func:`output_fields`); ``text_chars`` the free text it generated.
    Usage fields are None unless the log records them. ``decisions`` names the ``onboard`` decisions that
    replace the call's choice (empty: the component's own name). ``inferred`` marks a step the log does not
    mark as an LLM call that was counted as one.
    """

    component: str
    choices: tuple[str, ...] | None = None
    catalog: tuple[str, ...] | None = None
    fields: dict[str, tuple[str, ...]] = field(default_factory=dict)
    text_chars: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_s: float | None = None
    cost_usd: float | None = None
    model: str | None = None
    decisions: tuple[str, ...] = ()
    inferred: bool = False

    @property
    def tokens(self) -> int | None:
        """Input plus output tokens, else the recorded total, else None."""
        if self.input_tokens is not None or self.output_tokens is not None:
            return (self.input_tokens or 0) + (self.output_tokens or 0)
        return self.total_tokens


def redact(
    raw: dict[str, Any],
    steps_key: str,
    cut: int,
    outcome_keys: Iterable[str],
    outcome_steps: Iterable[str],
    step_name_key: str,
) -> dict[str, Any]:
    """Copy of ``raw`` with steps ``[:cut]`` only, top-level outcome keys removed, outcome steps dropped."""
    out = {k: copy.deepcopy(v) for k, v in raw.items() if k != steps_key and k not in set(outcome_keys)}
    drop = set(outcome_steps)
    out[steps_key] = [
        copy.deepcopy(s) for s in raw.get(steps_key, [])[:cut] if s.get(step_name_key) not in drop
    ]
    return out


class GenericJsonAdapter:
    """Reads one JSON trajectory per file: ``{..., "steps": [{"name": <component>, ...}, ...], <outcome>}``.

    Candidates and labels are benchmark-specific; pass ``candidates_fn(dp)`` and ``labels_fn(dp)`` (for
    example, the catalog of tools and the tools the successful run actually called), or use a dedicated
    adapter (``tabagent.adapters``).
    """

    #: What ``tabagent.scan`` should tell the reader about how these logs record LLM calls.
    scan_notes: tuple[str, ...] = ()

    def __init__(
        self,
        steps_key: str = "steps",
        step_name_key: str = "name",
        success: Callable[[dict], bool] | None = None,
        group_key: str | None = "task_id",
        outcome_keys: Iterable[str] = ("score", "eval", "evaluation", "reward", "success", "outcome"),
        outcome_steps: Iterable[str] = ("EvaluationResult",),
        candidates_fn: Callable[[DecisionPoint], list[Candidate]] | None = None,
        labels_fn: Callable[[DecisionPoint, dict], set[str]] | None = None,
    ):
        self.steps_key = steps_key
        self.step_name_key = step_name_key
        self.success = success or (lambda raw: float(raw.get("score", 1.0) or 0.0) > 0)
        self.group_key = group_key
        self.outcome_keys = tuple(outcome_keys)
        self.outcome_steps = tuple(outcome_steps)
        self.candidates_fn = candidates_fn
        self.labels_fn = labels_fn
        self._raw: dict[str, dict] = {}  # full trajectories, for labels (post-decision information)

    def load(self, path: str | Path) -> list[Trajectory]:
        """Trajectories from one JSON file or every ``*.json`` of a directory. Files whose name contains
        ``results`` or ``metadata`` (run summaries) and JSON without the steps key are skipped."""
        p = Path(path)
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        out = []
        for f in files:
            if "results" in f.name or "metadata" in f.name:  # run summaries, not trajectories
                continue
            raw = json.loads(f.read_text())
            if not isinstance(raw, dict) or self.steps_key not in raw:
                continue
            group = str(raw.get(self.group_key, f.stem)) if self.group_key else f.stem
            out.append(Trajectory(f.stem, raw, bool(self.success(raw)), group, f.name))
        return out

    def decision_points(self, traj: Trajectory, target: str) -> list[DecisionPoint]:
        """Every step named ``target``, with its redacted prefix (see :func:`redact`)."""
        self._raw[traj.id] = traj.raw
        steps = traj.raw.get(self.steps_key, [])
        return [
            DecisionPoint(
                traj.id,
                i,
                redact(
                    traj.raw, self.steps_key, i, self.outcome_keys, self.outcome_steps, self.step_name_key
                ),
                copy.deepcopy(step),
                traj.group,
                traj.source,
            )
            for i, step in enumerate(steps)
            if step.get(self.step_name_key) == target
        ]

    def components(self, trajs: list[Trajectory]) -> dict[str, int]:
        """Step name -> number of successful trajectories containing it (the replaceable components)."""
        counts: dict[str, int] = {}
        drop = set(self.outcome_steps)
        for t in trajs:
            if not t.success:
                continue
            names = {s.get(self.step_name_key) for s in t.raw.get(self.steps_key, [])} - drop - {None}
            for n in names:
                counts[str(n)] = counts.get(str(n), 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def candidates(self, dp: DecisionPoint) -> list[Candidate]:
        """The legal candidates of a decision point (from ``candidates_fn``)."""
        if self.candidates_fn is None:
            raise NotImplementedError("pass candidates_fn= to GenericJsonAdapter or use a dedicated adapter")
        return self.candidates_fn(dp)

    def labels(self, dp: DecisionPoint) -> set[str]:
        """The chosen candidates, read from the *full* trajectory (e.g. the tools the successful run used).
        Available for decision points produced by this adapter's ``decision_points``."""
        if self.labels_fn is None:
            raise NotImplementedError("pass labels_fn= to GenericJsonAdapter or use a dedicated adapter")
        return self.labels_fn(dp, self._raw[dp.trajectory_id])

    def llm_calls(self, traj: Trajectory) -> list[LLMCall]:
        """The LLM calls of a run, for ``tabagent.scan``: one per step that records an LLM call (a usage,
        model, prompt or messages field). When no step of the run records one, every named step counts as a
        call, marked ``inferred``. The output is the step's ``output``, ``data``, ``result``, ``response``,
        ``completion``, ``answer``, ``decision`` or ``choice`` field (JSON text is parsed), else its other
        fields; usage is read with :func:`read_usage`."""
        drop = set(self.outcome_steps)
        steps = [
            s
            for s in traj.raw.get(self.steps_key, [])
            if isinstance(s, dict) and s.get(self.step_name_key) and s.get(self.step_name_key) not in drop
        ]
        marked = [s for s in steps if any(s.get(k) not in (None, "", [], {}) for k in _LLM_KEYS)]
        return [self._llm_call(s, inferred=not marked) for s in marked or steps]

    def _llm_call(self, step: dict[str, Any], inferred: bool) -> LLMCall:
        key = next((k for k in _OUTPUT_KEYS if step.get(k) not in (None, "")), None)
        if key is None:
            output: Any = {k: v for k, v in step.items() if k != self.step_name_key and k not in _STEP_META}
        else:
            parsed = parse_json(step[key])
            output = step[key] if parsed is None else parsed
        fields, text = output_fields(output, "" if isinstance(output, dict) else key or "output")
        name = str(step[self.step_name_key])
        return LLMCall(name, fields=fields, text_chars=text, inferred=inferred, **read_usage(step))


# --------------------------------------------------------------------------- LLM-call utilities
# Step keys that show a step called an LLM, keys that hold its output, and bookkeeping keys that are neither.
_LLM_KEYS = (
    "usage",
    "usage_metadata",
    "token_usage",
    "model",
    "model_name",
    "prompt",
    "prompts",
    "messages",
    "llm",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)
_OUTPUT_KEYS = ("output", "data", "result", "response", "completion", "answer", "decision", "choice")
_STEP_META = frozenset(
    {
        *_LLM_KEYS,
        "id",
        "step",
        "type",
        "input",
        "inputs",
        "timestamp",
        "start_time",
        "end_time",
        "latency",
        "latency_s",
        "latency_ms",
        "duration",
        "duration_s",
        "duration_ms",
        "elapsed",
        "elapsed_s",
        "elapsed_ms",
        "cost",
        "cost_usd",
        "response_cost",
        "tokens",
        "response_metadata",
    }
)
_USAGE_KEYS = ("usage", "usage_metadata", "token_usage")
_IN_TOKENS = ("input_tokens", "prompt_tokens")
_CACHE_TOKENS = ("cache_creation_input_tokens", "cache_read_input_tokens")  # Anthropic: not in input_tokens
_OUT_TOKENS = ("output_tokens", "completion_tokens")
_TOTAL_TOKENS = ("total_tokens", "tokens")
_COST = ("cost_usd", "cost", "response_cost")
_SECONDS = ("latency_s", "latency", "duration_s", "duration", "elapsed_s", "elapsed")
_MILLISECONDS = ("latency_ms", "duration_ms", "elapsed_ms")
_REASONING = frozenset({"thoughts", "thought", "reasoning", "rationale", "explanation", "reflection"})
_NAME_KEYS = ("name", "id", "api_name", "tool", "label")
#: A categorical output value: a string of at most this many characters, on one line.
SHORT_VALUE = 64
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: Any) -> Any:
    """``text`` parsed as JSON (tolerating Markdown code fences); dicts and lists pass through; else None."""
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str) or not text.strip():
        return None
    for s in (text, *_FENCE.findall(text)):
        try:
            return json.loads(s)
        except ValueError:
            continue
    return None


def _number(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 else None


def read_usage(record: Any) -> dict[str, Any]:
    """The tokens, latency, cost and model recorded on one LLM call's log record, as ``LLMCall`` keyword
    arguments (only what is recorded).

    Reads ``usage`` (OpenAI ``prompt_tokens``/``completion_tokens``; the Anthropic Messages API format's
    ``input_tokens``/``output_tokens``, whose cache-read and cache-write tokens count as input), LangChain
    ``usage_metadata`` and ``response_metadata`` (``token_usage``, ``model_name``), and flat keys: token
    counts, ``cost_usd``/``cost``/``response_cost`` (USD), ``latency_s``/``latency``/``duration``/``elapsed``
    (seconds), ``latency_ms``/``duration_ms``/``elapsed_ms``, ``model``/``model_name``.
    """
    if not isinstance(record, dict):
        return {}
    meta = record.get("response_metadata")
    meta = meta if isinstance(meta, dict) else {}
    sources = [record, meta]
    sources += [s[k] for s in (record, meta) for k in _USAGE_KEYS if isinstance(s.get(k), dict)]

    def first(keys: Iterable[str]) -> tuple[float | None, dict]:
        for s in sources:
            for k in keys:
                v = _number(s.get(k))
                if v is not None:
                    return v, s
        return None, {}

    out: dict[str, Any] = {}
    tin, src = first(_IN_TOKENS)
    if tin is not None:
        out["input_tokens"] = int(tin + sum(_number(src.get(k)) or 0 for k in _CACHE_TOKENS))
    for name, keys in (("output_tokens", _OUT_TOKENS), ("total_tokens", _TOTAL_TOKENS)):
        v, _ = first(keys)
        if v is not None:
            out[name] = int(v)
    cost, _ = first(_COST)
    if cost is not None:
        out["cost_usd"] = cost
    seconds, _ = first(_SECONDS)
    ms, _ = first(_MILLISECONDS)
    if seconds is not None or ms is not None:
        out["latency_s"] = seconds if seconds is not None else ms / 1000
    model = next(
        (s[k] for s in (record, meta) for k in ("model", "model_name") if isinstance(s.get(k), str) and s[k]),
        None,
    )
    if model:
        out["model"] = model
    return out


def _item_name(v: Any) -> str | None:
    if isinstance(v, str):
        return v if _short(v) else None
    if isinstance(v, dict):
        return next((v[k] for k in _NAME_KEYS if _short(v.get(k))), None)
    return None


def _short(v: Any) -> bool:
    return isinstance(v, str) and 0 < len(v) <= SHORT_VALUE and "\n" not in v


def _text_chars(v: Any) -> int:
    if isinstance(v, str):
        return len(v)
    if isinstance(v, dict):
        return sum(_text_chars(x) for k, x in v.items() if str(k).lower() not in _REASONING)
    if isinstance(v, list):
        return sum(_text_chars(x) for x in v)
    return 0


def output_fields(value: Any, name: str = "", depth: int = 2) -> tuple[dict[str, tuple[str, ...]], int]:
    """The short categorical values of an LLM output and the length of its free text, for ``LLMCall``.

    A one-line string of at most ``SHORT_VALUE`` characters, a boolean, or a list of such strings (or of
    objects with a short ``name``, ``id``, ``api_name``, ``tool`` or ``label``) is a categorical value, keyed
    by its dotted field path (``name`` for a bare value); longer text counts as free text. Objects are read
    ``depth`` levels deep; numbers and reasoning fields (``thoughts``, ``reasoning``, ...) are neither.
    """
    key = name or "output"
    if isinstance(value, bool):
        return {key: (str(value).lower(),)}, 0
    if isinstance(value, str):
        return ({key: (value,)}, 0) if _short(value) else ({}, len(value))
    if isinstance(value, list):
        names = [_item_name(v) for v in value]
        if all(n is not None for n in names):
            return {key: tuple(n for n in names if n is not None)}, 0
        return {}, _text_chars(value)
    if not isinstance(value, dict):
        return {}, 0
    if depth <= 0:
        return {}, _text_chars(value)
    fields: dict[str, tuple[str, ...]] = {}
    text = 0
    for k, v in value.items():
        if str(k).lower() in _REASONING:
            continue
        f, t = output_fields(v, f"{name}.{k}" if name else str(k), depth - 1)
        fields.update(f)
        text += t
    return fields, text


# --------------------------------------------------------------------------- JSON utilities
def get_json_schema(data: Any, path: str = "") -> Any:
    """Schema of a JSON value as the TabSchema prompts show it: nested ``{"type", "properties"}`` dicts,
    with each list described by its first element."""
    if isinstance(data, dict):
        schema = {}
        for key, value in data.items():
            current = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                schema[key] = {"type": "dict", "properties": get_json_schema(value, current)}
            elif isinstance(value, list):
                if value:
                    schema[key] = {"type": "list", "items": get_json_schema(value[0], f"{current}[0]")}
                else:
                    schema[key] = {"type": "list", "items": "unknown (empty array)"}
            else:
                schema[key] = {"type": type(value).__name__}
        return schema
    if isinstance(data, list):
        return get_json_schema(data[0], f"{path}[0]") if data else "unknown (empty array)"
    return {"type": type(data).__name__}


def field_paths(data: Any, prefix: tuple = ()) -> set[tuple]:
    """Every key path that occurs in a JSON value; list positions are written ``"[]"``."""
    out = {prefix} if prefix else set()
    if isinstance(data, dict):
        for k, v in data.items():
            out |= field_paths(v, (*prefix, str(k)))
    elif isinstance(data, list):
        for v in data:
            out |= field_paths(v, (*prefix, "[]"))
    return out


def union_paths(values: Iterable[Any]) -> set[tuple]:
    """Union of ``field_paths`` over the onboarding prefixes: the fields the agent's schema can contain."""
    out: set[tuple] = set()
    for v in values:
        out |= field_paths(v)
    return out


def flatten_json(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts with ``_``-joined keys (the helper the CodeExtractor prompt prescribes)."""
    flat: dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            flat.update(flatten_json(value, f"{prefix}{key}_"))
        else:
            flat[f"{prefix}{key}"] = value
    return flat
