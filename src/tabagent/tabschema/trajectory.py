"""Trajectories, decision points and the JSON utilities TabSchema's prompts are built on (paper App. D).

TabSchema works on an agent's *native* logs: the extraction code it writes reads the agent's own logging
schema, which only has to be stable across the onboarding pool (Sec. 3.1). A trajectory is a JSON object with
a list of steps; each step names the component that produced it. A *decision point* is one occurrence of the
component being replaced (for example ``ShortlisterAgent``) inside a successful trajectory.

Features use only what precedes the decision (App. D.2, FeatureAnalyzer and CodeExtractor prompts), and this
is enforced structurally: the ``prefix`` a decision point exposes is a copy of the trajectory truncated
before the decision step, holding only whitelisted task fields (``task_keys``; never outcome fields such as
scores, evaluator results or final answers). Extraction code never receives the decision step, later steps,
or outcomes.
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
    keep: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Copy of ``raw`` with steps ``[:cut]`` only, top-level outcome keys removed (and, with ``keep``, every
    top-level key not in it), outcome steps dropped. A missing or malformed steps value counts as no steps."""
    allowed = None if keep is None else set(keep)
    out = {
        k: copy.deepcopy(v)
        for k, v in raw.items()
        if k != steps_key and k not in set(outcome_keys) and (allowed is None or k in allowed)
    }
    drop = set(outcome_steps)
    out[steps_key] = [
        copy.deepcopy(s)
        for s in steps_of(raw, steps_key)[:cut]
        if not (isinstance(s, dict) and s.get(step_name_key) in drop)
    ]
    return out


def steps_of(raw: dict[str, Any], steps_key: str) -> list[Any]:
    """The steps list of a raw trajectory; a missing or non-list value counts as no steps."""
    steps = raw.get(steps_key)
    return steps if isinstance(steps, list) else []


#: Top-level fields a decision-point prefix keeps by default: what describes the task before the agent acts.
TASK_KEYS = ("instruction", "task", "query", "question", "goal", "intent", "domain")
_TRUE = {"true", "yes", "pass", "passed", "success", "succeeded", "solved", "resolved"}


def positive_outcome(value: Any) -> bool:
    """Whether an outcome value means success: ``True``, a positive number (or numeric string), or a word
    such as ``pass``/``solved``; anything else (``False``, 0, ``"n/a"``, an object) means failure."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        try:
            return float(value) > 0
        except ValueError:
            return value.strip().lower() in _TRUE
    return False


class GenericJsonAdapter:
    """Reads one JSON trajectory per file: ``{..., "steps": [{"name": <component>, ...}, ...], <outcome>}``.

    A run succeeds by its first outcome field among ``success``, ``reward`` and ``score`` (see
    :func:`positive_outcome`); runs without one count as successful and are reported in ``warnings``. Pass
    ``success`` (a field name or a function of the run) to decide otherwise. Prefixes keep only the
    top-level fields in ``task_keys`` (``None`` keeps every field but the outcome keys); the fields left out
    are reported in ``warnings``. The task id (``group_key``) is never in a prefix: it is the
    cross-validation group, not a feature.

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
        success: str | Callable[[dict], bool] | None = None,
        group_key: str | None = "task_id",
        outcome_keys: Iterable[str] = ("score", "eval", "evaluation", "reward", "success", "outcome"),
        outcome_steps: Iterable[str] = ("EvaluationResult",),
        candidates_fn: Callable[[DecisionPoint], list[Candidate]] | None = None,
        labels_fn: Callable[[DecisionPoint, dict], set[str]] | None = None,
        task_keys: Iterable[str] | None = TASK_KEYS,
    ):
        self.steps_key = steps_key
        self.step_name_key = step_name_key
        extra = [success] if isinstance(success, str) else []
        self.success = success if callable(success) else self._outcome
        self._success_key = success if isinstance(success, str) else None
        self.group_key = group_key
        self.outcome_keys = tuple(dict.fromkeys([*outcome_keys, *extra]))
        self.outcome_steps = tuple(outcome_steps)
        self.candidates_fn = candidates_fn
        self.labels_fn = labels_fn
        self._hidden = (*self.outcome_keys, *([group_key] if group_key else []))  # never in a prefix
        self.task_keys = (
            None if task_keys is None else tuple(k for k in dict.fromkeys(task_keys) if k not in self._hidden)
        )
        self._raw: dict[str, dict] = {}  # full trajectories, for labels (post-decision information)
        self._no_outcome = self._loaded = 0
        self._left_out: set[str] = set()  # top-level fields kept out of prefixes by ``task_keys``

    def _outcome(self, raw: dict) -> bool:
        keys = (self._success_key,) if self._success_key else ("success", "reward", "score")
        for k in keys:
            if raw.get(k) is not None:
                return positive_outcome(raw[k])
        self._no_outcome += 1
        return True

    @property
    def warnings(self) -> list[str]:
        """Data notes for the onboarding report: runs without an outcome, fields kept out of prefixes."""
        out = []
        if self._no_outcome:
            keys = self._success_key or "success/reward/score"
            out.append(
                f"{self._no_outcome} of {self._loaded} runs have no outcome field ({keys}); they were "
                "treated as successful"
            )
        if self._left_out:
            out.append(
                "top-level fields not copied into decision-point prefixes (not in task_keys): "
                f"{', '.join(sorted(self._left_out))}; pass task_keys= to keep fields known before the "
                "decision"
            )
        return out

    def load(self, path: str | Path) -> list[Trajectory]:
        """Trajectories from one JSON file or every ``*.json`` below a directory (recursive; hidden files,
        ``results.json`` and ``metadata.json`` run summaries skipped). Files that are not valid JSON and JSON
        without the steps key are skipped. A trajectory's id is its path below the directory."""
        p = Path(path)
        if p.is_dir():
            files = sorted(
                f
                for f in p.rglob("*.json")
                if f.is_file() and not any(x.startswith(".") for x in f.relative_to(p).parts)
            )
        else:
            files = [p]
        out = []
        self._no_outcome = self._loaded = 0
        for f in files:
            if f.name in ("results.json", "metadata.json"):  # run summaries, not trajectories
                continue
            try:
                raw = json.loads(f.read_text())
            except ValueError:
                continue
            if not isinstance(raw, dict) or self.steps_key not in raw:
                continue
            tid = f.relative_to(p).with_suffix("").as_posix() if p.is_dir() else f.stem
            group = str(raw.get(self.group_key, tid)) if self.group_key else tid
            self._loaded += 1
            out.append(Trajectory(tid, raw, bool(self.success(raw)), group, f.name))
        return out

    def _prefix(self, raw: dict[str, Any], cut: int) -> dict[str, Any]:
        if self.task_keys is not None:
            self._left_out |= {
                str(k)
                for k in raw
                if k not in self.task_keys and k != self.steps_key and k not in self._hidden
            }
        return redact(
            raw, self.steps_key, cut, self._hidden, self.outcome_steps, self.step_name_key, self.task_keys
        )

    def decision_points(self, traj: Trajectory, target: str) -> list[DecisionPoint]:
        """Every step named ``target``, with its redacted prefix (see :func:`redact`). Steps that are not
        objects are kept in prefixes but are never decisions."""
        self._raw[traj.id] = traj.raw
        steps = steps_of(traj.raw, self.steps_key)
        return [
            DecisionPoint(
                traj.id,
                i,
                self._prefix(traj.raw, i),
                copy.deepcopy(step),
                traj.group,
                traj.source,
            )
            for i, step in enumerate(steps)
            if isinstance(step, dict) and step.get(self.step_name_key) == target
        ]

    def live_prefix(self, raw: dict) -> dict[str, Any]:
        """The prefix of a live decision: the trajectory so far, redacted exactly as in training."""
        return self._prefix(raw, len(steps_of(raw, self.steps_key)))

    def components(self, trajs: list[Trajectory]) -> dict[str, int]:
        """Step name -> number of successful trajectories containing it (the replaceable components)."""
        counts: dict[str, int] = {}
        drop = set(self.outcome_steps)
        for t in trajs:
            if not t.success:
                continue
            steps = steps_of(t.raw, self.steps_key)
            names = {
                str(s[self.step_name_key]) for s in steps if isinstance(s, dict) and s.get(self.step_name_key)
            }
            names -= drop
            for n in names:
                counts[n] = counts.get(n, 0) + 1
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
            for s in steps_of(traj.raw, self.steps_key)
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


def merge_examples(values: list[Any]) -> Any:
    """Deep-merge JSON values so one example shows every key seen in any of them, in every item of every list
    (for the schema prompts); a scalar keeps its first value."""
    if not values:
        return {}
    if all(isinstance(v, dict) for v in values):
        keys = list(dict.fromkeys(k for v in values for k in v))
        return {k: merge_examples([v[k] for v in values if k in v]) for k in keys}
    if all(isinstance(v, list) for v in values):
        items = [x for v in values for x in v]
        return [merge_examples(items)] if items else []
    return copy.deepcopy(values[0])


def union_schema(values: list[Any]) -> Any:
    """``get_json_schema`` of the merged ``values``: every field that occurs in any of them, at any list
    position (``get_json_schema`` alone describes a list by its first item)."""
    return get_json_schema(merge_examples(values))


def clip_strings(value: Any, max_chars: int) -> Any:
    """A copy of a JSON value with every string longer than ``max_chars`` cut, noting how much was cut."""
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return f"{value[:max_chars]} ...[{len(value) - max_chars} more characters]"
    if isinstance(value, dict):
        return {k: clip_strings(v, max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [clip_strings(v, max_chars) for v in value]
    return value


def clip_for_prompt(value: Any, budget: int = 400_000, floor: int = 200) -> Any:
    """``value`` itself when its JSON fits ``budget`` characters; otherwise a copy with long strings clipped,
    the cap halved from 20,000 characters until the JSON fits or the cap reaches ``floor``."""
    if len(json.dumps(value, default=str, ensure_ascii=False)) <= budget:
        return value
    cap = 20_000
    while True:
        out = clip_strings(value, cap)
        if cap <= floor or len(json.dumps(out, default=str, ensure_ascii=False)) <= budget:
            return out
        cap //= 2


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
