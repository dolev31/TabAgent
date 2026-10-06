"""Trajectories, decision points and the JSON utilities TabSchema's prompts are built on (paper App. D).

TabSchema works on an agent's *native* logs: the extraction code it writes reads the agent's own logging
schema, which only has to be stable across the onboarding pool (Sec. 3.1). A trajectory is a JSON object with
a list of steps; each step names the component that produced it. A *decision point* is one occurrence of the
component being replaced (for example ``ShortlisterAgent``) inside a successful trajectory.

Features use only what precedes the decision (App. D.2, FeatureAnalyzer and CodeExtractor prompts), and this
is enforced structurally: the ``prefix`` a decision point exposes is a copy of the trajectory truncated
before the decision step, holding only whitelisted task fields (``task_keys``; never outcome fields such as
scores, evaluator results or final answers). Extraction code never receives the decision step, later steps,
or outcomes. The per-candidate function of an extractor receives the prefix and one candidate as the
decision offers it (``candidate_dict``: id, name, description, argument hints and its catalog entry), never
its label.
"""

from __future__ import annotations

import bisect
import copy
import json
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Candidate:
    """One legal choice at a decision: a tool, skill, API, application, model or arm.

    ``spec`` is the candidate's catalog entry as the decision offers it (a tool's JSON schema, a CUGA API's
    method, path, parameters and response schemas), as JSON text (:func:`spec_text`), so a candidate stays
    hashable; "" when the catalog has none."""

    id: str
    name: str = ""
    description: str = ""
    arg_hints: str = ""
    spec: str = ""


#: The optional second function of a compiled extractor: the features of one candidate of the decision, in
#: the context of the prefix (``extract_candidate_features(trajectory_data, candidate)``), one row of a
#: ranking or multi-label decision table each.
CANDIDATE_FUNCTION = "extract_candidate_features"
#: The keys of the candidate dict that function receives: the candidate as the decision offers it, at training
#: and at serving alike. It never holds whether the candidate was chosen.
CANDIDATE_KEYS = ("id", "name", "description", "args", "spec")


def spec_text(entry: Any, volatile: Iterable[str] = ()) -> str:
    """A catalog entry as ``Candidate.spec``: canonical JSON (sorted keys, no spaces); a string is read as
    JSON first. ``volatile`` names top-level keys of the entry that change from run to run without changing
    the candidate (CUGA writes a ``canary_string`` into each API entry), which are left out. "" for no entry
    (None, "", or text that is not JSON)."""
    if isinstance(entry, str):
        if not entry.strip():
            return ""
        try:
            entry = json.loads(entry)
        except ValueError:
            return ""
    if entry is None:
        return ""
    drop = set(volatile)
    if drop and isinstance(entry, dict):
        entry = {k: v for k, v in entry.items() if k not in drop}
    return json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def candidate_dict(c: Candidate) -> dict[str, Any]:
    """The candidate as ``extract_candidate_features`` receives it: its id, name (the id when it has none),
    description and argument hints, the values of its decision-table row, and its catalog entry (``spec``,
    a dict, or None when the catalog has none); see ``CANDIDATE_KEYS``. It is built from these fields alone,
    so it cannot hold a label. ``decide`` and a served head read it back to the same candidate dict."""
    return {
        "id": c.id,
        "name": c.name or c.id,
        "description": c.description,
        "args": c.arg_hints,
        "spec": json.loads(c.spec) if c.spec else None,
    }


class _Lazy:
    """The raw log of a lazy trajectory: a function that reads it (see :meth:`Trajectory.lazy`)."""

    __slots__ = ("load",)

    def __init__(self, load: Callable[[], dict[str, Any]]):
        self.load = load


class _RawField:
    """The ``raw`` field of :class:`Trajectory`: a dict held in memory, or read through a loader every time it
    is accessed (a lazy trajectory). A data descriptor without a default, so ``raw`` stays a required
    dataclass field (``dataclasses.fields``, ``replace`` and ``asdict`` keep working)."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.slot = f"_{name}"

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        if obj is None:
            raise AttributeError("raw")  # no class-level default: a required field
        value = obj.__dict__[self.slot]
        return value.load() if isinstance(value, _Lazy) else value

    def __set__(self, obj: Any, value: Any) -> None:
        obj.__dict__[self.slot] = value  # the frozen dataclass sets its fields once, in __init__


@dataclass(frozen=True, repr=False)
class Trajectory:
    """One agent run: its raw log, whether it succeeded, its task (group) and source file.

    ``Trajectory(id, raw, success, group, source)`` holds ``raw`` in memory. Adapters reading a large input
    make :meth:`lazy` trajectories instead, which keep only where the run is: each access to ``raw`` reads the
    run again from its file, checks that its bytes have not changed, and returns a fresh copy (a few parsed
    runs are kept per load), so an onboarding holds the decision points' prefixes, not every log. ``raw`` is
    the same value either way. Trajectories are immutable dataclasses; ``dataclasses.replace`` on a lazy one
    returns an in-memory one (:meth:`with_raw` keeps it lazy).
    """

    id: str
    raw: dict[str, Any] = _RawField()  # type: ignore[assignment]
    success: bool = True
    group: str | None = None  # task id; decision points of one task never straddle CV folds
    source: str = ""  # file name, for provenance (the analyzer prompt's ``trajectory_file``)

    @classmethod
    def lazy(
        cls,
        id: str,
        load: Callable[[], dict[str, Any]],
        success: bool = True,
        group: str | None = None,
        source: str = "",
    ) -> Trajectory:
        """A trajectory whose raw log is ``load()``, called every time ``raw`` is read (``load`` must return
        an equal value every time, or raise)."""
        return cls(id, _Lazy(load), success, group, source)  # type: ignore[arg-type]

    @property
    def is_lazy(self) -> bool:
        """Whether ``raw`` is read from the run's file when it is needed."""
        return isinstance(self.__dict__["_raw"], _Lazy)

    def with_raw(self, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> Trajectory:
        """The same run with raw log ``fn(raw)`` (a redaction, say). ``fn`` gets a copy, so it may edit it in
        place; a lazy trajectory stays lazy and applies ``fn`` each time its log is read, so ``fn`` must be
        deterministic."""
        held = self.__dict__["_raw"]
        if not isinstance(held, _Lazy):
            return Trajectory(self.id, fn(json_copy(held)), self.success, self.group, self.source)
        load = held.load
        return Trajectory.lazy(self.id, lambda: fn(load()), self.success, self.group, self.source)

    def __repr__(self) -> str:
        return (
            f"Trajectory(id={self.id!r}, success={self.success!r}, group={self.group!r}, "
            f"source={self.source!r}{', lazy' if self.is_lazy else ''})"
        )

    def __reduce__(self):  # pickled (and copied) with its raw log, in memory
        return (Trajectory, (self.id, self.raw, self.success, self.group, self.source))


@dataclass(frozen=True)
class DecisionPoint:
    """One occurrence of the replaced component: the redacted prefix before it and the step itself.

    The prefixes of one run's decision points share the step objects they have in common (a prefix is a new
    list of the steps before its decision, not a deep copy per decision point), so that a run with many
    decisions does not take memory quadratic in its length: treat a prefix as read-only, and copy it before
    changing it (``json.loads(json.dumps(prefix))``).
    """

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


_ATOMIC = (str, int, float, bool, type(None))


def json_copy(value: Any) -> Any:
    """A copy of a JSON value: new dicts and lists, the same (immutable) strings and numbers. Faster than
    ``copy.deepcopy`` and equal to it on JSON; other objects are deep-copied."""
    if isinstance(value, dict):
        return {k: v if isinstance(v, _ATOMIC) else json_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [v if isinstance(v, _ATOMIC) else json_copy(v) for v in value]
    if isinstance(value, _ATOMIC):
        return value
    if isinstance(value, tuple):
        return tuple(json_copy(v) for v in value)
    return copy.deepcopy(value)


def step_name(step: Any, key: str = "name") -> str | None:
    """The name of a step: its ``key`` field when that is a non-empty string, else None (a step that is not an
    object, or whose name is missing, empty or not a string, is never a decision)."""
    if not isinstance(step, dict):
        return None
    name = step.get(key)
    return name if isinstance(name, str) and name else None


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
    return _Prefixes(raw, steps_key, outcome_keys, outcome_steps, step_name_key, keep).before([cut])[0]


class _Prefixes:
    """The redacted prefixes of one run (see :func:`redact`), sharing their copied fields and steps."""

    def __init__(
        self,
        raw: dict[str, Any],
        steps_key: str,
        outcome_keys: Iterable[str],
        outcome_steps: Iterable[str],
        step_name_key: str,
        keep: Iterable[str] | None,
    ):
        hidden, allowed = set(outcome_keys), None if keep is None else set(keep)
        self.head = {
            k: json_copy(v)
            for k, v in raw.items()
            if k != steps_key and k not in hidden and (allowed is None or k in allowed)
        }
        self.raw, self.steps_key = raw, steps_key
        self.drop, self.name_key = set(outcome_steps), step_name_key

    def before(self, cuts: list[int]) -> list[dict[str, Any]]:
        steps = steps_of(self.raw, self.steps_key)
        end = max(cuts, default=0)
        kept, at = [], []  # copies of the steps before the last cut (outcome steps dropped), their indices
        for j, s in enumerate(steps[: max(end, 0)]):
            if step_name(s, self.name_key) in self.drop:
                continue
            kept.append(json_copy(s))
            at.append(j)
        return [
            {**self.head, self.steps_key: kept[: bisect.bisect_left(at, c) if c >= 0 else 0]} for c in cuts
        ]


def steps_of(raw: Any, steps_key: str) -> list[Any]:
    """The steps list of a raw trajectory; a missing or non-list value counts as no steps."""
    steps = raw.get(steps_key) if isinstance(raw, dict) else None
    return steps if isinstance(steps, list) else []


#: Top-level fields a decision-point prefix keeps by default: what describes the task before the agent acts.
TASK_KEYS = ("instruction", "task", "query", "question", "goal", "intent", "domain")
#: Outcome words read as success and as failure (any other word is an unreadable outcome, read as failure).
TRUE_WORDS = frozenset(
    {"true", "yes", "y", "pass", "passed", "success", "succeeded", "successful", "solved", "resolved", "ok"}
)
FALSE_WORDS = frozenset(
    {
        "false", "no", "n", "fail", "failed", "failure", "error", "errored", "unsolved", "unresolved",
        "incorrect", "wrong", "timeout", "none", "null",
    }
)  # fmt: skip


def read_outcome(value: Any) -> bool | None:
    """An outcome value as success (True), failure (False), or unreadable (None).

    ``True``/``False``; a finite number (success when positive); a numeric string the same way; a word
    (:data:`TRUE_WORDS`, :data:`FALSE_WORDS`, any case). Anything else (another word, an empty string, NaN or
    an infinity, an object, a list) is unreadable: adapters count it as a failure and report it, and
    ``success=`` (a field name or a function of the run) reads it instead.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0 if math.isfinite(value) else None
    if isinstance(value, str):
        word = value.strip().lower()
        if word in TRUE_WORDS:
            return True
        if word in FALSE_WORDS:
            return False
        try:
            number = float(word)
        except ValueError:
            return None
        return number > 0 if math.isfinite(number) else None
    return None


def positive_outcome(value: Any) -> bool:
    """Whether an outcome value means success (see :func:`read_outcome`; an unreadable value is a failure)."""
    return read_outcome(value) is True


def _example(value: Any) -> str:
    text = json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= 40 else text[:37] + "..."


class GenericJsonAdapter:
    """Reads JSON trajectories with named steps, ``{..., "steps": [{"name": <component>, ...}, ...],
    <outcome>}``: one per ``.json`` file, a list of them in one file, or one per line of a ``.jsonl`` file.

    A run succeeds by its first outcome field among ``success``, ``reward`` and ``score`` (see
    :func:`read_outcome`); runs without one count as successful and are reported in ``warnings``, and so are
    unreadable outcomes, which count as failures. Pass ``success`` (a field name or a function of the run) to
    decide otherwise. Prefixes keep only the top-level fields in ``task_keys`` (``None`` keeps every field but
    the outcome keys); the fields left out are reported in ``warnings``. The task id (``group_key``) is never
    in a prefix: it is the cross-validation group, not a feature.

    Candidates and labels are benchmark-specific; pass ``candidates_fn(dp)`` and ``labels_fn(dp)`` (for
    example, the catalog of tools and the tools the successful run actually called), or use a dedicated
    adapter (``tabagent.adapters``).

    ``load`` reads the files as described in ``tabagent.adapters.logfiles`` (encodings, damaged JSON, JSON
    Lines, linked folders) and reports in ``warnings`` every file, line or record it skipped or repaired; it
    refuses a folder that also holds runs of another log format. A step that is not an object, or has no
    string name, stays in the prefixes but is never a decision.
    """

    #: What ``tabagent.scan`` should tell the reader about how these logs record LLM calls.
    scan_notes: tuple[str, ...] = ()
    #: The log format, as messages name it.
    format_name = "JSON runs with named steps"

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
        self._no_outcome = self._loaded = 0
        self._bad_outcomes: list[str] = []  # runs whose outcome value is unreadable, and the first values
        self._n_bad_outcomes = 0
        self._left_out: set[str] = set()  # top-level fields kept out of prefixes by ``task_keys``
        self.notes: Any = None  # the ``LoadNotes`` of the last ``load``
        self._reset()

    # ------------------------------------------------------------------ outcomes
    #: How a run without an outcome field counts, and the fields read (for ``warnings``).
    _missing_outcome = "successful"

    @property
    def _outcome_fields(self) -> str:
        return self._success_key or "success/reward/score"

    def _outcome(self, raw: dict) -> bool:
        keys = (self._success_key,) if self._success_key else ("success", "reward", "score")
        for k in keys:
            if raw.get(k) is not None:
                return self._read_outcome(k, raw[k])
        self._no_outcome += 1
        return True

    def _read_outcome(self, key: str, value: Any) -> bool:
        """``value`` read as an outcome; an unreadable value counts as a failure and is reported."""
        ok = read_outcome(value)
        if ok is None:
            self._n_bad_outcomes += 1
            if len(self._bad_outcomes) < 3:
                self._bad_outcomes.append(f"{key}={_example(value)}")
        return ok is True

    def _run_success(self, raw: dict, where: str) -> bool:
        """``success(raw)``, with the place of a run on which a ``success=`` function fails. A function of
        the user's gets a copy of the run, so that an edit in place never changes what was loaded."""
        own = getattr(self.success, "__self__", None) is self  # this adapter's own outcome reader
        try:
            return bool(self.success(raw if own else json_copy(raw)))
        except Exception as e:  # noqa: BLE001 - the user's function: say where it failed
            raise ValueError(
                f"{where}: the success function raised {type(e).__name__}: {e}; make it return True or False "
                "for every run (or pass success= with the name of the outcome field)"
            ) from e

    @property
    def warnings(self) -> list[str]:
        """Data notes for the onboarding report: files and records skipped or repaired, runs without an
        outcome or with an unreadable one, fields kept out of prefixes."""
        out = [] if self.notes is None else self.notes.warnings()
        if self._no_outcome:
            out.append(
                f"{self._no_outcome} of {self._loaded} runs have no outcome field ({self._outcome_fields}); "
                f"they were treated as {self._missing_outcome}"
            )
        if self._n_bad_outcomes:
            out.append(
                f"{self._n_bad_outcomes} of {self._loaded} runs have an outcome that is not a boolean, a "
                f"finite number or a pass/fail word (first: {', '.join(self._bad_outcomes)}); they were "
                "treated as failed (pass success= to read them)"
            )
        if self._left_out:
            out.append(
                "top-level fields not copied into decision-point prefixes (not in task_keys): "
                f"{', '.join(sorted(self._left_out))}; pass task_keys= to keep fields known before the "
                "decision"
            )
        return out

    # ------------------------------------------------------------------ loading
    def _is_run(self, record: Any) -> bool:
        """Whether a JSON record is a run in this adapter's format."""
        return isinstance(record, dict) and self.steps_key in record

    def _group(self, raw: dict, rid: str, f: Path, single: bool) -> str:
        """The run's task (its cross-validation group): the ``group_key`` field, else the run id. ``single``:
        the run is the whole file ``f``."""
        return str(raw.get(self.group_key, rid)) if self.group_key else rid

    def _on_run(self, raw: dict, rid: str) -> None:
        """Called on every run that is loaded, with its id (the CUGA adapter collects its catalogs here)."""

    #: What ``load`` does with runs of another log format: ``"skip"`` them with a warning (an adapter given
    #: explicitly always does), or ``"error"`` (``detect_adapter(..., on_other_formats="error")``).
    on_other_formats = "skip"

    def load(self, path: str | Path, *, lazy: bool | None = None) -> list[Trajectory]:
        """Trajectories from one file or every ``.json``/``.jsonl`` file below a directory (recursive, linked
        folders followed; hidden files and the ``results.json``/``metadata.json`` run summaries skipped). A
        ``.json`` file holds one run or a list of runs, a ``.jsonl`` file one run per line. A trajectory's id
        is its path below the directory (``:<index>`` for a run in a list or on a line). Files, lines and
        records that cannot be read, records without the steps key and runs of another log format are skipped
        and reported in ``warnings`` (see ``on_other_formats``). ``lazy``: read runs again from their files
        when needed instead of holding them (default: from 256 MB of logs, ``logfiles.LAZY_INPUT_BYTES``)."""
        from tabagent.adapters import logfiles as lf

        root = Path(path)
        self.notes = notes = lf.LoadNotes()
        self._no_outcome = self._loaded = self._n_bad_outcomes = 0
        self._bad_outcomes = []
        self._left_out = set()
        self._reset()
        files = lf.iter_files(root)
        lazy = lf.use_lazy(files, lazy)
        cache = lf.RunCache()  # the parsed runs read last; it goes away with these trajectories
        ids = lf.RunIds()
        out: list[Trajectory] = []
        for f in files:
            if f.name in lf.SKIP_FILES:
                notes.add("summaries", lf.relative(root, f))
                continue
            base = lf.run_id(root, f)
            for doc in lf.read_docs(root, f, notes):
                many = isinstance(doc.data, list) and doc.line is None
                records = doc.data if many else [doc.data]
                for i, rec in enumerate(records):
                    where = f"{doc.where}[{i}]" if many else doc.where
                    if not self._is_run(rec):
                        _not_a_run(self, rec, where, notes)
                        continue
                    index = i if many else doc.line
                    rid = ids(base, f, index)
                    self._on_run(rec, rid)
                    ok = self._run_success(rec, where)
                    group = self._group(rec, rid, f, not many and doc.line is None)
                    load = None if many or not lazy else lf.lazy_loader(doc, cache=cache)
                    traj = (
                        Trajectory(rid, rec, ok, group, f.name)
                        if load is None
                        else Trajectory.lazy(rid, load, ok, group, f.name)
                    )
                    self._loaded += 1
                    self._names[rid] = (traj, self._step_names(rec))
                    out.append(traj)
        _refuse_mixed(self, root, notes, len(out), len(files))
        return out

    def _reset(self) -> None:
        """Forget the runs of an earlier ``load``."""
        self._raw: dict[str, Trajectory] = {}  # trajectory id -> its run, for labels (they read the full run)
        # trajectory id -> (the loaded trajectory, its step names), so components() need not read it again
        self._names: dict[str, tuple[Trajectory, frozenset[str]]] = {}

    def _step_names(self, raw: Any) -> frozenset[str]:
        names = {step_name(s, self.step_name_key) for s in steps_of(raw, self.steps_key)}
        return frozenset(names - set(self.outcome_steps) - {None})

    # ------------------------------------------------------------------ decision points
    def _prefixer(self, raw: dict[str, Any]) -> _Prefixes:
        if self.task_keys is not None:
            self._left_out |= {
                str(k)
                for k in raw
                if k not in self.task_keys and k != self.steps_key and k not in self._hidden
            }
        return _Prefixes(
            raw, self.steps_key, self._hidden, self.outcome_steps, self.step_name_key, self.task_keys
        )

    def _prefix(self, raw: dict[str, Any], cut: int) -> dict[str, Any]:
        return self._prefixer(raw).before([cut])[0]

    def decision_points(self, traj: Trajectory, target: str) -> list[DecisionPoint]:
        """Every step named ``target``, with its redacted prefix (see :func:`redact`; the prefixes of one run
        share their steps, see :class:`DecisionPoint`). Steps that are not objects, or have no string name,
        are kept in prefixes but are never decisions."""
        self._raw[traj.id] = traj
        raw = traj.raw
        steps = steps_of(raw, self.steps_key)
        at = [i for i, s in enumerate(steps) if step_name(s, self.step_name_key) == target]
        if not at:
            return []
        prefixes = self._prefixer(raw).before(at)
        return [
            DecisionPoint(traj.id, i, prefix, json_copy(steps[i]), traj.group, traj.source)
            for i, prefix in zip(at, prefixes, strict=True)
        ]

    def live_prefix(self, raw: dict) -> dict[str, Any]:
        """The prefix of a live decision: the trajectory so far, redacted exactly as in training."""
        if not isinstance(raw, dict):
            raise TypeError(
                f"a live run is the trajectory so far, a dict like {{{self.steps_key!r}: [...]}}, not "
                f"{type(raw).__name__}"
            )
        return self._prefix(raw, len(steps_of(raw, self.steps_key)))

    def components(self, trajs: list[Trajectory]) -> dict[str, int]:
        """Step name -> number of successful trajectories containing it (the replaceable components), most
        frequent first (ties by name)."""
        counts: dict[str, int] = {}
        for t in trajs:
            if not t.success:
                continue
            seen = self._names.get(t.id)
            for n in seen[1] if seen is not None and seen[0] is t else self._step_names(t.raw):
                counts[n] = counts.get(n, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def candidates(self, dp: DecisionPoint) -> list[Candidate]:
        """The legal candidates of a decision point (from ``candidates_fn``)."""
        if self.candidates_fn is None:
            raise NotImplementedError("pass candidates_fn= to GenericJsonAdapter or use a dedicated adapter")
        return self.candidates_fn(dp)

    def _run_of(self, dp: DecisionPoint) -> dict[str, Any]:
        """The full run of a decision point built by this adapter's ``decision_points``."""
        traj = self._raw.get(dp.trajectory_id)
        if traj is None:
            raise KeyError(
                f"labels need the full run of {dp.id}, and this adapter has not built decision points from "
                f"run {dp.trajectory_id!r}: call decision_points(trajectory, target) on it first"
            )
        return traj.raw

    def labels(self, dp: DecisionPoint) -> set[str]:
        """The chosen candidates, read from the *full* trajectory (e.g. the tools the successful run used).
        Available for decision points produced by this adapter's ``decision_points``."""
        if self.labels_fn is None:
            raise NotImplementedError("pass labels_fn= to GenericJsonAdapter or use a dedicated adapter")
        traj = self._raw.get(dp.trajectory_id)
        run = self._run_of(dp)
        return self.labels_fn(dp, run if traj is not None and traj.is_lazy else json_copy(run))  # a copy

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
            if step_name(s, self.step_name_key) not in (None, *drop)
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


def _not_a_run(adapter: Any, record: Any, where: str, notes: Any) -> None:
    """Note a record that is not a run of ``adapter``'s format: a run of another log format (skipped with a
    warning, an error with ``on_other_formats="error"``; see :func:`_refuse_mixed`), or not a run at all."""
    from tabagent.adapters.detect import compatible, detect_format, format_name

    cls = detect_format(record)
    if cls is not None and not compatible(type(adapter), cls):
        name = f"{format_name(cls)}, {cls.__name__}"
        strict = getattr(adapter, "on_other_formats", "skip") == "error"
        notes.add(
            f"other_format:{name}",
            where,
            "records",
            level="error" if strict else "warning",
            found=f"runs of another log format ({name}), " + ("not read" if strict else "skipped"),
            fix="" if strict else "point at a folder or file with only that format to onboard them",
        )
    else:
        notes.add("not_a_run", where, "records")


def _refuse_mixed(adapter: Any, root: Path, notes: Any, n_runs: int, n_files: int = 0) -> None:
    """Raise ``MixedFormatsError`` when ``load`` found runs of another log format, and ``ValueError`` when it
    read files but found no run in them (saying what it skipped)."""
    from tabagent.adapters.logfiles import MixedFormatsError

    others = [i for code, i in notes.issues.items() if code.startswith("other_format:")]
    if not others or getattr(adapter, "on_other_formats", "skip") != "error":
        if n_files and not n_runs:
            found = "; ".join(i.message(fix=False) for i in list(notes.issues.values())[:3])
            files = f"{n_files:,} file{'s' if n_files != 1 else ''}"
            raise ValueError(
                f"no runs found in {root} ({files} read by {type(adapter).__name__}, which reads "
                f"{type(adapter).format_name}): {found or 'no file holds a run'}. See the supported formats "
                "in the docs (Log formats), or run tabagent check-logs on the folder"
            )
        return
    listed = "; ".join(i.message(fix=False).replace(", not read:", ":") for i in others)
    raise MixedFormatsError(
        f"{root} holds runs of more than one log format: {n_runs:,} runs of {type(adapter).format_name} "
        f"({type(adapter).__name__}), and {listed}. One onboarding reads one format: point at a folder or "
        'file with one format, pass the adapter of the format to read, or use on_other_formats="skip" (the '
        "default; CLI --on-other-formats skip) to read this format and skip the others with a warning"
    )


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
