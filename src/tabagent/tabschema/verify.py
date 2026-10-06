"""Verification of a compiled extractor's values: the release's addition to TabSchema, not in the paper.

The paper's CodeValidator (App. D.1) and the deterministic checks (``checks``) review the extraction *code*.
Nothing in the paper checks the *values* the code computes. Before a compiled extractor ships, the compile
step runs it, sandboxed as everywhere else, and checks four properties (:func:`verify_extractor`):

1. **Determinism.** The code runs once more on every onboarding decision point, in fresh interpreters and
   in a shuffled order, and must reproduce the compile step's features. A column that differs cannot be
   reproduced at serving time and is dropped. Typical causes are a set joined in iteration order (string
   hashing is seeded per process), a counter kept across calls, the clock and randomness. Reading the clock
   or calling ``hash``/``id`` is also refused statically (``checks``).
2. **Outcome invariance.** On a sample of decision points, a prefix gets the adapter's outcome fields
   (``reward``, ``success``, ...), once with success values and once with failure values, and every feature
   must stay as it was. A run's outcome exists only after the run ends, so no decision-time prefix holds one.
   A column that changes reads data that would leak the label wherever a prefix carried it, and is dropped.
3. **Future invariance** (:class:`FutureCheck`), a check of the *adapter*. For each sampled decision point the
   raw run is changed after the decision and the prefix is rebuilt by a fresh copy of the adapter: every step
   after the decision, and the decision step itself, is replaced by a perturbed copy (every string, including
   tool names, arguments and ids, and every number; only the step-name key and the ``role``/``type`` format
   keys are kept, so the decision is still found), the outcome fields are changed, and, separately, the run
   is cut right after the decision step. The rebuilt prefix must equal the original. A difference means the
   adapter puts post-decision data into prefixes, for example by cutting a step too late, and onboarding
   **fails closed** with an error naming the adapter and the first differing path. The built-in adapters never
   trigger it. Steps are changed in the *run*, never appended to the prefix: the extractor's contract is that
   every step it sees precedes the decision, so a prefix with appended steps looks like a longer history.
4. **Robustness.** The extractor runs on the shortest onboarding prefixes, on a prefix with no steps, and
   on prefixes with one field removed (a task field a live prefix may lack, or a step field that only some
   onboarding steps have). The report says which features raise or go missing. Nothing is dropped for
   raising: raising is the extractor's contract, and a head defers a decision whose extraction fails.

The samples come from the coverage sampler (``sampling``): new tasks first, then uncovered sources, positions
in the run (first, middle, last decision), prefix lengths and labels, so recurring decisions are checked at
every position, not only at a run's first decision.

:func:`make_tests` records a small set of golden tests: snapshots of onboarding prefixes with the features the
extractor computes on them, plus the determinism and outcome-invariance checks. ``extractor_tests.json`` holds
them next to ``extract.py``, and a saved head carries the file. :func:`verify_head` (``tabagent verify-head
PATH``) checks the head's artifact and runs the tests again in the sandbox, for CI and deployment gates. The
snapshots are trajectory text: they come from the trajectories after ``redact=``, strings that look like
credentials, e-mail addresses or home-directory paths are masked (:func:`mask_secrets`), and each snapshot
fits ``GOLDEN_MAX_CHARS``. Even so, treat the file like the trajectories. Everything else these checks record
(examples of differing values, error lines, dropped reasons) is masked the same way.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tabagent.defaults import DEFAULT_SEED
from tabagent.table import LeakageError
from tabagent.tabschema.sampling import sample_decision_points
from tabagent.tabschema.sandbox import DockerSandbox, ExecResult, SubprocessSandbox
from tabagent.tabschema.trajectory import DecisionPoint, Trajectory, clip_strings

logger = logging.getLogger(__name__)

TESTS_FILE = "extractor_tests.json"
TESTS_FORMAT = 1
#: Decision points of the outcome, future and robustness checks (coverage-sampled).
DEFAULT_SAMPLES = 20
#: Golden tests recorded with an extractor.
DEFAULT_GOLDENS = 8
#: The robustness check's shortest onboarding prefixes and removed fields.
SHORTEST, MAX_FIELDS = 3, 6
#: A golden prefix: strings longer than this are clipped, and the whole snapshot fits this many characters.
GOLDEN_STRING_CHARS, GOLDEN_MAX_CHARS = 2_000, 20_000
#: The text appended to every string of a run's changed future.
FUTURE_MARK = " [changed after the decision]"
#: Keys that say what kind of raw step (or message) a step is: the changed future keeps their values, with the
#: adapter's step-name key, so the adapter still recognizes the decision. Everything else is perturbed.
FORMAT_KEYS = frozenset({"role", "type"})
#: Kept as well by the conservative fallback, for formats whose ids or names decide how a step is read.
STRUCTURAL_KEYS = frozenset(
    {"role", "type", "name", "id", "tool_call_id", "tool_name", "arguments", "requestor", "timestamp", "lc"}
)
_MISSING = object()
_SECRET_NAME = re.compile(
    r"(?i)(api[_-]?key|apikey|secret|token|passw(or)?d|passwd|pwd|access[_-]?key|private[_-]?key|credential"
    r"|authorization|auth[_-]?token|session[_-]?id|cookie)"
)
#: Key names that count tokens rather than hold one (``total_tokens``, ``max_tokens``, ``token_count``).
_COUNT_NAME = re.compile(r"(?i)tokens|token_?(count|usage|limit|budget)")
#: Strings masked wherever these checks record trajectory text: credentials, e-mail addresses, home paths.
_SECRETS = (
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
        "[masked key]",
    ),
    (re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}"), "[masked key]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "[masked key]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}"), "[masked key]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "[masked key]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "[masked key]"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), "[masked key]"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[masked key]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "[masked key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "[masked token]"),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/-]{16,}=*"), r"\1 [masked token]"),
    # key=value and "key": "value" pairs whose key names a credential
    (
        re.compile(
            r"(?i)(\b[\w.-]*(?:api[_-]?key|apikey|secret|token(?!s|_?count|_?usage|_?limit)|passw(?:or)?d|passwd|pwd|access[_-]?key"
            r"|private[_-]?key|credential)[\w.-]*[\"']?\s*[:=]\s*[\"']?)([^\s\"',;&}]{3,})"
        ),
        r"\1[masked]",
    ),
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@\"']+:[^/\s@\"']+@"), r"\1[masked]@"),  # URL userinfo
    (
        re.compile(
            r"(?i)([?&](?:api[_-]?key|key|token|access_token|secret|sig|signature|password|auth)=)[^&\s#\"']+"
        ),
        r"\1[masked]",
    ),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"), r"masked@\1"),  # domain kept
    (re.compile(r"(?<![\w.])/(?:Users|home)/[^/\s\"'<>]+"), "/home/[user]"),
    (re.compile(r"(?<![\w.])/root(?=/|\b)"), "/home/[user]"),
    (re.compile(r"\b[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"'<>]+"), r"C:\\Users\\[user]"),
    (re.compile(r"\\\\[^\\\s\"'<>]+\\+Users\\+[^\\\s\"'<>]+"), r"\\\\[host]\\Users\\[user]"),
    (re.compile(r"(?<![\w~/])~[A-Za-z_][\w.-]*(?=/|\b)"), "~[user]"),
)


# --------------------------------------------------------------------------- helpers
def canonical(value: Any) -> str:
    """The canonical JSON of a value (what the sandbox sees and returns)."""
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def json_copy(value: Any) -> Any:
    """A JSON round trip of ``value``: the copy the sandbox runs the extractor on."""
    return json.loads(json.dumps(value, default=str))


def same(a: Any, b: Any) -> bool:
    """Whether two feature values agree: numbers up to rounding, anything else as canonical JSON."""
    numbers = (int, float)
    if (
        isinstance(a, numbers)
        and isinstance(b, numbers)
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    ):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return canonical(a) == canonical(b)


def mask_secrets(value: Any, by_key: bool = True) -> Any:
    """``value`` with strings (dict keys included) that look like credentials, e-mail addresses or
    home-directory paths masked (an e-mail keeps its domain). With ``by_key`` (for trajectory data), every
    string under a key that names a credential (``api_key``, ``password``, ``token``, ...; not a token count
    such as ``total_tokens``) is masked too; numbers are never masked, and records keyed by feature names pass
    ``by_key=False``."""
    if isinstance(value, str):
        for pattern, repl in _SECRETS:
            value = pattern.sub(repl, value)
        return value
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = mask_secrets(k, by_key) if isinstance(k, str) else k
            secret = by_key and isinstance(k, str) and _SECRET_NAME.search(k) and not _COUNT_NAME.search(k)
            out[key] = "[masked]" if secret and isinstance(v, str) and v else mask_secrets(v, by_key)
        return out
    if isinstance(value, list):
        return [mask_secrets(v, by_key) for v in value]
    return value


def _show(value: Any, limit: int = 80) -> str:
    text = "missing" if value is _MISSING else repr(mask_secrets(value, by_key=False))
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _last_line(error: str | None) -> str:
    lines = (error or "").strip().splitlines()
    return mask_secrets(lines[-1][:300]) if lines else "unknown error"


def _names(keys: list[str], n: int = 4) -> str:
    return ", ".join(keys[:n]) + (", ..." if len(keys) > n else "")


def _get(features: dict | None, col: str) -> Any:
    return _MISSING if features is None or col not in features else features[col]


def outcome_value(key: str, success: bool) -> Any:
    """A plausible value of the outcome field ``key``: a number for rewards and scores, a boolean for success
    flags, an object for evaluator records, otherwise a sentence."""
    k = key.lower()
    if k in ("success", "passed", "resolved", "solved", "succeeded"):
        return success
    if any(w in k for w in ("info", "eval", "criteria", "scenario", "result")):
        return {
            "reward": 1.0 if success else 0.0,
            "passed": success,
            "note": "checks passed" if success else "checks failed",
        }
    if any(w in k for w in ("reward", "score")):
        return 1.0 if success else 0.0
    return "success: the task was solved" if success else "failure: the task was not solved"


def with_outcomes(prefix: dict, keys: Iterable[str], success: bool) -> dict:
    """A copy of ``prefix`` with every outcome field in ``keys`` set (see :func:`outcome_value`)."""
    out = json_copy(prefix)
    for k in keys:
        out[k] = outcome_value(k, success)
    return out


def first_difference(a: Any, b: Any, path: str = "prefix") -> str | None:
    """The first JSON path where ``a`` and ``b`` differ (None when they are equal)."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in [*a, *(k for k in b if k not in a)]:
            if k not in a or k not in b:
                return f"{path}.{k} ({'added' if k not in a else 'removed'})"
            d = first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None if len(a) == len(b) else f"{path} (length {len(a)} -> {len(b)})"
    return None if canonical(a) == canonical(b) else path


# --------------------------------------------------------------------------- the changed future
def perturb(value: Any, keep: frozenset[str], key: str = "") -> Any:
    """``value`` with every string (JSON inside strings included) marked with :data:`FUTURE_MARK` and every
    number shifted, except the values of the keys in ``keep``; booleans and nulls are kept."""
    if key in keep or value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip()[:1] in ("{", "["):
            try:
                inner = json.loads(value)
            except ValueError:
                inner = None
            if isinstance(inner, (dict, list)):
                return json.dumps(perturb(inner, keep))
        return value + FUTURE_MARK
    if isinstance(value, int):
        return value + 7
    if isinstance(value, float):
        return value + 0.5
    if isinstance(value, dict):
        return {k: perturb(v, keep, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [perturb(v, keep, key) for v in value]
    return value


def steps_key_of(raw: dict, adapter: Any = None) -> str | None:
    """The key of a raw run's step list: the adapter's ``steps_key``, else ``messages`` or ``steps``."""
    for key in (getattr(adapter, "steps_key", None), "messages", "steps"):
        if key and isinstance(raw.get(key), list):
            return key
    return None


def changed_futures(
    raw: dict,
    steps_key: str,
    cut: int,
    outcome_keys: Iterable[str],
    name_key: str | None = None,
    decision_input_keys: Iterable[str] = (),
) -> list[tuple[str, dict]]:
    """The changed runs of the future-invariance check, as (label, run): ``raw[steps_key][cut - 1]`` is the
    decision step. Two perturb the decision step and every later step, append perturbed copies of them and
    set the outcome fields to failure values (keeping the format keys, then all structural keys); one cuts
    the run right after the decision and drops its outcome fields. Everything before the decision is
    untouched, so a prefix rebuilt from any of them must equal the original. ``decision_input_keys`` are the
    decision step's keys written before the decision (its prompt, say), which an adapter may put in prefixes:
    the decision step keeps them."""
    steps = raw[steps_key]
    keys = [k for k in outcome_keys if k != steps_key]
    inputs = frozenset(decision_input_keys)
    out = []
    for label, keep in (
        ("future changed", FORMAT_KEYS),
        ("future changed (structural keys kept)", STRUCTURAL_KEYS),
    ):
        keep = keep | ({name_key} if name_key else set())
        decision = perturb(copy.deepcopy(steps[cut - 1]), keep | inputs)
        after = [decision, *(perturb(copy.deepcopy(s), keep) for s in steps[cut:])]
        run = {
            **raw,
            steps_key: [
                *steps[: cut - 1],
                *after,
                *(perturb(copy.deepcopy(s), keep) for s in steps[cut - 1 :]),
            ],
        }
        run.update({k: outcome_value(k, False) for k in keys})
        out.append((label, run))
    cut_run = {k: v for k, v in raw.items() if k not in keys}
    cut_run[steps_key] = list(steps[:cut])
    out.append(("run cut after the decision", cut_run))
    return out


def _containers(value: Any, depth: int) -> Any:
    if depth <= 0:
        return value
    if isinstance(value, dict):
        return {k: _containers(v, depth - 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_containers(v, depth - 1) for v in value]
    if isinstance(value, set):
        return set(value)
    return value


def isolated_adapter(adapter: Any) -> Any:
    """A copy of ``adapter`` for rebuilding prefixes from changed runs, so that nothing it records while
    reading them (runs, catalogs, tool pools) reaches the original: a fresh adapter rebuilt from its
    description for the built-in adapters, else a shallow copy whose dicts, lists and sets are copied three
    levels deep (the objects in them are shared, never deep-copied)."""
    from tabagent.adapters import adapter_from_description, describe_adapter

    fresh = adapter_from_description(describe_adapter(adapter))
    if fresh is not None and type(fresh) is type(adapter):
        return fresh
    out = copy.copy(adapter)
    for k, v in vars(adapter).items():
        out.__dict__[k] = _containers(v, 3)
    return out


class FutureCheck:
    """The future-invariance check of an adapter on its onboarding runs (see the module docstring).

    ``check(dp)`` returns ``{"ok": n}`` (``n`` changed runs rebuilt the same prefix), ``{"skipped": reason}``
    or
    ``{"leak": {...}}`` with the variant and the first differing path. A run that cannot be read once is not
    read again."""

    def __init__(
        self, adapter: Any, trajectories: Iterable[Trajectory], target: str, outcome_keys: Iterable[str]
    ):
        self.source = adapter
        self.by_id = {t.id: t for t in trajectories}
        self.target = target
        self.outcome_keys = tuple(outcome_keys)
        self.name_key = getattr(adapter, "step_name_key", None)
        self.decision_input_keys = tuple(getattr(adapter, "decision_input_keys", ()) or ())
        self.adapter_name = type(adapter).__name__
        self._copy: Any = None
        self.failed: dict[str, str] = {}

    def _points(self, t: Trajectory, raw: dict) -> dict[int, DecisionPoint]:
        if self._copy is None:
            self._copy = isolated_adapter(self.source)
        changed = Trajectory(t.id, raw, t.success, t.group, t.source)
        return {d.step_index: d for d in self._copy.decision_points(changed, self.target)}

    def __call__(self, dp: DecisionPoint) -> dict[str, Any]:
        t = self.by_id.get(dp.trajectory_id)
        if t is None:
            return {"skipped": "its run is not among the onboarding runs"}
        if t.id in self.failed:
            return {"skipped": self.failed[t.id]}
        from tabagent.adapters.logfiles import LogChangedError

        try:
            return self._check(t, dp)
        except LogChangedError:  # a log changed under the check: say so, never report the point as skipped
            raise
        except Exception as e:  # an adapter that cannot read a run: not read again
            self.failed[t.id] = f"the adapter could not read the run ({type(e).__name__}: {e})"[:300]
            return {"skipped": self.failed[t.id]}

    def _check(self, t: Trajectory, dp: DecisionPoint) -> dict[str, Any]:
        base = self._points(t, t.raw).get(dp.step_index)
        if base is None or canonical(base.prefix) != canonical(dp.prefix):
            return {"skipped": "the adapter does not rebuild the same prefix from the unchanged run"}
        key = steps_key_of(t.raw, self.source)
        if key is None:
            variants = [
                (
                    "outcome fields changed",
                    {**t.raw, **{k: outcome_value(k, False) for k in self.outcome_keys}},
                )
            ]
        else:
            cut = self._cut(t, key, dp)
            if cut is None:
                return {"skipped": "the decision step could not be located in the run"}
            variants = changed_futures(
                t.raw, key, cut, self.outcome_keys, self.name_key, self.decision_input_keys
            )
        usable = 0
        for label, raw in variants:
            try:
                rebuilt = self._points(t, raw).get(dp.step_index)
            except Exception:  # noqa: BLE001 - a changed run the adapter cannot read: try the next one
                continue
            if rebuilt is None:
                continue
            usable += 1
            path = first_difference(dp.prefix, rebuilt.prefix)
            if path:
                return {"leak": {"decision_point": dp.id, "variant": label, "path": path}}
        if not usable:
            return {"skipped": "no changed run kept the decision"}
        return {"ok": usable}

    def _cut(self, t: Trajectory, key: str, dp: DecisionPoint) -> int | None:
        """The index right after the decision's raw step: ``step_index + 1`` when the raw step is the decision
        step, else the shortest run prefix in which the adapter finds the decision (binary search)."""
        steps = t.raw[key]
        if 0 <= dp.step_index < len(steps) and canonical(steps[dp.step_index]) == canonical(dp.target_step):
            return dp.step_index + 1

        def found(c: int) -> bool:
            return dp.step_index in self._points(t, {**t.raw, key: steps[:c]})

        lo, hi = 1, len(steps)
        if not hi or not found(hi):
            return None
        while lo < hi:
            mid = (lo + hi) // 2
            if found(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo


def check_adapter(
    adapter: Any,
    trajectories: Iterable[Trajectory],
    decision_points: list[DecisionPoint],
    target: str,
    *,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    label_of: Callable[[DecisionPoint], Iterable[Any]] | None = None,
) -> dict[str, Any]:
    """Run the future-invariance check of ``adapter`` (:class:`FutureCheck`) on ``samples`` coverage-sampled
    decision points; no LLM is involved. Raises :class:`PrefixLeakError` when a prefix changes with the run's
    future; returns the record otherwise (``checked``, ``skipped``, ``reasons``, ``leaks``, ``notes``)."""
    sample = sample_decision_points(decision_points, samples, seed, label_of, "coverage")
    check = FutureCheck(adapter, trajectories, target, getattr(adapter, "outcome_keys", ()))
    notes: list[str] = []
    rec = _future(sample, check, notes)
    rec["notes"] = notes
    if rec["leaks"]:
        raise PrefixLeakError(leak_message(type(adapter).__name__, rec), {"future": rec})
    return rec


def skipped_future(reason: str) -> dict[str, Any]:
    """The future-invariance record of a check that was turned off."""
    return {"checked": 0, "skipped": 0, "reasons": {reason: 0}, "leaks": [], "notes": []}


class PrefixLeakError(LeakageError, ValueError):
    """The adapter puts post-decision data into decision-point prefixes (see :class:`FutureCheck`)."""

    def __init__(self, message: str, record: dict | None = None):
        super().__init__(message)
        self.record = record or {}


def leak_message(adapter: str, future: dict[str, Any]) -> str:
    """The error message of a future-invariance failure."""
    leak = future["leaks"][0]
    return (
        f"the adapter {adapter} puts post-decision data into decision-point prefixes: on "
        f"{len(future['leaks'])} of {future['checked']} checked decision points the prefix changed when only "
        f"the run after the decision changed (first: {leak['decision_point']}, variant '{leak['variant']}', "
        f"at {leak['path']}). A prefix must hold only what precedes the decision, at training and at "
        "serving: fix the adapter's decision_points (its cut, or the fields it copies) and onboard again"
    )


# --------------------------------------------------------------------------- compile-time verification
def _n_steps(prefix: dict, steps_key: str) -> int:
    steps = prefix.get(steps_key)
    return len(steps) if isinstance(steps, list) else 0


def edge_fields(prefixes: list[dict], steps_key: str, limit: int = MAX_FIELDS) -> list[tuple[str, str]]:
    """Fields a live prefix may lack: the top-level task fields (most common first), then the step fields
    that only some onboarding steps have. ``(scope, name)`` with scope ``task`` or ``step``."""
    top = Counter(k for p in prefixes for k in p if k != steps_key)
    in_steps: Counter = Counter()
    n_steps = 0
    for p in prefixes:
        for s in p.get(steps_key) or []:
            if isinstance(s, dict):
                n_steps += 1
                in_steps.update(s.keys())
    fields = [("task", k) for k, _ in sorted(top.items(), key=lambda kv: (-kv[1], str(kv[0])))]
    fields += [
        ("step", k) for k, c in sorted(in_steps.items(), key=lambda kv: (-kv[1], str(kv[0]))) if c < n_steps
    ]
    return fields[:limit]


def _has(prefix: dict, field_: tuple[str, str], steps_key: str) -> bool:
    scope, name = field_
    if scope == "task":
        return name in prefix
    return any(isinstance(s, dict) and name in s for s in prefix.get(steps_key) or [])


def _without(prefix: dict, field_: tuple[str, str], steps_key: str) -> dict:
    scope, name = field_
    out = json_copy(prefix)
    if scope == "task":
        out.pop(name, None)
    else:
        for s in out.get(steps_key) or []:
            if isinstance(s, dict):
                s.pop(name, None)
    return out


class _Box:
    """The sandbox, counting the interpreter runs the checks make."""

    def __init__(self, sandbox: SubprocessSandbox | None):
        self.sandbox = sandbox or SubprocessSandbox()
        self.batches = 0
        self.prefixes = 0

    def run(self, code: str, prefixes: list[dict]) -> list[ExecResult]:
        if not prefixes:
            return []
        self.batches += 1
        self.prefixes += len(prefixes)
        return self.sandbox.run_batch(code, prefixes)


def _changed(
    columns: Iterable[str], pairs: Iterable[tuple[dict | None, dict | None]]
) -> dict[str, dict[str, Any]]:
    """Per column, the pairs of successful runs where its values differ: their count and the first example."""
    out: dict[str, dict[str, Any]] = {}
    for a, b in pairs:
        if a is None or b is None:
            continue
        for c in columns:
            va, vb = _get(a, c), _get(b, c)
            if va is _MISSING and vb is _MISSING:
                continue
            if va is _MISSING or vb is _MISSING or not same(va, vb):
                rec = out.setdefault(c, {"n": 0, "example": [_show(va), _show(vb)]})
                rec["n"] += 1
    return out


def verify_extractor(
    code: str,
    columns: list[str],
    decision_points: list[DecisionPoint],
    *,
    outputs: list[dict | None] | None = None,
    output_columns: list[str] | None = None,
    sandbox: SubprocessSandbox | None = None,
    outcome_keys: Iterable[str] = (),
    future: Callable[[DecisionPoint], dict[str, Any]] | dict[str, Any] | None = None,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    steps_key: str = "steps",
    label_of: Callable[[DecisionPoint], Iterable[Any]] | None = None,
) -> dict[str, Any]:
    """Run the determinism, outcome-invariance, future-invariance and robustness checks (see the module
    docstring).

    ``outputs`` are the features the compile step computed on every decision point (aligned with
    ``decision_points``; None where the code failed), for the ``output_columns`` they hold (default:
    ``columns``): the determinism pass is compared with them on every point. ``future`` is a
    :class:`FutureCheck`. ``samples`` decision points are coverage-sampled (``label_of``: the adapter's
    labels) for the other checks. Returns a JSON-able record, masked: ``dropped`` maps each column that failed
    a check to the reason, and ``future["leaks"]`` lists the decision points whose prefix changed with the
    run's future."""
    t0 = time.monotonic()
    box = _Box(sandbox)
    sample = sample_decision_points(decision_points, samples, seed, label_of, "coverage")
    index = {dp.id: i for i, dp in enumerate(decision_points)}
    prefixes = [dp.prefix for dp in sample]
    out: dict[str, Any] = {"sample": len(sample), "dropped": {}, "notes": []}
    dropped: dict[str, str] = out["dropped"]
    keys = list(dict.fromkeys(str(k) for k in outcome_keys if k != steps_key))
    outcome = [with_outcomes(p, keys, ok) for p in prefixes for ok in (True, False)] if keys else []
    cases = _edge_cases(decision_points, sample, steps_key)
    order = np.random.default_rng(seed).permutation(len(decision_points))
    # the sample, alone; every decision point again, shuffled, in other interpreters; then the variants
    base = box.run(code, prefixes)
    shuffled = box.run(code, [decision_points[i].prefix for i in order])
    again: list[ExecResult | None] = [None] * len(decision_points)
    for i, r in zip(order, shuffled, strict=True):
        again[int(i)] = r
    rest = box.run(code, outcome + [p for _, p in cases])
    outcome_res, case_res = rest[: len(outcome)], rest[len(outcome) :]
    ok_base = [r.features if r.ok else None for r in base]
    ok_again = [r.features if r and r.ok else None for r in again]

    # 1. determinism, on every decision point
    unstable = _changed(columns, [(ok_base[k], ok_again[index[dp.id]]) for k, dp in enumerate(sample)])
    if outputs is not None:
        for c, rec in _changed(output_columns or columns, zip(outputs, ok_again, strict=True)).items():
            unstable.setdefault(c, {"n": 0, "example": rec["example"]})["n"] += rec["n"]
    flaky = sum(
        r.ok != (again[index[dp.id]] and again[index[dp.id]].ok) for r, dp in zip(base, sample, strict=True)
    )
    out["determinism"] = {
        "prefixes": len(decision_points),
        "shuffled": True,
        "unstable": unstable,
        "raised_in_one_run": int(flaky),
    }
    for c, rec in unstable.items():
        dropped[c] = (
            f"nondeterministic: running it again (fresh interpreters, shuffled order) changed its value on "
            f"{rec['n']} decision point(s) ({rec['example'][0]} and {rec['example'][1]}), so it cannot be "
            "reproduced at serving time"
        )
    if flaky:
        out["notes"].append(f"the extractor raised in one run and not in the other on {flaky} prefix(es)")

    # 2. outcome invariance
    if keys:
        pairs, crashed = [], []
        for i, b in enumerate(ok_base):
            for r in outcome_res[2 * i : 2 * i + 2]:
                if b is not None and not r.ok:
                    crashed.append(_last_line(r.error))
                pairs.append((b, r.features if r.ok else None))
        changed = _changed([c for c in columns if c not in dropped], pairs)
        out["outcome"] = {"keys": keys, "prefixes": len(outcome), "changed": changed, "raised": len(crashed)}
        for c, rec in changed.items():
            dropped[c] = (
                f"reads the run's outcome fields: adding {_names(keys)} to a prefix changed its value on "
                f"{rec['n']} of {len(outcome)} variants ({rec['example'][0]} -> {rec['example'][1]}); no "
                "decision-time prefix holds an outcome, so it would leak the label wherever one did"
            )
        if crashed:
            out["notes"].append(
                f"the extractor raised on {len(crashed)} prefix(es) once outcome fields were added "
                f"({crashed[0]}): it iterates over fields it does not know"
            )
    else:
        out["outcome"] = {"skipped": "the adapter names no outcome fields"}

    # 3. future invariance, through the adapter
    out["future"] = dict(future) if isinstance(future, dict) else _future(sample, future, out["notes"])

    # 4. robustness (reported, never dropped)
    out["robustness"] = _robustness(columns, cases, case_res)
    out["batches"] = box.batches
    out["prefixes_run"] = box.prefixes
    out["seconds"] = round(time.monotonic() - t0, 2)
    return mask_secrets(out, by_key=False)


def _future(
    sample: list[DecisionPoint], check: Callable[[DecisionPoint], dict] | None, notes: list[str]
) -> dict[str, Any]:
    if check is None:
        return {
            "checked": 0,
            "skipped": len(sample),
            "reasons": {"no adapter and runs to rebuild prefixes": 1},
        }
    results = [check(dp) for dp in sample]
    rec = {
        "checked": sum("ok" in r or "leak" in r for r in results),
        "skipped": sum("skipped" in r for r in results),
        "reasons": dict(Counter(r["skipped"] for r in results if "skipped" in r)),
        "leaks": [r["leak"] for r in results if "leak" in r],
    }
    if not rec["checked"]:
        reason = next(iter(rec["reasons"]), "no decision point")
        msg = f"the future-invariance check was skipped: no decision point could be checked ({reason})"
        logger.warning("TabSchema: %s", msg)
        notes.append(msg)
    return rec


def _edge_cases(
    decision_points: list[DecisionPoint], sample: list[DecisionPoint], steps_key: str
) -> list[tuple[str, dict]]:
    """The robustness check's prefixes: the shortest onboarding prefixes, one without steps and, per field a
    live prefix may lack (:func:`edge_fields`), a prefix without it."""
    fewest = sorted(decision_points, key=lambda dp: (_n_steps(dp.prefix, steps_key), dp.id))[: 10 * SHORTEST]
    order = sorted(fewest, key=lambda dp: (_n_steps(dp.prefix, steps_key), len(canonical(dp.prefix)), dp.id))
    cases: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for dp in order:
        key = canonical(dp.prefix)
        if key not in seen:
            seen.add(key)
            cases.append(
                (f"a shortest onboarding prefix ({_n_steps(dp.prefix, steps_key)} steps)", dp.prefix)
            )
        if len(cases) == SHORTEST:
            break
    if order and isinstance(order[0].prefix.get(steps_key), list):
        cases.append(("no steps", {**json_copy(order[0].prefix), steps_key: []}))
    pool = [dp.prefix for dp in sample] + [dp.prefix for dp in order]
    for f in edge_fields([dp.prefix for dp in decision_points], steps_key):
        having = next((p for p in pool if _has(p, f, steps_key)), None)
        if having is not None:
            cases.append((f"without the {f[0]} field {f[1]!r}", _without(having, f, steps_key)))
    return cases


def _robustness(
    columns: list[str], cases: list[tuple[str, dict]], results: list[ExecResult]
) -> dict[str, Any]:
    """Which features raise or go missing on each edge case."""
    records, failing = [], Counter()
    for (label, _), r in zip(cases, results, strict=True):
        if not r.ok:
            records.append({"case": label, "raised": _last_line(r.error)})
            failing.update(columns)
            continue
        missing = [c for c in columns if c not in r.features]
        failing.update(missing)
        records.append(
            {
                "case": label,
                "raised": None,
                "missing": missing,
                "null": [c for c in columns if c in r.features and r.features[c] is None],
            }
        )
    return {"cases": records, "failing": dict(sorted(failing.items(), key=lambda kv: (-kv[1], kv[0])))}


# --------------------------------------------------------------------------- golden tests
def _largest_list(value: Any, skip: Any = None) -> list | None:
    """The list with the longest JSON inside ``value`` that has more than one item (not ``skip``)."""
    best, size = None, 0
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, list):
            if len(v) > 1 and v is not skip:
                n = len(canonical(v))
                if n > size:
                    best, size = v, n
            stack.extend(v)
    return best


def snapshot(prefix: dict, steps_key: str = "steps", max_chars: int = GOLDEN_MAX_CHARS) -> dict | None:
    """A golden test's prefix: a JSON copy of ``prefix`` with credentials, e-mail addresses and home paths
    masked (:func:`mask_secrets`) and strings clipped to ``GOLDEN_STRING_CHARS``; while its JSON exceeds
    ``max_chars``, the middle steps are left out (the first two and as many of the latest as fit), then the
    largest other lists are halved, then strings are clipped shorter. None when it cannot be made to fit. The
    expected features are computed on the snapshot itself, so they reproduce exactly."""
    out = clip_strings(mask_secrets(json_copy(prefix)), GOLDEN_STRING_CHARS)
    steps = out.get(steps_key)
    if isinstance(steps, list):
        keep = len(steps)
        while len(canonical(out)) > max_chars and keep > 3:
            keep = max(3, keep // 2)
            out[steps_key] = steps[:2] + steps[len(steps) - (keep - 2) :]
    cap = GOLDEN_STRING_CHARS
    for _ in range(200):
        if len(canonical(out)) <= max_chars:
            return out
        big = _largest_list(out)
        if big is not None:
            del big[max(1, len(big) // 2) :]
        elif cap > 50:
            cap //= 2
            out = clip_strings(out, cap)
        else:
            break
    return out if len(canonical(out)) <= max_chars else None


def make_tests(
    code: str,
    columns: list[str],
    decision_points: list[DecisionPoint],
    *,
    sandbox: SubprocessSandbox | None = None,
    outcome_keys: Iterable[str] = (),
    n: int = DEFAULT_GOLDENS,
    seed: int = DEFAULT_SEED,
    steps_key: str = "steps",
    verification: dict | None = None,
    label_of: Callable[[DecisionPoint], Iterable[Any]] | None = None,
) -> dict[str, Any]:
    """The extractor's tests: up to ``n`` coverage-sampled snapshots (:func:`snapshot`) with the features
    ``code`` computes on them (``columns`` only), the checks :func:`verify_tests` re-runs, and the
    compile-time ``verification`` record (masked, without timings). With ``n`` 0 the file holds no trajectory
    text. ``golden_stats`` says how many goldens were requested and recorded, and how many snapshots were too
    large or made the code raise (masking can remove what the code reads): :func:`verify_tests` fails a file
    with fewer goldens than requested because snapshots raised."""
    goldens: list[dict] = []
    stats: dict[str, Any] = {"requested": n, "recorded": 0, "snapshots": 0, "too_large": 0, "raised": 0}
    if n > 0:
        snaps: list[dict] = []
        seen: set[str] = set()
        for dp in sample_decision_points(decision_points, 2 * n, seed, label_of, "coverage"):
            s = snapshot(dp.prefix, steps_key)
            if s is None:
                stats["too_large"] += 1
            elif canonical(s) not in seen:
                seen.add(canonical(s))
                snaps.append(s)
        results = _Box(sandbox).run(code, snaps)
        goldens = [
            {"prefix": s, "features": {c: r.features[c] for c in columns if c in r.features}}
            for s, r in zip(snaps, results, strict=True)
            if r.ok
        ][:n]
        failed = [r for r in results if not r.ok]
        stats.update(snapshots=len(snaps), raised=len(failed), recorded=len(goldens))
        if failed:
            stats["first_error"] = _last_line(failed[0].error)
    return {
        "format": TESTS_FORMAT,
        "about": (
            "Tests of extract.py: snapshots of onboarding decision-point prefixes and the features the "
            "extractor computed on them, plus the checks `tabagent verify-head` re-runs. The snapshots are "
            "trajectory text (taken after redact=, with credentials, e-mail addresses and home paths "
            "masked): treat this file like the trajectories."
        ),
        "extractor_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "columns": list(columns),
        "steps_key": steps_key,
        "outcome_keys": [k for k in dict.fromkeys(map(str, outcome_keys)) if k != steps_key],
        "checks": ["goldens", "determinism", "outcome_invariance"],
        "goldens": goldens,
        "golden_stats": stats,
        "verification": mask_secrets(
            {k: v for k, v in (verification or {}).items() if k != "seconds"}, by_key=False
        ),
    }


def restrict_tests(tests: dict | None, columns: Iterable[str]) -> dict | None:
    """``tests`` for the columns that are kept (after drops later in discovery)."""
    if tests is None:
        return None
    keep = [c for c in tests["columns"] if c in set(columns)]
    out = {**tests, "columns": keep}
    out["goldens"] = [
        {**g, "features": {c: v for c, v in g["features"].items() if c in keep}} for g in tests["goldens"]
    ]
    return out


def check_tests_file(tests: Any) -> dict:
    """``tests`` (a parsed ``extractor_tests.json``) when its shape is valid, else ``ValueError``."""
    problems = []
    if not isinstance(tests, dict):
        raise ValueError(f"{TESTS_FILE} must hold a JSON object")
    if not isinstance(tests.get("format"), int) or isinstance(tests.get("format"), bool):
        problems.append("'format' is not an integer")
    elif tests["format"] > TESTS_FORMAT:
        problems.append(f"format {tests['format']} is newer than this library reads ({TESTS_FORMAT})")
    if not isinstance(tests.get("columns"), list) or not all(isinstance(c, str) for c in tests["columns"]):
        problems.append("'columns' is not a list of names")
    goldens = tests.get("goldens")
    if not isinstance(goldens, list) or not all(
        isinstance(g, dict) and isinstance(g.get("prefix"), dict) and isinstance(g.get("features"), dict)
        for g in goldens
    ):
        problems.append("'goldens' is not a list of {prefix, features} objects")
    if not isinstance(tests.get("outcome_keys", []), list):
        problems.append("'outcome_keys' is not a list")
    if not isinstance(tests.get("extractor_sha256", ""), str):
        problems.append("'extractor_sha256' is not a string")
    stats = tests.get("golden_stats", {})
    if not isinstance(stats, dict) or not all(
        isinstance(stats.get(k, 0), int) for k in ("requested", "recorded", "raised", "snapshots")
    ):
        problems.append("'golden_stats' is not an object of counts")
    if problems:
        raise ValueError(f"{TESTS_FILE} is invalid: {'; '.join(problems)}")
    return tests


@dataclass
class VerifyReport:
    """The result of :func:`verify_tests` or :func:`verify_head`: one record per check (``name``,
    ``passed``, ``detail``) and warnings that do not fail it. ``passed`` is True when every check passed."""

    checks: list[dict[str, Any]] = field(default_factory=list)
    path: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c["passed"] for c in self.checks)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    def __str__(self) -> str:
        where = f" of {self.path}" if self.path else ""
        head = f"{'PASSED' if self.passed else 'FAILED'}: extractor checks{where}"
        lines = [f"  [{'ok' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}" for c in self.checks]
        lines += [f"  [warn] {w}" for w in self.warnings]
        return "\n".join([head, *lines]) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "path": self.path, "checks": self.checks, "warnings": self.warnings}


def verify_tests(
    code: str, tests: dict, sandbox: SubprocessSandbox | None = None, require_goldens: bool = False
) -> VerifyReport:
    """Re-run an extractor's tests (:func:`make_tests`) in the sandbox: the code's hash, each golden prefix's
    expected features, determinism (a second run in fresh interpreters, in reverse order) and outcome
    invariance. A file without goldens (``goldens=0``) passes with a warning unless ``require_goldens``.
    An invalid file raises ``ValueError``."""
    tests = check_tests_file(tests)
    rep = VerifyReport()
    sha = hashlib.sha256(code.encode()).hexdigest()
    rep.add(
        "extractor hash",
        sha == tests.get("extractor_sha256"),
        "extract.py is the code the tests were recorded for"
        if sha == tests.get("extractor_sha256")
        else "extract.py changed since the tests were recorded",
    )
    columns, goldens = list(tests["columns"]), list(tests["goldens"])
    stats = tests.get("golden_stats") or {}
    requested, recorded = stats.get("requested", len(goldens)), stats.get("recorded", len(goldens))
    if stats.get("raised") and recorded < requested:
        rep.add(
            "golden coverage",
            False,
            f"{recorded} of {requested} golden prefixes recorded: {stats['raised']} of "
            f"{stats.get('snapshots')} snapshots made the extractor raise "
            f"({stats.get('first_error', 'unknown error')}); masking or clipping removed what the code "
            "reads, so the goldens cover too little",
        )
    if not goldens:
        if requested:
            msg = (
                f"0 of {requested} golden prefixes recorded ({stats.get('too_large', 0)} snapshot(s) too "
                "large): only the hashes and artifact were checked"
            )
        else:
            msg = "no golden prefixes recorded (goldens=0): only the hashes and artifact were checked"
        if require_goldens:
            rep.add("goldens", False, msg)
        else:
            rep.warnings.append(msg)
        return rep
    box = _Box(sandbox)
    prefixes = [g["prefix"] for g in goldens]
    keys = list(tests.get("outcome_keys") or [])
    variants = [with_outcomes(p, keys, ok) for p in prefixes for ok in (True, False)]
    first = box.run(code, prefixes)
    rest = box.run(code, prefixes[::-1] + variants)  # the second run, reversed, in other interpreters
    second, outcome_res = rest[: len(prefixes)][::-1], rest[len(prefixes) :]
    errors, mismatches = [], []
    for i, (g, r) in enumerate(zip(goldens, first, strict=True)):
        if not r.ok:
            errors.append(f"golden {i}: raised {_last_line(r.error)}")
            continue
        for c in columns:
            want, got = _get(g["features"], c), _get(r.features, c)
            if (want is _MISSING) != (got is _MISSING) or (want is not _MISSING and not same(want, got)):
                mismatches.append(f"golden {i}, {c}: expected {_show(want)}, got {_show(got)}")
    bad = errors + mismatches
    rep.add(
        "goldens",
        not bad,
        f"{len(goldens)} golden prefixes reproduce their {len(columns)} recorded features"
        if not bad
        else f"{len(bad)} difference(s); " + "; ".join(bad[:5]),
    )
    unstable = _changed(
        columns,
        zip(
            [r.features if r.ok else None for r in first],
            [r.features if r.ok else None for r in second],
            strict=True,
        ),
    )
    rep.add(
        "determinism",
        not unstable,
        "a second run in fresh interpreters gave identical features"
        if not unstable
        else "values differ between two runs: "
        + ", ".join(f"{c} ({v['example'][0]} vs {v['example'][1]})" for c, v in unstable.items()),
    )
    if keys:
        pairs = [
            (first[i].features if first[i].ok else None, r.features if r.ok else None)
            for i in range(len(prefixes))
            for r in outcome_res[2 * i : 2 * i + 2]
        ]
        changed = _changed([c for c in columns if c not in unstable], pairs)
        rep.add(
            "outcome invariance",
            not changed,
            f"adding the outcome fields ({_names(keys)}) changed no feature"
            if not changed
            else "outcome fields change: "
            + ", ".join(f"{c} ({v['example'][0]} -> {v['example'][1]})" for c, v in changed.items()),
        )
    return rep


def _sandbox_from(spec: Any) -> SubprocessSandbox | None:
    if spec is None or spec == "subprocess":
        return None
    if isinstance(spec, str) and (spec == "docker" or spec.startswith("docker:")):
        image = spec.partition(":")[2]
        return DockerSandbox(image=image) if image else DockerSandbox()
    if hasattr(spec, "run_batch"):
        return spec
    raise ValueError(f"sandbox: 'subprocess', 'docker', 'docker:<image>' or a sandbox, not {spec!r}")


def verify_head(path: str | Path, sandbox: Any = None, require_goldens: bool = False) -> VerifyReport:
    """Check a head saved by ``onboard`` before deploying it: the model artifact matches its manifest, the
    card pins the hash of ``extract.py`` and the code matches it, the tests in ``extractor_tests.json`` cover
    the card's extractor features, and re-running them in the sandbox (``"subprocess"``, ``"docker"``,
    ``"docker:<image>"`` or a sandbox instance) reproduces every recorded feature, twice, with the outcome
    fields making no difference. A head without golden prefixes passes with a warning unless
    ``require_goldens``. ``path`` is a directory or ``hf://OWNER/REPO[@REVISION]``. A path that is not a head,
    or an invalid tests file, raises ``ValueError``. CLI: ``tabagent verify-head PATH`` (exit status 1 when a
    check fails, 2 on such an error)."""
    from tabagent import artifact, hub

    src = hub.resolve(path)
    rep = VerifyReport(path=str(path))
    if hub.is_hub_uri(path) and _sandbox_from(sandbox) is None:
        rep.warnings.append(
            "a head from the Hub runs its extractor in the subprocess sandbox, which is not a security "
            "boundary; pass --sandbox docker (sandbox='docker') for code you do not trust"
        )
    meta_file = src / "head.json"
    if not meta_file.is_file():
        raise ValueError(f"{path} is not a head saved by onboard (it has no head.json)")
    meta = json.loads(meta_file.read_text())
    try:
        agent = artifact.load(src / "model", verify=True)
        rep.add("artifact", True, "the model files match their manifest")
    except (ValueError, OSError, KeyError, PermissionError) as e:
        rep.add("artifact", False, f"the model artifact is not intact: {e}")
        return rep
    if not meta.get("has_extractor"):
        rep.add("extractor", True, "this head has no TabSchema extractor: nothing more to verify")
        return rep
    if not (src / "extract.py").is_file():
        rep.add("card hash", False, "extract.py is missing")
        return rep
    code = (src / "extract.py").read_text()
    expected = (agent.card_.extractor or {}).get("sha256")
    sha = hashlib.sha256(code.encode()).hexdigest()
    rep.add(
        "card hash",
        bool(expected) and sha == expected,
        "extract.py matches the card's extractor hash"
        if sha == expected
        else "extract.py does not match the card's hash"
        if expected
        else "the card pins no hash for extract.py (TabAgentHead.load refuses it unless allow_unpinned=True; "
        "saving the head again pins it)",
    )
    tests_file = src / TESTS_FILE
    if not tests_file.is_file():
        msg = f"no {TESTS_FILE}: no golden tests were saved with this head"
        if require_goldens:
            rep.add("tests", False, msg)
        else:
            rep.warnings.append(msg + " (only the hashes and artifact were checked)")
        return rep
    try:
        tests = json.loads(tests_file.read_text())
    except ValueError as e:
        raise ValueError(f"{TESTS_FILE} is not valid JSON: {e}") from e
    tests = check_tests_file(tests)
    card_cols = [f.name for f in agent.card_.features if f.source == "extractor"]
    uncovered = [c for c in card_cols if c not in set(tests["columns"])]
    rep.add(
        "coverage",
        not uncovered,
        f"the tests cover the card's {len(card_cols)} extractor features"
        if not uncovered
        else f"card features without golden values: {', '.join(uncovered)}",
    )
    sub = verify_tests(code, tests, _sandbox_from(sandbox), require_goldens)
    rep.checks += sub.checks
    rep.warnings += sub.warnings
    return rep
