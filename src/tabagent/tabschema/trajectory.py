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
