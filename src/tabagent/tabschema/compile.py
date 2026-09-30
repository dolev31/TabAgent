"""Compile the accepted features into one reusable extractor (paper Sec. 3.1; Sec. 4.3).

The paper's TabSchema produces "reusable extractors: Python feature-extraction functions that can be
re-applied when onboarding a new agent", and "only the resulting extractors are applied to the held-out
fold". The per-trajectory code of the analysis stage proves each proposal feasible on one trace; this
step writes a single extractor for the MetaJudge-accepted features and holds it to the standard a reusable
component needs:

1. the CodeExtractor writes code for the accepted features against the merged schema of all onboarding
   decision points (its prompt is unchanged);
2. the code must pass the deterministic checks and run on all onboarding decision points, with at most
   ``1 - min_success`` failures; otherwise it is retried (up to three times) with the failing tracebacks;
3. the CodeValidator reviews the final code and a sample of its outputs; features it flags as reading a
   hallucinated field or not matching their plan are dropped;
4. each output column must pass quality gates (non-constant, null rate, value type), and gets a frozen type.
"""

from __future__ import annotations

import copy
import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tabagent.tabschema.agents import (
    CodeExecutor,
    CodeExtractor,
    CodeValidator,
    FeatureSpec,
    extract_with_retries,
    normalize_name,
)
from tabagent.tabschema.sandbox import SubprocessSandbox
from tabagent.tabschema.trajectory import DecisionPoint

TEXT_CAP = 20_000


class ExtractionError(RuntimeError):
    """The extractor could not be made to run on the onboarding trajectories."""


@dataclass
class CompiledExtractor:
    """The compiled extractor: its code, the output columns with their frozen types, the feature plan each
    column implements, what was dropped and why, and per-column quality statistics."""

    code: str
    features: dict[str, str]  # output column -> frozen type (text / num / cat)
    specs: dict[str, dict] = field(default_factory=dict)  # output column -> the feature plan it implements
    dropped: dict[str, str] = field(default_factory=dict)  # output column or feature -> reason
    quality: dict[str, dict] = field(default_factory=dict)
    attempts: int = 1

    @property
    def sha256(self) -> str:
        """SHA-256 of the code (recorded in the card, checked when a head is loaded)."""
        return hashlib.sha256(self.code.encode()).hexdigest()


def merge_examples(values: list[Any]) -> Any:
    """Deep-merge JSON values so one example shows every key seen in any of them (for the schema prompt)."""
    if not values:
        return {}
    if all(isinstance(v, dict) for v in values):
        keys = list(dict.fromkeys(k for v in values for k in v))
        return {k: merge_examples([v[k] for v in values if k in v]) for k in keys}
    if all(isinstance(v, list) for v in values):
        items = [x for v in values for x in v[:3]]
        return [merge_examples(items)] if items else []
    return copy.deepcopy(values[0])


def _resolve_type(spec_type: str, col: pd.Series) -> str:
    from tabagent.auto import infer_type

    values = col.dropna()
    numeric = pd.to_numeric(values, errors="coerce").notna().mean() if len(values) else 0.0
    t = (spec_type or "").lower()
    if t.startswith("numeric") and numeric >= 0.95:
        return "num"
    if t.startswith("categor"):
        return "cat"
    if t.startswith("text"):
        return "text"
    if pd.api.types.is_bool_dtype(values) or numeric >= 0.95:
        return "num"
    return infer_type(values.astype(str))


def _match_spec(column: str, specs: dict[str, FeatureSpec]) -> FeatureSpec | None:
    key = normalize_name(column)
    if key in specs:
        return specs[key]
    prefixes = [k for k in specs if key.startswith(k + "_")]  # flattened sub-keys of a planned feature
    return specs[max(prefixes, key=len)] if prefixes else None


def compile_extractor(
    accepted: list[FeatureSpec],
    decision_points: list[DecisionPoint],
    extractor: CodeExtractor,
    validator: CodeValidator | None,
    target: str,
    *,
    sandbox: SubprocessSandbox | None = None,
    known_paths: set | None = None,
    outcome_keys: tuple[str, ...] = (),
    max_retries: int = 3,
    min_success: float = 0.95,
    max_null_rate: float = 0.5,
) -> CompiledExtractor:
    """Write, check and quality-gate one extractor for the ``accepted`` features (see the module docstring).

    Raises ``ExtractionError`` when no code runs on enough decision points or no column passes the gates.
    """
    if not accepted:
        raise ExtractionError("no accepted features to compile")
    prefixes = [dp.prefix for dp in decision_points]
    example = merge_examples(prefixes)
    executor = CodeExecutor(sandbox, known_paths, outcome_keys, max_failure_rate=1 - min_success)
    attempt, history = extract_with_retries(
        extractor, executor, accepted, prefixes, example, target, max_retries
    )
    if not attempt.ok:
        raise ExtractionError(f"extractor failed after {len(history)} attempts: {attempt.error[:500]}")

    outputs = [r.features if r.ok else None for r in attempt.results]
    frame = pd.DataFrame([o or {} for o in outputs])
    ok_rows = np.array([o is not None for o in outputs])
    by_key = {s.key: s for s in accepted}
    verdicts = (
        validator(accepted, attempt.code, example, [o for o in outputs if o][:3], target) if validator else {}
    )

    result = CompiledExtractor(attempt.code, {}, attempts=len(history))
    produced = set()
    for col in frame.columns:
        spec = _match_spec(col, by_key)
        verdict = verdicts.get(spec.key) if spec else None
        values = frame.loc[ok_rows, col]
        as_text = values.astype(str)
        empty = values.isna() | as_text.str.strip().isin(["", "None", "nan", "[]", "{}"])
        q = {
            "null_rate": float(empty.mean()) if len(values) else 1.0,
            "n_unique": int(values.nunique(dropna=True)),
            "spec": spec.feature_name if spec else None,
            "validator": verdict,
        }
        result.quality[col] = q
        if spec is not None:
            produced.add(spec.key)
        if verdict in ("hallucinated_field", "plan_mismatch"):
            result.dropped[col] = f"CodeValidator: {verdict}"
        elif q["n_unique"] <= 1:
            result.dropped[col] = "constant across onboarding decision points"
        elif q["null_rate"] > max_null_rate:
            result.dropped[col] = f"null rate {q['null_rate']:.2f} > {max_null_rate}"
        elif as_text.str.len().mean() > TEXT_CAP:
            result.dropped[col] = f"mean length above {TEXT_CAP} characters"
        else:
            result.features[col] = _resolve_type(spec.feature_type if spec else "computed", values)
            result.specs[col] = (
                spec.to_dict() if spec else {"feature_name": col, "note": "not in the feature plan"}
            )
    for key, spec in by_key.items():
        if key not in produced:
            result.dropped.setdefault(
                spec.feature_name, "accepted but not produced by the compiled extractor"
            )
    if not result.features:
        raise ExtractionError("every compiled feature failed the quality gates")
    return result


class ExtractorRunner:
    """Apply a compiled extractor to decision-point prefixes.

    By default the code runs in the sandbox. ``trust_extractor=True`` runs it in-process, which is fast but
    gives the code the caller's privileges; use it only for an extractor you have reviewed. A prefix on
    which the code fails yields ``None`` (no features); failures are counted in ``n_failed`` and reported
    with a ``RuntimeWarning`` naming the first error.
    """

    def __init__(
        self, code: str, trust_extractor: bool = False, sandbox: SubprocessSandbox | None = None
    ) -> None:
        self.code, self.trust_extractor = code, trust_extractor
        self.sandbox = sandbox or SubprocessSandbox()
        self._compiled = compile(code, "<extract.py>", "exec") if trust_extractor else None
        self.n_runs = 0
        self.n_failed = 0

    def run(self, prefixes: list[dict]) -> list[dict | None]:
        """The features of each prefix, or None where the extractor failed."""
        errors: list[str] = []
        if not self.trust_extractor:
            out = []
            for r in self.sandbox.run_batch(self.code, prefixes):
                out.append(r.features if r.ok else None)
                if not r.ok:
                    errors.append(str(r.error))
        else:
            out = []
            for p in prefixes:
                ns: dict[str, Any] = {"__name__": "__main__", "trajectory": p}
                try:
                    exec(self._compiled, ns)  # noqa: S102 - trusted, reviewed extractor
                    out.append(ns["features"])
                except Exception as e:  # the extractor is arbitrary code: any failure means no features
                    out.append(None)
                    errors.append(f"{type(e).__name__}: {e}")
        self.n_runs += len(prefixes)
        self.n_failed += len(errors)
        if errors:
            last = errors[0].strip().splitlines()[-1] if errors[0].strip() else "unknown error"
            warnings.warn(
                f"the TabSchema extractor failed on {len(errors)} of {len(prefixes)} decision point(s); "
                f"their features are empty. First error: {last}",
                RuntimeWarning,
                stacklevel=2,
            )
        return out
