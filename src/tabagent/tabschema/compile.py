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
   hallucinated field or not matching their plan are dropped (an unusable validator answer leaves the
   deterministic checks alone in charge, with a note);
4. each output column must implement an accepted feature (its name, a sub-key of it, or it under a group
   prefix), must not reuse a name of the decision table, and must pass quality gates (non-constant, null
   rate, length); it then gets a frozen type: the plan's type for the feature itself, the data's for a
   sub-key;
5. a kept column that lists earlier candidates (the tools called before the decision, say) becomes the
   context of the card's fitted co-usage statistic (``co_usage_context``).
"""

from __future__ import annotations

import hashlib
import json
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
    LLMFormatError,
    extract_with_retries,
    normalize_name,
)
from tabagent.tabschema.fitted import context_items
from tabagent.tabschema.sandbox import SubprocessSandbox
from tabagent.tabschema.trajectory import DecisionPoint, merge_examples

TEXT_CAP = 20_000
CAT_MAX_CHARS = 64  # a "categorical" plan whose values are longer and mostly distinct is typed as text
#: A co-usage context: most of its items are candidate ids, on enough rows, and it lists few of them per row.
CO_USAGE_SHARE, CO_USAGE_ROWS, CO_USAGE_MAX_ITEMS = 0.8, 0.2, 0.5


class ExtractionError(RuntimeError):
    """The extractor could not be made to run on the onboarding trajectories."""


#: Appended to every compiled extractor (``context_text=True``): the decision's context as plain text, not
#: written by the LLM. It reads the prefix's task-level text fields and the texts of its steps (system
#: prompts, identical across decisions, are skipped) and keeps the start and the latest part of a long
#: history. Being part of ``extract.py``, it runs identically at training and at inference. It gives the card
#: a context text for the context-candidate similarity features, TabSynth's paraphrases and BM25.
CONTEXT_TEXT_CODE = """

# --- appended by TabAgent (not LLM-written): the decision's context as text ---
def _tabagent_context_text(trajectory_data, limit=4000):
    parts = []
    for key, value in trajectory_data.items():
        if key != "steps" and isinstance(value, str) and value.strip():
            parts.append(value.strip())
    for step in trajectory_data.get("steps") or []:
        if not isinstance(step, dict) or step.get("name") == "system":
            continue
        for key in ("content", "text", "thoughts", "intent", "data"):
            value = step.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    text = "\\n".join(parts)
    if len(text) <= limit:
        return text
    return text[: limit // 4] + "\\n...\\n" + text[-(limit - limit // 4) :]


if isinstance(features, dict):
    features[__TABAGENT_CONTEXT_COLUMN__] = _tabagent_context_text(trajectory)
"""
CONTEXT_TEXT_COLUMN = "context_text"


def context_text_code(column: str) -> str:
    """The appended context-text code, writing ``column``."""
    return CONTEXT_TEXT_CODE.replace("__TABAGENT_CONTEXT_COLUMN__", repr(column))


def context_texts(prefixes: list[dict], column: str = CONTEXT_TEXT_COLUMN) -> list[str]:
    """The context texts the appended code computes for ``prefixes`` (run here, on JSON copies)."""
    code = compile(context_text_code(column), "<context_text>", "exec")
    out = []
    for p in prefixes:
        ns: dict[str, Any] = {"features": {}, "trajectory": json.loads(json.dumps(p, default=str))}
        exec(code, ns)  # noqa: S102 - the library's own fixed code
        out.append(ns["features"][column])
    return out


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
    notes: list[str] = field(default_factory=list)
    co_usage_context: str | None = None  # the column listing earlier candidates, if any

    @property
    def sha256(self) -> str:
        """SHA-256 of the code (recorded in the card, checked when a head is loaded)."""
        return hashlib.sha256(self.code.encode()).hexdigest()


def _resolve_type(spec_type: str, col: pd.Series) -> str:
    from tabagent.auto import infer_type

    values = col.dropna()
    numeric = pd.to_numeric(values, errors="coerce").notna().mean() if len(values) else 0.0
    t = (spec_type or "").lower()
    if t.startswith("numeric") and numeric >= 0.95:
        return "num"
    if t.startswith("categor"):
        text = values.astype(str)
        free_text = len(text) and text.str.len().mean() > CAT_MAX_CHARS and text.nunique() > len(text) / 2
        return "text" if free_text else "cat"
    if t.startswith("text"):
        return "text"
    if pd.api.types.is_bool_dtype(values) or numeric >= 0.95:
        return "num"
    return infer_type(values.astype(str))


def _match_spec(column: str, specs: dict[str, FeatureSpec]) -> tuple[FeatureSpec | None, bool]:
    """The planned feature an output column implements and whether it is that feature itself: its name, or
    its name under a group prefix (``text_features_user_intent``, from ``flatten_json``); a sub-key of it
    (``user_intent_length``) is derived from it. (None, False) when the column is not in the plan."""
    key = normalize_name(column)
    if key in specs:
        return specs[key], True
    grouped = [k for k in specs if key.endswith("_" + k)]
    if grouped:
        return specs[max(grouped, key=len)], True
    derived = [k for k in specs if key.startswith(k + "_")]
    return (specs[max(derived, key=len)], False) if derived else (None, False)


def _is_flag(values: pd.Series) -> bool:
    """A presence flag: one boolean value where present and missing elsewhere (``True``/``None``)."""
    present = values.dropna()
    return (
        bool(len(present) and len(present) < len(values) and present.map(lambda v: isinstance(v, bool)).all())
        and present.nunique() == 1
    )


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
    reserved: frozenset[str] | set[str] = frozenset(),
    candidate_ids: frozenset[str] | set[str] = frozenset(),
    context_text: bool = True,
) -> CompiledExtractor:
    """Write, check and quality-gate one extractor for the ``accepted`` features (see the module docstring).
    Output columns named like a column of the decision table (``reserved``) are dropped; ``candidate_ids``
    (the decision's candidates) find the co-usage context.

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
    result = CompiledExtractor(attempt.code, {}, attempts=len(history))
    try:
        sample = [o for o in outputs if o][:3]
        verdicts = validator(accepted, attempt.code, example, sample, target) if validator else {}
    except LLMFormatError as e:  # the deterministic checks and the quality gates still apply
        verdicts = {}
        result.notes.append(f"the CodeValidator's answer was unusable, so no feature was judged by it ({e})")

    produced = set()
    for col in frame.columns:
        spec, itself = _match_spec(col, by_key)
        verdict = verdicts.get(spec.key) if spec else None
        values = frame.loc[ok_rows, col]
        as_text = values.astype(str)
        flag = _is_flag(values)
        empty = values.isna() | as_text.str.strip().isin(["", "None", "nan", "[]", "{}"])
        q = {
            "null_rate": 0.0 if flag else float(empty.mean()) if len(values) else 1.0,
            "n_unique": int(values.nunique(dropna=False)),
            "spec": spec.feature_name if spec else None,
            "validator": verdict,
        }
        result.quality[col] = q
        if spec is not None:
            produced.add(spec.key)
        if col in reserved:
            result.dropped[col] = "name reserved by the decision table"
        elif spec is None:
            result.dropped[col] = "not in the accepted feature plan"
        elif verdict in ("hallucinated_field", "plan_mismatch"):
            result.dropped[col] = f"CodeValidator: {verdict}"
        elif q["n_unique"] <= 1:
            result.dropped[col] = "constant across onboarding decision points"
        elif q["null_rate"] > max_null_rate:
            result.dropped[col] = f"null rate {q['null_rate']:.2f} > {max_null_rate}"
        elif as_text.str.len().mean() > TEXT_CAP:
            result.dropped[col] = f"mean length above {TEXT_CAP} characters"
        else:
            result.features[col] = _resolve_type(spec.feature_type if itself else "computed", values)
            result.specs[col] = spec.to_dict()
    for key, spec in by_key.items():
        if key not in produced:
            result.dropped.setdefault(
                spec.feature_name, "accepted but not produced by the compiled extractor"
            )
    if not result.features:
        reasons = "; ".join(f"{k}: {v}" for k, v in result.dropped.items())
        raise ExtractionError(f"every compiled feature failed the quality gates ({reasons[:1000]})")
    if candidate_ids:
        result.co_usage_context = _co_usage_context(result, frame.loc[ok_rows], set(candidate_ids))
    if context_text:  # after the co-usage search: the context text is free text, never a candidate list
        _add_context_text(result, prefixes, set(reserved) | set(frame.columns))
    return result


def _add_context_text(result: CompiledExtractor, prefixes: list[dict], taken: set[str]) -> None:
    """Append the context-text code to the extractor when its text varies across decision points."""
    column = CONTEXT_TEXT_COLUMN if CONTEXT_TEXT_COLUMN not in taken else f"tabagent_{CONTEXT_TEXT_COLUMN}"
    values = pd.Series(context_texts(prefixes, column))
    if values.nunique() <= 1:
        result.notes.append("no context text added: the prefixes hold no text that varies across decisions")
        return
    result.code += context_text_code(column)
    result.features[column] = "text"
    result.specs[column] = {
        "feature_name": column,
        "feature_type": "text",
        "description": "the decision's context as text: the prefix's task-level text fields and step texts "
        "(added by TabAgent, not LLM-written)",
        "added_by": "tabagent",
    }
    result.quality[column] = {
        "null_rate": float((values.str.strip() == "").mean()),
        "n_unique": int(values.nunique()),
    }


def _co_usage_context(result: CompiledExtractor, frame: pd.DataFrame, candidates: set[str]) -> str | None:
    """The kept text or categorical column that best lists earlier candidates: at least ``CO_USAGE_SHARE`` of
    its items are candidate ids, on at least ``CO_USAGE_ROWS`` of the rows, with at most
    ``CO_USAGE_MAX_ITEMS`` of the candidates per row (a list of every offered candidate tells nothing). Its
    statistics are recorded in ``quality``."""
    best, best_share = None, 0.0
    for col, typ in result.features.items():
        if typ == "num":
            continue
        items = [context_items(v) for v in frame[col]]
        flat = [x for it in items for x in it]
        if not flat:
            continue
        share = sum(x in candidates for x in flat) / len(flat)
        rows = sum(bool(it) for it in items) / len(items)
        per_row = len(flat) / max(1, sum(bool(it) for it in items)) / len(candidates)
        result.quality[col].update(candidate_share=round(share, 3), rows_with_candidates=round(rows, 3))
        fits = share >= CO_USAGE_SHARE and rows >= CO_USAGE_ROWS and per_row <= CO_USAGE_MAX_ITEMS
        if fits and share > best_share:
            best, best_share = col, share
    return best


class ExtractorRunner:
    """Apply a compiled extractor to decision-point prefixes.

    By default the code runs in the sandbox. ``trust_extractor=True`` runs it in-process, which is fast but
    gives the code the caller's privileges; use it only for an extractor you have reviewed. In-process, the
    code sees a JSON copy of the prefix and its features are read back through JSON, exactly as in the
    sandbox, so both modes give identical features and the caller's prefix is never changed. A prefix on
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
                ns: dict[str, Any] = {
                    "__name__": "__main__",
                    "trajectory": json.loads(json.dumps(p, default=str)),
                }
                try:
                    exec(self._compiled, ns)  # noqa: S102 - trusted, reviewed extractor
                    feats = ns.get("features")
                    if not isinstance(feats, dict):
                        raise TypeError("the code must leave a dict named `features` in its global scope")
                    out.append(json.loads(json.dumps(feats)))
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
