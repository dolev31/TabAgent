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
   context of the card's fitted co-usage statistic (``co_usage_context``);
6. the release's verification (``tabagent.tabschema.verify``, not in the paper) runs the final code on a
   sample of decision points: columns that are nondeterministic or change when outcome fields or the
   run's future change are dropped, and how the code copes with short or incomplete prefixes is reported.
   Golden tests (``extractor_tests.json``) are recorded for ``tabagent verify-head``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tabagent.defaults import DEFAULT_SEED
from tabagent.tabschema import verify as V
from tabagent.tabschema.agents import (
    CodeExecutor,
    CodeExtractor,
    CodeValidator,
    FeatureSpec,
    LLMContentFiltered,
    LLMFormatError,
    extract_with_retries,
    generate_retry_prompt,
    normalize_name,
)
from tabagent.tabschema.fitted import context_items
from tabagent.tabschema.sandbox import ExecResult, SubprocessSandbox
from tabagent.tabschema.trajectory import DecisionPoint, merge_examples

TEXT_CAP = 20_000
CAT_MAX_CHARS = 64  # a "categorical" plan whose values are longer and mostly distinct is typed as text
#: Columns one planned feature may expand into, and columns the whole extractor may keep. Code that returns a
#: dict per row (word counts, one entry per entity) is flattened into one column per key; past these caps the
#: columns that vary most across decision points are kept.
MAX_COLUMNS_PER_FEATURE, MAX_EXTRACTOR_COLUMNS = 64, 512
#: A co-usage context: most of its items are candidate ids, on enough rows, and it lists few of them per row.
CO_USAGE_SHARE, CO_USAGE_ROWS, CO_USAGE_MAX_ITEMS = 0.8, 0.2, 0.5
#: Duplicate text columns: two text or categorical columns whose normalized token sequences (``_sequence``:
#: the lowercased ``\w+`` tokens, in order, digits kept) are equal on every decision point hold the same
#: content in two forms (raw text and its tokens, say); one is kept. The comparison is screened on a seeded
#: sample of ``DUPLICATE_SCREEN_ROWS`` decision points and confirmed on all of them.
DUPLICATE_SCREEN_ROWS = 2_000
_TOKEN = re.compile(r"\w+")
#: Cells (decision points x output columns) the quality gates hold in memory at once. Past it (a dict per row
#: over many decision points), every column first goes through the cheap gates on statistics gathered over all
#: decision points; presence flags are then decided without the table, and the other columns are ranked on a
#: seeded sample of ``PRECAP_ROWS`` decision points, at most ``max(64, CELL_BUDGET / decision points)`` of
#: them going through the full gates.
CELL_BUDGET, PRECAP_ROWS = 20_000_000, 2_000


class ExtractionError(RuntimeError):
    """The extractor could not be made to run on the onboarding trajectories."""


#: Appended to every extractor the sandbox runs and to every compiled ``extract.py``, after the LLM's code:
#: the CodeExtractor prompt's own ``flatten_json`` applied to the output ("use flatten_json to flatten every
#: dict that goes into the final csv file", App. D.2), so a nested dict becomes ``parent_child`` columns
#: instead of a failed attempt; and a value that is not a finite number (NaN, an infinity from a division by
#: zero) becomes missing (None), as the head treats it, instead of a value JSON cannot carry. Being part of
#: ``extract.py``, it runs identically at training and inference.
FLATTEN_MARK = "# --- appended by TabAgent (not LLM-written): flatten nested dicts"
FLATTEN_CODE = f"""

{FLATTEN_MARK}, as the CodeExtractor prompt asks ---
def _tabagent_flatten(obj, prefix=""):
    flat = {{}}
    for key, value in obj.items():
        if isinstance(value, dict):
            flat.update(_tabagent_flatten(value, f"{{prefix}}{{key}}_"))
        elif isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            flat[f"{{prefix}}{{key}}"] = None
        else:
            flat[f"{{prefix}}{{key}}"] = value
    return flat


if isinstance(features, dict):
    features = _tabagent_flatten(features)
"""


def with_flatten(code: str) -> str:
    """``code`` with ``FLATTEN_CODE`` appended once."""
    return code if FLATTEN_MARK in code else code + FLATTEN_CODE


#: Appended to every compiled extractor (``context_text=True``): the decision's context as plain text, not
#: written by the LLM. It reads the prefix's task-level text fields (never the adapter's outcome fields,
#: should a prefix carry one) and the texts of its steps (system prompts, identical across decisions, are
#: skipped) and keeps the start and the latest part of a long history. Being part of ``extract.py``, it runs
#: identically at training and at inference. It gives the card a context text for the context-candidate
#: similarity features, TabSynth's paraphrases and BM25.
CONTEXT_TEXT_CODE = """

# --- appended by TabAgent (not LLM-written): the decision's context as text ---
def _tabagent_context_text(trajectory_data, limit=4000, skip=__TABAGENT_SKIP__):
    parts = []
    for key, value in trajectory_data.items():
        if key != "steps" and key not in skip and isinstance(value, str) and value.strip():
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


def context_text_code(column: str, skip: Iterable[str] = ()) -> str:
    """The appended context-text code, writing ``column`` and ignoring the top-level fields in ``skip`` (the
    adapter's outcome fields)."""
    keys = repr(tuple(dict.fromkeys(str(k) for k in skip)))
    return CONTEXT_TEXT_CODE.replace("__TABAGENT_CONTEXT_COLUMN__", repr(column)).replace(
        "__TABAGENT_SKIP__", keys
    )


def context_texts(
    prefixes: list[dict], column: str = CONTEXT_TEXT_COLUMN, skip: Iterable[str] = ()
) -> list[str]:
    """The context texts the appended code computes for ``prefixes`` (run here, on JSON copies)."""
    code = compile(context_text_code(column, skip), "<context_text>", "exec")
    out = []
    for p in prefixes:
        ns: dict[str, Any] = {"features": {}, "trajectory": json.loads(json.dumps(p, default=str))}
        exec(code, ns)  # noqa: S102 - the library's own fixed code
        out.append(ns["features"][column])
    return out


@dataclass
class CompiledExtractor:
    """The compiled extractor: its code, the output columns with their frozen types, the feature plan each
    column implements, what was dropped and why, and per-column quality statistics.

    ``verification`` is the record of the release's compile-time checks (``tabagent.tabschema.verify``),
    ``tests`` the golden tests saved as ``extractor_tests.json``, and ``outputs`` the code's values of the
    kept columns on each onboarding decision point, in order (None where it failed), before the context text
    was appended."""

    code: str
    features: dict[str, str]  # output column -> frozen type (text / num / cat)
    specs: dict[str, dict] = field(default_factory=dict)  # output column -> the feature plan it implements
    dropped: dict[str, str] = field(default_factory=dict)  # output column or feature -> reason
    quality: dict[str, dict] = field(default_factory=dict)
    attempts: int = 1
    attempt_errors: list[str] = field(default_factory=list)  # why each failed attempt failed, in order
    notes: list[str] = field(default_factory=list)
    co_usage_context: str | None = None  # the column listing earlier candidates, if any
    verification: dict[str, Any] = field(default_factory=dict)
    tests: dict[str, Any] | None = None
    outputs: list[dict | None] = field(default_factory=list, repr=False, compare=False)

    @property
    def sha256(self) -> str:
        """SHA-256 of the code (recorded in the card, checked when a head is loaded)."""
        return hashlib.sha256(self.code.encode()).hexdigest()

    @property
    def llm_columns(self) -> list[str]:
        """The kept columns the LLM wrote (every column but the ones TabAgent appended)."""
        return [c for c in self.features if not self.specs.get(c, {}).get("added_by")]

    def drop(self, column: str, reason: str) -> None:
        """Leave ``column`` out of the card, recording ``reason`` (the code keeps computing it)."""
        self.features.pop(column, None)
        self.specs.pop(column, None)
        self.dropped[column] = reason
        if self.co_usage_context == column:
            self.co_usage_context = None

    def without(self, columns: Iterable[str]) -> CompiledExtractor:
        """A copy without ``columns`` (the code unchanged), for cards that leave features out."""
        out = set(columns)
        return dataclasses.replace(
            self,
            features={c: t for c, t in self.features.items() if c not in out},
            specs={c: s for c, s in self.specs.items() if c not in out},
            co_usage_context=None if self.co_usage_context in out else self.co_usage_context,
        )


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


def _flagged(verdicts: dict[str, str], by_key: dict[str, FeatureSpec]) -> dict[str, str]:
    return {k: v for k, v in verdicts.items() if k in by_key and v in ("hallucinated_field", "plan_mismatch")}


def _repair(
    extractor,
    executor,
    validator,
    judged,
    accepted,
    prefixes,
    example,
    target,
    attempt,
    verdicts,
    evidence,
    by_key,
    history,
):
    """Ask the CodeExtractor once more with the CodeValidator's objections as the error of the retry prompt;
    keep the new code only if it runs and fewer accepted features are flagged. Returns (attempt, verdicts,
    note)."""
    flagged = _flagged(verdicts, by_key)
    objections = "\n".join(
        f"- {by_key[k].feature_name}: {v}. {evidence.get(k, '')}".strip() for k, v in sorted(flagged.items())
    )
    error = (
        "The code runs, but the CodeValidator found that these features do not follow their plan or read "
        f"fields the trajectories do not have:\n{objections}\nFix these features; keep the others unchanged."
    )
    try:
        code = extractor(accepted, example, target, generate_retry_prompt(1, error, attempt.code))
        new = executor.run(code, prefixes)
    except (LLMFormatError, LLMContentFiltered) as e:
        return (
            attempt,
            verdicts,
            f"the repair round for {len(flagged)} flagged feature(s) failed ({type(e).__name__})",
        )
    history.append(new)
    if not new.ok:
        return (
            attempt,
            verdicts,
            f"the repair round's code did not run ({new.error[:200]}); kept the first code",
        )
    new_verdicts, _ = judged(new)
    still = _flagged(new_verdicts, by_key)
    if len(still) < len(flagged):
        fixed = sorted(by_key[k].feature_name for k in flagged if k not in still)
        return (
            new,
            new_verdicts,
            f"the repair round fixed {len(fixed)} of {len(flagged)} flagged feature(s): {', '.join(fixed)}",
        )
    return (
        attempt,
        verdicts,
        f"the repair round did not reduce the {len(flagged)} flagged feature(s); kept the first code",
    )


def _variation(values: pd.Series) -> float:
    """Share of rows whose value differs from the column's most common one (missing counts as a value)."""
    if not len(values):
        return 0.0
    counts = values.astype(str).where(values.notna(), "<missing>").value_counts()
    return float(1 - counts.iloc[0] / len(values))


def _cap_columns(
    result: CompiledExtractor,
    frame: pd.DataFrame,
    per_feature: int,
    total: int,
    known: dict[str, float] | None = None,
) -> None:
    """Keep at most ``per_feature`` columns of one planned feature and ``total`` columns in all, those that
    vary most across decision points (``known``: the variation of columns that are not in ``frame``); the rest
    are dropped with the reason."""
    known = known or {}
    variation = {c: known[c] if c in known else _variation(frame[c]) for c in result.features}
    groups: dict[str, list[str]] = {}
    for col in result.features:
        groups.setdefault(str(result.specs[col].get("feature_name") or col), []).append(col)
    drop: dict[str, str] = {}
    for name, cols in groups.items():
        if len(cols) > per_feature:
            ranked = sorted(cols, key=lambda c: (-variation[c], c))
            for c in ranked[per_feature:]:
                drop[c] = (
                    f"one of {len(cols)} columns of {name}: only the {per_feature} that vary most are kept "
                    "(max_columns_per_feature)"
                )
            result.notes.append(
                f"{name} expanded into {len(cols)} columns (a dict per row, such as word counts); kept the "
                f"{per_feature} that vary most"
            )
    kept = [c for c in result.features if c not in drop]
    if len(kept) > total:
        for c in sorted(kept, key=lambda c: (-variation[c], c))[total:]:
            drop[c] = (
                f"the extractor's columns are capped at {total} (max_extractor_columns); this one varies less"
            )
        result.notes.append(f"the extractor produced {len(kept)} columns; kept the {total} that vary most")
    for c, why in drop.items():
        result.features.pop(c, None)
        result.specs.pop(c, None)
        result.dropped[c] = why


def _sample_variation(keys: list[str], rows: list[dict]) -> dict[str, float]:
    """``_variation`` of each key over ``rows`` (a key missing from a row counts as a value), in one pass over
    the rows' entries, for wide sparse outputs."""
    counts: dict[str, Counter] = {k: Counter() for k in keys}
    for row in rows:
        for k, v in row.items():
            c = counts.get(k)
            if c is not None:
                c["<missing>" if v is None else str(v)] += 1
    n = max(1, len(rows))
    out = {}
    for k, c in counts.items():
        present = sum(c.values())
        top = max(c.most_common(1)[0][1] if c else 0, n - present)
        out[k] = 1 - top / n
    return out


_EMPTY = frozenset({"", "None", "nan", "[]", "{}"})


def _is_null(v: Any) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def _column_stats(keys: list[str], rows: list[dict]) -> dict[str, dict[str, Any]]:
    """Per column, over every row, in one pass over the rows' entries: the rows with a value (``nonnull``),
    the empty-looking values among them (``empty``, as the null-rate gate counts them) and up to two distinct
    values (``values``)."""
    stats = {k: {"nonnull": 0, "empty": 0, "values": []} for k in keys}
    for row in rows:
        for k, v in row.items():
            st = stats.get(k)
            if st is None or _is_null(v):
                continue
            st["nonnull"] += 1
            if isinstance(v, str) and v.strip() in _EMPTY:
                st["empty"] += 1
            vals = st["values"]
            if len(vals) < 2 and v not in vals:
                vals.append(v)
    return stats


def _true(v: Any) -> bool:
    return (isinstance(v, bool) and v) or (isinstance(v, (int, float)) and not isinstance(v, bool) and v == 1)


def _precap(
    result: CompiledExtractor,
    keys: list[str],
    outputs: list[dict | None],
    by_key: dict[str, FeatureSpec],
    reserved: frozenset[str] | set[str],
    verdicts: dict[str, str],
    max_null_rate: float,
    per_feature: int,
    produced: set[str],
    seed: int = DEFAULT_SEED,
) -> tuple[list[str], dict[str, tuple], dict[str, float]]:
    """Narrow the output columns before the quality gates when they would not fit ``CELL_BUDGET``:

    1. the cheap gates, on statistics over every decision point: reserved and unplanned names, features the
       CodeValidator flags, constant columns and columns too often empty (as the full gates decide them);
    2. presence flags (one true value where present, missing elsewhere) are decided from those statistics,
       without the table, and ranked by their presence, so a rare flag that varies is never crowded out;
    3. the other columns are ranked by their variation on a seeded sample of ``PRECAP_ROWS`` decision points;
       of each feature at most twice ``per_feature`` and in all ``max(64, CELL_BUDGET / decision points)``
       go on to the full gates (the memory cap: a note says when it binds).

    Returns the columns for the full gates, the flags (column -> (plan, itself, value, rows present)) and the
    flags' variation."""
    ok = [o for o in outputs if o is not None]
    n = max(1, len(ok))
    stats = _column_stats(keys, ok)
    groups: dict[str, list[str]] = {}
    flags: dict[str, tuple] = {}
    flag_var: dict[str, float] = {}
    for k in keys:
        spec, itself = _match_spec(k, by_key)
        st = stats[k]
        verdict = verdicts.get(spec.key) if spec else None
        nonnull, values = st["nonnull"], st["values"]
        flag = 0 < nonnull < n and len(values) == 1 and _true(values[0])
        n_unique = len(values) + (nonnull < n)
        if spec is not None:
            produced.add(spec.key)
        if k in reserved:
            result.dropped[k] = "name reserved by the decision table"
        elif spec is None:
            result.dropped[k] = "not in the accepted feature plan"
        elif verdict in ("hallucinated_field", "plan_mismatch"):
            result.dropped[k] = f"CodeValidator: {verdict}"
        elif n_unique <= 1 or (len(values) <= 1 and not flag):
            result.dropped[k] = "constant across onboarding decision points"
        elif not flag and (n - nonnull + st["empty"]) / n > max_null_rate:
            result.dropped[k] = f"null rate {(n - nonnull + st['empty']) / n:.2f} > {max_null_rate}"
        elif flag:
            flags[k] = (spec, itself, values[0], nonnull)
            flag_var[k] = min(nonnull, n - nonnull) / n
        else:
            groups.setdefault(spec.key, []).append(k)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ok), size=min(PRECAP_ROWS, len(ok)), replace=False)) if ok else []
    sample = [ok[i] for i in idx]
    candidates = [k for cols in groups.values() for k in cols]
    variation = _sample_variation(candidates, sample)
    rank = {k: (-variation[k], k) for k in candidates}
    kept: list[str] = []
    for spec_key, cols in groups.items():
        cols = sorted(cols, key=rank.__getitem__)
        for k in cols[2 * per_feature :]:
            result.dropped[k] = (
                f"one of {len(cols)} columns of {by_key[spec_key].feature_name}: only the {per_feature} that "
                "vary most are kept (max_columns_per_feature; ranked on a sample of "
                f"{len(sample)} decision points)"
            )
        kept += cols[: 2 * per_feature]
    limit = max(64, CELL_BUDGET // max(1, len(outputs)))
    if len(kept) > limit:
        for k in sorted(kept, key=rank.__getitem__)[limit:]:
            result.dropped[k] = (
                f"the extractor's columns other than presence flags are capped at {limit} for {len(outputs)} "
                "decision points (memory: max(64, 20,000,000 / decision points)); this one varies less"
            )
        result.notes.append(
            f"the memory cap bound: of {len(kept)} columns other than presence flags, the {limit} that vary "
            f"most went through the quality gates (max(64, 20,000,000 / {len(outputs)} decision points))"
        )
        kept = sorted(kept, key=rank.__getitem__)[:limit]
    result.notes.append(
        f"the extractor produced {len(keys)} distinct columns on {len(outputs)} decision points: the quality "
        f"gates ran on statistics over every decision point ({len(flags)} presence flag(s) decided from "
        "them) to bound memory"
    )
    keep = set(kept)
    return [k for k in keys if k in keep], flags, flag_var


def _is_flag(values: pd.Series) -> bool:
    """A presence flag: one true value where present (``True`` or ``1``) and missing elsewhere. A column
    whose only present value is false (``0``, ``False``) says that nothing happened anywhere: not a flag."""
    present = values.dropna()
    if not len(present) or len(present) == len(values) or present.nunique() != 1:
        return False
    v = present.iloc[0]
    return (isinstance(v, bool) and v) or (isinstance(v, (int, float)) and not isinstance(v, bool) and v == 1)


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
    max_columns_per_feature: int = MAX_COLUMNS_PER_FEATURE,
    max_columns: int = MAX_EXTRACTOR_COLUMNS,
    repair: bool = True,
    reserved: frozenset[str] | set[str] = frozenset(),
    candidate_ids: frozenset[str] | set[str] = frozenset(),
    context_text: bool = True,
    verify: bool = True,
    verify_samples: int = V.DEFAULT_SAMPLES,
    future: Callable[[DecisionPoint], dict] | None = None,
    goldens: int = V.DEFAULT_GOLDENS,
    seed: int = DEFAULT_SEED,
    steps_key: str = "steps",
    label_of: Callable[[DecisionPoint], Iterable[Any]] | None = None,
    adapter_name: str = "",
    drop_redundant_text: bool = True,
) -> CompiledExtractor:
    """Write, check and quality-gate one extractor for the ``accepted`` features (see the module docstring).
    Output columns named like a column of the decision table (``reserved``) are dropped; ``candidate_ids``
    (the decision's candidates) find the co-usage context.

    With ``verify``, the final code runs again on every decision point and is checked on ``verify_samples``
    coverage-sampled ones (:func:`tabagent.tabschema.verify.verify_extractor`; ``future`` is the adapter's
    :class:`~tabagent.tabschema.verify.FutureCheck`, ``label_of`` its labels), and failing columns are
    dropped. Up to ``goldens`` golden tests are recorded in ``tests``.

    With ``drop_redundant_text``, a text column whose normalized token sequence equals another kept text
    column's on every decision point is dropped (:func:`_drop_duplicate_text`).

    Raises ``ExtractionError`` when no code runs on enough decision points or no column passes the gates, and
    :class:`~tabagent.tabschema.verify.PrefixLeakError` when the adapter's prefixes change with the run's
    future (``adapter_name`` names it in the message).
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
        error = ExtractionError(f"extractor failed after {len(history)} attempts: {attempt.error[:500]}")
        # a rerun asks for new code instead of replaying the attempts that failed
        error.forget = [(extractor, [p for h in history for p in h.prompts])]  # type: ignore[attr-defined]
        raise error

    by_key = {s.key: s for s in accepted}
    notes: list[str] = []

    def judged(att) -> tuple[dict[str, str], dict[str, str]]:
        sample = [r.features for r in att.results if r.ok][:3]
        return validator.judge(accepted, att.code, example, sample, target)

    verdicts: dict[str, str] = {}
    try:
        if validator:
            verdicts, evidence = judged(attempt)
            flagged = _flagged(verdicts, by_key)
            if (
                flagged and repair
            ):  # one repair round with the validator's objections (App. D.1's retry prompt)
                attempt, verdicts, note = _repair(
                    extractor,
                    executor,
                    validator,
                    judged,
                    accepted,
                    prefixes,
                    example,
                    target,
                    attempt,
                    verdicts,
                    evidence,
                    by_key,
                    history,
                )
                notes.append(note)
    except LLMFormatError as e:  # the deterministic checks and the quality gates still apply
        verdicts = {}
        notes.append(f"the CodeValidator's answer was unusable, so no feature was judged by it ({e})")

    outputs = [r.features if r.ok else None for r in attempt.results]
    result = CompiledExtractor(
        with_flatten(attempt.code),
        {},
        attempts=len(history),
        attempt_errors=[h.error[:4000] for h in history if not h.ok],
        notes=notes,
    )
    for h in history:  # the outputs are read: the attempts' per-row results are not held any longer
        h.results = []
    all_keys = list(dict.fromkeys(k for o in outputs if o for k in o))
    keys, produced = all_keys, set()
    flags: dict[str, tuple] = {}
    flag_var: dict[str, float] = {}
    if len(keys) * max(1, len(outputs)) > CELL_BUDGET:  # bound the memory of the gates
        keys, flags, flag_var = _precap(
            result,
            keys,
            outputs,
            by_key,
            reserved,
            verdicts,
            max_null_rate,
            max_columns_per_feature,
            produced,
            seed,
        )
    frame = pd.DataFrame([{k: o[k] for k in keys if k in o} if o else {} for o in outputs], columns=keys)
    ok_rows = np.array([o is not None for o in outputs])
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
        elif q["n_unique"] <= 1 or (values.dropna().nunique() <= 1 and not flag):
            # missing where the code skipped a key is no second value: a 0-or-missing column is constant
            result.dropped[col] = "constant across onboarding decision points"
        elif q["null_rate"] > max_null_rate:
            result.dropped[col] = f"null rate {q['null_rate']:.2f} > {max_null_rate}"
        elif as_text.str.len().mean() > TEXT_CAP:
            result.dropped[col] = f"mean length above {TEXT_CAP} characters"
        else:
            result.features[col] = _resolve_type(spec.feature_type if itself else "computed", values)
            result.specs[col] = spec.to_dict()
    for col, (spec, itself, value, present) in flags.items():  # decided from statistics over every point
        result.quality[col] = {
            "null_rate": 0.0,
            "n_unique": 2,
            "spec": spec.feature_name,
            "validator": verdicts.get(spec.key),
        }
        typed = pd.Series([value] * max(2, min(present, 1000)))
        result.features[col] = _resolve_type(spec.feature_type if itself else "computed", typed)
        result.specs[col] = spec.to_dict()
    if flags:  # in the outputs' column order, as the full gates leave them
        result.features = {k: result.features[k] for k in all_keys if k in result.features}
        result.specs = {k: result.specs[k] for k in all_keys if k in result.specs}
    for key, spec in by_key.items():
        if key not in produced:
            result.dropped.setdefault(
                spec.feature_name, "accepted but not produced by the compiled extractor"
            )
    _cap_columns(result, frame.loc[ok_rows], max_columns_per_feature, max_columns, flag_var)
    if not result.features:
        reasons = "; ".join(f"{k}: {v}" for k, v in result.dropped.items())
        raise ExtractionError(f"every compiled feature failed the quality gates ({reasons[:1000]})")
    if candidate_ids:
        result.co_usage_context = _co_usage_context(result, frame.loc[ok_rows], set(candidate_ids))
    taken = set(reserved) | set(all_keys)
    del frame
    context = None
    if context_text:  # after the co-usage search: the context text is free text, never a candidate list
        context = _add_context_text(result, prefixes, taken, outcome_keys)
    kept = list(result.features)  # later checks read only the kept columns: the rest is not held in memory
    result.outputs = [None if o is None else {k: o[k] for k in kept if k in o} for o in outputs]
    del outputs, attempt
    if verify:
        _verify(
            result,
            decision_points,
            sandbox,
            outcome_keys,
            future,
            verify_samples,
            seed,
            steps_key,
            label_of,
            adapter_name,
        )
    if drop_redundant_text:  # after verification, so a column is never dropped for one verification drops
        _drop_duplicate_text(result, context, seed)
    result.tests = V.make_tests(
        result.code,
        list(result.features),
        decision_points,
        sandbox=sandbox,
        outcome_keys=outcome_keys,
        n=goldens,
        seed=seed,
        steps_key=steps_key,
        verification=result.verification,
        label_of=label_of,
    )
    stats = result.tests["golden_stats"]
    if stats["raised"]:
        result.notes.append(
            f"golden tests: {stats['recorded']} of {stats['requested']} recorded; {stats['raised']} of "
            f"{stats['snapshots']} masked snapshots made the extractor raise "
            f"({stats.get('first_error', '')}), "
            "and verify-head fails a head whose goldens fall short because of that"
        )
    return result


def _verify(
    result: CompiledExtractor,
    decision_points: list[DecisionPoint],
    sandbox: SubprocessSandbox | None,
    outcome_keys: tuple[str, ...],
    future: Callable[[DecisionPoint], dict] | None,
    samples: int,
    seed: int,
    steps_key: str,
    label_of: Callable[[DecisionPoint], Iterable[Any]] | None,
    adapter_name: str,
) -> None:
    """Run the compile-time checks on the final code and drop the columns that fail them; fail closed when
    the adapter's prefixes change with the run's future."""
    rec = V.verify_extractor(
        result.code,
        list(result.features),
        decision_points,
        outputs=result.outputs,
        output_columns=result.llm_columns,
        sandbox=sandbox,
        outcome_keys=outcome_keys,
        future=future,
        samples=samples,
        seed=seed,
        steps_key=steps_key,
        label_of=label_of,
    )
    result.verification = rec
    if rec["future"].get("leaks"):
        raise V.PrefixLeakError(V.leak_message(adapter_name or "used", rec["future"]), rec)
    for col, reason in rec["dropped"].items():
        result.drop(col, reason)
    result.notes += [f"verification: {n}" for n in rec["notes"]]
    if not result.llm_columns:
        reasons = "; ".join(f"{k}: {v}" for k, v in rec["dropped"].items())
        raise ExtractionError(f"every compiled feature failed the verification checks ({reasons[:1000]})")


def _sequence(value: Any) -> tuple[str, ...]:
    r"""The normalized token sequence of a text value: its ``\w+`` tokens, lowercased, in order. Digits and
    word order count; case, punctuation, brackets and quotes do not, so a text and the list of its tokens
    (``"Find my friends."``, ``'["find", "my", "friends"]'``) give the same sequence, while a bag of words
    with counts, n-grams joined by ``_``, or a negation do not."""
    if _is_null(value) or isinstance(value, bool):
        return ()
    return tuple(_TOKEN.findall(str(value).lower()))


def _drop_duplicate_text(
    result: CompiledExtractor, context: tuple[str, list[str]] | None = None, seed: int = DEFAULT_SEED
) -> None:
    """Drop the LLM-written text and categorical columns that duplicate another kept one: an equal normalized
    token sequence (``_sequence``) on every decision point, the appended context text (``context``: its column
    and values) included. Only exact duplicates go; a column that differs from the other on any decision
    point, in any word, number or order, is kept, whatever the overlap. Of a group of duplicates the co-usage
    context is kept first, then the context text, then a categorical form over a text one, then by name. Pairs
    are screened on a seeded sample of ``DUPLICATE_SCREEN_ROWS`` decision points and confirmed on all of
    them."""
    n = len(result.outputs)
    if not n:
        return

    def values(col: str) -> list[Any]:
        if context is not None and col == context[0]:
            return [None if o is None else t for o, t in zip(result.outputs, context[1], strict=True)]
        return [None if o is None else o.get(col) for o in result.outputs]

    cols = [c for c, t in result.features.items() if t in ("text", "cat")]
    cols = [c for c in cols if c in result.llm_columns or (context is not None and c == context[0])]
    if len(cols) < 2:
        return

    def priority(c: str) -> tuple:
        return (
            c != result.co_usage_context,
            context is None or c != context[0],
            result.features[c] != "cat",
            c,
        )

    rng = np.random.default_rng(seed)
    screen = np.sort(rng.choice(n, size=min(DUPLICATE_SCREEN_ROWS, n), replace=False))
    full: dict[str, list[tuple[str, ...]]] = {}

    def sequences(c: str) -> list[tuple[str, ...]]:
        if c not in full:
            full[c] = [_sequence(v) for v in values(c)]
        return full[c]

    sampled = {c: [_sequence(values(c)[i]) for i in screen] for c in cols} if n > len(screen) else None
    keepers: list[str] = []
    dropped = 0
    for col in sorted(cols, key=priority):
        twin = None
        for k in keepers:
            if sampled is not None and sampled[col] != sampled[k]:
                continue
            if sequences(col) == sequences(k):
                twin = k
                break
        if twin is None:
            keepers.append(col)
            continue
        result.drop(
            col,
            f"redundant with {twin}: the same words and numbers in the same order on every one of the {n} "
            "onboarding decision points (case and punctuation aside)",
        )
        dropped += 1
    if dropped:
        result.notes.append(
            f"{dropped} text column(s) duplicated another kept text column word for word on every decision "
            "point and were dropped (redundant with ...; see compile.json)"
        )


def _add_context_text(
    result: CompiledExtractor, prefixes: list[dict], taken: set[str], skip: Iterable[str] = ()
) -> tuple[str, list[str]] | None:
    """Append the context-text code to the extractor when its text varies across decision points; returns its
    column and values (None when it is not added)."""
    column = CONTEXT_TEXT_COLUMN if CONTEXT_TEXT_COLUMN not in taken else f"tabagent_{CONTEXT_TEXT_COLUMN}"
    texts = context_texts(prefixes, column, skip)
    values = pd.Series(texts)
    if values.nunique() <= 1:
        result.notes.append("no context text added: the prefixes hold no text that varies across decisions")
        return None
    result.code += context_text_code(column, skip)
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
    return column, texts


def _co_usage_context(result: CompiledExtractor, frame: pd.DataFrame, candidates: set[str]) -> str | None:
    """The kept text or categorical column that best lists earlier candidates: at least ``CO_USAGE_SHARE`` of
    its items are candidate ids, on at least ``CO_USAGE_ROWS`` of the rows, with at most
    ``CO_USAGE_MAX_ITEMS`` of the candidates per row (a list of every offered candidate tells nothing). Its
    statistics are recorded in ``quality``."""
    best, best_share = None, 0.0
    for col, typ in result.features.items():
        if typ == "num" or col not in frame:  # a presence flag decided from statistics lists nothing
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

    def run(self, prefixes: list[dict], results_out: list[ExecResult] | None = None) -> list[dict | None]:
        """The features of each prefix, or None where the extractor failed. ``results_out``, when given, is
        extended with each prefix's ``ExecResult`` (the error and its kind where the extractor failed)."""
        errors: list[str] = []
        per_prefix: list[ExecResult] = []
        if not self.trust_extractor:
            out = []
            for r in self.sandbox.run_batch(self.code, prefixes):
                out.append(r.features if r.ok else None)
                per_prefix.append(r)
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
                    per_prefix.append(ExecResult(True, out[-1]))
                except Exception as e:  # the extractor is arbitrary code: any failure means no features
                    out.append(None)
                    errors.append(f"{type(e).__name__}: {e}")
                    per_prefix.append(ExecResult(False, error=errors[-1], kind="error"))
        if results_out is not None:
            results_out.extend(per_prefix)
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
