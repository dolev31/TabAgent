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

**Per-candidate features** (the release's addition). A ranking or multi-label decision table has one row per
candidate of a decision, and the extractor may define a second function,
``extract_candidate_features(trajectory_data, candidate)``, whose columns are computed for each (decision,
candidate) row: the words the request shares with the candidate's description, whether earlier steps mention
it, and so on. ``candidate`` is the candidate as the decision offers it (``trajectory.candidate_dict``: id,
name, description, argument hints and catalog entry), never its label. Its columns go through the same checks
and gates as the prefix's, over the (decision, candidate) rows, and a name the prefix's function or the
decision table already uses is dropped. When the decision has candidates, prefix columns that index the
candidates by position (``candidate_3_overlap``; :func:`positional_columns` decides from their values, not
their names alone) are asked for again in the per-candidate function by the repair round, and dropped once
that function computes the feature; until then they are kept, with a note. An extractor without the function
compiles exactly as before.

**One form per text feature.** Of the text columns of one planned feature, one that only restates another's
words in a shorter derived form on every row is dropped (:func:`_drop_derived_text`): a bag of words or a set
of its tokens, as a JSON list or dict, or its words sorted and de-duplicated. The longer form, and an ordered
sequence next to a set, are never dropped.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import re
import traceback
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
    SPEC_FIELDS,
    CodeExecutor,
    CodeExtractor,
    CodeValidator,
    FeatureSpec,
    LLMContentFiltered,
    LLMFormatError,
    candidate_samples,
    extract_with_retries,
    generate_retry_prompt,
    normalize_name,
    pick_examples,
)
from tabagent.tabschema.checks import defines_candidate_function
from tabagent.tabschema.fitted import context_items
from tabagent.tabschema.sandbox import (
    CANDIDATE_CALLS,
    MAX_CANDIDATE_ERRORS,
    ExecResult,
    SubprocessSandbox,
)
from tabagent.tabschema.trajectory import DecisionPoint, candidate_dict, merge_examples

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
#: Words that name a decision's candidates. An indexed column family (``candidate_3_overlap``,
#: ``api2_method``, ``available_api_1_path``) with one of them anywhere in its name (a plural ``s``
#: allowed) may list the candidates by position when the family's size follows the candidates' count;
#: without one, only when its texts name the candidates in their order (:func:`positional_columns`).
CANDIDATE_WORDS = frozenset(
    {
        "candidate",
        "cand",
        "api",
        "tool",
        "app",
        "application",
        "item",
        "option",
        "choice",
        "arm",
        "model",
        "skill",
        "agent",
        "subagent",
        "function",
        "action",
        "endpoint",
        "route",
        "element",
        "doc",
        "document",
    }
)
#: Words of a name about the run's history (the tool called before, the last app used, a step of the run):
#: such a column lists earlier calls or steps, not the decision's candidates, and is never positional.
HISTORY_WORDS = frozenset(
    {
        "prev",
        "previous",
        "last",
        "history",
        "ago",
        "recent",
        "past",
        "earlier",
        "step",
        "steps",
        "turn",
        "turns",
    }
)
_NUMBER = re.compile(r"\d+")
#: A family's evidence: its size follows the candidates' count (exactly on at least this share of the decision
#: points, or, for a family emitted only where a value is set, within the candidates on this share and up to
#: the largest decision's last candidate), or its texts name the candidates in their order on this share of
#: its values; over at least ``POSITIONAL_MIN_POINTS`` points or values.
POSITIONAL_SHARE, POSITIONAL_MIN_POINTS = 0.8, 3
POSITIONAL_REASON = (
    "indexes candidates by position; extract_candidate_features computes the feature for each candidate"
)
#: The global dict a CodeExtractor answer may hold for accepted features it does not compute, ``{feature name:
#: "merged into <feature>" or "declined: <reason>"}``; the repair round asks for it (:func:`not_extracted`).
NOT_EXTRACTED = "NOT_EXTRACTED"
NOT_PRODUCED = "accepted but not produced by the compiled extractor"
TAKEN_REASON = (
    "name taken: extract_features_for_classification or the decision table already has a column of that name"
)


class ExtractionError(RuntimeError):
    """The extractor could not be made to run on the onboarding trajectories."""


def _families(keys: Iterable[str]) -> dict[str, dict[str, dict[int, list[str]]]]:
    """The indexed columns of ``keys``: by enumeration (the name before one of its numbers:
    ``candidate_3_overlap`` and ``candidate_3_name`` are both in ``candidate_``), then by column template
    (the name with that number replaced by ``#``), then by index. A name with several numbers is in an
    enumeration for each; a name with a history word (``HISTORY_WORDS``) is in none."""
    out: dict[str, dict[str, dict[int, list[str]]]] = {}
    for k in keys:
        low = str(k).lower()
        if HISTORY_WORDS & set(re.findall(r"[a-z]+", low)):
            continue
        for m in _NUMBER.finditer(low):
            if m.start() == 0:
                continue
            template = f"{low[: m.start()]}#{low[m.end() :]}"
            group = out.setdefault(low[: m.start()], {}).setdefault(template, {})
            group.setdefault(int(m.group()), []).append(k)
    return out


def _names_candidates(names: Iterable[str]) -> bool:
    """Whether a name holds a word for a candidate (``CANDIDATE_WORDS``, a plural ``s`` allowed)."""
    words = {w for n in names for w in re.findall(r"[a-z]+", n)}
    return any(w in CANDIDATE_WORDS or w.removesuffix("s") in CANDIDATE_WORDS for w in words)


def _follows_count(indices: dict[int, list[str]], rows: list, candidates: list) -> bool:
    """Whether a family's size follows the number of the decision's candidates, which must vary across the
    decision points (a constant count cannot tell a fixed-size list from the catalog): the indices present
    at a point are exactly as many as its candidates; or, for a family emitted only where a value is set,
    they lie within the point's candidates and reach the last candidate of the largest decision (a fixed
    list, or the first few of a longer enumeration, never does)."""
    if len(indices) < 2:
        return False
    present = []
    for row, cands in zip(rows, candidates, strict=True):
        if row is None or not cands:
            continue
        have = [i for i, cols in indices.items() if any(not _is_null(row.get(c)) for c in cols)]
        if have:
            present.append((have, len(cands)))
    if len(present) < POSITIONAL_MIN_POINTS or len({n for _, n in present}) < 2:
        return False
    if np.mean([len(have) == n for have, n in present]) >= POSITIONAL_SHARE:
        return True
    offset = 1 if min(min(have) for have, _ in present) >= 1 else 0
    fits = np.mean([max(have) - offset < n for have, n in present]) >= POSITIONAL_SHARE
    return fits and max(max(have) for have, _ in present) - offset == max(n for _, n in present) - 1


def _lines_up(indices: dict[int, list[str]], rows: list, candidates: list) -> bool:
    """Whether a family's text values name the decision's candidates in their order: the value of index
    ``i`` is (or holds) the id or name of candidate ``i`` (0-based) or ``i - 1`` (1-based)."""
    best = 0.0
    for offset in (0, 1):
        hits = total = 0
        for row, cands in zip(rows, candidates, strict=True):
            if row is None or not cands:
                continue
            for i, cols in indices.items():
                if not 0 <= i - offset < len(cands):
                    continue
                names = {str(cands[i - offset].get(k) or "").lower() for k in ("id", "name")} - {""}
                for c in cols:
                    v = row.get(c)
                    if isinstance(v, str) and v.strip():
                        total += 1
                        low = v.strip().lower()
                        hits += low in names or any(len(n) >= 3 and n in low for n in names)
        if total >= POSITIONAL_MIN_POINTS:
            best = max(best, hits / total)
    return best >= POSITIONAL_SHARE


def positional_columns(rows: list[dict | None], candidates: list[list[dict]]) -> list[str]:
    """The prefix columns, among the keys of ``rows`` (one output per decision point, aligned with the points'
    ``candidates``), that list the decision's candidates by position. An indexed enumeration (no history word
    in its names, ``HISTORY_WORDS``) is positional, all its columns, when one of its columns' texts name the
    candidates in their order (:func:`_lines_up`), or, when its names hold a word for a candidate
    (``CANDIDATE_WORDS``), when its size follows the candidates' count across decision points of different
    sizes (:func:`_follows_count`). A list of earlier calls (``prev_tool_1``, ``tool_1_ago``), a fixed-size
    list (``item_1_price`` to ``item_3_price``, even with three candidates at every point) or the first few
    of a longer enumeration with no candidate named is never one."""
    keys = list(dict.fromkeys(k for r in rows if r for k in r))
    out: set[str] = set()
    for prefix, templates in _families(keys).items():
        indices: dict[int, list[str]] = {}
        for by_index in templates.values():
            for i, cols in by_index.items():
                indices.setdefault(i, []).extend(cols)
        if len(indices) < 2:
            continue
        counted = _names_candidates([prefix, *templates]) and _follows_count(indices, rows, candidates)
        if counted or any(len(t) > 1 and _lines_up(t, rows, candidates) for t in templates.values()):
            out.update(c for cols in indices.values() for c in cols)
    return [k for k in keys if k in out]


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


#: The step fields the appended texts read, in this order, and the characters they keep of a longer text: its
#: first quarter and its latest three quarters.
STEP_TEXT_KEYS = ("content", "text", "thoughts", "intent", "data")
TEXT_LIMIT = 4000

#: Appended to every compiled extractor (``context_text=True``): the decision's context as plain text, not
#: written by the LLM. It reads the prefix's task-level text fields (never the adapter's outcome fields,
#: should a prefix carry one) and the texts of its steps (system prompts, identical across decisions, are
#: skipped) and keeps the start and the latest part of a long history. Being part of ``extract.py``, it runs
#: identically at training and at inference. It gives the card a context text for the context-candidate
#: similarity features, TabSynth's paraphrases and BM25.
CONTEXT_TEXT_CODE = """

# --- appended by TabAgent (not LLM-written): the decision's context as text ---
def _tabagent_context_text(trajectory_data, limit=__TABAGENT_LIMIT__, skip=__TABAGENT_SKIP__):
    parts = []
    for key, value in trajectory_data.items():
        if key != __TABAGENT_STEPS__ and key not in skip and isinstance(value, str) and value.strip():
            parts.append(value.strip())
    for step in trajectory_data.get(__TABAGENT_STEPS__) or []:
        if not isinstance(step, dict) or step.get("name") == "system":
            continue
        for key in __TABAGENT_TEXT_KEYS__:
            value = step.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    text = "\\n".join(parts)
    if len(text) <= limit:
        return text
    return text[: limit // 4] + "\\n...\\n" + text[-(limit - limit // 4) :]


if isinstance(features, dict):
    features[__TABAGENT_COLUMN__] = _tabagent_context_text(trajectory)
"""
CONTEXT_TEXT_COLUMN = "context_text"

#: Appended to every compiled extractor (``last_step_text=True``) after the context text: the text of the
#: prefix's last step, not written by the LLM. ``context_text`` concatenates every step and keeps the start
#: and the latest part of a long history, so the step that handed control to the decision can be diluted by
#: long payloads; this column holds that step alone. It is the last step with text (steps that are not dicts
#: or are named ``system`` skipped; one without text is passed for the nearest earlier one that has some):
#: its ``STEP_TEXT_KEYS`` strings in that order (never a field the adapter names as an outcome field), joined
#: by newlines and clipped as ``context_text`` is ("" for a prefix without such a step). Being part of
#: ``extract.py``, it runs identically at training and at inference.
LAST_STEP_TEXT_CODE = """

# --- appended by TabAgent (not LLM-written): the text of the decision's last step ---
def _tabagent_last_step_text(trajectory_data, limit=__TABAGENT_LIMIT__, skip=__TABAGENT_SKIP__):
    for step in reversed(trajectory_data.get(__TABAGENT_STEPS__) or []):
        if not isinstance(step, dict) or step.get("name") == "system":
            continue
        parts = []
        for key in __TABAGENT_TEXT_KEYS__:
            value = step.get(key)
            if key not in skip and isinstance(value, str) and value.strip():
                parts.append(value.strip())
        if parts:
            text = "\\n".join(parts)
            if len(text) <= limit:
                return text
            return text[: limit // 4] + "\\n...\\n" + text[-(limit - limit // 4) :]
    return ""


if isinstance(features, dict):
    features[__TABAGENT_COLUMN__] = _tabagent_last_step_text(trajectory)
"""
LAST_STEP_TEXT_COLUMN = "last_step_text"
_APPENDED = {CONTEXT_TEXT_COLUMN: CONTEXT_TEXT_CODE, LAST_STEP_TEXT_COLUMN: LAST_STEP_TEXT_CODE}


def appended_code(kind: str, column: str, skip: Iterable[str] = (), steps_key: str = "steps") -> str:
    """The appended code of ``kind`` (``context_text`` or ``last_step_text``), writing ``column``, reading the
    adapter's step list ``steps_key`` and never the fields in ``skip`` (the adapter's outcome fields)."""
    keys = repr(tuple(dict.fromkeys(str(k) for k in skip)))
    fields = "(" + ", ".join(json.dumps(k) for k in STEP_TEXT_KEYS) + ")"
    return (
        _APPENDED[kind]
        .replace("__TABAGENT_COLUMN__", repr(column))
        .replace("__TABAGENT_SKIP__", keys)
        .replace("__TABAGENT_STEPS__", json.dumps(steps_key))
        .replace("__TABAGENT_TEXT_KEYS__", fields)
        .replace("__TABAGENT_LIMIT__", str(TEXT_LIMIT))
    )


def context_text_code(column: str, skip: Iterable[str] = (), steps_key: str = "steps") -> str:
    """The appended context-text code (:func:`appended_code`)."""
    return appended_code(CONTEXT_TEXT_COLUMN, column, skip, steps_key)


def last_step_text_code(column: str, skip: Iterable[str] = (), steps_key: str = "steps") -> str:
    """The appended last-step-text code (:func:`appended_code`)."""
    return appended_code(LAST_STEP_TEXT_COLUMN, column, skip, steps_key)


def appended_texts(
    prefixes: list[dict],
    column: str,
    kind: str = CONTEXT_TEXT_COLUMN,
    skip: Iterable[str] = (),
    steps_key: str = "steps",
) -> list[str]:
    """The values the appended code of ``kind`` computes for ``prefixes`` (run here, on JSON copies)."""
    code = compile(appended_code(kind, column, skip, steps_key), f"<{kind}>", "exec")
    out = []
    for p in prefixes:
        ns: dict[str, Any] = {"features": {}, "trajectory": json.loads(json.dumps(p, default=str))}
        exec(code, ns)  # noqa: S102 - the library's own fixed code
        out.append(ns["features"][column])
    return out


def context_texts(
    prefixes: list[dict],
    column: str = CONTEXT_TEXT_COLUMN,
    skip: Iterable[str] = (),
    steps_key: str = "steps",
) -> list[str]:
    """The context texts the appended code computes for ``prefixes``."""
    return appended_texts(prefixes, column, CONTEXT_TEXT_COLUMN, skip, steps_key)


def last_step_texts(
    prefixes: list[dict],
    column: str = LAST_STEP_TEXT_COLUMN,
    skip: Iterable[str] = (),
    steps_key: str = "steps",
) -> list[str]:
    """The last-step texts the appended code computes for ``prefixes``."""
    return appended_texts(prefixes, column, LAST_STEP_TEXT_COLUMN, skip, steps_key)


@dataclass
class CompiledExtractor:
    """The compiled extractor: its code, the output columns with their frozen types, the feature plan each
    column implements, what was dropped and why, and per-column quality statistics.

    ``verification`` is the record of the release's compile-time checks (``tabagent.tabschema.verify``),
    ``tests`` the golden tests saved as ``extractor_tests.json``, and ``outputs`` the code's values of the
    kept columns on each onboarding decision point, in order (None where it failed), before the context text
    was appended.

    ``not_produced`` maps each accepted feature that no output column implements to the reason: merged into
    another feature or declined, as the code's ``NOT_EXTRACTED`` says, else not produced.

    ``per_candidate`` names the columns of the per-candidate function (``candidate_columns``, the kept ones),
    and ``candidate_outputs`` holds their values on each onboarding decision point: per candidate of the
    decision, in the adapter's order, a dict (None where the call failed); None for a decision point without
    candidates or when the code has no per-candidate function."""

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
    per_candidate: set[str] = field(default_factory=set)
    candidate_outputs: list[list[dict | None] | None] = field(default_factory=list, repr=False, compare=False)
    not_produced: dict[str, str] = field(default_factory=dict)  # accepted feature -> why no column has it

    @property
    def sha256(self) -> str:
        """SHA-256 of the code (recorded in the card, checked when a head is loaded)."""
        return hashlib.sha256(self.code.encode()).hexdigest()

    @property
    def llm_columns(self) -> list[str]:
        """The kept columns the LLM wrote (every column but the ones TabAgent appended)."""
        return [c for c in self.features if not self.specs.get(c, {}).get("added_by")]

    @property
    def candidate_columns(self) -> list[str]:
        """The kept columns of the per-candidate function."""
        return [c for c in self.features if c in self.per_candidate]

    @property
    def context_columns(self) -> list[str]:
        """The kept columns computed on the prefix alone."""
        return [c for c in self.features if c not in self.per_candidate]

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


def not_extracted(code: str) -> dict[str, str]:
    """The ``NOT_EXTRACTED`` literal dict at the top level of ``code`` (feature name -> why the code does not
    compute it), {} when it has none."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == NOT_EXTRACTED for t in node.targets
        ):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                return {}
            return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}
    return {}


def declared_reason(text: str) -> str:
    """A ``NOT_EXTRACTED`` reason as recorded: ``merged into <feature>`` or ``declined: <reason>``."""
    t = " ".join(str(text).split())[:300]
    if t.lower().startswith("merged into"):
        return "merged into" + t[len("merged into") :]
    if t.lower().startswith("declined"):
        return "declined: " + t[len("declined") :].lstrip(" :")
    return f"declined: {t}"


def _not_produced(attempt, by_key: dict[str, FeatureSpec]) -> dict[str, FeatureSpec]:
    """The accepted features no output column of ``attempt`` implements (either function's) and its code does
    not account for in ``NOT_EXTRACTED``."""
    keys: set[str] = set()
    for r in attempt.results:
        if not r.ok:
            continue
        keys.update(r.features or ())
        for v in r.candidates or ():
            keys.update(v or ())
    produced = {spec.key for k in keys if (spec := _match_spec(k, by_key)[0]) is not None}
    declared = {normalize_name(n) for n in not_extracted(attempt.code)}
    return {k: s for k, s in by_key.items() if k not in produced and k not in declared}


def _positional(
    attempt, by_key: dict[str, FeatureSpec], candidates: list[list[dict]]
) -> tuple[dict[str, list[str]], set[str]]:
    """The prefix columns of ``attempt`` that index candidates by position (:func:`positional_columns`): by
    the planned feature they implement (their template, ``candidate_#_text``, when none), those whose
    feature the per-candidate function does not compute yet; and, as a set, those whose feature it computes
    (to drop)."""
    rows = [r.features if r.ok else None for r in attempt.results]
    cols = positional_columns(rows, candidates)
    if not cols:
        return {}, set()
    per = {k for r in attempt.results if r.ok for v in r.candidates or () for k in v or ()}
    computed = {spec.key for k in per if (spec := _match_spec(k, by_key)[0]) is not None}
    pending: dict[str, list[str]] = {}
    replaced: set[str] = set()
    for col in cols:
        spec, _ = _match_spec(col, by_key)
        if spec is not None and spec.key in computed:
            replaced.add(col)
        else:
            # a column of no planned feature: by its template (``candidate_#_text``)
            pending.setdefault(spec.feature_name if spec else _NUMBER.sub("#", col, count=1), []).append(col)
    return pending, replaced


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
    guarded=None,
    candidates=None,
    examples=None,
):
    """Ask the CodeExtractor once more, in one call, with the error of the retry prompt listing the
    CodeValidator's objections, the accepted features the code does not produce (to add, or to list in
    ``NOT_EXTRACTED`` as merged into another feature or declined, with the reason) and, with ``guarded``
    (the decision's candidates, when the positional guard is on), the features whose columns index candidates
    by position (to be computed by the per-candidate function); the prompt shows the example candidates
    ``examples``. Keep the new code only if it runs and fewer accepted features are flagged, missing or
    positional. Returns (attempt, verdicts, note)."""
    flagged = _flagged(verdicts, by_key)
    moved = _positional(attempt, by_key, guarded)[0] if guarded is not None else {}
    missing = _not_produced(attempt, by_key)
    parts = []
    if flagged:
        objections = "\n".join(
            f"- {by_key[k].feature_name}: {v}. {evidence.get(k, '')}".strip()
            for k, v in sorted(flagged.items())
        )
        parts.append(
            "The code runs, but the CodeValidator found that these features do not follow their plan or read "
            f"fields the trajectories do not have:\n{objections}"
        )
    if moved:
        listed = "\n".join(
            f"- {name}: {', '.join(cols[:8])}" + (f" and {len(cols) - 8} more" if len(cols) > 8 else "")
            for name, cols in sorted(moved.items())
        )
        parts.append(
            "These features index the candidates by position, so their columns do not describe the "
            f"candidate each row scores:\n{listed}\nCompute each of them in "
            "extract_candidate_features(trajectory_data, candidate), for the candidate it is given, and do "
            "not enumerate the candidates in extract_features_for_classification."
        )
    if missing:
        plans = [{k: v for k, v in s.to_dict().items() if k in SPEC_FIELDS} for s in missing.values()]
        parts.append(
            "The code does not produce these accepted features:\n"
            + json.dumps(plans, indent=2, ensure_ascii=False)
            + f"\nAdd each of them, under its name. If one would only repeat a feature the code produces, or "
            "cannot be computed from the trajectory, do not add it: list it instead in a global dict "
            f'{NOT_EXTRACTED} = {{"<feature_name>": "merged into <the feature that carries the same '
            'information>"}, or {"<feature_name>": "declined: <a one-line reason>"}.'
        )
    error = "\n".join(parts) + "\nFix these features; keep the others unchanged."
    n = len(flagged) + len(moved) + len(missing)
    more = {"candidates": examples} if candidates is not None and examples else {}
    try:
        code = extractor(accepted, example, target, generate_retry_prompt(1, error, attempt.code), **more)
        new = executor.run(code, prefixes) if candidates is None else executor.run(code, prefixes, candidates)
    except (LLMFormatError, LLMContentFiltered) as e:
        return (attempt, verdicts, f"the repair round for {n} flagged feature(s) failed ({type(e).__name__})")
    history.append(new)
    if not new.ok:
        return (
            attempt,
            verdicts,
            f"the repair round's code did not run ({new.error[:200]}); kept the first code",
        )
    if new.candidate_error and not attempt.candidate_error:
        return (
            attempt,
            verdicts,
            "the repair round's per-candidate function failed too often "
            f"({new.candidate_error.splitlines()[0][:200]}); kept the first code",
        )
    new_verdicts = judged(new)[0] if validator else {}
    still = _flagged(new_verdicts, by_key)
    still_moved = _positional(new, by_key, guarded)[0] if guarded is not None else {}
    still_missing = _not_produced(new, by_key)
    if len(still) + len(still_moved) + len(still_missing) < n:
        fixed = sorted(
            {by_key[k].feature_name for k in flagged if k not in still}
            | {m for m in moved if m not in still_moved}
            | {s.feature_name for k, s in missing.items() if k not in still_missing}
        )
        return (
            new,
            new_verdicts,
            f"the repair round fixed {len(fixed)} of {n} flagged feature(s): {', '.join(fixed)}",
        )
    return (
        attempt,
        verdicts,
        f"the repair round did not reduce the {n} flagged feature(s); kept the first code",
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
    positional: frozenset[str] | set[str] = frozenset(),
    taken: frozenset[str] | set[str] = frozenset(),
) -> tuple[list[str], dict[str, tuple], dict[str, float]]:
    """Narrow the output columns before the quality gates when they would not fit ``CELL_BUDGET``:

    1. the cheap gates, on statistics over every decision point: reserved, taken (``_gate``), positional and
       unplanned names, features the CodeValidator flags, constant columns and columns too often empty (as
       the full gates decide them);
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
        if k in taken:  # as in _gate
            result.dropped[f"{k} (per-candidate)"] = TAKEN_REASON
            continue
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
        elif k in positional:
            result.dropped[k] = POSITIONAL_REASON
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


def _gate(
    result: CompiledExtractor,
    rows: list[dict | None],
    by_key: dict[str, FeatureSpec],
    verdicts: dict[str, str],
    reserved: frozenset[str] | set[str],
    max_null_rate: float,
    per_feature: int,
    produced: set[str],
    seed: int,
    positional: frozenset[str] | set[str] = frozenset(),
    taken: frozenset[str] | set[str] = frozenset(),
) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    """The quality gates of step 4 (see the module docstring) on the output ``rows`` of one function: the
    prefix's, one row per decision point, or the per-candidate function's, one row per (decision, candidate)
    (None where the code failed). A column named in ``reserved`` (the decision table's), ``taken`` (the
    prefix function's, for per-candidate columns; recorded as ``<name> (per-candidate)``, the prefix's
    column standing) or ``positional`` is dropped first. Kept columns are typed
    into ``result``. Returns the frame of the rows the code ran on, the presence flags' variation and every
    output key in order."""
    all_keys = list(dict.fromkeys(k for o in rows if o for k in o))
    keys = all_keys
    flags: dict[str, tuple] = {}
    flag_var: dict[str, float] = {}
    if len(keys) * max(1, len(rows)) > CELL_BUDGET:  # bound the memory of the gates
        keys, flags, flag_var = _precap(
            result,
            keys,
            rows,
            by_key,
            reserved,
            verdicts,
            max_null_rate,
            per_feature,
            produced,
            seed,
            positional,
            taken,
        )
    frame = pd.DataFrame([{k: o[k] for k in keys if k in o} if o else {} for o in rows], columns=keys)
    ok_rows = np.array([o is not None for o in rows], dtype=bool)
    before = list(result.features)
    for col in frame.columns:
        if col in taken:  # the prefix's column of that name stands; this one is recorded under its own key
            result.dropped[f"{col} (per-candidate)"] = TAKEN_REASON
            continue
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
        elif col in positional:
            result.dropped[col] = POSITIONAL_REASON
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
        order = [c for c in before if c in result.features] + [k for k in all_keys if k in result.features]
        result.features = {k: result.features[k] for k in dict.fromkeys(order)}
        result.specs = {k: result.specs[k] for k in result.features}
    return frame.loc[ok_rows], flag_var, all_keys


def _candidate_lists(
    decision_points: list[DecisionPoint], candidates_of: Callable[[DecisionPoint], Iterable] | None
) -> list[list[dict]] | None:
    """The candidate dicts of each decision point (``trajectory.candidate_dict``), or None when no decision
    point has any (a single-class decision, or no candidates function). A candidate offered at many decision
    points is one dict (they are read, never changed), so large catalog entries are held once."""
    if candidates_of is None:
        return None
    out = candidate_lists([candidates_of(dp) or () for dp in decision_points])
    return out if any(out) else None


def candidate_lists(per_decision: Iterable[Iterable]) -> list[list[dict]]:
    """``trajectory.candidate_dict`` of each candidate of each decision, one dict per distinct candidate."""
    memo: dict[Any, dict] = {}
    out = []
    for cands in per_decision:
        row = []
        for c in cands:
            d = memo.get(c)
            if d is None:
                d = memo[c] = candidate_dict(c)
            row.append(d)
        out.append(row)
    return out


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
    candidates_of: Callable[[DecisionPoint], Iterable] | None = None,
    last_step_text: bool = True,
    positional_guard: bool = True,
) -> CompiledExtractor:
    """Write, check and quality-gate one extractor for the ``accepted`` features (see the module docstring).
    Output columns named like a column of the decision table (``reserved``) are dropped; ``candidate_ids``
    (the decision's candidates) find the co-usage context.

    ``candidates_of`` gives a decision point's candidates (the adapter's ``candidates``; ``Candidate``
    objects). With them, the CodeExtractor may write the per-candidate function, which runs on each decision
    point's candidates and whose columns are gated over the (decision, candidate) rows. With
    ``positional_guard``, prefix columns that index candidates by position are asked for again as
    per-candidate features by the repair round, and dropped once the per-candidate function computes them.

    With ``verify``, the final code runs again on every decision point and is checked on ``verify_samples``
    coverage-sampled ones (:func:`tabagent.tabschema.verify.verify_extractor`; ``future`` is the adapter's
    :class:`~tabagent.tabschema.verify.FutureCheck`, ``label_of`` its labels), and failing columns are
    dropped. Up to ``goldens`` golden tests are recorded in ``tests``.

    With ``drop_redundant_text``, a text column whose normalized token sequence equals another kept text
    column's on every decision point is dropped (:func:`_drop_duplicate_text`), and so is one that restates
    another column of its feature in a derived form (:func:`_drop_derived_text`).

    With ``context_text`` and ``last_step_text``, the fixed context-text and last-step-text functions are
    appended to the extractor (``CONTEXT_TEXT_CODE``, ``LAST_STEP_TEXT_CODE``) when their texts vary.

    Raises ``ExtractionError`` when no code runs on enough decision points or no column passes the gates, and
    :class:`~tabagent.tabschema.verify.PrefixLeakError` when the adapter's prefixes change with the run's
    future (``adapter_name`` names it in the message).
    """
    if not accepted:
        raise ExtractionError("no accepted features to compile")
    prefixes = [dp.prefix for dp in decision_points]
    example = merge_examples(prefixes)
    candidates = _candidate_lists(decision_points, candidates_of)
    guard = positional_guard and candidates is not None
    # the example candidates the prompts show: three of the first decision point's that has some, picked from
    # the seed and its id, never by the adapter's order or a label
    examples = None
    if candidates is not None:
        first = next(i for i, c in enumerate(candidates) if c)
        examples = pick_examples(candidates[first], seed, decision_points[first].id)
    keys = [dp.id for dp in decision_points]
    executor = CodeExecutor(sandbox, known_paths, outcome_keys, max_failure_rate=1 - min_success)
    attempt, history = extract_with_retries(
        extractor,
        executor,
        accepted,
        prefixes,
        example,
        target,
        max_retries,
        candidates=candidates,
        examples=examples,
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
        if candidates is None:
            return validator.judge(accepted, att.code, example, sample, target)
        shown = candidate_samples(att.results, candidates, keys, seed)
        return validator.judge(accepted, att.code, example, sample, target, examples, shown)

    verdicts: dict[str, str] = {}
    try:
        evidence: dict[str, str] = {}
        if validator:
            verdicts, evidence = judged(attempt)
        flagged = _flagged(verdicts, by_key)
        moved = _positional(attempt, by_key, candidates)[0] if guard else {}
        if (flagged or moved or _not_produced(attempt, by_key)) and repair:
            # one repair round with the validator's objections (App. D.1's retry prompt), the accepted
            # features the code does not produce and the positional features
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
                candidates if guard else None,
                candidates,
                examples,
            )
            notes.append(note)
    except LLMFormatError as e:  # the deterministic checks and the quality gates still apply
        verdicts = {}
        notes.append(f"the CodeValidator's answer was unusable, so no feature was judged by it ({e})")

    outputs = [r.features if r.ok else None for r in attempt.results]
    pending, positional = _positional(attempt, by_key, candidates) if guard else ({}, set())
    per_candidate = None
    if candidates is not None and defines_candidate_function(attempt.code):
        per_candidate = [
            (list(r.candidates) if r.ok and r.candidates is not None else [None] * len(c))
            for r, c in zip(attempt.results, candidates, strict=True)
        ]
    if attempt.candidate_error:
        notes.append(f"the per-candidate function still failed too often: {attempt.candidate_error[:300]}")
    result = CompiledExtractor(
        with_flatten(attempt.code),
        {},
        attempts=len(history),
        attempt_errors=[h.error[:4000] for h in history if not h.ok],
        notes=notes,
    )
    for h in history:  # the outputs are read: the attempts' per-row results are not held any longer
        h.results = []
    produced: set[str] = set()
    frame, flag_var, all_keys = _gate(
        result,
        outputs,
        by_key,
        verdicts,
        reserved,
        max_null_rate,
        max_columns_per_feature,
        produced,
        seed,
        positional=positional,
    )
    if positional:
        result.notes.append(
            f"{len(positional)} column(s) of extract_features_for_classification indexed candidates by "
            "position and were dropped: extract_candidate_features computes their features (compile.json)"
        )
    if pending:
        result.notes.append(
            f"{sum(map(len, pending.values()))} column(s) of extract_features_for_classification index "
            f"candidates by position ({', '.join(sorted(pending))}); kept, since extract_candidate_features "
            "does not compute their features"
        )
    cand_keys: list[str] = []
    known = dict(flag_var)
    if per_candidate is not None:
        rows = [v for per in per_candidate for v in per]
        done = sum(v is not None for v in rows)
        cand_keys = list(dict.fromkeys(k for v in rows if v for k in v))
        if done < min_success * len(rows):
            for k in cand_keys:
                spec, _ = _match_spec(k, by_key)
                if spec is not None:
                    produced.add(spec.key)
                result.dropped[f"{k} (per-candidate)" if k in all_keys else k] = (
                    f"the per-candidate function ran on {done} of {len(rows)} (decision, candidate) rows, "
                    f"under compile_min_success ({min_success:.0%})"
                )
        else:
            before = set(result.features)
            cand_frame, cand_flags, _ = _gate(
                result,
                rows,
                by_key,
                verdicts,
                reserved,
                max_null_rate,
                max_columns_per_feature,
                produced,
                seed,
                taken=set(all_keys),
            )
            result.per_candidate = set(result.features) - before
            known.update(cand_flags)
            known.update({c: _variation(cand_frame[c]) for c in result.per_candidate if c in cand_frame})
            for c in result.per_candidate:
                result.quality[c]["per_candidate"] = True
                result.quality[c]["success_rate"] = round(done / max(1, len(rows)), 4)
            del cand_frame
        del rows
    declared = {normalize_name(k): declared_reason(v) for k, v in not_extracted(result.code).items()}
    for key, spec in by_key.items():
        if key not in produced:
            reason = result.dropped.setdefault(spec.feature_name, declared.get(key, NOT_PRODUCED))
            result.not_produced[spec.feature_name] = reason
    _cap_columns(result, frame, max_columns_per_feature, max_columns, known)
    if not result.features:
        reasons = "; ".join(f"{k}: {v}" for k, v in result.dropped.items())
        raise ExtractionError(f"every compiled feature failed the quality gates ({reasons[:1000]})")
    if candidate_ids:
        result.co_usage_context = _co_usage_context(result, frame, set(candidate_ids))
    taken = set(reserved) | set(all_keys) | set(cand_keys)
    del frame
    appended: dict[str, list[str]] = {}  # the appended texts' columns and values, in order
    # after the co-usage search: the appended texts are free text, never a candidate list
    for kind, on in ((CONTEXT_TEXT_COLUMN, context_text), (LAST_STEP_TEXT_COLUMN, last_step_text)):
        added = (
            _add_appended(result, kind, prefixes, taken | set(appended), outcome_keys, steps_key)
            if on
            else None
        )
        if added is not None:
            appended[added[0]] = added[1]
    kept = result.context_columns  # later checks read only the kept columns: the rest is not held in memory
    result.outputs = [None if o is None else {k: o[k] for k in kept if k in o} for o in outputs]
    kept_c = result.candidate_columns
    if kept_c:
        result.candidate_outputs = [
            None if per is None else [None if v is None else {k: v[k] for k in kept_c if k in v} for v in per]
            for per in per_candidate
        ]
    del outputs, attempt, per_candidate
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
            candidates,
        )
    if drop_redundant_text:  # after verification, so a column is never dropped for one verification drops
        _drop_duplicate_text(result, appended, seed)
        if result.candidate_columns:
            _drop_duplicate_text(result, None, seed, candidate_rows(result))
        _drop_derived_text(result, seed=seed)
    kept_c = result.candidate_columns
    result.tests = V.make_tests(
        result.code,
        result.context_columns,
        decision_points,
        sandbox=sandbox,
        outcome_keys=outcome_keys,
        n=goldens,
        seed=seed,
        steps_key=steps_key,
        verification=result.verification,
        label_of=label_of,
        **({"candidates": candidates, "candidate_columns": kept_c} if kept_c else {}),
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


def candidate_rows(result: CompiledExtractor) -> list[dict | None]:
    """The per-candidate outputs of a compiled extractor as one list of (decision, candidate) rows."""
    return [v for per in result.candidate_outputs if per is not None for v in per]


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
    candidates: list[list[dict]] | None = None,
) -> None:
    """Run the compile-time checks on the final code and drop the columns that fail them; fail closed when
    the adapter's prefixes change with the run's future."""
    per = result.candidate_columns
    rec = V.verify_extractor(
        result.code,
        result.context_columns,
        decision_points,
        outputs=result.outputs,
        output_columns=[c for c in result.llm_columns if c not in result.per_candidate],
        sandbox=sandbox,
        outcome_keys=outcome_keys,
        future=future,
        samples=samples,
        seed=seed,
        steps_key=steps_key,
        label_of=label_of,
        **(
            {
                "candidates": candidates,
                "candidate_columns": per,
                "candidate_outputs": result.candidate_outputs,
            }
            if per
            else {}
        ),
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
    result: CompiledExtractor,
    context: tuple[str, list[str]] | dict[str, list[str]] | None = None,
    seed: int = DEFAULT_SEED,
    rows: list[dict | None] | None = None,
) -> None:
    """Drop the LLM-written text and categorical columns that duplicate another kept one: an equal normalized
    token sequence (``_sequence``) on every decision point, the appended texts (``context``: the context
    text's column and values, or ``{column: values}`` of every appended text, in order) included. Only exact
    duplicates go; a column that differs from the other on any decision point, in any word, number or order,
    is kept, whatever the overlap. Of a group of duplicates the co-usage context is kept first, then the
    appended texts, then a categorical form over a text one, then by name. Pairs are screened on a seeded
    sample of ``DUPLICATE_SCREEN_ROWS`` decision points and confirmed on all of them.

    With ``rows`` (the per-candidate outputs, one per (decision, candidate) row), the per-candidate columns
    are compared on those rows instead; prefix columns and per-candidate columns are never compared."""
    outputs = result.outputs if rows is None else rows
    n = len(outputs)
    if not n:
        return
    appended = dict([context] if isinstance(context, tuple) else (context or {}))
    unit = "onboarding decision points" if rows is None else "onboarding (decision, candidate) rows"

    def values(col: str) -> list[Any]:
        if col in appended:
            return [None if o is None else t for o, t in zip(outputs, appended[col], strict=True)]
        return [None if o is None else o.get(col) for o in outputs]

    scope = result.per_candidate if rows is not None else set(result.features) - result.per_candidate
    cols = [c for c, t in result.features.items() if t in ("text", "cat") and c in scope]
    cols = [c for c in cols if c in result.llm_columns or c in appended]
    if len(cols) < 2:
        return
    order = list(appended)

    def priority(c: str) -> tuple:
        return (
            c != result.co_usage_context,
            order.index(c) if c in appended else len(order),
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
            f"{unit} (case and punctuation aside)",
        )
        dropped += 1
    if dropped:
        where = "decision point" if rows is None else "(decision, candidate) row"
        result.notes.append(
            f"{dropped} text column(s) duplicated another kept text column word for word on every {where} "
            "and were dropped (redundant with ...; see compile.json)"
        )


_WORD = re.compile(r"[^\W_]+")


def _form(value: Any) -> tuple[str, list[str]] | None:
    """A text value's form for :func:`_drop_derived_text`: ``text`` (plain text), ``sequence`` (a JSON list,
    or a string holding one, in an order of its own or with repeats) or ``set`` (a JSON dict, a bag of words,
    or a sorted list without repeats), with its lowercased word tokens (``_`` splits words, so ``get_the`` is
    two; a dict contributes its keys, a list its string items, nested ones included). None when missing."""
    if _is_null(value) or isinstance(value, bool):
        return None
    parsed: Any = None
    if isinstance(value, str) and value.strip()[:1] in ("[", "{"):
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = None
    if not isinstance(parsed, (list, dict)):
        return "text", _WORD.findall(str(value).lower())
    words: list[str] = []
    stack = [parsed]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            words += _WORD.findall(" ".join(map(str, v)).lower())
        elif isinstance(v, list):
            stack.extend(reversed(v))
        elif isinstance(v, str):
            words += _WORD.findall(v.lower())
    is_set = isinstance(parsed, dict) or words == sorted(set(words))
    return ("set" if is_set else "sequence"), words


def _derived(form: tuple[str, list[str]] | None, base: tuple[str, list[str]] | None) -> bool:
    """Whether a value restates ``base`` in a shorter derived form: nothing at all; a JSON list or dict of a
    subset of its words, never longer than it and never an ordered sequence next to a set (or next to text
    whose words are sorted and de-duplicated); or plain text that is its words sorted and de-duplicated.
    Plain text is never derived from a structured form."""
    if form is None or not form[1]:
        return True
    if base is None or len(form[1]) > len(base[1]):
        return False
    (kind, words), (base_kind, base_words) = form, base
    if kind == "text":
        return base_kind == "text" and words == sorted(set(base_words))
    if kind == "sequence" and (base_kind == "set" or base_words == sorted(set(base_words))):
        return False
    return set(words) <= set(base_words)


def _order_rank(forms: list[tuple[str, list[str]] | None]) -> int:
    """Which of one feature's text columns is compared first by :func:`_drop_derived_text`: plain text with an
    order of its own (0), plain text that is sorted and de-duplicated on every row (1), an ordered sequence
    (2), a set (3)."""
    seen = [f for f in forms if f is not None and f[1]]
    kinds = {f[0] for f in seen}
    if kinds <= {"text"}:
        return 1 if seen and all(f[1] == sorted(set(f[1])) for f in seen) else 0
    return 3 if kinds == {"set"} else 2


def _drop_derived_text(result: CompiledExtractor, seed: int = DEFAULT_SEED) -> None:
    """Of the LLM-written text and categorical columns of one planned feature, drop each that restates
    another of them in a shorter derived form on every row (:func:`_derived`: a bag of words or a subset of
    its tokens as a JSON list or dict, or its words sorted and de-duplicated): the CodeExtractor is asked to
    emit a text once, since the head represents text itself. Only within one feature's columns and on every
    row (every decision point, or every (decision, candidate) row for the per-candidate columns), never across
    features. The longer form, and an ordered sequence next to a set, are always kept: columns are compared
    plain text with an order of its own first, then sorted plain text, sequences and sets
    (:func:`_order_rank`), longer before shorter, then by name; a dropped column names the plainest, shortest
    column it restates. The appended texts are not LLM-written and are left alone."""
    groups: dict[tuple[str, bool], list[str]] = {}
    for col, t in result.features.items():
        if t in ("text", "cat") and col in result.llm_columns:
            name = str((result.specs.get(col) or {}).get("feature_name") or col)
            groups.setdefault((name, col in result.per_candidate), []).append(col)
    dropped = 0
    for (_, per), cols in groups.items():
        if len(cols) < 2:
            continue
        rows = candidate_rows(result) if per else result.outputs
        n = len(rows)
        if not n:
            continue
        forms = {c: [None if o is None else _form(o.get(c)) for o in rows] for c in cols}
        mean = {c: np.mean([len(f[1]) for f in forms[c] if f is not None] or [0]) for c in cols}
        rank = {c: _order_rank(forms[c]) for c in cols}
        rng = np.random.default_rng(seed)
        screen = np.sort(rng.choice(n, size=min(DUPLICATE_SCREEN_ROWS, n), replace=False))
        keepers: list[str] = []
        for col in sorted(cols, key=lambda c: (rank[c], -mean[c], c)):
            base = next(
                (
                    k
                    for k in sorted(keepers, key=lambda k: (rank[k], mean[k], k))
                    if all(_derived(forms[col][i], forms[k][i]) for i in screen)
                    and all(_derived(a, b) for a, b in zip(forms[col], forms[k], strict=True))
                ),
                None,
            )
            if base is None:
                keepers.append(col)
                continue
            result.drop(col, f"a derived form of {base}; the head represents text itself")
            dropped += 1
    if dropped:
        result.notes.append(
            f"{dropped} text column(s) restated another column of their feature in a derived form (its "
            "tokens, bag of words or n-grams) and were dropped (a derived form of ...; see compile.json)"
        )


def _add_appended(
    result: CompiledExtractor,
    kind: str,
    prefixes: list[dict],
    taken: set[str],
    skip: Iterable[str] = (),
    steps_key: str = "steps",
) -> tuple[str, list[str]] | None:
    """Append the code of ``kind`` (``context_text`` or ``last_step_text``) to the extractor when its text
    varies across decision points, as column ``kind`` (``tabagent_<kind>`` when that name is taken); returns
    its column and values (None when it is not added)."""
    column = kind if kind not in taken else f"tabagent_{kind}"
    texts = appended_texts(prefixes, column, kind, skip, steps_key)
    values = pd.Series(texts)
    if values.nunique() <= 1:
        what = "context text" if kind == CONTEXT_TEXT_COLUMN else "last-step text"
        result.notes.append(f"no {what} added: the prefixes hold no such text that varies across decisions")
        return None
    result.code += appended_code(kind, column, skip, steps_key)
    result.features[column] = "text"
    result.specs[column] = {
        "feature_name": column,
        "feature_type": "text",
        "description": (
            "the decision's context as text: the prefix's task-level text fields and step texts"
            if kind == CONTEXT_TEXT_COLUMN
            else "the text of the decision's last step: the latest step with text, its content, text, "
            "thoughts, intent and data"
        )
        + " (added by TabAgent, not LLM-written)",
        "added_by": "tabagent",
        **({"appended": kind} if kind != CONTEXT_TEXT_COLUMN else {}),
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


#: The per-candidate calls of the in-process runner: the sandbox's own code (``sandbox.CANDIDATE_CALLS``), so
#: that in-process and sandboxed runs treat candidates alike.
_IN_PROCESS: dict[str, Any] = {}
exec(compile(CANDIDATE_CALLS, "<candidate_calls>", "exec"), _IN_PROCESS)  # noqa: S102 - the library's own code
_candidate_calls = _IN_PROCESS["_tabagent_candidate_calls"]


class ExtractorRunner:
    """Apply a compiled extractor to decision-point prefixes.

    By default the code runs in the sandbox. ``trust_extractor=True`` runs it in-process, which is fast but
    gives the code the caller's privileges; use it only for an extractor you have reviewed. In-process, the
    code sees a JSON copy of the prefix and its features are read back through JSON, exactly as in the
    sandbox, so both modes give identical features and the caller's prefix is never changed. A prefix on
    which the code fails yields ``None`` (no features); failures are counted in ``n_failed`` and reported
    with a ``RuntimeWarning`` naming the first error.

    An extractor that defines the per-candidate function (``has_candidate_function``) also gives the
    features of each candidate of a decision (:meth:`run_items`): the code runs once on the prefix, then the
    function once per candidate, in one sandbox call per decision. A candidate whose call fails gets None,
    counted in ``n_candidate_failed``.
    """

    def __init__(
        self, code: str, trust_extractor: bool = False, sandbox: SubprocessSandbox | None = None
    ) -> None:
        self.code, self.trust_extractor = code, trust_extractor
        self.sandbox = sandbox or SubprocessSandbox()
        self._compiled = compile(code, "<extract.py>", "exec") if trust_extractor else None
        self.has_candidate_function = defines_candidate_function(code)
        self.n_runs = 0
        self.n_failed = 0
        self.n_candidate_runs = 0
        self.n_candidate_failed = 0

    def run(
        self,
        prefixes: list[dict],
        results_out: list[ExecResult] | None = None,
        candidates: list[list[dict] | None] | None = None,
    ) -> list[dict | None]:
        """The features of each prefix, or None where the extractor failed. ``results_out``, when given, is
        extended with each prefix's ``ExecResult`` (the error and its kind where the extractor failed).
        ``candidates`` (aligned with ``prefixes``: candidate dicts, ``trajectory.candidate_dict``) has the
        per-candidate function run too, when the code defines it; its features are in the ``ExecResult``s
        (``candidates``; see :meth:`run_items`)."""
        errors: list[str] = []
        per_prefix: list[ExecResult] = []
        if candidates is not None and not self.has_candidate_function:
            candidates = None  # nothing to call: the run is the single-function run, unchanged
        if not self.trust_extractor:
            out = []
            batch = (
                self.sandbox.run_batch(self.code, prefixes)
                if candidates is None
                else self.sandbox.run_batch(self.code, prefixes, candidates=candidates)
            )
            for r in batch:
                out.append(r.features if r.ok else None)
                per_prefix.append(r)
                if not r.ok:
                    errors.append(str(r.error))
        else:
            out = []
            for n, p in enumerate(prefixes):
                trajectory = json.loads(json.dumps(p, default=str))
                ns: dict[str, Any] = {"__name__": "__main__", "trajectory": trajectory}
                try:
                    exec(self._compiled, ns)  # noqa: S102 - trusted, reviewed extractor
                    feats = ns.get("features")
                    if not isinstance(feats, dict):
                        raise TypeError("the code must leave a dict named `features` in its global scope")
                    out.append(json.loads(json.dumps(feats)))
                    result = ExecResult(True, out[-1])
                    if candidates is not None and candidates[n] is not None:
                        cands = json.loads(json.dumps(candidates[n], default=str))
                        per, cand_errors = _candidate_calls(
                            ns,
                            trajectory,
                            cands,
                            json.dumps,
                            traceback.format_exc,
                            lambda: False,
                            MAX_CANDIDATE_ERRORS,
                        )
                        result.candidates = None if per is None else json.loads(json.dumps(per))
                        result.candidate_errors = cand_errors
                    per_prefix.append(result)
                except Exception as e:  # the extractor is arbitrary code: any failure means no features
                    out.append(None)
                    errors.append(f"{type(e).__name__}: {e}")
                    per_prefix.append(ExecResult(False, error=errors[-1], kind="error"))
        if results_out is not None:
            results_out.extend(per_prefix)
        self.n_runs += len(prefixes)
        self.n_failed += len(errors)
        failed_candidates, cand_error = 0, None
        if candidates is not None:
            for r, cands in zip(per_prefix, candidates, strict=True):
                n = len(cands or ())
                self.n_candidate_runs += n
                bad = n if r.candidates is None else sum(v is None for v in r.candidates)
                failed_candidates += bad
                if bad and cand_error is None and r.candidate_errors:
                    cand_error = r.candidate_errors[0]
            self.n_candidate_failed += failed_candidates
        if errors:
            last = errors[0].strip().splitlines()[-1] if errors[0].strip() else "unknown error"
            warnings.warn(
                f"the TabSchema extractor failed on {len(errors)} of {len(prefixes)} decision point(s); "
                f"their features are empty. First error: {last}",
                RuntimeWarning,
                stacklevel=2,
            )
        elif cand_error is not None:
            last = cand_error.strip().splitlines()[-1] if cand_error.strip() else "unknown error"
            warnings.warn(
                f"the TabSchema extractor's per-candidate function failed on {failed_candidates} "
                f"candidate(s); their features are empty. First error: {last}",
                RuntimeWarning,
                stacklevel=2,
            )
        return out

    def run_items(
        self,
        prefixes: list[dict],
        candidates: list[list[dict] | None],
        results_out: list[ExecResult] | None = None,
    ) -> tuple[list[dict | None], list[list[dict | None] | None]]:
        """The features of each prefix and of each of its candidates (candidate dicts, aligned with
        ``prefixes``): ``(per prefix, per prefix the per-candidate features in candidate order)``. A candidate
        gets None where its call failed or the prefix failed; a prefix's candidate list is None when the code
        defines no per-candidate function. ``results_out`` as in :meth:`run`."""
        results: list[ExecResult] = []
        out = self.run(prefixes, results, candidates)
        if results_out is not None:
            results_out.extend(results)
        per: list[list[dict | None] | None] = []
        for r, cands in zip(results, candidates, strict=True):
            if not self.has_candidate_function or cands is None:
                per.append(None)
            elif r.candidates is None:
                per.append([None] * len(cands))
            else:
                per.append(list(r.candidates))
        return out, per
