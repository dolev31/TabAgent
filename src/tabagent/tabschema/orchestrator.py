"""TabSchema, the FeatureOrchestrator (paper Sec. 3.1; App. D).

``TabSchema.discover`` turns an agent's successful trajectories into a feature card for one decision
component:

1. **sample** decision points (the component's occurrences in successful runs), one per task first;
2. per sampled decision point, in parallel: **FeatureAnalyzer** proposes features; **CodeExtractor** and
   **CodeExecutor** make them executable on that trace (three retries); **CodeValidator** keeps the
   features the executed code implements faithfully from real fields;
3. **FeatureAggregator** merges the proposals of all decision points;
4. the **Relevance**, **Generality** and **Impact and Uniqueness** judges score them concurrently, and the
   **MetaJudge** accepts, conditionally accepts or rejects each one;
5. **compile**: one reusable extractor for the accepted features, run and quality-checked on every
   onboarding decision point;
6. the **card**: extractor features (state), candidate name/description/argument hints (schema,
   Sec. 3.3), and dependency statistics (priors, context-candidate similarity) that the card fits on the
   training rows of each fold (Sec. 4.3). For a ranking or multi-label decision the recipe search may add,
   for each numeric per-candidate feature and dependency statistic, its rank and its gap to the best within
   the decision (``card.with_relative``), which a pointwise head cannot see otherwise; the discovered card
   never holds them.

For a ranking or multi-label decision, the extractor may also define a per-candidate function,
``extract_candidate_features(trajectory_data, candidate)``, whose features describe one candidate in the
context of the prefix (the release's addition; ``compile``): the CodeExtractor and CodeValidator prompts get a
TabAgent section on it, and its columns enter the card as schema features computed per (decision, candidate)
row (``source: extractor_candidate``).

The release adds three checks the paper does not have, between compiling and the card: the compile step's
**verification** of the values (determinism, outcome and future invariance, robustness; ``verify``) with
golden tests saved as ``extractor_tests.json``; the **OutputAuditor**, an LLM judge of the values on a sample
of decision points (``audit``); and a **usefulness** estimate of each feature by grouped cross-validation on
the onboarding rows (``usefulness``). Each can be turned off in ``TabSchemaConfig``.

Discovery runs once, on the onboarding trajectories it is given. The compiled extractor is a stateless
function of one trajectory prefix, so applying it to held-out decisions leaks nothing; every label-derived
statistic is fitted by the card inside ``cross_validate``'s training folds. Running discovery itself inside
each fold means calling ``discover`` on that fold's trajectories.

A decision point whose analysis fails (an unusable answer, a content filter, code that never runs) is skipped
and recorded in ``analysis.json``; the run goes on with the others. A batch of a whole-run stage whose answer
is unusable is asked again in halves (``agents.in_halves``). A provider error that is transient is retried
with backoff by the stage (``llm.RetryPolicy``); one that persists stops the run (``LLMUnavailable``).

``run.json`` in the run directory follows the run as it progresses: each stage's status, timing, progress,
output, drops, failures and LLM cost (``tabagent.tabschema.runlog``; ``tabagent runs show``). A run that stops
(a spent budget, a provider that stays down, an interruption) or fails (no feature accepted, an extractor that
cannot be compiled) leaves a partial ``report.md``, ``run.json`` and every paid answer in its run directory:
rerunning with that directory resumes, every answer it holds costs nothing, and the resumed run produces the
card an uninterrupted run would have.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import logging
import math
import numbers
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabagent.defaults import DEFAULT_SEED
from tabagent.table import DecisionSpec, DecisionTable
from tabagent.tabschema import prompts
from tabagent.tabschema import verify as V
from tabagent.tabschema.agents import (
    REJECTING,
    STOP,
    VALID,
    VERDICTS,
    CodeExecutor,
    CodeExtractor,
    CodeValidator,
    FeatureAggregator,
    FeatureAnalyzer,
    FeatureSpec,
    LLMFormatError,
    candidate_samples,
    extract_with_retries,
    forget_cause,
    normalize_name,
    premerge,
    shutdown_interrupted,
)
from tabagent.tabschema.audit import OutputAuditor, run_audit
from tabagent.tabschema.card import DEPENDENCY_FEATURES, Feature, FeatureCard
from tabagent.tabschema.compile import (
    CONTEXT_TEXT_COLUMN,
    CompiledExtractor,
    ExtractionError,
    ExtractorRunner,
    appended_texts,
    candidate_lists,
    compile_extractor,
)
from tabagent.tabschema.judges import DECISIONS, Judge, MetaJudge, judge_table, run_judges
from tabagent.tabschema.llm import LLM, BudgetExceeded, CostMeter, LLMUnavailable, auto_llm, status_code
from tabagent.tabschema.runlog import RunLog, drop_reason, stopped_status
from tabagent.tabschema.runstore import RunStore, atomic_write
from tabagent.tabschema.sampling import STRATEGIES, coverage, sample_decision_points
from tabagent.tabschema.sandbox import SubprocessSandbox
from tabagent.tabschema.trajectory import (
    CANDIDATE_FUNCTION,
    DecisionPoint,
    GenericJsonAdapter,
    Trajectory,
    candidate_dict,
    union_paths,
)
from tabagent.tabschema.usefulness import (
    ALL_LLM,
    DEFAULT_MAX_DECISIONS,
    DEFAULT_MAX_ROWS,
    MIN_DECISIONS,
    feature_groups,
    feature_usefulness,
)

logger = logging.getLogger(__name__)

GROUP, DECISION, CANDIDATE, LABEL = "task", "decision_id", "candidate", "chosen"
#: Columns of the decision table and of the card that an extractor column must not shadow.
RESERVED = frozenset(
    {
        GROUP,
        DECISION,
        CANDIDATE,
        LABEL,
        "label",
        "candidate_name",
        "candidate_description",
        "candidate_args",
        "context_candidate_lex",
        "context_candidate_sim",
        "candidate_prior",
    }
)


@dataclass
class TabSchemaConfig:
    """Settings of a TabSchema discovery run.

    Parameters
    ----------
    model: the LLM (default: the paper's GPT-5.4 on OpenAI, or the provider whose key is in the environment).
    n_analyze: decision points the FeatureAnalyzer reads.
    sample_seed: seed of that sample.
    sample_strategy: how the sample is drawn (see ``tabagent.tabschema.sampling``): ``coverage`` (one per
        task first, covering the sources, positions in the run, prefix lengths and labels of the
        onboarding data) or ``tasks`` (one per task, the first decision of each run).
    max_code_retries: CodeExecutor retries with the error fed back (three, App. D.1).
    json_retries: re-asks when an answer is not parseable JSON.
    accept: MetaJudge outcomes compiled into the card: accepted and conditionally accepted features by
        default. ``("accept",)`` keeps only accepted ones. On the next-tool-call decision of public τ²-bench
        logs, compiling the conditional features too raised head P@R by 0.066 and 0.029 in two discovery runs
        (10 and 20 analyzed decision points); adding the rejected ones changed it by under 0.004 more.
    compile_min_success: share of onboarding decision points the compiled extractor must run on.
    max_null_rate: extractor columns emptier than this are dropped.
    max_columns_per_feature, max_extractor_columns: caps on the columns one planned feature expands into
        (code that returns a dict per row, such as word counts, gets one column per key) and on the
        extractor's columns in all; past them the columns that vary most across decision points are kept.
        On many decision points memory caps the columns other than presence flags further, to
        ``max(64, 20,000,000 / decision points)`` (512 at 39,000 points, 200 at 100,000); presence flags are
        decided from statistics over every point and only ever capped by these two; a compile note says when
        the memory cap binds.
    compile_repair: when the CodeValidator flags accepted features in the compiled code, ask the CodeExtractor
        once more with its objections (the paper's retry prompt), and keep the new code only if it runs and
        fewer features are flagged.
    drop_redundant_text: drop a compiled text column that duplicates another kept text column (the appended
        context text included) word for word on every onboarding decision point: the same lowercased tokens,
        numbers and order, so a text and the list of its tokens collapse into one. Overlap, containment or
        the same words in another order never drop a column. Of duplicates a categorical form is kept over a
        text one. Within the columns of one planned feature, a column that restates another in a derived form
        on every row (its tokens, bag of words or n-grams as a JSON list or dict, or the same words in another
        order) is dropped too: the head represents text itself.
    dependency_features: dependency statistics added to the card. ``co_usage`` is added when an extractor
        column lists earlier candidates (``co_usage_context``).
    pair_embeddings: add the e5 context-candidate similarity; None adds it when the embed extra is installed.
    workers: parallel analyzer calls.
    timeout_s: sandbox time limit per trajectory.
    max_cost_usd: LLM budget; needs a price for the model in ``prices``.
    prices: ``{model: (USD per 1M input tokens, USD per 1M output tokens)}``.
    max_prompt_chars: a decision-point prefix longer than this (as JSON) reaches the FeatureAnalyzer with its
        longest strings clipped.
    stage_models: a model per stage, overriding ``model``: keys are stage roles (``analyzer``, ``extractor``,
        ``validator``, ``aggregator``, ``meta_judge``, ``judge_relevance``, ...) or ``judges`` for all three
        judges. For example ``{"extractor": "azure/gpt-6-sol"}`` when a provider's content filter stops one
        model's code answers. Every model needs a price in ``prices`` when ``max_cost_usd`` is set.
    fallback_model: asked once, for any stage, when the provider's content filter stops an answer.
    context_text: append to the extractor a fixed, non-LLM function that returns the decision's context as
        text (``context_text``), so every card has a context text for the context-candidate similarity
        features, TabSynth and the BM25 baseline, even when the LLM's features are all flags and counts.
    last_step_text: append to the extractor a fixed, non-LLM function that returns the text of the prefix's
        last step (``last_step_text``): the step that handed control to the decision, which the context text
        can dilute among long payloads. It is a context text too (TabSynth, the context-candidate
        similarities).
    positional_guard: for a decision with candidates, ask the CodeExtractor to compute in its per-candidate
        function the features it wrote as columns that index the candidates by position
        (``compile.positional_columns``, decided from the columns' values), and drop those columns once it
        does.
    relative_features: let the recipe search (``auto.select``) try, for a ranking or multi-label decision,
        each numeric per-candidate feature's and dependency statistic's rank and gap to the best within the
        decision (``card.with_relative``) as a challenger, kept only when it scores strictly better in the
        inner validation. The discovered card never holds them, so the usefulness estimate and a
        ``TabAgent`` built on the card directly do not use them; the saved head records the choice.

    The release's checks beyond the paper (see ``docs/onboarding.md``, "How the generated code is verified"):

    verify: run the compiled extractor, sandboxed, on ``verify_samples`` decision points and drop the columns
        that are nondeterministic or change when outcome fields or the run's future change; report how it
        copes with short and incomplete prefixes (``tabagent.tabschema.verify``).
    verify_samples: decision points of those checks (coverage-sampled).
    verify_adapter: check, before any LLM call, that the adapter's prefixes do not change with the run's
        future (``verify.FutureCheck``); a prefix that does stops discovery with ``PrefixLeakError``. Turning
        it off (or ``verify``) skips the check with a warning; an adapter whose prefixes legitimately hold the
        decision step's own input (its prompt) declares those keys in ``decision_input_keys`` instead.
    goldens: golden tests recorded in ``extractor_tests.json`` and saved with the head (0: none).
    audit: ask the OutputAuditor (stage ``auditor``) whether the extracted values are correct on
        ``audit_samples`` decision points, and drop features judged incorrect on more than
        ``audit_max_incorrect`` of the points (at least 3) where they were judged
        (``tabagent.tabschema.audit``).
    usefulness: estimate each feature's contribution by drop-column grouped cross-validation on the onboarding
        rows, on at most ``usefulness_max_decisions`` decision points (``tabagent.tabschema.usefulness``); a
        report only, unless ``prune_harmful`` also drops the features whose removal clearly helps.

    Operations:

    llm_retries: retries of an LLM call that fails transiently (a rate limit, a dropped connection, a server
        error, an empty answer), with exponential backoff that honors the provider's ``Retry-After``; past
        them the run stops with ``LLMUnavailable``, resumable from its run directory. A call that times out
        is retried once, then handled like an answer cut at the output limit.
    request_timeout_s: seconds one LLM request may take, for the backends ``auto_llm`` builds; None scales it
        with the stage's output limit: ``max(300, 120 + max_tokens / 40)``, room for the longest answer at 40
        tokens per second (520 s for the per-decision-point stages' 16,000 tokens, 920 s for the whole-run
        stages' 32,000).

    Every field is checked when the config is made (types and ranges), with a message naming the field;
    :meth:`from_dict` builds one from a mapping (a JSON or YAML file) and names unknown keys.
    """

    model: str | None = None
    n_analyze: int = 20
    sample_seed: int = DEFAULT_SEED
    sample_strategy: str = "coverage"
    max_code_retries: int = 3
    json_retries: int = 1
    accept: tuple[str, ...] = ("accept", "conditional")
    compile_min_success: float = 0.95
    max_null_rate: float = 0.5
    max_columns_per_feature: int = 64
    max_extractor_columns: int = 512
    compile_repair: bool = True
    drop_redundant_text: bool = True
    dependency_features: tuple[str, ...] = (
        "candidate_prior",
        "context_candidate_lex",
        "context_candidate_sim",
        "co_usage",
    )
    pair_embeddings: bool | None = None
    workers: int = 8
    timeout_s: float = 30
    max_cost_usd: float | None = None
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)
    max_prompt_chars: int = 400_000
    stage_models: dict[str, str] = field(default_factory=dict)
    fallback_model: str | None = None
    context_text: bool = True
    last_step_text: bool = True
    positional_guard: bool = True
    relative_features: bool = True
    verify: bool = True
    verify_adapter: bool = True
    verify_samples: int = V.DEFAULT_SAMPLES
    goldens: int = V.DEFAULT_GOLDENS
    audit: bool = True
    audit_samples: int = 12
    audit_max_incorrect: float = 0.25
    usefulness: bool = True
    prune_harmful: bool = False
    usefulness_max_decisions: int = DEFAULT_MAX_DECISIONS
    usefulness_max_rows: int = DEFAULT_MAX_ROWS
    llm_retries: int = 5
    request_timeout_s: float | None = None

    def __post_init__(self) -> None:
        _validate(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TabSchemaConfig:
        """A config from a mapping (a JSON or YAML file's content), checked like any config; an unknown key is
        an error that suggests the closest field name."""
        if not isinstance(d, dict):
            raise ValueError(
                f"a TabSchemaConfig is a mapping of field names to values, not {type(d).__name__}"
            )
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - fields)
        if unknown:
            hints = []
            for k in unknown:
                close = difflib.get_close_matches(str(k), sorted(fields), n=1)
                hints.append(f"{k!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
            raise ValueError(f"unknown TabSchemaConfig field(s): {', '.join(hints)}")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        """The config as JSON-able values (``from_dict`` reads it back)."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(self).items()}


STAGE_ROLES = (
    "analyzer",
    "extractor",
    "validator",
    "aggregator",
    "judge_relevance",
    "judge_generality",
    "judge_impact",
    "meta_judge",
    "auditor",
)


def _timeout(cfg: TabSchemaConfig) -> dict[str, float]:
    """``auto_llm``'s timeout argument for ``cfg`` (none when the request timeout is left to the backend)."""
    return {} if cfg.request_timeout_s is None else {"timeout": cfg.request_timeout_s}


def _is_bool(v: Any) -> bool:
    return isinstance(v, (bool, np.bool_))


def _is_int(v: Any) -> bool:
    return isinstance(v, numbers.Integral) and not _is_bool(v)


def _is_number(v: Any) -> bool:
    return isinstance(v, numbers.Real) and not _is_bool(v) and math.isfinite(float(v))


def _validate(cfg: TabSchemaConfig) -> None:
    """Check every field of ``cfg`` (types and ranges), normalizing lists to tuples; ``ValueError`` names the
    field, the expected value and the one given."""

    def bad(name: str, expected: str) -> ValueError:
        return ValueError(f"TabSchemaConfig.{name} must be {expected}, not {getattr(cfg, name)!r}")

    ints = {
        "n_analyze": 1,
        "max_code_retries": 0,
        "json_retries": 0,
        "max_columns_per_feature": 1,
        "max_extractor_columns": 1,
        "workers": 1,
        "max_prompt_chars": 1_000,
        "verify_samples": 1,
        "goldens": 0,
        "audit_samples": 1,
        "usefulness_max_decisions": 2,
        "usefulness_max_rows": 1,
        "llm_retries": 0,
    }
    for name, low in ints.items():
        v = getattr(cfg, name)
        if not _is_int(v) or v < low:
            raise bad(name, f"an integer of at least {low}")
        setattr(cfg, name, int(v))  # numpy integers become Python ones
    if not _is_int(cfg.sample_seed):
        raise bad("sample_seed", "an integer")
    cfg.sample_seed = int(cfg.sample_seed)
    for name in (
        "compile_repair",
        "drop_redundant_text",
        "context_text",
        "last_step_text",
        "positional_guard",
        "relative_features",
        "verify",
        "verify_adapter",
        "audit",
        "usefulness",
        "prune_harmful",
    ):
        if not _is_bool(getattr(cfg, name)):
            raise bad(name, "True or False")
        setattr(cfg, name, bool(getattr(cfg, name)))
    if cfg.pair_embeddings is not None:
        if not _is_bool(cfg.pair_embeddings):
            raise bad("pair_embeddings", "True, False or None")
        cfg.pair_embeddings = bool(cfg.pair_embeddings)
    for name in ("model", "fallback_model"):
        v = getattr(cfg, name)
        if v is not None and (not isinstance(v, str) or not v.strip()):
            raise bad(name, "a model name or None")
    if cfg.sample_strategy not in STRATEGIES:
        raise bad("sample_strategy", f"one of {STRATEGIES}")
    for name, allowed in (("accept", DECISIONS), ("dependency_features", DEPENDENCY_FEATURES)):
        v = getattr(cfg, name)
        if isinstance(v, str) or not isinstance(v, (tuple, list, set, frozenset)):
            raise bad(name, f"a tuple of {allowed}, such as {allowed[:2]}")
        unknown = [x for x in v if x not in allowed]
        if unknown:
            raise ValueError(f"TabSchemaConfig.{name}: unknown value(s) {unknown}; use some of {allowed}")
        setattr(cfg, name, tuple(dict.fromkeys(v)))
    if not cfg.accept:
        raise bad("accept", f"a non-empty tuple of {DECISIONS}")
    if not _is_number(cfg.compile_min_success) or not 0 < cfg.compile_min_success <= 1:
        raise bad("compile_min_success", "a number in (0, 1]")
    if not _is_number(cfg.max_null_rate) or not 0 <= cfg.max_null_rate <= 1:
        raise bad("max_null_rate", "a number in [0, 1]")
    if not _is_number(cfg.audit_max_incorrect) or not 0 <= cfg.audit_max_incorrect < 1:
        raise bad("audit_max_incorrect", "a number in [0, 1)")
    if not _is_number(cfg.timeout_s) or cfg.timeout_s <= 0:
        raise bad("timeout_s", "a positive number of seconds")
    if cfg.max_cost_usd is not None and (not _is_number(cfg.max_cost_usd) or cfg.max_cost_usd < 0):
        raise bad("max_cost_usd", "a non-negative number of USD or None")
    if cfg.request_timeout_s is not None and (
        not _is_number(cfg.request_timeout_s) or cfg.request_timeout_s <= 0
    ):
        raise bad("request_timeout_s", "a positive number of seconds or None")
    for name in ("compile_min_success", "max_null_rate", "audit_max_incorrect", "timeout_s"):
        setattr(cfg, name, float(getattr(cfg, name)))
    for name in ("max_cost_usd", "request_timeout_s"):
        if getattr(cfg, name) is not None:
            setattr(cfg, name, float(getattr(cfg, name)))
    if not isinstance(cfg.prices, dict):
        raise bad("prices", "a mapping {model: (USD per 1M input tokens, USD per 1M output tokens)}")
    prices = {}
    for model, price in cfg.prices.items():
        ok = (
            isinstance(price, (tuple, list))
            and len(price) == 2
            and all(_is_number(x) and x >= 0 for x in price)
        )
        if not isinstance(model, str) or not ok:
            raise ValueError(
                f"TabSchemaConfig.prices[{model!r}] must be (USD per 1M input tokens, USD per 1M output "
                f"tokens), two non-negative numbers, not {price!r}"
            )
        prices[model] = (float(price[0]), float(price[1]))
    cfg.prices = prices
    if not isinstance(cfg.stage_models, dict):
        raise bad("stage_models", "a mapping {stage: model}")
    roles = (*STAGE_ROLES, "judges")
    unknown = sorted(set(cfg.stage_models) - set(roles))
    if unknown:
        raise ValueError(f"stage_models: unknown stages {unknown}; use {list(roles)}")
    for role, model in cfg.stage_models.items():
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"TabSchemaConfig.stage_models[{role!r}] must be a model name, not {model!r}")


@dataclass
class DiscoveryReport:
    """What a discovery run did: the sampled and analyzed decision points, the aggregated, judged, compiled
    and dropped features, the release's verification, output audit and usefulness records, the LLM cost and
    the run directory."""

    target: str
    n_trajectories: int
    n_successful: int
    n_decision_points: int
    analyzed: list[dict[str, Any]] = field(default_factory=list)
    aggregated: list[str] = field(default_factory=list)
    rejected_by_aggregator: list[str] = field(default_factory=list)
    judged: list[dict[str, Any]] = field(default_factory=list)
    compiled: dict[str, str] = field(default_factory=dict)
    dropped: dict[str, str] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    run_dir: str = ""
    notes: list[str] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    usefulness: dict[str, Any] = field(default_factory=dict)
    per_candidate: list[str] = field(default_factory=list)
    not_produced: dict[str, str] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """The report as Markdown (written to ``discovery.md``)."""
        c = self.cost
        usd = c.get("total_usd")
        unpriced = ", ".join(c.get("unpriced_models") or []) or "the model"
        cost = f"${usd:.2f}" if usd is not None else f"unknown (no price for {unpriced})"
        tokens = f"{c.get('input_tokens', 0):,} input / {c.get('output_tokens', 0):,} output tokens"
        cached = (c.get("cached") or {}).get("calls", 0)
        reused = f"; {cached} more answers reused from the run directory" if cached else ""
        failed = sum(bool(a.get("error")) for a in self.analyzed)
        lines = [
            f"# TabSchema discovery: `{self.target}`",
            "",
            f"* trajectories: {self.n_trajectories} ({self.n_successful} successful); decision points: "
            f"{self.n_decision_points}; analyzed: {len(self.analyzed)}"
            + (f" ({failed} failed, see analysis.json)" if failed else ""),
            f"* LLM cost: {cost}; {c.get('calls', 0)} calls, {tokens}{reused}",
            *(f"* {n}" for n in self.notes),
            "",
            "## Judged features",
            "",
            "| feature | relevance | generality | impact | decision | meta score |",
            "|---|---|---|---|---|---|",
        ]
        for j in self.judged:
            lines.append(
                f"| {j['feature_name']} | {j.get('relevance', '')} | {j.get('generality', '')} | "
                f"{j.get('impact', '')} | {j.get('final_decision', '')} | {j.get('meta_score', '')} |"
            )
        lines += _compiled_md(self.compiled, self.per_candidate, self.not_produced)
        lines += _verification_md(self.verification) + _audit_md(self.audit) + _usefulness_md(self.usefulness)
        if self.dropped:
            lines += ["", "## Dropped", ""] + [f"* `{k}`: {v}" for k, v in self.dropped.items()]
        return "\n".join(lines) + "\n"


def not_produced_counts(not_produced: dict[str, str]) -> dict[str, int]:
    """Accepted features without a column, counted by reason: merged into another feature, declined, or not
    produced."""
    out = Counter(
        "merged into another feature"
        if r.startswith("merged into")
        else "declined"
        if r.startswith("declined")
        else "not produced"
        for r in not_produced.values()
    )
    return dict(sorted(out.items()))


def _compiled_md(
    compiled: dict[str, str], per_candidate: list[str], not_produced: dict[str, str]
) -> list[str]:
    """The report's section on the compiled columns: per decision, per candidate, and the accepted features
    that have none."""
    per = set(per_candidate)
    decision = [f"* `{k}` ({v})" for k, v in compiled.items() if k not in per]
    candidate = [f"* `{k}` ({v})" for k, v in compiled.items() if k in per]
    out = ["", "## Compiled extractor features", ""]
    out += [
        f"### Per-decision columns ({len(decision)}, `extract_features_for_classification`)",
        "",
        *decision,
    ]
    if candidate:
        out += [
            "",
            f"### Per-candidate columns ({len(candidate)}, `extract_candidate_features`)",
            "",
            *candidate,
        ]
    if not_produced:
        counts = "; ".join(f"{k}: {n}" for k, n in not_produced_counts(not_produced).items())
        out += ["", f"### Accepted features without a column ({len(not_produced)}: {counts})", ""]
        out += [f"* `{k}`: {v}" for k, v in not_produced.items()]
    return out


def _verification_md(v: dict[str, Any]) -> list[str]:
    """The report's section on the compile-time checks (``verify``)."""
    if not v:
        return []
    out = ["", "## Verification of the extracted values (TabAgent's addition, not in the paper)", ""]
    det = v.get("determinism", {})
    unstable = det.get("unstable") or {}
    out.append(
        f"* determinism: every decision point ({det.get('prefixes', 0)}) run again in fresh interpreters, "
        "in a shuffled order: "
        + (
            f"{len(unstable)} column(s) differ and are dropped ({', '.join(unstable)})"
            if unstable
            else "identical"
        )
    )
    oc = v.get("outcome", {})
    if "skipped" in oc:
        out.append(f"* outcome invariance: skipped ({oc['skipped']})")
    else:
        changed = oc.get("changed") or {}
        keys = ", ".join(oc.get("keys", [])[:6])
        out.append(
            f"* outcome invariance: adding the outcome fields ({keys}) to {v.get('sample', 0)} prefixes "
            + (
                f"changed {len(changed)} column(s), dropped ({', '.join(changed)})"
                if changed
                else "changed no feature"
            )
        )
    fu = v.get("future", {})
    if not fu.get("checked"):
        reasons = "; ".join(fu.get("reasons") or {}) or "no decision point could be checked"
        out.append(f"* future invariance: **skipped**, not passed ({reasons})")
    elif fu.get("leaks"):
        leak = fu["leaks"][0]
        out.append(
            f"* future invariance: **FAILED**: the adapter's prefix changed with the run's future on "
            f"{len(fu['leaks'])} of {fu['checked']} decision points (first {leak['decision_point']}, at "
            f"{leak['path']})"
        )
    else:
        skipped = f"; {fu['skipped']} skipped" if fu.get("skipped") else ""
        out.append(
            f"* future invariance: {fu['checked']} prefixes rebuilt by the adapter from runs changed after "
            f"the decision (and cut after it) are unchanged{skipped}"
        )
    rob = v.get("robustness", {})
    cases = rob.get("cases") or []
    raised = [c for c in cases if c.get("raised")]
    line = f"* robustness (reported, not dropped): {len(cases)} edge prefixes; the extractor raised on "
    line += str(len(raised))
    if raised:
        line += " (" + "; ".join(f"{c['case']}: {c['raised']}" for c in raised[:4]) + ")"
    out.append(line)
    gone = [
        f"{c['case']}: {', '.join(c['missing'][:5])}"
        for c in cases
        if not c.get("raised") and c.get("missing")
    ]
    if gone:
        out.append("* columns missing from the output: " + "; ".join(gone[:4]))
    out.append(f"* {v.get('batches', 0)} sandbox runs on {v.get('prefixes_run', 0)} prefixes")
    return out


def _audit_md(a: dict[str, Any]) -> list[str]:
    """The report's section on the OutputAuditor (``audit``)."""
    if not a:
        return []
    out = ["", "## Output audit (TabAgent's addition, not in the paper)", ""]
    if a.get("skipped"):
        return [*out, f"* skipped: {a['skipped']}"]
    out.append(
        f"* {a['answered']} of {a['samples']} sampled decision points judged by "
        f"{a.get('model', 'the auditor')}; a feature is dropped when it was judged on at least "
        f"{a['min_judged']} points, at least {a.get('min_incorrect', 2)} of them incorrect, and the "
        f"one-sided 95% Wilson lower bound of its incorrect rate exceeds {a['max_incorrect']:.0%} (see "
        "audit.json)"
    )
    if a.get("errors"):
        out.append(f"* {len(a['errors'])} point(s) without a usable answer (first: {a['errors'][0][:200]})")
    out += [
        "",
        "| feature | correct | incorrect | cannot tell | incorrect rate | kept |",
        "|---|---|---|---|---|---|",
    ]
    for col, r in a.get("features", {}).items():
        rate = "" if r["incorrect_rate"] is None else f"{r['incorrect_rate']:.0%}"
        kept = "no" if col in a.get("dropped", {}) else "yes"
        out.append(f"| {col} | {r['correct']} | {r['incorrect']} | {r['cannot_tell']} | {rate} | {kept} |")
    return out


def _usefulness_md(u: dict[str, Any]) -> list[str]:
    """The report's section on the usefulness estimate (``usefulness``)."""
    if not u:
        return []
    out = ["", "## Usefulness of the features (TabAgent's addition, not in the paper)", ""]
    if u.get("skipped"):
        return [*out, f"* skipped: {u['skipped']}"]
    m = u["metric"]
    out.append(
        f"* drop-feature importance (a feature's columns dropped together) on the onboarding rows: "
        f"{u['decisions']} decision points "
        f"({u.get('rows', '?')} rows) of {u['tasks']} tasks, {u['folds']} task-grouped folds, CatBoost with "
        f"{u['iterations']} iterations, {u.get('seeds', 1)} seed(s); {m} {u.get('base', float('nan')):.3f} "
        "with every feature"
    )
    pruning = u.get("pruned") is not None
    out.append(
        f"* delta: the head's {m} with the feature minus without it (95% bootstrap interval over tasks); "
        "useful or harmful when the Holm-adjusted p-value (paired t-test over tasks) is below 0.05"
        + (
            " under every seed; harmful features are pruned"
            if pruning
            else "; a report only (prune_harmful=False)"
        )
    )
    out += ["", f"| feature | delta {m} | 95% interval | Holm p | verdict |", "|---|---|---|---|---|"]
    for col, r in sorted(u.get("features", {}).items(), key=lambda kv: -kv[1]["delta"]):
        n = len(r.get("columns") or [col])
        name = f"{col} ({n} columns)" if n > 1 else col
        ci = f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
        out.append(f"| {name} | {r['delta']:+.4f} | {ci} | {r.get('p_holm', '')} | {r['verdict']} |")
    return out


def compile_record(compiled: CompiledExtractor) -> dict[str, Any]:
    """The compile step as ``compile.json``: the attempts and why each failed one failed, the kept columns per
    decision and per candidate, every kept column (type and the planned feature it implements), the accepted
    features without a column with the reason (``not_produced``: merged into another feature, declined, not
    produced), every column's quality statistics, every dropped column or feature with the reason, and the
    notes."""
    return {
        "attempts": compiled.attempts,
        "attempt_errors": compiled.attempt_errors,
        "per_decision": compiled.context_columns,
        "per_candidate": compiled.candidate_columns,
        "not_produced": {
            "count": len(compiled.not_produced),
            "by_reason": not_produced_counts(compiled.not_produced),
            "features": dict(compiled.not_produced),
        },
        "kept": {
            col: {
                "type": t,
                "feature": (compiled.specs.get(col) or {}).get("feature_name"),
                **({"per_candidate": True} if col in compiled.per_candidate else {}),
            }
            for col, t in compiled.features.items()
        },
        "dropped": dict(compiled.dropped),
        "quality": compiled.quality,
        "co_usage_context": compiled.co_usage_context,
        "notes": list(compiled.notes),
    }


@dataclass
class DiscoveryResult:
    """The output of ``TabSchema.discover``: the feature card, the compiled extractor and the report."""

    card: FeatureCard
    extractor: CompiledExtractor
    report: DiscoveryReport

    def save(self, path: str | Path) -> Path:
        """Write ``extract.py``, its golden tests (``extractor_tests.json``), ``card.yaml``, ``discovery.md``
        and ``discovery.json`` to ``path``."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        atomic_write(out / "extract.py", self.extractor.code)
        if self.extractor.tests:
            atomic_write(out / V.TESTS_FILE, json.dumps(self.extractor.tests, indent=1, ensure_ascii=False))
        self.card.save(out / "card.yaml")
        atomic_write(out / "discovery.md", self.report.to_markdown())
        report = dataclasses.asdict(self.report)
        atomic_write(out / "discovery.json", json.dumps(report, indent=1, default=str))
        return out


class TabSchema:
    """The FeatureOrchestrator: discovers a feature card for one decision component of an agent.

    Parameters
    ----------
    llm: the model for every stage (default: ``auto_llm(config.model)``, from environment keys).
    config: a ``TabSchemaConfig``.
    adapter: reads the agent's logs into trajectories and decision points (default ``GenericJsonAdapter``).
    store: the run directory (default: a new one under ``$TABAGENT_CACHE/runs``).
    sandbox: runs LLM-written extraction code (default ``SubprocessSandbox``).
    stage_llms: LLM backends by model name, for the models ``config.stage_models`` and
        ``config.fallback_model`` name (default: built from the environment with ``auto_llm``).

    With ``config.max_cost_usd`` set, the model needs a price in ``config.prices``; otherwise the budget
    could not be enforced and the constructor raises ``ValueError``.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        config: TabSchemaConfig | None = None,
        adapter=None,
        store: RunStore | None = None,
        sandbox: SubprocessSandbox | None = None,
        stage_llms: dict[str, LLM] | None = None,
    ):
        self.config = config or TabSchemaConfig()
        self.llm = llm or auto_llm(self.config.model, **_timeout(self.config))
        self.adapter = adapter or GenericJsonAdapter()
        self.store = store
        self.sandbox = sandbox or SubprocessSandbox(timeout_s=self.config.timeout_s)
        self.meter = CostMeter(prices=dict(self.config.prices), max_cost_usd=self.config.max_cost_usd)
        unknown = sorted(set(self.config.stage_models) - set(STAGE_ROLES) - {"judges"})
        if unknown:
            raise ValueError(f"stage_models: unknown stages {unknown}; use {[*STAGE_ROLES, 'judges']}")
        strategy = self.config.sample_strategy
        if strategy not in STRATEGIES:
            raise ValueError(f"sample_strategy must be one of {STRATEGIES}, not {strategy!r}")
        self.llms = {self.llm.model: self.llm, **(stage_llms or {})}
        for name in {*self.config.stage_models.values(), self.config.fallback_model} - {None}:
            if name not in self.llms:
                self.llms[name] = auto_llm(name, **_timeout(self.config))
        for name in self.llms:
            self.meter.require_price(name)

    def llm_for(self, role: str) -> LLM:
        """The LLM of a stage: its ``stage_models`` entry (``judges`` for all three judges), else ``llm``."""
        models = self.config.stage_models
        name = models.get(role) or (models.get("judges") if role.startswith("judge_") else None)
        return self.llms[name] if name else self.llm

    def models(self) -> dict[str, str]:
        """The model of every stage, and the fallback (``"fallback"``) when one is set."""
        out = {role: self.llm_for(role).model for role in STAGE_ROLES}
        if self.config.fallback_model:
            out["fallback"] = self.config.fallback_model
        return out

    def _stage(self, cls, *args):
        st = cls(*args, self.llm, self.meter, self.store, self.config.json_retries)
        st.llm = self.llm_for(st.role)
        st.workers = self.config.workers
        st.retry = dataclasses.replace(st.retry, attempts=self.config.llm_retries + 1)
        fb = self.config.fallback_model
        st.fallback = self.llms[fb] if fb else None
        self._stages.append(st)
        return st

    def _candidates(self, dp: DecisionPoint, kind: str) -> list:
        """The adapter's candidates of ``dp``, [] for a single-class decision or an adapter that has none
        (it raises ``NotImplementedError``). Any other error of the adapter is logged and raised: it must not
        quietly turn the per-candidate features off."""
        if kind == "single":
            return []
        try:
            return list(self.adapter.candidates(dp) or ())
        except NotImplementedError:  # an adapter without candidates
            return []
        except Exception as e:
            logger.error(
                "TabSchema: the adapter's candidates of %s failed: %s: %s", dp.id, type(e).__name__, e
            )
            raise

    def _progress(self, stage: str, what: str):
        """A callback ``(done, total, failed)`` that records a stage's progress and logs it."""
        log = getattr(self, "log", None)

        def report(done: int, total: int, failed: int) -> None:
            if log is not None:
                logger.info("TabSchema: %s: %s", what, log.progress(stage, done, total, failed))

        return report

    def _events(self, since: int = 0) -> list[dict[str, Any]]:
        """The events of the stages made since the ``since``-th one (unusable answers, split batches, ...)."""
        return [e for st in self._stages[since:] for e in st.events]

    # ------------------------------------------------------------------ discovery
    def discover(
        self,
        trajectories: list[Trajectory] | str | Path,
        target: str,
        kind: str = "ranking",
        name: str | None = None,
    ) -> DiscoveryResult:
        """Run the discovery loop for the decision component ``target``.

        ``trajectories`` are ``Trajectory`` objects or a path the adapter reads; only successful ones are
        used. ``kind`` is the decision pattern and ``name`` the card name. Every intermediate artifact is
        written to the run directory, with ``run.json`` (the run's progress and anatomy,
        ``tabagent runs show``); a run that stops leaves a partial ``report.md`` there, and rerunning with the
        same directory resumes it.
        """
        cfg = self.config
        trajs = (
            self.adapter.load(trajectories) if isinstance(trajectories, (str, Path)) else list(trajectories)
        )
        good = [t for t in trajs if t.success]
        if not good:
            raise ValueError(
                f"no successful trajectories among the {len(trajs)} given: TabSchema learns from successful "
                "runs only (check the outcome field the adapter reads)"
            )
        dps = [dp for t in good for dp in self.adapter.decision_points(t, target)]
        if not dps:
            raise ValueError(f"no occurrences of {target!r} in the {len(good)} successful trajectories")
        if len(dps) < 2:
            raise ValueError(
                f"TabSchema needs at least 2 decision points of {target!r}, found {len(dps)}: no feature can "
                "vary across fewer"
            )
        store = self.store or RunStore()
        self.store = store
        self._stages = []
        STOP.clear()  # a backoff wait ends early only on an interruption of this run
        report = DiscoveryReport(target, len(trajs), len(good), len(dps), run_dir=str(store.dir))
        log = RunLog(store, self.meter, target)
        self.log = log
        log.set(
            "data",
            {
                "trajectories": len(trajs),
                "successful": len(good),
                "decision_points": len(dps),
                "tasks": len({str(dp.group or dp.trajectory_id) for dp in dps}),
            },
        )
        store.write_json(
            "config.json",
            {
                **cfg.to_dict(),
                "target": target,
                "kind": kind,
                "model": self.llm.model,
                "stage_models_resolved": self.models(),
                "prompts_sha256": {p: prompts.sha256(p) for p in prompts.PROMPTS},
            },
        )
        t0 = time.monotonic()
        try:
            with log.stage("adapter_check") as rec:
                future = self.check_adapter(good, target, dps)  # before any LLM call: a leak costs nothing
                rec["produced"] = {
                    "checked": future.get("checked", 0),
                    "leaks": len(future.get("leaks") or []),
                }
            if future["notes"]:
                report.notes += future["notes"]
            compiled, decisions = self._run(
                dps, target, store, report, candidate_ids(self.adapter, dps, kind=kind), good, kind, future
            )
            report.cost = self.meter.summary()
            with log.stage("card") as rec:
                card = self._card(compiled, decisions, kind, name or f"{target}_tabschema", len(dps))
                card.save(store.path("card.yaml"))
                rec["produced"] = {
                    "features": len(card.features),
                    "extractor columns": len(compiled.features),
                }
        except BaseException as e:  # leave a partial report and run.json next to the paid answers, then fail
            if isinstance(e, V.PrefixLeakError):
                report.verification = e.record
                report.notes.append(f"onboarding stopped before any LLM call: {e}")
            forgotten = forget_cause(e) if isinstance(e, Exception) else 0
            report.cost = self.meter.summary()
            store.write_text("report.md", report.to_markdown())
            status = stopped_status(e)
            log.finish(status, e, {"drops_by_reason": drops_by_reason(report, cfg), "seconds": _secs(t0)})
            if isinstance(e, (BudgetExceeded, ExtractionError, LLMFormatError, LLMUnavailable)):
                if status == "stopped":
                    hint = "rerun with that run directory to resume (answers it holds cost nothing)"
                else:
                    hint = "rerun with that run directory: the answers it holds cost nothing" + (
                        f", and the {forgotten} answer(s) that caused this failure are asked again"
                        if forgotten
                        else ""
                    )
                raise type(e)(
                    f"{e}. The partial report, run.json and every paid answer are in {store.dir}: {hint}"
                ) from e
            if isinstance(e, Exception) and hasattr(e, "add_note"):  # a provider's own error keeps its type
                stage = next((k for k, r in log.data["stages"].items() if r.get("status") == status), "?")
                e.add_note(
                    f"TabSchema stopped in stage {stage}; the partial report, run.json and every paid answer "
                    f"are in {store.dir} (rerun with it to resume)"
                )
            raise
        logger.info(
            "TabSchema: compiled an extractor with %d features (%d dropped); LLM: %s",
            len(compiled.features),
            len(compiled.dropped),
            self.meter.describe(self.llm.model),
        )
        store.write_text("report.md", report.to_markdown())
        log.finish(
            "completed",
            extra={
                "drops_by_reason": drops_by_reason(report, cfg),
                "seconds": _secs(t0),
                "outputs": {"card": "card.yaml", "extractor": "extract.py", "report": "report.md"},
            },
        )
        return DiscoveryResult(card, compiled, report)

    def _analyze_point(
        self,
        dp: DecisionPoint,
        target: str,
        analyzer: FeatureAnalyzer,
        extractor: CodeExtractor,
        validator: CodeValidator,
        known: set,
        outcome_keys: tuple[str, ...],
        kind: str = "ranking",
    ) -> dict[str, Any]:
        """One decision point's analysis: proposals, feasible code, the CodeValidator's verdicts. A failure
        that is this point's own (an unusable answer, a content filter, code that never runs) loses the point
        and is recorded; a spent budget, a provider that stays down, a provider's error (an invalid key, a
        model that does not exist, a bad request) or an interruption stops the run."""
        cfg = self.config
        rec: dict[str, Any] = {"decision_point": dp.id, "proposed": [], "kept": [], "error": ""}
        try:
            specs = analyzer(dp, target)
            rec["proposed"] = [s.feature_name for s in specs]
            executor = CodeExecutor(self.sandbox, known, outcome_keys)
            cands = [candidate_dict(c) for c in self._candidates(dp, kind)]
            attempt, history = extract_with_retries(
                extractor,
                executor,
                specs,
                [dp.prefix],
                dp.prefix,
                target,
                cfg.max_code_retries,
                candidates=[cands] if cands else None,
            )
            rec["attempts"] = len(history)
            if not attempt.ok:
                rec["error"] = f"infeasible after {len(history)} attempts: {attempt.error[:300]}"
                rec["_forget"] = [(extractor, [p for h in history for p in h.prompts])]
                return {**rec, "specs": []}
            if attempt.candidate_error:
                rec["candidate_error"] = attempt.candidate_error[:300]
            more = {}
            if cands:  # the per-candidate function's outputs, judged next to the prefix's
                more = {"candidates": cands, "candidate_outputs": candidate_samples(attempt.results, [cands])}
            try:
                verdicts = validator(
                    specs, attempt.code, dp.prefix, [attempt.results[0].features], target, **more
                )
            except LLMFormatError as e:  # the deterministic checks passed; keep the proposals unjudged
                verdicts, rec["validator_error"] = {}, str(e)[:300]
            kept = [s for s in specs if verdicts.get(s.key, VALID) not in REJECTING]
            rec["kept"] = [s.feature_name for s in kept]
            rec["validator"] = verdicts
            odd = sorted({v for v in verdicts.values() if v not in VERDICTS})
            if odd:
                rec["validator_unrecognized"] = odd[:10]
            return {**rec, "specs": [s.to_dict() for s in kept]}
        except (BudgetExceeded, LLMUnavailable):
            raise
        except Exception as e:  # one decision point's own failure (an unusable answer, a filter) is skipped
            if (
                status_code(e) is not None
            ):  # a provider's error (an invalid key, a bad request): not the point's
                raise
            logger.warning("TabSchema: analyzing %s failed: %s: %s", dp.id, type(e).__name__, e)
            rec["error"] = f"{type(e).__name__}: {e}"[:500]
            rec["_forget"] = list(getattr(e, "forget", None) or [])
            return {**rec, "specs": []}

    def _run(
        self,
        dps: list[DecisionPoint],
        target: str,
        store: RunStore,
        report: DiscoveryReport,
        candidates: frozenset[str] = frozenset(),
        trajectories: list[Trajectory] | None = None,
        kind: str = "ranking",
        future: dict[str, Any] | None = None,
    ) -> tuple[CompiledExtractor, list[dict]]:
        """The stages of :meth:`discover` after its checks: analysis, aggregation, judging, compilation and
        the release's audit and usefulness estimate. Fills ``report`` as it goes; returns the compiled
        extractor and the MetaJudge's decisions."""
        cfg = self.config
        log = getattr(self, "log", None) or RunLog(store, self.meter, target)
        if not hasattr(self, "_stages"):
            self._stages = []
        known = union_paths(dp.prefix for dp in dps)
        outcome_keys = tuple(getattr(self.adapter, "outcome_keys", ()))
        labels = getattr(self.adapter, "labels", None)
        with log.stage("sample") as rec:
            sample = sample_decision_points(dps, cfg.n_analyze, cfg.sample_seed, labels, cfg.sample_strategy)
            store.write_json("sample.json", [dp.id for dp in sample])
            covered = coverage(dps, sample, labels)
            store.write_json("sample_coverage.json", covered)
            rec["produced"] = {"points": len(sample), **{k: f"{a}/{b}" for k, (a, b) in covered.items()}}
        report.notes.append(
            f"analyzed sample ({cfg.sample_strategy}) covers "
            + ", ".join(f"{a}/{b} {kind}" for kind, (a, b) in covered.items())
        )
        logger.info(
            "TabSchema: %d decision points in %d successful trajectories; analyzing %d (run directory %s)",
            len(dps),
            report.n_successful,
            len(sample),
            store.dir,
        )

        # --- analysis, one decision point per worker
        mark = len(self._stages)
        with log.stage("analyze") as rec:
            analyzer, extractor = self._stage(FeatureAnalyzer), self._stage(CodeExtractor)
            analyzer.max_chars = cfg.max_prompt_chars
            validator = self._stage(CodeValidator)
            pool = ThreadPoolExecutor(max_workers=cfg.workers)
            try:
                futures = [
                    pool.submit(
                        self._analyze_point,
                        dp,
                        target,
                        analyzer,
                        extractor,
                        validator,
                        known,
                        outcome_keys,
                        kind,
                    )
                    for dp in sample
                ]
                failed = 0
                for done, f in enumerate(as_completed(futures), 1):
                    a = f.result()
                    failed += bool(a["error"])
                    logger.info(
                        "TabSchema: analyzed decision points: %s",
                        log.progress("analyze", done, len(sample), failed),
                    )
                analyses = [f.result() for f in futures]
            except KeyboardInterrupt:  # return at once; threads waiting in a backoff are woken
                shutdown_interrupted(pool)
                raise
            except BaseException:  # a spent budget, say: drop the decision points not yet started
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            pool.shutdown(wait=True)
            marks = [m for a in analyses for m in a.pop("_forget", None) or [] if a["error"]]
            unjudged = sum(bool(a.get("validator_error")) for a in analyses)
            if unjudged:
                report.notes.append(
                    f"the CodeValidator's answer was unusable at {unjudged} analyzed decision point(s); "
                    "their proposals went on unjudged by it (analysis.json)"
                )
            report.analyzed = [{k: v for k, v in a.items() if k != "specs"} for a in analyses]
            store.write_json("analysis.json", analyses)
            proposals = premerge([FeatureSpec.from_dict(x) for a in analyses for x in a["specs"]])
            rec["produced"] = {
                "points": len(analyses),
                "proposals": sum(len(a["proposed"]) for a in analyses),
                "feasible": sum(len(a["specs"]) for a in analyses),
                "distinct": len(proposals),
            }
            rec["failures"] = dict(Counter(_error_kind(a["error"]) for a in analyses if a["error"]))
            rec["events"] = _event_counts(self._events(mark))
            logger.info(
                "TabSchema: analyzed %d decision points (%d failed); %d feasible proposals, %d distinct",
                len(analyses),
                failed,
                sum(len(a["specs"]) for a in analyses),
                len(proposals),
            )
            if not proposals:
                errors = [a["error"] for a in analyses if a["error"]]
                first = f"; first error: {errors[0]}" if errors else ""
                err = RuntimeError(
                    f"no feasible feature proposals from the {len(analyses)} analyzed decision points"
                    f"{first}; see analysis.json in {store.dir}"
                )
                err.forget = marks  # type: ignore[attr-defined]
                raise err

        # --- aggregation
        mark = len(self._stages)
        with log.stage("aggregate") as rec:
            aggregator = self._stage(FeatureAggregator)
            filtered, rejected = aggregator(proposals)
            unaggregated = list(getattr(aggregator, "unaggregated", []))
            logger.info("TabSchema: aggregated %d proposals into %d features", len(proposals), len(filtered))
            report.aggregated = [f.feature_name for f in filtered]
            report.rejected_by_aggregator = [f.feature_name for f in rejected]
            events = self._events(mark)
            store.write_json(
                "aggregator.json",
                {
                    "filtered": [f.to_dict() for f in filtered],
                    "rejected": [f.to_dict() for f in rejected],
                    **({"unaggregated": unaggregated} if unaggregated else {}),
                    **({"events": events} if events else {}),
                },
            )
            rec["produced"] = {
                "proposals": len(proposals),
                "features": len(filtered),
                "rejected": len(rejected),
            }
            rec["events"] = _event_counts(events)
            if unaggregated:
                report.notes.append(
                    "**the FeatureAggregator gave no usable answer for "
                    f"{len(unaggregated)} proposal(s); they were judged as proposed, unaggregated** "
                    "(aggregator.json)"
                )
            proposed = {p.key for p in proposals}
            renamed = [f.feature_name for f in filtered if f.key not in proposed]
            if renamed:
                report.notes.append(
                    "the FeatureAggregator kept features under names no analyzer proposed (judged without "
                    f"the analyzer's view): {', '.join(renamed)}"
                )
            if not filtered:
                raise ExtractionError(
                    f"the FeatureAggregator kept none of the {len(proposals)} proposals; see aggregator.json"
                )

        # --- the three judges, then the MetaJudge
        mark = len(self._stages)
        with log.stage("judge") as rec:
            verdicts = run_judges(
                filtered,
                lambda d: self._stage(Judge, d),
                workers=max(3, cfg.workers),
                stages=[],
            )
            table = judge_table(verdicts, proposals)
            unscored = {d: len(filtered) - len(v) for d, v in verdicts.items() if len(v) < len(filtered)}
            rec["produced"] = {
                "features": len(filtered),
                **{f"{d} scores": len(v) for d, v in verdicts.items()},
            }
            rec["events"] = _event_counts(self._events(mark))
            if unscored:
                rec["failures"] = {f"no {d} score": n for d, n in unscored.items()}
                report.notes += [
                    f"**the {d} judge gave no usable score for {n} of {len(filtered)} features; the "
                    "MetaJudge decided on them without it** (judges.json)"
                    for d, n in unscored.items()
                ]
        judge_events = self._events(mark)
        mark = len(self._stages)
        with log.stage("meta_judge") as rec:
            decisions = self._stage(MetaJudge)(verdicts, table)
            events = [*judge_events, *self._events(mark)]
            store.write_json(
                "judges.json",
                {
                    "verdicts": verdicts,
                    "table": table,
                    "meta": decisions,
                    **({"events": events} if events else {}),
                },
            )
            accept = {d for d in cfg.accept if d in DECISIONS}
            by_key = {normalize_name(d["feature_name"]): d for d in decisions}
            accepted = [f for f in filtered if f.key in by_key and by_key[f.key]["final_decision"] in accept]
            for e in self._events(mark):  # decisions on names the MetaJudge was not given: ignored
                for name in e.get("features", []) if e.get("event") == "ignored" else []:
                    report.dropped[str(name)] = (
                        "the MetaJudge decided on a name it was not given (renamed or invented); ignored"
                    )
            for f in filtered:
                if f.key not in by_key:
                    report.dropped[f.feature_name] = "the MetaJudge gave no decision on it"
            logger.info(
                "TabSchema: judged %d features; the MetaJudge accepted %d", len(filtered), len(accepted)
            )
            scores = {normalize_name(row["feature_name"]): row for row in table}
            for d in decisions:
                row = scores.get(normalize_name(d["feature_name"]), {})
                report.judged.append(
                    {
                        "feature_name": d["feature_name"],
                        "final_decision": d["final_decision"],
                        "meta_score": d.get("meta_score"),
                        **{k: row.get(f"judge_{k}_score") for k in ("relevance", "generality", "impact")},
                    }
                )
            counts = {
                k: sum(by_key[f.key]["final_decision"] == k for f in filtered if f.key in by_key)
                for k in DECISIONS
            }
            rec["produced"] = {**counts, "compiled": len(accepted)}
            rec["events"] = _event_counts(self._events(mark))
            if not accepted:
                raise ExtractionError(
                    f"the MetaJudge accepted none of the {len(filtered)} aggregated features ({counts}); see "
                    f"judges.json. TabSchemaConfig.accept (now {cfg.accept}) sets which of accept, "
                    "conditional and reject are compiled"
                )

        # --- one extractor for the accepted features
        mark = len(self._stages)
        with log.stage("compile") as rec:
            if future is None and trajectories:  # a caller that skipped discover's check
                future = self.check_adapter(trajectories, target, dps)
            extractor, validator = self._stage(CodeExtractor), self._stage(CodeValidator)
            logger.info(
                "TabSchema: compiling one extractor for %d features on %d decision points",
                len(accepted),
                len(dps),
            )
            try:
                compiled = compile_extractor(
                    accepted,
                    dps,
                    extractor,
                    validator,
                    target,
                    sandbox=self.sandbox,
                    known_paths=known,
                    outcome_keys=outcome_keys,
                    max_retries=cfg.max_code_retries,
                    min_success=cfg.compile_min_success,
                    max_null_rate=cfg.max_null_rate,
                    max_columns_per_feature=cfg.max_columns_per_feature,
                    max_columns=cfg.max_extractor_columns,
                    repair=cfg.compile_repair,
                    reserved=RESERVED,
                    candidate_ids=candidates,
                    context_text=cfg.context_text,
                    verify=cfg.verify,
                    verify_samples=cfg.verify_samples,
                    future=future,
                    goldens=cfg.goldens,
                    seed=cfg.sample_seed,
                    steps_key=getattr(self.adapter, "steps_key", "steps"),
                    label_of=labels,
                    adapter_name=type(self.adapter).__name__,
                    drop_redundant_text=cfg.drop_redundant_text,
                    candidates_of=(lambda dp: self._candidates(dp, kind)) if candidates else None,
                    last_step_text=cfg.last_step_text,
                    positional_guard=cfg.positional_guard,
                )
            except V.PrefixLeakError as e:  # the report says so, next to every paid answer
                report.verification = e.record
                report.notes.append(f"onboarding stopped: {e}")
                raise
            store.write_text("extract.py", compiled.code)
            report.verification = {k: v for k, v in compiled.verification.items() if k != "seconds"}
            rec["produced"] = {
                "columns": len(compiled.features),
                "code attempts": compiled.attempts,
                "decision points without features": sum(o is None for o in compiled.outputs),
            }
            rec["dropped"] = dict(Counter(drop_reason(r) for r in compiled.dropped.values()))
            rec["events"] = _event_counts(self._events(mark))
        if cfg.audit:
            with log.stage("audit") as rec:
                mark = len(self._stages)
                analyzed = frozenset(dp.id for dp in sample)
                self._audit(compiled, dps, target, store, report, analyzed, labels, kind)
                rec["events"] = _event_counts(self._events(mark))
                a = report.audit
                rec["produced"] = {"points": a.get("samples", 0), "answered": a.get("answered", 0)}
                rec["dropped"] = {"judged incorrect": len(a.get("dropped") or {})} if a.get("dropped") else {}
                if a.get("skipped"):
                    rec["note"] = a["skipped"]
                if a.get("errors"):
                    rec["failures"] = dict(Counter(_error_kind(e) for e in a["errors"]))
        if cfg.usefulness:
            with log.stage("usefulness") as rec:
                self._usefulness(compiled, dps, decisions, kind, store, report, outcome_keys)
                u = report.usefulness
                rec["produced"] = dict(Counter(r.get("verdict") for r in (u.get("features") or {}).values()))
                if u.get("skipped"):
                    rec["note"] = u["skipped"]
        compiled.tests = V.restrict_tests(compiled.tests, compiled.features)
        if compiled.tests:
            store.write_json(V.TESTS_FILE, compiled.tests)
        store.write_json("compile.json", compile_record(compiled))
        report.compiled = dict(compiled.features)
        report.per_candidate = compiled.candidate_columns
        report.not_produced = dict(compiled.not_produced)
        report.dropped.update(compiled.dropped)
        report.notes += compiled.notes
        return compiled, decisions

    # ------------------------------------------------------------------ the release's checks
    def check_adapter(
        self, trajectories: list[Trajectory], target: str, dps: list[DecisionPoint] | None = None
    ) -> dict[str, Any]:
        """The adapter's future-invariance check (``verify.check_adapter``) on the successful
        ``trajectories``, before any LLM call: raises ``PrefixLeakError`` when the adapter's prefixes change
        with the run's future. With ``verify`` or ``verify_adapter`` off it is skipped with a warning. The
        result is kept, so discovery after a preflight does not run it twice."""
        cfg = self.config
        good = [t for t in trajectories if t.success]
        key = (target, tuple(t.id for t in good))
        if getattr(self, "_adapter_check", (None, None))[0] == key:
            return self._adapter_check[1]
        if not (cfg.verify and cfg.verify_adapter):
            which = "verify=False" if not cfg.verify else "verify_adapter=False"
            msg = (
                f"the adapter's leak check (future invariance) was skipped ({which}): post-decision data in "
                "prefixes would go unnoticed"
            )
            logger.warning("TabSchema: %s", msg)
            rec = {**V.skipped_future(f"turned off ({which})"), "notes": [msg]}
        else:
            if dps is None:
                dps = [dp for t in good for dp in self.adapter.decision_points(t, target)]
            rec = V.check_adapter(
                self.adapter,
                good,
                dps,
                target,
                samples=cfg.verify_samples,
                seed=cfg.sample_seed,
                label_of=getattr(self.adapter, "labels", None),
            )
        self._adapter_check = (key, rec)
        return rec

    def _audit(
        self,
        compiled: CompiledExtractor,
        dps: list[DecisionPoint],
        target: str,
        store: RunStore,
        report: DiscoveryReport,
        analyzed: frozenset[str],
        labels: Any = None,
        kind: str = "ranking",
    ) -> None:
        """Run the OutputAuditor on a sample of decision points and drop the features it judges incorrect too
        often; writes ``audit.json``. The audit only ever removes features: when it would remove them all, it
        removes none and says so."""
        cfg = self.config
        auditor = self._stage(OutputAuditor)
        columns = {c: compiled.specs[c] for c in compiled.llm_columns if c not in compiled.per_candidate}
        per = {c: compiled.specs[c] for c in compiled.candidate_columns}
        logger.info(
            "TabSchema: auditing the values of %d features on %d decision points",
            len(columns) + len(per),
            cfg.audit_samples,
        )
        rec = run_audit(
            auditor,
            columns,
            dps,
            compiled.outputs,
            target,
            samples=cfg.audit_samples,
            max_incorrect=cfg.audit_max_incorrect,
            seed=cfg.sample_seed,
            exclude=analyzed,
            workers=cfg.workers,
            label_of=labels,
            progress=self._progress("audit", "audited decision points"),
            **(
                {
                    "candidate_columns": per,
                    "candidate_outputs": compiled.candidate_outputs,
                    "candidates_of": lambda dp: [candidate_dict(c) for c in self._candidates(dp, kind)],
                }
                if per
                else {}
            ),
        )
        rec["model"] = auditor.llm.model
        if rec["dropped"] and len(rec["dropped"]) >= len(compiled.llm_columns):
            rec["not_applied"] = rec.pop("dropped")
            rec["dropped"] = {}
            compiled.notes.append(
                "**the OutputAuditor judged every LLM-written feature incorrect too often; none was dropped "
                "(the audit never empties a card): review the features and audit.json before deploying**"
            )
        for col, reason in rec["dropped"].items():
            compiled.drop(col, reason)
        if rec["skipped"]:
            compiled.notes.append(f"output audit skipped: {rec['skipped']}")
        if rec.get("analyzed_points"):
            compiled.notes.append(
                f"output audit: {rec['analyzed_points']} of the audited decision points were also analyzed "
                "(fewer unanalyzed points than audit_samples)"
            )
        store.write_json("audit.json", rec)
        report.audit = {k: v for k, v in rec.items() if k != "points"}

    def _usefulness(
        self,
        compiled: CompiledExtractor,
        dps: list[DecisionPoint],
        decisions: list[dict],
        kind: str,
        store: RunStore,
        report: DiscoveryReport,
        outcome_keys: tuple[str, ...],
    ) -> None:
        """Estimate each extractor feature's usefulness on the onboarding rows (and, with ``prune_harmful``,
        drop the clearly harmful LLM-written ones); writes ``usefulness.json``."""
        cfg = self.config
        idx = {dp.id: i for i, dp in enumerate(dps)}
        sample = (
            dps
            if len(dps) <= cfg.usefulness_max_decisions
            else sample_decision_points(dps, cfg.usefulness_max_decisions, cfg.sample_seed, strategy="tasks")
        )
        appended = [c for c in compiled.features if compiled.specs.get(c, {}).get("added_by")]

        def features_of(points: list[DecisionPoint]) -> list[dict]:
            feats = [dict(compiled.outputs[idx[dp.id]] or {}) for dp in points]
            for col in appended:  # the appended texts are library code: computed here, as in the sandbox
                which = compiled.specs[col].get("appended", CONTEXT_TEXT_COLUMN)
                steps = getattr(self.adapter, "steps_key", "steps")
                texts = appended_texts([dp.prefix for dp in points], col, which, outcome_keys, steps)
                for f, text in zip(feats, texts, strict=True):
                    f[col] = text
            return feats

        def candidates_of(points: list[DecisionPoint]) -> list | None:
            if not compiled.candidate_columns:
                return None
            out = []
            for dp in points:
                values = compiled.candidate_outputs[idx[dp.id]]
                ids = [c.id for c in self._candidates(dp, kind)]
                out.append(None if values is None else dict(zip(ids, values, strict=True)))
            return out

        def card_for(dropped: frozenset[str]) -> FeatureCard:
            return self._card(
                compiled.without(dropped), decisions, kind, "usefulness", len(dps), embeddings=False
            )

        try:
            table = rows_to_table(
                card_for(frozenset()), sample, features_of(sample), self.adapter, candidates_of(sample)
            )
            if len(table) > cfg.usefulness_max_rows and len(sample) > MIN_DECISIONS:  # bound the runtime
                n = max(MIN_DECISIONS, int(len(sample) * cfg.usefulness_max_rows / len(table)))
                sample = sample_decision_points(sample, n, cfg.sample_seed, strategy="tasks")
                table = rows_to_table(
                    card_for(frozenset()), sample, features_of(sample), self.adapter, candidates_of(sample)
                )
            logger.info(
                "TabSchema: estimating the usefulness of %d features on %d decision points (%d rows)",
                len(compiled.features),
                len(sample),
                len(table),
            )
            rec = feature_usefulness(
                table,
                card_for,
                feature_groups(compiled),
                seed=cfg.sample_seed,
                seeds=2 if cfg.prune_harmful else 1,
            )
        except Exception as e:  # a report only: no candidates or labels, one label value, a fit that fails
            logger.warning("TabSchema: the usefulness estimate was skipped: %s: %s", type(e).__name__, e)
            rec = {"skipped": f"{type(e).__name__}: {e}"[:300]}
        if cfg.prune_harmful and not rec.get("skipped"):
            harmful_columns = {
                c
                for name, r in rec["features"].items()
                if r.get("verdict") == "harmful" and name != ALL_LLM
                for c in r.get("columns", [name])
            }
            harmful = [c for c in compiled.llm_columns if c in harmful_columns]
            by_column = {
                c: r
                for name, r in rec["features"].items()
                if name != ALL_LLM
                for c in r.get("columns", [name])
            }
            if harmful and len(harmful) >= len(compiled.llm_columns):
                compiled.notes.append(
                    "prune_harmful: every LLM-written feature looked harmful; none was pruned"
                )
                harmful = []
            for col in harmful:
                r = by_column[col]
                compiled.drop(
                    col,
                    f"harmful (prune_harmful): removing it raised {rec['metric']} by {-r['delta']:.4f} "
                    f"(Holm-adjusted p {r['p_holm']}, under {rec['seeds']} seeds) in grouped CV on the "
                    "onboarding rows",
                )
            rec["pruned"] = harmful
            if harmful:
                compiled.notes.append(
                    f"prune_harmful dropped {len(harmful)} feature(s) chosen on the onboarding rows, so "
                    "cross-validation on those rows (the onboarding report) is optimistic; "
                    "tabagent.cross_validate_discovery prunes inside each training fold"
                )
        store.write_json("usefulness.json", rec)
        report.usefulness = {k: v for k, v in rec.items() if k != "seconds"}

    # ------------------------------------------------------------------ card and table
    def _card(
        self,
        compiled: CompiledExtractor,
        decisions: list[dict],
        kind: str,
        name: str,
        n_dps: int,
        embeddings: bool | None = None,
    ) -> FeatureCard:
        cfg = self.config
        meta = {normalize_name(d["feature_name"]): d for d in decisions}
        feats: list[Feature] = []
        per_candidate: list[Feature] = []
        for col, typ in compiled.features.items():
            plan = compiled.specs.get(col, {})
            prov = {"plan": plan, "meta_judge": meta.get(normalize_name(plan.get("feature_name", "")), {})}
            if col in compiled.per_candidate:  # computed per (decision, candidate) row: a schema feature
                per_candidate.append(
                    Feature(
                        col,
                        "schema",
                        typ,
                        "extractor_candidate",
                        description=plan.get("description", ""),
                        provenance=prov,
                    )
                )
                continue
            feats.append(
                Feature(
                    col, "state", typ, "extractor", description=plan.get("description", ""), provenance=prov
                )
            )
        if kind == "single":
            spec = DecisionSpec("single", GROUP, "label", decision=(DECISION,))
        else:
            spec = DecisionSpec(kind, GROUP, LABEL, CANDIDATE, decision=(DECISION,))
            feats += [
                Feature("candidate_name", "schema", "cat"),
                Feature("candidate_description", "schema", "text"),
                Feature("candidate_args", "schema", "text"),
                *per_candidate,
            ]
            context = [f.name for f in feats if f.source == "extractor" and f.type == "text"]
            cand_text = ["candidate_name", "candidate_description"]
            dep = set(cfg.dependency_features)
            if context and "context_candidate_lex" in dep:
                feats.append(
                    Feature(
                        "context_candidate_lex", "schema", "num", "jaccard", {"a": context, "b": cand_text}
                    )
                )
            use_emb = cfg.pair_embeddings if embeddings is None else embeddings
            if use_emb is None:
                from tabagent.auto import embeddings_available

                use_emb = embeddings_available()
            if context and use_emb and "context_candidate_sim" in dep:
                feats.append(
                    Feature(
                        "context_candidate_sim",
                        "dependency",
                        "num",
                        "fitted:embedding_cosine",
                        {"query": context, "passage": cand_text, "encoder": "e5-small-v2"},
                    )
                )
            if "candidate_prior" in dep:
                feats.append(
                    Feature(
                        "candidate_prior", "dependency", "num", "fitted:candidate_prior", {"key": CANDIDATE}
                    )
                )
            if compiled.co_usage_context and "co_usage" in dep:
                feats.append(
                    Feature(
                        "co_usage",
                        "dependency",
                        "num",
                        "fitted:co_usage",
                        {"context": compiled.co_usage_context, "key": CANDIDATE},
                    )
                )
        card = FeatureCard(
            name,
            spec,
            feats,
            synth_text=[f.name for f in feats if f.source == "extractor" and f.type == "text"],
            synth_k=4,
            provenance={
                "kind": "tabschema-llm",
                "model": self.llm.model,
                **({"models": self.models()} if len(set(self.models().values())) > 1 else {}),
                "decision_points": n_dps,
                "prompts_sha256": {p: prompts.sha256(p) for p in prompts.PROMPTS},
                # what the card cost, including answers reused from the run directory at their first cost
                "cost_usd": round(self.meter.total_usd + (self.meter.cached_usd or 0.0), 4),
            },
            extractor={
                "path": "extract.py",
                "sha256": compiled.sha256,
                "entrypoint": "extract_features_for_classification",
                **({"candidate_entrypoint": CANDIDATE_FUNCTION} if per_candidate else {}),
            },
        )
        return card  # the relative features are the recipe search's challenger (auto.select), never here

    def build_table(
        self,
        result: DiscoveryResult,
        trajectories: list[Trajectory],
        target: str,
        trust_extractor: bool = False,
    ) -> DecisionTable:
        """The decision table: one row per decision point x candidate, with the extractor's features computed
        on each prefix and labels from the adapter (for tool shortlisting, the tools the successful run
        used; Sec. 4.1). The extractor runs in the sandbox unless ``trust_extractor``."""
        dps = [dp for t in trajectories if t.success for dp in self.adapter.decision_points(t, target)]
        runner = ExtractorRunner(result.extractor.code, trust_extractor, self.sandbox)
        feats, per = extract_rows(runner, result.card, dps, self.adapter)
        return rows_to_table(result.card, dps, feats, self.adapter, per)


def candidate_ids(
    adapter, dps: list[DecisionPoint], probe: int = 200, kind: str = "single"
) -> frozenset[str]:
    """The candidate ids of the first ``probe`` decision points (empty when the adapter gives none), which
    tell the compile step which extractor column lists earlier candidates (the co-usage context). For a
    ranking or multi-label decision only an adapter without candidates (``NotImplementedError``) gives none;
    any other error of the adapter is logged and raised, as it must not quietly turn the per-candidate
    features off."""
    try:
        return frozenset(c.id for dp in dps[:probe] for c in adapter.candidates(dp))
    except NotImplementedError:  # an adapter without candidates
        return frozenset()
    except Exception as e:
        if kind == "single":  # a single-class decision need not have candidates
            return frozenset()
        logger.error("TabSchema: the adapter's candidates failed: %s: %s", type(e).__name__, e)
        raise


def extract_rows(
    runner: ExtractorRunner, card: FeatureCard, dps: list[DecisionPoint], adapter
) -> tuple[list[dict | None], list[dict[str, dict | None]] | None]:
    """The extractor's features for ``dps``, as :func:`rows_to_table` takes them: per decision point, and,
    when the card has per-candidate features, per candidate of each, by candidate id (one sandbox call per
    decision)."""
    prefixes = [dp.prefix for dp in dps]
    if not any(f.source == "extractor_candidate" for f in card.features) or not runner.has_candidate_function:
        return runner.run(prefixes), None
    candidates = candidate_lists([adapter.candidates(dp) for dp in dps])
    feats, per = runner.run_items(prefixes, candidates)
    by_id = [
        None if values is None else {c["id"]: v for c, v in zip(cands, values, strict=True)}
        for cands, values in zip(candidates, per, strict=True)
    ]
    return feats, by_id


def rows_to_table(
    card: FeatureCard,
    dps: list[DecisionPoint],
    feats: list[dict | None],
    adapter,
    candidate_feats: list[dict[str, dict | None] | None] | None = None,
) -> DecisionTable:
    """Assemble the decision table from decision points, their extracted features and the adapter's labels.
    ``candidate_feats`` holds the per-candidate features of each decision point by candidate id (None where
    there are none); they fill the card's per-candidate columns of each candidate's row."""
    cols = [f.name for f in card.features if f.source == "extractor"]
    per_cols = [f.name for f in card.features if f.source == "extractor_candidate"]
    rows = []
    for i, (dp, f) in enumerate(zip(dps, feats, strict=True)):
        base = {
            GROUP: str(dp.group or dp.trajectory_id),
            DECISION: dp.id,
            **{c: (f or {}).get(c) for c in cols},
        }
        if card.spec.kind == "single":
            labels = adapter.labels(dp) or {None}
            if len(labels) != 1:
                raise ValueError(
                    f"kind='single' needs one label per decision, but {dp.id} has {len(labels)} "
                    f"({sorted(map(str, labels))[:5]}); use kind='ranking' or 'multilabel'"
                )
            (label,) = labels
            rows.append({**base, "label": label})
            continue
        chosen = adapter.labels(dp)
        cands = adapter.candidates(dp)
        per = (candidate_feats[i] if candidate_feats is not None else None) or {}
        for c in cands:
            row = {
                **base,
                CANDIDATE: c.id,
                "candidate_name": c.name or c.id,
                "candidate_description": c.description,
                "candidate_args": c.arg_hints,
            }
            if per_cols:
                row.update({col: (per.get(c.id) or {}).get(col) for col in per_cols})
            row[LABEL] = int(c.id in chosen)
            rows.append(row)
    return DecisionTable(pd.DataFrame(rows), card.spec)


def _secs(t0: float) -> float:
    return round(time.monotonic() - t0, 2)


def _error_kind(error: str) -> str:
    """The kind of a recorded error (its exception name, or what failed)."""
    text = str(error)
    if text.startswith("infeasible after"):
        return "code never ran"
    head = text.split(":", 1)[0].strip()
    return head if head and " " not in head else text[:60]


def _event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(f"{e.get('stage')}: {e.get('event')}" for e in events))


def drops_by_reason(report: DiscoveryReport, cfg: TabSchemaConfig) -> dict[str, int]:
    """Every dropped decision point, proposal, feature and column of a run, counted by reason."""
    out: Counter = Counter()
    for a in report.analyzed:
        if a.get("error"):
            out[f"analysis: decision point failed ({_error_kind(a['error'])})"] += 1
        else:
            out["analysis: proposal flagged by the CodeValidator"] += len(a.get("proposed") or []) - len(
                a.get("kept") or []
            )
    out["FeatureAggregator: rejected"] += len(report.rejected_by_aggregator)
    for j in report.judged:
        if j.get("final_decision") not in cfg.accept:
            out[f"MetaJudge: {j.get('final_decision')}"] += 1
    for reason in report.dropped.values():
        out[drop_reason(reason)] += 1
    return {k: v for k, v in out.most_common() if v}
