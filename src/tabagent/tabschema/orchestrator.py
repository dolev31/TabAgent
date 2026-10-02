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
   training rows of each fold (Sec. 4.3).

The release adds three checks the paper does not have, between compiling and the card: the compile step's
**verification** of the values (determinism, outcome and future invariance, robustness; ``verify``) with
golden tests saved as ``extractor_tests.json``; the **OutputAuditor**, an LLM judge of the values on a sample
of decision points (``audit``); and a **usefulness** estimate of each feature by grouped cross-validation on
the onboarding rows (``usefulness``). Each can be turned off in ``TabSchemaConfig``.

Discovery runs once, on the onboarding trajectories it is given. The compiled extractor is a stateless
function of one trajectory prefix, so applying it to held-out decisions leaks nothing; every label-derived
statistic is fitted by the card inside ``cross_validate``'s training folds. Running discovery itself inside
each fold means calling ``discover`` on that fold's trajectories.

A decision point whose analysis fails (an unusable answer, a provider error, code that never runs) is
skipped and recorded in ``analysis.json``; the run goes on with the others. A run that stops (a spent budget,
no feature accepted, an extractor that cannot be compiled) leaves a partial ``report.md`` and every paid
answer in its run directory: rerunning with that directory resumes, and cached answers cost nothing.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from tabagent.defaults import DEFAULT_SEED
from tabagent.table import DecisionSpec, DecisionTable
from tabagent.tabschema import prompts
from tabagent.tabschema import verify as V
from tabagent.tabschema.agents import (
    CodeExecutor,
    CodeExtractor,
    CodeValidator,
    FeatureAggregator,
    FeatureAnalyzer,
    FeatureSpec,
    LLMFormatError,
    extract_with_retries,
    normalize_name,
    premerge,
)
from tabagent.tabschema.audit import OutputAuditor, run_audit
from tabagent.tabschema.card import Feature, FeatureCard
from tabagent.tabschema.compile import (
    CompiledExtractor,
    ExtractionError,
    ExtractorRunner,
    compile_extractor,
    context_texts,
)
from tabagent.tabschema.judges import DECISIONS, Judge, MetaJudge, judge_table, run_judges
from tabagent.tabschema.llm import LLM, BudgetExceeded, CostMeter, auto_llm
from tabagent.tabschema.runstore import RunStore
from tabagent.tabschema.sampling import STRATEGIES, coverage, sample_decision_points
from tabagent.tabschema.sandbox import SubprocessSandbox
from tabagent.tabschema.trajectory import (
    DecisionPoint,
    GenericJsonAdapter,
    Trajectory,
    union_paths,
)
from tabagent.tabschema.usefulness import (
    DEFAULT_MAX_DECISIONS,
    DEFAULT_MAX_ROWS,
    MIN_DECISIONS,
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
    accept: MetaJudge outcomes kept. The default keeps only ``accept``: conditionally accepted features are
        left out unless ``"conditional"`` is added.
    compile_min_success: share of onboarding decision points the compiled extractor must run on.
    max_null_rate: extractor columns emptier than this are dropped.
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
    """

    model: str | None = None
    n_analyze: int = 20
    sample_seed: int = DEFAULT_SEED
    sample_strategy: str = "coverage"
    max_code_retries: int = 3
    json_retries: int = 1
    accept: tuple[str, ...] = ("accept",)  # add "conditional" to widen
    compile_min_success: float = 0.95
    max_null_rate: float = 0.5
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

    def __post_init__(self) -> None:
        limits = {
            "verify_samples": (self.verify_samples, 1),
            "goldens": (self.goldens, 0),
            "audit_samples": (self.audit_samples, 1),
            "usefulness_max_decisions": (self.usefulness_max_decisions, 2),
            "usefulness_max_rows": (self.usefulness_max_rows, 1),
        }
        for name, (value, low) in limits.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < low:
                raise ValueError(
                    f"TabSchemaConfig.{name} must be an integer of at least {low}, not {value!r}"
                )
        if not 0 <= self.audit_max_incorrect < 1:
            raise ValueError(
                f"TabSchemaConfig.audit_max_incorrect must be in [0, 1), not {self.audit_max_incorrect!r}"
            )


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
        lines += ["", "## Compiled extractor features", ""] + [
            f"* `{k}` ({v})" for k, v in self.compiled.items()
        ]
        lines += _verification_md(self.verification) + _audit_md(self.audit) + _usefulness_md(self.usefulness)
        if self.dropped:
            lines += ["", "## Dropped", ""] + [f"* `{k}`: {v}" for k, v in self.dropped.items()]
        return "\n".join(lines) + "\n"


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
        f"* drop-column importance on the onboarding rows: {u['decisions']} decision points "
        f"({u.get('rows', '?')} rows) of {u['tasks']} tasks, {u['folds']} task-grouped folds, CatBoost with "
        f"{u['iterations']} iterations, {u.get('seeds', 1)} seed(s); {m} {u.get('base', float('nan')):.3f} "
        "with every feature"
    )
    pruning = u.get("pruned") is not None
    out.append(
        f"* delta: the head's {m} with the feature minus without it (95% bootstrap interval over tasks); "
        "useful or harmful when the Holm-adjusted bootstrap p-value is below 0.05"
        + (
            " under every seed; harmful features are pruned"
            if pruning
            else "; a report only (prune_harmful=False)"
        )
    )
    out += ["", f"| feature | delta {m} | 95% interval | Holm p | verdict |", "|---|---|---|---|---|"]
    for col, r in sorted(u.get("features", {}).items(), key=lambda kv: -kv[1]["delta"]):
        out.append(
            f"| {col} | {r['delta']:+.4f} | [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r.get('p_holm', '')} | "
            f"{r['verdict']} |"
        )
    return out


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
        (out / "extract.py").write_text(self.extractor.code)
        if self.extractor.tests:
            (out / V.TESTS_FILE).write_text(json.dumps(self.extractor.tests, indent=1, ensure_ascii=False))
        self.card.save(out / "card.yaml")
        (out / "discovery.md").write_text(self.report.to_markdown())
        report = dataclasses.asdict(self.report)
        (out / "discovery.json").write_text(json.dumps(report, indent=1, default=str))
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
        self.llm = llm or auto_llm(self.config.model)
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
                self.llms[name] = auto_llm(name)
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
        fb = self.config.fallback_model
        st.fallback = self.llms[fb] if fb else None
        return st

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
        written to the run directory; a run that stops leaves a partial ``report.md`` there, and rerunning
        with the same directory resumes it.
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
        report = DiscoveryReport(target, len(trajs), len(good), len(dps), run_dir=str(store.dir))
        store.write_json(
            "config.json",
            {
                **cfg.__dict__,
                "target": target,
                "kind": kind,
                "model": self.llm.model,
                "stage_models_resolved": self.models(),
                "prompts_sha256": {p: prompts.sha256(p) for p in prompts.PROMPTS},
            },
        )
        try:
            future = self.check_adapter(good, target, dps)  # before any LLM call: a leak costs nothing
            if future["notes"]:
                report.notes += future["notes"]
            compiled, decisions = self._run(
                dps, target, store, report, candidate_ids(self.adapter, dps), good, kind, future
            )
        except Exception as e:  # leave a partial report next to the paid answers, then fail
            if isinstance(e, V.PrefixLeakError):
                report.verification = e.record
                report.notes.append(f"onboarding stopped before any LLM call: {e}")
            report.cost = self.meter.summary()
            store.write_text("report.md", report.to_markdown())
            if isinstance(e, (BudgetExceeded, ExtractionError, LLMFormatError)):
                raise type(e)(
                    f"{e}. The partial report and every paid answer are in {store.dir}: rerun with that run "
                    "directory to resume (cached answers cost nothing)"
                ) from e
            raise
        report.cost = self.meter.summary()
        logger.info(
            "TabSchema: compiled an extractor with %d features (%d dropped); LLM cost %s",
            len(compiled.features),
            len(compiled.dropped),
            self.meter.describe(self.llm.model),
        )
        card = self._card(compiled, decisions, kind, name or f"{target}_tabschema", len(dps))
        card.save(store.path("card.yaml"))
        store.write_text("report.md", report.to_markdown())
        return DiscoveryResult(card, compiled, report)

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
        known = union_paths(dp.prefix for dp in dps)
        outcome_keys = tuple(getattr(self.adapter, "outcome_keys", ()))
        labels = getattr(self.adapter, "labels", None)
        sample = sample_decision_points(dps, cfg.n_analyze, cfg.sample_seed, labels, cfg.sample_strategy)
        store.write_json("sample.json", [dp.id for dp in sample])
        covered = coverage(dps, sample, labels)
        store.write_json("sample_coverage.json", covered)
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
        analyzer, extractor = self._stage(FeatureAnalyzer), self._stage(CodeExtractor)
        analyzer.max_chars = cfg.max_prompt_chars
        validator = self._stage(CodeValidator)

        def analyze(dp: DecisionPoint) -> dict[str, Any]:
            rec: dict[str, Any] = {"decision_point": dp.id, "proposed": [], "kept": [], "error": ""}
            try:
                specs = analyzer(dp, target)
                rec["proposed"] = [s.feature_name for s in specs]
                executor = CodeExecutor(self.sandbox, known, outcome_keys)
                attempt, history = extract_with_retries(
                    extractor, executor, specs, [dp.prefix], dp.prefix, target, cfg.max_code_retries
                )
                rec["attempts"] = len(history)
                if not attempt.ok:
                    rec["error"] = f"infeasible after {len(history)} attempts: {attempt.error[:300]}"
                    return {**rec, "specs": []}
                try:
                    verdicts = validator(
                        specs, attempt.code, dp.prefix, [attempt.results[0].features], target
                    )
                except LLMFormatError as e:  # the deterministic checks passed; keep the proposals unjudged
                    verdicts, rec["validator_error"] = {}, str(e)[:300]
                kept = [s for s in specs if verdicts.get(s.key, "valid") == "valid"]
                rec["kept"] = [s.feature_name for s in kept]
                rec["validator"] = verdicts
                return {**rec, "specs": [s.to_dict() for s in kept]}
            except BudgetExceeded:
                raise
            except Exception as e:  # one decision point's failure (a bad answer, a provider error) is skipped
                logger.warning("TabSchema: analyzing %s failed: %s: %s", dp.id, type(e).__name__, e)
                rec["error"] = f"{type(e).__name__}: {e}"[:500]
                return {**rec, "specs": []}

        pool = ThreadPoolExecutor(max_workers=cfg.workers)
        try:
            analyses = list(pool.map(analyze, sample))
        finally:  # on an error (a spent budget, say) drop the decision points not yet started
            pool.shutdown(wait=True, cancel_futures=True)
        logger.info(
            "TabSchema: analyzed %d decision points; %d feasible proposals",
            len(analyses),
            sum(len(a["specs"]) for a in analyses),
        )
        report.analyzed = [{k: v for k, v in a.items() if k != "specs"} for a in analyses]
        store.write_json("analysis.json", analyses)
        proposals = premerge([FeatureSpec.from_dict(s) for a in analyses for s in a["specs"]])
        if not proposals:
            errors = [a["error"] for a in analyses if a["error"]]
            first = f"; first error: {errors[0]}" if errors else ""
            raise RuntimeError(
                f"no feasible feature proposals from the {len(analyses)} analyzed decision points{first}; "
                f"see analysis.json in {store.dir}"
            )

        filtered, rejected = self._stage(FeatureAggregator)(proposals)
        logger.info("TabSchema: aggregated %d proposals into %d features", len(proposals), len(filtered))
        report.aggregated = [f.feature_name for f in filtered]
        report.rejected_by_aggregator = [f.feature_name for f in rejected]
        store.write_json(
            "aggregator.json",
            {"filtered": [f.to_dict() for f in filtered], "rejected": [f.to_dict() for f in rejected]},
        )
        proposed = {p.key for p in proposals}
        renamed = [f.feature_name for f in filtered if f.key not in proposed]
        if renamed:
            report.notes.append(
                "the FeatureAggregator kept features under names no analyzer proposed (judged without the "
                f"analyzer's view): {', '.join(renamed)}"
            )
        if not filtered:
            raise ExtractionError(
                f"the FeatureAggregator kept none of the {len(proposals)} proposals; see aggregator.json"
            )

        verdicts = run_judges(filtered, lambda d: self._stage(Judge, d), workers=max(3, cfg.workers))
        table = judge_table(verdicts, proposals)
        decisions = self._stage(MetaJudge)(verdicts, table)
        store.write_json("judges.json", {"verdicts": verdicts, "table": table, "meta": decisions})
        accept = {d for d in cfg.accept if d in DECISIONS}
        by_key = {normalize_name(d["feature_name"]): d for d in decisions}
        accepted = [f for f in filtered if f.key in by_key and by_key[f.key]["final_decision"] in accept]
        kept_keys = {f.key for f in filtered}
        for key, d in by_key.items():
            if key not in kept_keys:
                report.dropped[str(d["feature_name"])] = (
                    "the MetaJudge decided on a name the FeatureAggregator did not keep (renamed or invented)"
                )
        for f in filtered:
            if f.key not in by_key:
                report.dropped[f.feature_name] = "the MetaJudge gave no decision on it"
        logger.info("TabSchema: judged %d features; the MetaJudge accepted %d", len(filtered), len(accepted))
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
        if not accepted:
            counts = {
                k: sum(by_key[f.key]["final_decision"] == k for f in filtered if f.key in by_key)
                for k in DECISIONS
            }
            raise ExtractionError(
                f"the MetaJudge accepted none of the {len(filtered)} aggregated features ({counts}); see "
                f"judges.json. TabSchemaConfig(accept=('accept', 'conditional')) also keeps conditional ones"
            )

        if future is None and trajectories:  # a caller that skipped discover's check
            future = self.check_adapter(trajectories, target, dps)
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
            )
        except V.PrefixLeakError as e:  # the report says so, next to every paid answer
            report.verification = e.record
            report.notes.append(f"onboarding stopped: {e}")
            raise
        store.write_text("extract.py", compiled.code)
        report.verification = {k: v for k, v in compiled.verification.items() if k != "seconds"}
        if cfg.audit:
            self._audit(compiled, dps, target, store, report, frozenset(dp.id for dp in sample), labels)
        if cfg.usefulness:
            self._usefulness(compiled, dps, decisions, kind, store, report, outcome_keys)
        compiled.tests = V.restrict_tests(compiled.tests, compiled.features)
        if compiled.tests:
            store.write_json(V.TESTS_FILE, compiled.tests)
        report.compiled = dict(compiled.features)
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
    ) -> None:
        """Run the OutputAuditor on a sample of decision points and drop the features it judges incorrect too
        often; writes ``audit.json``. The audit only ever removes features: when it would remove them all, it
        removes none and says so."""
        cfg = self.config
        auditor = self._stage(OutputAuditor)
        columns = {c: compiled.specs[c] for c in compiled.llm_columns}
        logger.info(
            "TabSchema: auditing the values of %d features on %d decision points",
            len(columns),
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
            for (
                col
            ) in appended:  # the appended context text is library code: computed here, as in the sandbox
                texts = context_texts([dp.prefix for dp in points], col, outcome_keys)
                for f, text in zip(feats, texts, strict=True):
                    f[col] = text
            return feats

        def card_for(dropped: frozenset[str]) -> FeatureCard:
            return self._card(
                compiled.without(dropped), decisions, kind, "usefulness", len(dps), embeddings=False
            )

        try:
            table = rows_to_table(card_for(frozenset()), sample, features_of(sample), self.adapter)
            if len(table) > cfg.usefulness_max_rows and len(sample) > MIN_DECISIONS:  # bound the runtime
                n = max(MIN_DECISIONS, int(len(sample) * cfg.usefulness_max_rows / len(table)))
                sample = sample_decision_points(sample, n, cfg.sample_seed, strategy="tasks")
                table = rows_to_table(card_for(frozenset()), sample, features_of(sample), self.adapter)
            logger.info(
                "TabSchema: estimating the usefulness of %d features on %d decision points (%d rows)",
                len(compiled.features),
                len(sample),
                len(table),
            )
            rec = feature_usefulness(
                table,
                card_for,
                list(compiled.features),
                seed=cfg.sample_seed,
                seeds=2 if cfg.prune_harmful else 1,
            )
        except Exception as e:  # a report only: no candidates or labels, one label value, a fit that fails
            logger.warning("TabSchema: the usefulness estimate was skipped: %s: %s", type(e).__name__, e)
            rec = {"skipped": f"{type(e).__name__}: {e}"[:300]}
        if cfg.prune_harmful and not rec.get("skipped"):
            harmful = [
                c for c in compiled.llm_columns if rec["features"].get(c, {}).get("verdict") == "harmful"
            ]
            if harmful and len(harmful) >= len(compiled.llm_columns):
                compiled.notes.append(
                    "prune_harmful: every LLM-written feature looked harmful; none was pruned"
                )
                harmful = []
            for col in harmful:
                r = rec["features"][col]
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
        for col, typ in compiled.features.items():
            plan = compiled.specs.get(col, {})
            prov = {"plan": plan, "meta_judge": meta.get(normalize_name(plan.get("feature_name", "")), {})}
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
        return FeatureCard(
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
            },
        )

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
        return rows_to_table(result.card, dps, runner.run([dp.prefix for dp in dps]), self.adapter)


def candidate_ids(adapter, dps: list[DecisionPoint], probe: int = 200) -> frozenset[str]:
    """The candidate ids of the first ``probe`` decision points (empty when the adapter gives none), which
    tell the compile step which extractor column lists earlier candidates (the co-usage context)."""
    try:
        return frozenset(c.id for dp in dps[:probe] for c in adapter.candidates(dp))
    except Exception:  # an adapter without candidates (a single-class decision, say)
        return frozenset()


def rows_to_table(
    card: FeatureCard, dps: list[DecisionPoint], feats: list[dict | None], adapter
) -> DecisionTable:
    """Assemble the decision table from decision points, their extracted features and the adapter's labels."""
    cols = [f.name for f in card.features if f.source == "extractor"]
    rows = []
    for dp, f in zip(dps, feats, strict=True):
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
        for c in adapter.candidates(dp):
            rows.append(
                {
                    **base,
                    CANDIDATE: c.id,
                    "candidate_name": c.name or c.id,
                    "candidate_description": c.description,
                    "candidate_args": c.arg_hints,
                    LABEL: int(c.id in chosen),
                }
            )
    return DecisionTable(pd.DataFrame(rows), card.spec)
