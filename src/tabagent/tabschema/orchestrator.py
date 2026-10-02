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

import numpy as np
import pandas as pd

from tabagent.defaults import DEFAULT_SEED
from tabagent.table import DecisionSpec, DecisionTable
from tabagent.tabschema import prompts
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
from tabagent.tabschema.card import Feature, FeatureCard
from tabagent.tabschema.compile import CompiledExtractor, ExtractionError, ExtractorRunner, compile_extractor
from tabagent.tabschema.judges import DECISIONS, Judge, MetaJudge, judge_table, run_judges
from tabagent.tabschema.llm import LLM, BudgetExceeded, CostMeter, auto_llm
from tabagent.tabschema.runstore import RunStore
from tabagent.tabschema.sandbox import SubprocessSandbox
from tabagent.tabschema.trajectory import (
    DecisionPoint,
    GenericJsonAdapter,
    Trajectory,
    union_paths,
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
    n_analyze: decision points the FeatureAnalyzer reads, sampled one per task first.
    sample_seed: seed of that sample.
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
    """

    model: str | None = None
    n_analyze: int = 10
    sample_seed: int = DEFAULT_SEED
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


STAGE_ROLES = (
    "analyzer",
    "extractor",
    "validator",
    "aggregator",
    "judge_relevance",
    "judge_generality",
    "judge_impact",
    "meta_judge",
)


@dataclass
class DiscoveryReport:
    """What a discovery run did: the sampled and analyzed decision points, the aggregated, judged, compiled
    and dropped features, the LLM cost and the run directory."""

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
        if self.dropped:
            lines += ["", "## Dropped", ""] + [f"* `{k}`: {v}" for k, v in self.dropped.items()]
        return "\n".join(lines) + "\n"


@dataclass
class DiscoveryResult:
    """The output of ``TabSchema.discover``: the feature card, the compiled extractor and the report."""

    card: FeatureCard
    extractor: CompiledExtractor
    report: DiscoveryReport

    def save(self, path: str | Path) -> Path:
        """Write ``extract.py``, ``card.yaml``, ``discovery.md`` and ``discovery.json`` to ``path``."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "extract.py").write_text(self.extractor.code)
        self.card.save(out / "card.yaml")
        (out / "discovery.md").write_text(self.report.to_markdown())
        report = dataclasses.asdict(self.report)
        (out / "discovery.json").write_text(json.dumps(report, indent=1, default=str))
        return out


def sample_decision_points(dps: list[DecisionPoint], n: int, seed: int) -> list[DecisionPoint]:
    """Seeded sample that covers as many distinct tasks as possible before repeating one."""
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[DecisionPoint]] = {}
    for dp in dps:
        by_group.setdefault(str(dp.group), []).append(dp)
    groups = sorted(by_group)
    rng.shuffle(groups)
    out: list[DecisionPoint] = []
    rnd = 0
    while len(out) < n and any(len(v) > rnd for v in by_group.values()):
        for g in groups:
            if rnd < len(by_group[g]) and len(out) < n:
                out.append(by_group[g][rnd])
        rnd += 1
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
            compiled, decisions = self._run(dps, target, store, report, candidate_ids(self.adapter, dps))
        except Exception as e:  # leave a partial report next to the paid answers, then fail
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
    ) -> tuple[CompiledExtractor, list[dict]]:
        """The stages of :meth:`discover` after its checks: analysis, aggregation, judging and compilation.
        Fills ``report`` as it goes; returns the compiled extractor and the MetaJudge's decisions."""
        cfg = self.config
        known = union_paths(dp.prefix for dp in dps)
        outcome_keys = tuple(getattr(self.adapter, "outcome_keys", ()))
        sample = sample_decision_points(dps, cfg.n_analyze, cfg.sample_seed)
        store.write_json("sample.json", [dp.id for dp in sample])
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
        )
        store.write_text("extract.py", compiled.code)
        report.compiled = dict(compiled.features)
        report.dropped.update(compiled.dropped)
        report.notes += compiled.notes
        return compiled, decisions

    # ------------------------------------------------------------------ card and table
    def _card(
        self, compiled: CompiledExtractor, decisions: list[dict], kind: str, name: str, n_dps: int
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
            use_emb = cfg.pair_embeddings
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
