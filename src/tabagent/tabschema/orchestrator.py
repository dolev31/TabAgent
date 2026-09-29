"""TabSchema, the FeatureOrchestrator (paper Sec. 3.1, App. C).

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
6. the **card**: extractor features (state), candidate name/description/argument hints (schema, L321),
   and in-fold dependency statistics (priors, context-candidate similarity, L443).

``card_factory`` runs the whole loop inside each training fold, which is how the paper's statement that the
TabSchema loop is fitted inside the training portion of each fold (L443) is executed.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    premerge,
)
from tabagent.tabschema.card import Feature, FeatureCard
from tabagent.tabschema.compile import CompiledExtractor, ExtractorRunner, compile_extractor
from tabagent.tabschema.judges import DECISIONS, Judge, MetaJudge, judge_table, run_judges
from tabagent.tabschema.llm import LLM, CostMeter, auto_llm
from tabagent.tabschema.runstore import RunStore
from tabagent.tabschema.sandbox import SubprocessSandbox
from tabagent.tabschema.trajectory import (
    DecisionPoint,
    GenericJsonAdapter,
    Trajectory,
    union_paths,
)

GROUP, DECISION, CANDIDATE, LABEL = "task", "decision_id", "candidate", "chosen"


@dataclass
class TabSchemaConfig:
    model: str | None = None  # default: the paper's GPT-5.4 on OpenAI, or the provider found in the env
    n_analyze: int = 10
    sample_seed: int = 20260720
    max_code_retries: int = 3  # paper L1210
    json_retries: int = 1
    accept: tuple[str, ...] = ("accept",)  # add "conditional" to widen
    compile_min_success: float = 0.95
    max_null_rate: float = 0.5
    dependency_features: tuple[str, ...] = (
        "candidate_prior",
        "context_candidate_lex",
        "context_candidate_sim",
    )
    pair_embeddings: bool | None = None  # None: use e5 similarity when the embed extra is installed
    analyzer_hints: list[str] | None = None  # the original pipeline's example list; off (paper prompt)
    workers: int = 8
    timeout_s: float = 30
    max_cost_usd: float | None = None
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class DiscoveryReport:
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

    def to_markdown(self) -> str:
        lines = [
            f"# TabSchema discovery: `{self.target}`",
            "",
            f"* trajectories: {self.n_trajectories} ({self.n_successful} successful); decision points: "
            f"{self.n_decision_points}; analyzed: {len(self.analyzed)}",
            f"* LLM cost: ${self.cost.get('total_usd', 0):.2f}",
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
    card: FeatureCard
    extractor: CompiledExtractor
    report: DiscoveryReport

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "extract.py").write_text(self.extractor.code)
        self.card.save(out / "card.yaml")
        (out / "discovery.md").write_text(self.report.to_markdown())
        (out / "discovery.json").write_text(json.dumps(self.report.__dict__, indent=1, default=str))
        return out


def _sample(dps: list[DecisionPoint], n: int, seed: int) -> list[DecisionPoint]:
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
    def __init__(
        self,
        llm: LLM | None = None,
        config: TabSchemaConfig | None = None,
        adapter=None,
        store: RunStore | None = None,
        sandbox: SubprocessSandbox | None = None,
    ):
        self.config = config or TabSchemaConfig()
        self.llm = llm or auto_llm(self.config.model)
        self.adapter = adapter or GenericJsonAdapter()
        self.store = store
        self.sandbox = sandbox or SubprocessSandbox(timeout_s=self.config.timeout_s)
        self.meter = CostMeter(prices=dict(self.config.prices), max_cost_usd=self.config.max_cost_usd)

    def _stage(self, cls, *args):
        return cls(*args, self.llm, self.meter, self.store, self.config.json_retries)

    # ------------------------------------------------------------------ discovery
    def discover(
        self,
        trajectories: list[Trajectory] | str | Path,
        target: str,
        kind: str = "ranking",
        name: str | None = None,
    ) -> DiscoveryResult:
        cfg = self.config
        trajs = (
            self.adapter.load(trajectories) if isinstance(trajectories, (str, Path)) else list(trajectories)
        )
        store = self.store or RunStore()
        self.store = store
        good = [t for t in trajs if t.success]
        dps = [dp for t in good for dp in self.adapter.decision_points(t, target)]
        if not dps:
            raise ValueError(f"no occurrences of {target!r} in the successful trajectories")
        known = union_paths(dp.prefix for dp in dps)
        outcome_keys = tuple(getattr(self.adapter, "outcome_keys", ()))
        report = DiscoveryReport(target, len(trajs), len(good), len(dps), run_dir=str(store.dir))
        store.write_json(
            "config.json",
            {
                **cfg.__dict__,
                "target": target,
                "kind": kind,
                "model": self.llm.model,
                "prompts_sha256": {p: prompts.sha256(p) for p in prompts.PROMPTS},
            },
        )

        sample = _sample(dps, cfg.n_analyze, cfg.sample_seed)
        store.write_json("sample.json", [dp.id for dp in sample])
        analyzer, extractor = self._stage(FeatureAnalyzer), self._stage(CodeExtractor)
        validator = self._stage(CodeValidator)

        def analyze(dp: DecisionPoint) -> dict[str, Any]:
            rec: dict[str, Any] = {"decision_point": dp.id, "proposed": [], "kept": [], "error": ""}
            try:
                specs = analyzer(dp, target, cfg.analyzer_hints)
                rec["proposed"] = [s.feature_name for s in specs]
                executor = CodeExecutor(self.sandbox, known, outcome_keys)
                attempt, history = extract_with_retries(
                    extractor, executor, specs, [dp.prefix], dp.prefix, target, cfg.max_code_retries
                )
                rec["attempts"] = len(history)
                if not attempt.ok:
                    rec["error"] = f"infeasible after {len(history)} attempts: {attempt.error[:300]}"
                    return {**rec, "specs": []}
                verdicts = validator(specs, attempt.code, dp.prefix, [attempt.results[0].features], target)
                kept = [s for s in specs if verdicts.get(s.key, "valid") == "valid"]
                rec["kept"] = [s.feature_name for s in kept]
                rec["validator"] = verdicts
                return {**rec, "specs": [s.to_dict() for s in kept]}
            except LLMFormatError as e:
                rec["error"] = str(e)
                return {**rec, "specs": []}

        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            analyses = list(pool.map(analyze, sample))
        report.analyzed = [{k: v for k, v in a.items() if k != "specs"} for a in analyses]
        store.write_json("analysis.json", analyses)
        proposals = premerge([FeatureSpec.from_dict(s) for a in analyses for s in a["specs"]])
        if not proposals:
            raise RuntimeError("no feasible feature proposals; see analysis.json in the run directory")

        filtered, rejected = self._stage(FeatureAggregator)(proposals)
        report.aggregated = [f.feature_name for f in filtered]
        report.rejected_by_aggregator = [f.feature_name for f in rejected]
        store.write_json(
            "aggregator.json",
            {"filtered": [f.to_dict() for f in filtered], "rejected": [f.to_dict() for f in rejected]},
        )

        verdicts = run_judges(filtered, lambda d: self._stage(Judge, d))
        table = judge_table(verdicts, proposals)
        decisions = self._stage(MetaJudge)(verdicts, table)
        store.write_json("judges.json", {"verdicts": verdicts, "table": table, "meta": decisions})
        accept = {d for d in cfg.accept if d in DECISIONS}
        accepted_keys = {
            FeatureSpec(d["feature_name"]).key for d in decisions if d["final_decision"] in accept
        }
        accepted = [f for f in filtered if f.key in accepted_keys]
        scores = {row["feature_name"]: row for row in table}
        for d in decisions:
            row = scores.get(d["feature_name"], {})
            report.judged.append(
                {
                    "feature_name": d["feature_name"],
                    "final_decision": d["final_decision"],
                    "meta_score": d.get("meta_score"),
                    **{k: row.get(f"judge_{k}_score") for k in ("relevance", "generality", "impact")},
                }
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
        )
        store.write_text("extract.py", compiled.code)
        report.compiled, report.dropped = dict(compiled.features), dict(compiled.dropped)
        report.cost = self.meter.summary()
        card = self._card(compiled, decisions, kind, name or f"{target}_tabschema", len(dps))
        card.save(store.path("card.yaml"))
        store.write_text("report.md", report.to_markdown())
        return DiscoveryResult(card, compiled, report)

    # ------------------------------------------------------------------ card and table
    def _card(
        self, compiled: CompiledExtractor, decisions: list[dict], kind: str, name: str, n_dps: int
    ) -> FeatureCard:
        cfg = self.config
        meta = {d["feature_name"]: d for d in decisions}
        feats: list[Feature] = []
        for col, typ in compiled.features.items():
            plan = compiled.specs.get(col, {})
            prov = {"plan": plan, "meta_judge": meta.get(plan.get("feature_name"), {})}
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
                from tabagent.auto import _embeddings_available

                use_emb = _embeddings_available()
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
        return FeatureCard(
            name,
            spec,
            feats,
            synth_text=[f.name for f in feats if f.source == "extractor" and f.type == "text"],
            synth_k=4,
            provenance={
                "kind": "tabschema-llm",
                "model": self.llm.model,
                "decision_points": n_dps,
                "prompts_sha256": {p: prompts.sha256(p) for p in prompts.PAPER_PROMPTS},
                "cost_usd": round(self.meter.total_usd, 4),
                "run_dir": str(self.store.dir if self.store else ""),
            },
            extractor={
                "path": "extract.py",
                "sha256": compiled.sha256,
                "entrypoint": "extract_features_for_classification",
            },
        )

    def build_table(
        self, result: DiscoveryResult, trajectories: list[Trajectory], target: str, trusted: bool = False
    ) -> DecisionTable:
        """Rows = decision point x candidate, extractor features computed on each prefix, labels from the
        adapter (for tool shortlisting: the tools the successful run used, L371)."""
        dps = [dp for t in trajectories if t.success for dp in self.adapter.decision_points(t, target)]
        runner = ExtractorRunner(result.extractor.code, trusted, self.sandbox)
        return rows_to_table(result.card, dps, runner.run([dp.prefix for dp in dps]), self.adapter)


def rows_to_table(
    card: FeatureCard, dps: list[DecisionPoint], feats: list[dict | None], adapter
) -> DecisionTable:
    cols = [f.name for f in card.features if f.source == "extractor"]
    rows = []
    for dp, f in zip(dps, feats, strict=True):
        base = {
            GROUP: str(dp.group or dp.trajectory_id),
            DECISION: dp.id,
            **{c: (f or {}).get(c) for c in cols},
        }
        if card.spec.kind == "single":
            (label,) = adapter.labels(dp) or {None}
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
