"""One-call onboarding: agent logs + an LLM key -> a deployable decision head.

    import tabagent
    head = tabagent.onboard("./agent_logs", decision="ShortlisterAgent")   # the LLM comes from env keys
    print(head.report)            # grouped-CV quality vs baselines, latency, cost per decision
    head.save("my_head")
    head.decide(prefix, candidates, k=5)

``onboard`` runs the paper's lifecycle (Sec. 4: log the agent, extract TabSchema features from successful
runs, swap the generative module for TabHead): detect the log format, run TabSchema discovery, build the
decision table, select the training recipe by nested validation (TabSynth, prior, preset), evaluate with
grouped cross-validation, fit the final head and save a portable artifact. ``dry_run=True`` stops before any
LLM call and reports what discovery would process and cost.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabagent import auto
from tabagent.agent import TabAgent
from tabagent.tabhead import confidence, decode
from tabagent.table import DecisionTable
from tabagent.tabschema import prompts
from tabagent.tabschema.card import FeatureCard
from tabagent.tabschema.compile import ExtractorRunner
from tabagent.tabschema.llm import LLM, CostMeter, auto_llm
from tabagent.tabschema.orchestrator import (
    CANDIDATE,
    DECISION,
    GROUP,
    DiscoveryReport,
    TabSchema,
    TabSchemaConfig,
    _sample,
)
from tabagent.tabschema.runstore import RunStore
from tabagent.tabschema.trajectory import Candidate, Trajectory, get_json_schema

HEAD_FORMAT = 1


@dataclass
class DryRun:
    """What discovery would process, and an estimate of its size (4 characters per token)."""

    n_trajectories: int
    n_successful: int
    n_decision_points: int
    n_analyzed: int
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float | None
    components: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        cost = f"${self.est_cost_usd:.2f}" if self.est_cost_usd is not None else "unknown (pass prices=)"
        return (
            f"{self.n_trajectories} trajectories ({self.n_successful} successful), "
            f"{self.n_decision_points} decision points, {self.n_analyzed} analyzed by the LLM; "
            f"~{self.est_input_tokens:,} input / ~{self.est_output_tokens:,} output tokens; est. cost {cost}"
        )


def _as_candidates(items) -> list[Candidate]:
    out = []
    for c in items:
        if isinstance(c, Candidate):
            out.append(c)
        elif isinstance(c, str):
            out.append(Candidate(c, c))
        else:
            out.append(
                Candidate(
                    str(c["id"] if "id" in c else c["name"]),
                    c.get("name", ""),
                    c.get("description", ""),
                    c.get("arg_hints", c.get("args", "")),
                )
            )
    return out


@dataclass
class TabAgentHead:
    """A fitted TabAgent together with the TabSchema extractor that feeds it: the drop-in replacement."""

    agent: TabAgent
    target: str
    extractor_code: str | None = None
    report: Any = None
    discovery: DiscoveryReport | None = None
    gate_threshold: float | None = None
    trusted: bool = False

    @property
    def card(self) -> FeatureCard:
        return self.agent.card_

    def _runner(self) -> ExtractorRunner | None:
        if not self.extractor_code:
            return None
        if not hasattr(self, "_runner_cache"):
            self._runner_cache = ExtractorRunner(self.extractor_code, trusted=self.trusted)
        return self._runner_cache

    def features_for(self, prefix: dict[str, Any]) -> dict[str, Any]:
        runner = self._runner()
        return (runner.run([prefix])[0] or {}) if runner else {}

    def decide(
        self, prefix: dict[str, Any], candidates, k: int | None = None, defer_below: float | None = None
    ) -> dict[str, Any]:
        """Score one live decision: the trajectory prefix before the decision and the legal candidates."""
        cands = _as_candidates(candidates)
        feats = self.features_for(prefix)
        rows = [
            {
                GROUP: "live",
                DECISION: "live#0",
                **feats,
                CANDIDATE: c.id,
                "candidate_name": c.name or c.id,
                "candidate_description": c.description,
                "candidate_args": c.arg_hints,
            }
            for c in cands
        ]
        spec = self.card.spec
        table = DecisionTable(pd.DataFrame(rows), spec, has_labels=False)
        scores = self.agent.score(table)
        ids, cand_ids = ["live#0"] * len(rows), [c.id for c in cands]
        ranked = decode(spec.kind, scores, ids, cand_ids, k=k, threshold=self.agent.head_.threshold)
        conf = float(confidence(scores, ids, cand_ids).iloc[0])
        tau = defer_below if defer_below is not None else self.gate_threshold
        return {
            "choice": ranked["candidate"].tolist(),
            "scores": {c.id: float(s) for c, s in zip(cands, scores, strict=True)},
            "confidence": conf,
            "defer": bool(tau is not None and conf < tau),
        }

    def calibrate_gate(self, cv_result, target_coverage: float = 0.9) -> float:
        """Choose the deferral threshold so that ``target_coverage`` of decisions stay with the head (the
        least confident ``1 - target_coverage`` go to the LLM; App. D gate)."""
        t = cv_result.table
        ids = t.decision_ids.to_numpy()
        cands = t.df[t.spec.candidate].to_numpy()
        conf = np.concatenate(
            [confidence(cv_result.oof[s].to_numpy(float), ids, cands).to_numpy() for s in cv_result.seeds]
        )
        self.gate_threshold = float(np.quantile(conf, 1 - target_coverage))
        return self.gate_threshold

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        out = Path(path)
        self.agent.save(out / "model")
        if self.extractor_code:
            (out / "extract.py").write_text(self.extractor_code)
        meta = {
            "format_version": HEAD_FORMAT,
            "target": self.target,
            "gate_threshold": self.gate_threshold,
            "has_extractor": bool(self.extractor_code),
        }
        (out / "head.json").write_text(json.dumps(meta, indent=1))
        if self.report is not None:
            self.report.save(out)
        if self.discovery is not None:
            (out / "discovery.md").write_text(self.discovery.to_markdown())
        return out

    @classmethod
    def load(cls, path: str | Path, trust_extractor: bool = False) -> TabAgentHead:
        """Load a saved head. ``trust_extractor=True`` runs the TabSchema extractor in-process (fast); only
        do this for extractors you have reviewed. Otherwise it runs in the sandbox."""
        src = Path(path)
        meta = json.loads((src / "head.json").read_text())
        agent = TabAgent.load(src / "model")
        code = (src / "extract.py").read_text() if meta.get("has_extractor") else None
        if code is not None:
            import hashlib

            expected = (agent.card_.extractor or {}).get("sha256")
            if expected and hashlib.sha256(code.encode()).hexdigest() != expected:
                raise ValueError("extract.py does not match the card's extractor hash")
        return cls(
            agent, meta["target"], code, gate_threshold=meta.get("gate_threshold"), trusted=trust_extractor
        )


# --------------------------------------------------------------------------- onboarding
def _estimate(dps, sample, target: str, prices: dict | None, model: str) -> tuple[int, int, float | None]:
    per_dp = [
        len(prompts.text("feature_analyzer"))
        + 2 * len(json.dumps(dp.prefix, default=str))
        + len(json.dumps(dp.target_step, default=str))
        for dp in sample
    ]
    analyzer = sum(per_dp)
    extractor = int(
        1.5
        * sum(
            len(prompts.text("code_extractor")) + len(str(get_json_schema(dp.prefix))) + 4000 for dp in sample
        )
    )
    validator = extractor
    global_stages = 6 * (len(prompts.text("meta_judge")) + 8000)
    compile_ = 2 * (len(prompts.text("code_extractor")) + 12000)
    chars_in = analyzer + extractor + validator + global_stages + compile_
    tokens_in, tokens_out = chars_in // 4, (len(sample) * 3 + 8) * 1500
    if prices and model in prices:
        pin, pout = prices[model]
        return tokens_in, tokens_out, (tokens_in * pin + tokens_out * pout) / 1e6
    return tokens_in, tokens_out, None


def onboard(
    logs: str | Path | list[Trajectory],
    decision: str | None = None,
    *,
    adapter: Any = "auto",
    llm: LLM | None = None,
    model: str | None = None,
    kind: str = "ranking",
    out: str | Path | None = None,
    n_analyze: int = 10,
    max_cost_usd: float | None = 100.0,
    prices: dict[str, tuple[float, float]] | None = None,
    dry_run: bool = False,
    redact: Callable[[dict], dict] | None = None,
    search: str = "fast",
    n_splits: int = 5,
    llm_cost_per_call: float | None = None,
    trust_extractor: bool = True,
    run_dir: str | Path | None = None,
    config: TabSchemaConfig | None = None,
    gate_coverage: float | None = None,
    iterations: int | None = None,
) -> TabAgentHead | DryRun:
    """Onboard an agent decision component from its logs. See the module docstring."""
    if adapter == "auto":
        from tabagent.adapters import detect_adapter

        adapter = detect_adapter(logs)
    trajs = adapter.load(logs) if isinstance(logs, (str, Path)) else list(logs)
    if redact is not None:  # applied before anything can reach the LLM provider
        trajs = [Trajectory(t.id, redact(t.raw), t.success, t.group, t.source) for t in trajs]
    good = [t for t in trajs if t.success]
    comps = adapter.components(good) if hasattr(adapter, "components") else {}
    if decision is None:
        if len(comps) == 1:
            decision = next(iter(comps))
        else:
            raise ValueError(f"choose the decision component to replace with decision=...; found {comps}")
    dps = [dp for t in good for dp in adapter.decision_points(t, decision)]
    cfg = config or TabSchemaConfig(
        model=model, n_analyze=n_analyze, max_cost_usd=max_cost_usd, prices=dict(prices or {})
    )
    if dry_run:
        sample = _sample(dps, cfg.n_analyze, cfg.sample_seed)
        tin, tout, cost = _estimate(dps, sample, decision, cfg.prices, model or "gpt-5.4")
        return DryRun(len(trajs), len(good), len(dps), len(sample), tin, tout, cost, comps)

    llm = llm or auto_llm(model)
    store = RunStore(run_dir) if run_dir else RunStore()
    schema = TabSchema(llm, cfg, adapter, store=store)
    schema.meter = CostMeter(prices=dict(cfg.prices), max_cost_usd=cfg.max_cost_usd)
    result = schema.discover(good, decision, kind)
    table = schema.build_table(result, good, decision, trusted=trust_extractor)

    head_kw = {} if iterations is None else {"iterations": iterations}
    recipe, _ = auto.select(table, result.card, search=search, **head_kw)
    agent = auto.build_agent(result.card, recipe, **head_kw)
    from tabagent.report import evaluate

    rep = evaluate(agent, table, n_splits=n_splits, llm_cost_per_call=llm_cost_per_call)
    rep.notes.append(f"recipe: {recipe.label()} (nested inner validation)")
    rep.notes.append(
        "the feature card was discovered on all onboarding trajectories; CV estimates the head, "
        "not the discovery step"
    )
    rep.notes.append(f"TabSchema onboarding LLM cost: ${schema.meter.total_usd:.2f} ({llm.model})")
    rep.notes += [f"data: {w}" for w in getattr(adapter, "warnings", [])]
    head = TabAgentHead(
        agent.fit(table), decision, result.extractor.code, rep, result.report, trusted=trust_extractor
    )
    if gate_coverage is not None and kind != "single":
        head.calibrate_gate(rep.cv, gate_coverage)
    if out is not None:
        head.save(out)
    return head


def load(path: str | Path, trust_extractor: bool = False):
    """Load a saved head (``onboard`` output) or a bare TabAgent artifact."""
    p = Path(path)
    if (p / "head.json").exists():
        return TabAgentHead.load(p, trust_extractor)
    return TabAgent.load(p)
