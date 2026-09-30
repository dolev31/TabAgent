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

import hashlib
import json
import logging
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
from tabagent.tabschema.llm import DEFAULT_MODEL, LLM, auto_llm
from tabagent.tabschema.orchestrator import (
    CANDIDATE,
    DECISION,
    GROUP,
    DiscoveryReport,
    TabSchema,
    TabSchemaConfig,
    sample_decision_points,
)
from tabagent.tabschema.runstore import RunStore
from tabagent.tabschema.trajectory import Candidate, Trajectory, get_json_schema

logger = logging.getLogger(__name__)

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
    model: str = DEFAULT_MODEL

    def __str__(self) -> str:
        cost = (
            f"${self.est_cost_usd:.2f}"
            if self.est_cost_usd is not None
            else f"unknown (no price for {self.model}; pass prices= or --price-in/--price-out)"
        )
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
    """A fitted ``TabAgent`` with the TabSchema extractor that feeds it: the drop-in replacement for one
    agent decision.

    ``decide(prefix, candidates)`` runs the extractor on the trajectory prefix, builds one row per candidate
    and returns the head's decision. Heads come from :func:`onboard` and are persisted with ``save`` and
    :meth:`load`.

    Attributes
    ----------
    agent: the fitted ``TabAgent``.
    target: the replaced decision component (for example ``ShortlisterAgent``).
    extractor_code: the compiled TabSchema extractor (``extract.py``), or None.
    report: the onboarding evaluation report (``tabagent.report.Report``).
    discovery: the TabSchema discovery report.
    gate_threshold: decisions with a lower confidence are marked ``deferred`` (App. N.3).
    trust_extractor: run the extractor in-process (fast; for reviewed code) instead of in the sandbox.
    """

    agent: TabAgent
    target: str
    extractor_code: str | None = None
    report: Any = None
    discovery: DiscoveryReport | None = None
    gate_threshold: float | None = None
    trust_extractor: bool = False

    @property
    def card(self) -> FeatureCard:
        """The fitted feature card."""
        return self.agent.card_

    def _runner(self) -> ExtractorRunner | None:
        if not self.extractor_code:
            return None
        if not hasattr(self, "_runner_cache"):
            self._runner_cache = ExtractorRunner(self.extractor_code, trust_extractor=self.trust_extractor)
        return self._runner_cache

    def features_for(self, prefix: dict[str, Any]) -> dict[str, Any]:
        """The extractor's features for one trajectory prefix (empty, with a warning, if it fails)."""
        runner = self._runner()
        return (runner.run([prefix])[0] or {}) if runner else {}

    def decide(
        self,
        prefix: dict[str, Any],
        candidates=None,
        k: int | None = None,
        defer_below: float | None = None,
    ) -> dict[str, Any]:
        """Score one live decision.

        Parameters
        ----------
        prefix: the trajectory before the decision, in the format the head was onboarded on.
        candidates: the legal candidates: ids, ``{"id"/"name", "description", "args"}`` dicts or
            ``Candidate`` objects. Not used by single-class heads, which choose among their training classes.
        k: return the top ``k`` of a ranking.
        defer_below: confidence threshold of the gate; defaults to the calibrated ``gate_threshold``.

        Returns
        -------
        ``{"candidates": [ids], "scores": {id: p}, "confidence": c, "deferred": bool}``, or for a single-class
        head ``{"choice": label, "scores": {label: p}, "confidence": p(choice), "deferred": bool}``. Tied
        scores are ordered by candidate id.
        """
        spec = self.card.spec
        feats = self.features_for(prefix)
        tau = defer_below if defer_below is not None else self.gate_threshold
        if spec.kind == "single":
            if candidates:
                raise ValueError("a single-class head chooses among its training classes; pass no candidates")
            table = DecisionTable(pd.DataFrame([{GROUP: "live", DECISION: "live#0", **feats}]), spec, False)
            proba = self.agent.head_.predict_proba(self.agent.features(table))[0]
            classes = list(self.agent.head_.classes_)
            best = int(np.argmax(proba))
            return {
                "choice": classes[best],
                "scores": {str(c): float(p) for c, p in zip(classes, proba, strict=True)},
                "confidence": float(proba[best]),
                "deferred": bool(tau is not None and proba[best] < tau),
            }
        if not candidates:
            raise ValueError(f"a {spec.kind} head needs the decision's candidates")
        cands = _as_candidates(candidates)
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
        table = DecisionTable(pd.DataFrame(rows), spec, has_labels=False)
        scores = self.agent.score(table)
        ids, cand_ids = ["live#0"] * len(rows), [c.id for c in cands]
        ranked = decode(spec.kind, scores, ids, cand_ids, k=k, threshold=self.agent.head_.threshold)
        conf = float(confidence(scores, ids, cand_ids).iloc[0])
        return {
            "candidates": ranked["candidate"].tolist(),
            "scores": {c.id: float(s) for c, s in zip(cands, scores, strict=True)},
            "confidence": conf,
            "deferred": bool(tau is not None and conf < tau),
        }

    def calibrate_gate(self, cv_result, target_coverage: float = 0.9) -> float:
        """Set ``gate_threshold`` so that ``target_coverage`` of the cross-validated decisions stay with the
        head and the least confident ``1 - target_coverage`` are deferred to the LLM (App. N.3). Returns the
        threshold. Rankings only: a cross-validated single-class head keeps no scores to calibrate on."""
        t = cv_result.table
        if t.spec.kind == "single":
            raise ValueError("the confidence gate is calibrated on ranking scores; this head is single-class")
        if not 0 < target_coverage <= 1:
            raise ValueError("target_coverage must be in (0, 1]")
        ids = t.decision_ids.to_numpy()
        cands = t.df[t.spec.candidate].to_numpy()
        conf = np.concatenate(
            [confidence(cv_result.oof[s].to_numpy(float), ids, cands).to_numpy() for s in cv_result.seeds]
        )
        self.gate_threshold = float(np.quantile(conf, 1 - target_coverage))
        return self.gate_threshold

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        """Write the head to the directory ``path``: the model artifact (``model/``), ``extract.py``,
        ``head.json`` and, when present, the evaluation and discovery reports."""
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
    def load(cls, path: str | Path, trust_extractor: bool = False, trust_code: bool = False) -> TabAgentHead:
        """Load a saved head. ``extract.py`` must match the hash recorded in the card.

        ``trust_extractor=True`` runs the extractor in-process (fast); only do this for an extractor you have
        reviewed. Otherwise it runs in the sandbox. ``trust_code`` allows card features computed by modules
        outside ``tabagent``.
        """
        src = Path(path)
        meta = json.loads((src / "head.json").read_text())
        agent = TabAgent.load(src / "model", trust_code=trust_code)
        code = (src / "extract.py").read_text() if meta.get("has_extractor") else None
        if code is not None:
            expected = (agent.card_.extractor or {}).get("sha256")
            if expected and hashlib.sha256(code.encode()).hexdigest() != expected:
                raise ValueError("extract.py does not match the card's extractor hash")
        return cls(
            agent,
            meta["target"],
            code,
            gate_threshold=meta.get("gate_threshold"),
            trust_extractor=trust_extractor,
        )


# --------------------------------------------------------------------------- onboarding
# Rough sizes for the dry-run estimate (characters; 4 characters per token). Per sampled decision point the
# analyzer sees the prefix twice (JSON and schema) and the target step; the extractor sees its prompt, the
# prefix schema and the feature plans and is retried about every other time; the validator sees as much as
# the extractor. The whole-run calls are the aggregator, the three judges, the MetaJudge and the compile
# validator; compiling takes about two extractor calls. Every call answers about 1,500 tokens.
_PLAN_CHARS = 4000  # feature plans in an extractor prompt
_RETRY_FACTOR = 1.5  # extractor calls per sampled decision point, retries included
_WHOLE_RUN_CALLS, _WHOLE_RUN_CHARS = 6, 8000
_COMPILE_CALLS, _COMPILE_CHARS = 2, 12000
_CALLS_PER_POINT, _EXTRA_CALLS, _ANSWER_TOKENS = 3, 8, 1500


def _estimate(sample, prices: dict | None, model: str) -> tuple[int, int, float | None]:
    """Estimated (input tokens, output tokens, USD or None without a price) of discovering on ``sample``."""
    analyzer = sum(
        len(prompts.text("feature_analyzer"))
        + 2 * len(json.dumps(dp.prefix, default=str))
        + len(json.dumps(dp.target_step, default=str))
        for dp in sample
    )
    extractor = int(
        _RETRY_FACTOR
        * sum(
            len(prompts.text("code_extractor")) + len(str(get_json_schema(dp.prefix))) + _PLAN_CHARS
            for dp in sample
        )
    )
    whole_run = _WHOLE_RUN_CALLS * (len(prompts.text("meta_judge")) + _WHOLE_RUN_CHARS)
    compile_ = _COMPILE_CALLS * (len(prompts.text("code_extractor")) + _COMPILE_CHARS)
    chars_in = analyzer + 2 * extractor + whole_run + compile_
    tokens_in = chars_in // 4
    tokens_out = (len(sample) * _CALLS_PER_POINT + _EXTRA_CALLS) * _ANSWER_TOKENS
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
    n_analyze: int | None = None,
    max_cost_usd: float | None = None,
    prices: dict[str, tuple[float, float]] | None = None,
    dry_run: bool = False,
    redact: Callable[[dict], dict] | None = None,
    search: str = "fast",
    n_splits: int = 5,
    llm_cost_per_call: float | None = None,
    trust_extractor: bool = False,
    run_dir: str | Path | None = None,
    config: TabSchemaConfig | None = None,
    gate_coverage: float | None = None,
    iterations: int | None = None,
) -> TabAgentHead | DryRun:
    """Onboard an agent decision component from its logs (see the module docstring).

    Parameters
    ----------
    logs: a file or directory of agent logs, or a list of ``Trajectory``.
    decision: the component to replace (see ``tabagent components``); optional when the logs have one.
    adapter: the log adapter, or ``"auto"`` to detect it from the files.
    llm, model: the TabSchema LLM, or the model name to build one from environment keys (default GPT-5.4).
    kind: the decision pattern: ``ranking``, ``multilabel`` or ``single``.
    out: save the head to this directory.
    n_analyze: decision points analyzed by the LLM (default 10).
    max_cost_usd: stop TabSchema when its LLM cost exceeds this. Needs a known price for the model
        (``prices``); a budget without a price raises ``ValueError`` before any LLM call.
    prices: ``{model: (USD per 1M input tokens, USD per 1M output tokens)}``.
    dry_run: return a ``DryRun`` (work and cost estimate) without calling the LLM.
    redact: applied to every raw trajectory before anything can reach the LLM provider.
    search: ``fast`` or ``full`` recipe search (``tabagent.auto.select``).
    n_splits: folds of the grouped cross-validation in the report.
    llm_cost_per_call: the replaced LLM call's cost, to report the savings.
    trust_extractor: run the LLM-written extractor in-process. Off by default: the extractor then runs in
        the sandbox, since trajectories are untrusted text that reaches the LLM writing the code.
    run_dir: TabSchema run directory (default under ``$TABAGENT_CACHE/runs``); a rerun resumes from it.
    config: a full ``TabSchemaConfig``; it then also sets ``model``, ``n_analyze``, ``max_cost_usd`` and
        ``prices``, which must not be passed separately.
    gate_coverage: calibrate the confidence gate so this share of decisions stays with the head (rankings).
    iterations: override the CatBoost iteration count (for quick runs).

    Returns
    -------
    The fitted ``TabAgentHead``, or a ``DryRun`` with ``dry_run=True``.
    """
    given = {
        "model": model,
        "n_analyze": n_analyze,
        "max_cost_usd": max_cost_usd,
        "prices": prices,
    }
    if config is not None and any(v is not None for v in given.values()):
        clash = sorted(k for k, v in given.items() if v is not None)
        raise ValueError(f"config= already sets {clash}; set them in the TabSchemaConfig instead")
    if gate_coverage is not None:
        if kind == "single":
            raise ValueError("gate_coverage calibrates a ranking gate; it does not apply to kind='single'")
        if not 0 < gate_coverage <= 1:
            raise ValueError("gate_coverage must be in (0, 1]")
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
    logger.info(
        "onboarding %s: %d trajectories (%d successful), %d decision points",
        decision,
        len(trajs),
        len(good),
        len(dps),
    )
    cfg = config or TabSchemaConfig(**{k: v for k, v in given.items() if v is not None})
    if dry_run:
        sample = sample_decision_points(dps, cfg.n_analyze, cfg.sample_seed)
        name = cfg.model or (llm.model if llm is not None else DEFAULT_MODEL)
        tin, tout, cost = _estimate(sample, cfg.prices, name)
        return DryRun(len(trajs), len(good), len(dps), len(sample), tin, tout, cost, comps, name)

    llm = llm or auto_llm(cfg.model)
    store = RunStore(run_dir) if run_dir else RunStore()
    schema = TabSchema(llm, cfg, adapter, store=store)  # checks the budget can be enforced
    result = schema.discover(good, decision, kind)
    logger.info(
        "building the decision table (extractor %s)", "in-process" if trust_extractor else "sandboxed"
    )
    table = schema.build_table(result, good, decision, trust_extractor=trust_extractor)

    head_kw = {} if iterations is None else {"iterations": iterations}
    logger.info("selecting the training recipe (%s search, nested validation)", search)
    recipe, _ = auto.select(table, result.card, search=search, **head_kw)
    agent = auto.build_agent(result.card, recipe, **head_kw)
    from tabagent.report import evaluate

    logger.info("evaluating %s with %d-fold grouped cross-validation", recipe.label(), n_splits)
    rep = evaluate(agent, table, n_splits=n_splits, llm_cost_per_call=llm_cost_per_call)
    rep.notes.append(f"recipe: {recipe.label()} (nested inner validation)")
    rep.notes.append(
        "the feature card was discovered on all onboarding trajectories; CV estimates the head, "
        "not the discovery step"
    )
    rep.notes.append(f"TabSchema onboarding LLM cost: {schema.meter.describe(llm.model)}")
    rep.notes += [f"data: {w}" for w in getattr(adapter, "warnings", [])]
    logger.info("fitting the final head on all %d decision points", len(dps))
    head = TabAgentHead(
        agent.fit(table),
        decision,
        result.extractor.code,
        rep,
        result.report,
        trust_extractor=trust_extractor,
    )
    if gate_coverage is not None:
        head.calibrate_gate(rep.cv, gate_coverage)
    if out is not None:
        head.save(out)
        logger.info("saved the head to %s", out)
    return head


def load(
    path: str | Path, trust_extractor: bool = False, trust_code: bool = False
) -> TabAgentHead | TabAgent:
    """Load a saved head: a ``TabAgentHead`` for a directory written by ``onboard`` (it has ``head.json``),
    else the ``TabAgent`` artifact written by ``TabAgent.save``.

    A ``TabAgentHead`` decides from a trajectory prefix (``decide(prefix, candidates, k=...)``); a bare
    ``TabAgent`` decides from candidate rows (``decide(rows, k=...)``). ``trust_extractor`` and
    ``trust_code`` are described in :meth:`TabAgentHead.load`.
    """
    p = Path(path)
    if (p / "head.json").exists():
        return TabAgentHead.load(p, trust_extractor, trust_code)
    return TabAgent.load(p, trust_code=trust_code)
