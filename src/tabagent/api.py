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

import copy
import hashlib
import json
import logging
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabagent import auto, hub
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
)
from tabagent.tabschema.runstore import RunStore
from tabagent.tabschema.sampling import sample_decision_points
from tabagent.tabschema.sandbox import DockerSandbox, SubprocessSandbox
from tabagent.tabschema.trajectory import Candidate, Trajectory, clip_for_prompt, union_schema

logger = logging.getLogger(__name__)

HEAD_FORMAT = 1


@dataclass
class DryRun:
    """What discovery would process, an estimate of its size (4 characters per token), and the problems that
    would stop a real run before any LLM call (``problems``)."""

    n_trajectories: int
    n_successful: int
    n_decision_points: int
    n_analyzed: int
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float | None
    components: dict[str, int] = field(default_factory=dict)
    model: str = DEFAULT_MODEL
    problems: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        cost = (
            f"${self.est_cost_usd:.2f}"
            if self.est_cost_usd is not None
            else f"unknown (no price for {self.model}; pass prices= or --price-in/--price-out)"
        )
        problems = "".join(f"\nA real run would stop before any LLM call: {p}" for p in self.problems)
        return (
            f"{self.n_trajectories} trajectories ({self.n_successful} successful), "
            f"{self.n_decision_points} decision points, {self.n_analyzed} analyzed by the LLM; "
            f"~{self.est_input_tokens:,} input / ~{self.est_output_tokens:,} output tokens; est. cost {cost}"
            f"{problems}"
        )


def _as_candidates(items) -> list[Candidate]:
    """The decision's candidates as ``Candidate`` objects. A tool schema (OpenAI ``{"type": "function",
    "function": {...}}``, a bare function, the Anthropic Messages API format's ``input_schema``, MCP
    ``inputSchema``) is read as the adapters read a catalog (``parse_catalog``), so its argument hints are
    the parameter names, as in training."""
    from tabagent.adapters.common import is_tool_schema, tool_candidate

    out = []
    for c in items:
        if isinstance(c, Candidate):
            out.append(c)
        elif isinstance(c, str):
            out.append(Candidate(c, c))
        elif isinstance(c, dict) and is_tool_schema(c):
            out.append(tool_candidate(c))
        elif isinstance(c, dict) and ("id" in c or "name" in c):
            args = c.get("arg_hints", c.get("args", ""))
            out.append(
                Candidate(
                    str(c["id"] if "id" in c else c["name"]),
                    str(c.get("name") or ""),
                    str(c.get("description") or ""),
                    ", ".join(map(str, args)) if isinstance(args, (list, tuple)) else str(args or ""),
                )
            )
        else:
            raise ValueError(
                "a candidate is an id, a {'id' or 'name', 'description', 'args'} dict, a tool schema or a "
                f"Candidate, not {str(c)[:120]!r}"
            )
    return out


@dataclass
class TabAgentHead:
    """A fitted ``TabAgent`` with the TabSchema extractor that feeds it: the drop-in replacement for one
    agent decision.

    ``decide(prefix, candidates)`` runs the extractor on the trajectory prefix, builds one row per candidate
    and returns the head's decision; ``decide(run=..., candidates=...)`` first builds the prefix from the live
    run with the adapter the head was trained with (``live_prefix``). Heads come from :func:`onboard` and are
    persisted with ``save`` and :meth:`load`.

    Attributes
    ----------
    agent: the fitted ``TabAgent``.
    target: the replaced decision component (for example ``ShortlisterAgent``).
    extractor_code: the compiled TabSchema extractor (``extract.py``), or None.
    report: the onboarding evaluation report (``tabagent.report.Report``).
    discovery: the TabSchema discovery report.
    gate_threshold: decisions with a lower confidence are marked ``deferred`` (App. N.3).
    trust_extractor: run the extractor in-process (fast; for reviewed code) instead of in the sandbox.
    adapter: the log adapter of the training trajectories (an adapter or its ``describe_adapter``
        description); ``live_prefix`` builds live prefixes with it, and ``save`` records it in ``head.json``.
    sandbox: where an untrusted extractor runs: ``"subprocess"`` (the default, ``SubprocessSandbox``),
        ``"docker"`` or ``"docker:<image>"`` (``DockerSandbox``: no network, read-only filesystem), or a
        sandbox instance.
    redact: the function applied to each training trajectory before discovery, as a callable or a
        ``"module:function"`` import path; ``live_prefix`` applies it to a live run too.
    """

    agent: TabAgent
    target: str
    extractor_code: str | None = None
    report: Any = None
    discovery: DiscoveryReport | None = None
    gate_threshold: float | None = None
    trust_extractor: bool = False
    adapter: Any = None
    sandbox: Any = None
    redact: Callable[[dict], dict] | str | None = None

    def __post_init__(self) -> None:
        if self.trust_extractor and self.sandbox not in (None, "subprocess"):
            raise ValueError(
                "trust_extractor=True runs the extractor in-process, outside any sandbox; drop sandbox= or "
                "trust_extractor="
            )

    @property
    def card(self) -> FeatureCard:
        """The fitted feature card."""
        return self.agent.card_

    def _sandbox(self) -> SubprocessSandbox | None:
        spec = self.sandbox
        if spec is None or spec == "subprocess":
            return None
        if isinstance(spec, str) and (spec == "docker" or spec.startswith("docker:")):
            image = spec.partition(":")[2]
            return DockerSandbox(image=image) if image else DockerSandbox()
        if hasattr(spec, "run_batch"):
            return spec
        raise ValueError(f"sandbox: 'subprocess', 'docker', 'docker:<image>' or a sandbox, not {spec!r}")

    def _runner(self) -> ExtractorRunner | None:
        if not self.extractor_code:
            return None
        if getattr(self, "_runner_cache", None) is None:
            self._runner_cache = ExtractorRunner(
                self.extractor_code, trust_extractor=self.trust_extractor, sandbox=self._sandbox()
            )
        return self._runner_cache

    # ------------------------------------------------------------------ live prefixes
    def live_prefix(self, run: dict[str, Any], adapter: Any = None) -> dict[str, Any]:
        """The prefix of a live decision, built from ``run`` (the run so far, in the format of the training
        logs, with its run-level task fields) exactly as the training prefixes were: ``redact`` first, then
        the head's adapter (or ``adapter``; see ``tabagent.adapters.live_prefix``)."""
        from tabagent.adapters import adapter_from_description, live_prefix

        adapter = adapter if adapter is not None else self.adapter
        if isinstance(adapter, dict):
            desc, adapter = adapter, adapter_from_description(adapter)
            if adapter is None:
                raise ValueError(
                    f"this head was trained with {desc.get('class')!r}, which head.json cannot rebuild; "
                    "pass adapter= (the adapter used in training)"
                )
        if adapter is None:
            raise ValueError(
                "this head records no log adapter (head.json has none); pass adapter= (the adapter used in "
                "training), or build the prefix yourself and call decide(prefix, ...)"
            )
        if self.redact is not None:
            run = self._redact_fn()(run)
        return live_prefix(adapter, run, self.target)

    def _redact_fn(self) -> Callable[[dict], dict]:
        return import_object(self.redact) if isinstance(self.redact, str) else self.redact

    # ------------------------------------------------------------------ extraction
    def _extract(self, prefix: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        """The card's extractor columns for ``prefix``, filled as in training (None where the extractor gave
        no value), and the error when the extractor failed on it."""
        cols = [f.name for f in self.card.features if f.source == "extractor"]
        runner = self._runner()
        if runner is None:
            error = "this head has no extractor (extract.py) for its extractor features" if cols else None
            return dict.fromkeys(cols), error
        error = None
        try:
            out = runner.run([prefix])[0]
        except Exception as e:  # a sandbox that cannot start (no Docker, say): defer, do not crash
            out, error = None, f"{type(e).__name__}: {e}"
        if out is None:
            error = _explain(error or self._diagnose(prefix), prefix)
            logger.warning("TabSchema extractor failed on a live prefix; deferring the decision: %s", error)
        return {c: (out or {}).get(c) for c in cols}, error

    def _diagnose(self, prefix: dict[str, Any]) -> str:
        """The extractor's error on ``prefix``, from a second run (only after a failure)."""
        try:
            if self.trust_extractor:
                ns: dict[str, Any] = {"__name__": "__main__", "trajectory": copy.deepcopy(prefix)}
                exec(compile(self.extractor_code, "<extract.py>", "exec"), ns)  # noqa: S102 - reviewed code
                return "the extractor produced no features"
            res = (self._sandbox() or SubprocessSandbox()).run(self.extractor_code, prefix)
            text = (res.error or "").strip()
            return text.splitlines()[-1] if text else "the extractor produced no features"
        except Exception as e:  # the extractor is arbitrary code
            return f"{type(e).__name__}: {e}"

    def features_for(self, prefix: dict[str, Any]) -> dict[str, Any]:
        """The extractor's features for one trajectory prefix: every extractor column of the card, None where
        the extractor gave no value or failed (as in training)."""
        return self._extract(prefix)[0]

    def decide(
        self,
        prefix: dict[str, Any] | None = None,
        candidates=None,
        k: int | None = None,
        defer_below: float | None = None,
        *,
        run: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score one live decision.

        Parameters
        ----------
        prefix: the trajectory before the decision, normalized by the adapter the head was onboarded with
            (what the adapter's ``decision_points`` put in ``DecisionPoint.prefix``; ``live_prefix`` builds it
            from a live run).
        candidates: the legal candidates: ids, ``{"id"/"name", "description", "args"}`` dicts or
            ``Candidate`` objects. Not used by single-class heads, which choose among their training classes.
        k: return the top ``k`` of a ranking.
        defer_below: confidence threshold of the gate; defaults to the calibrated ``gate_threshold``.
        run: instead of ``prefix``, the live run so far in the format of the training logs, with its run-level
            task fields; the prefix is built with ``live_prefix``.

        Returns
        -------
        ``{"candidates": [ids], "scores": {id: p}, "confidence": c, "deferred": bool, "error": None}``, or for
        a single-class head ``{"choice": label, "scores": {label: p}, "confidence": p(choice), "deferred":
        bool, "error": None}``. Tied scores are ordered by candidate id. When the extractor fails on the
        prefix, its features are empty (None, as for a training decision point it failed on), the decision is
        ``deferred`` to the original LLM whatever its confidence, and ``error`` says why.
        """
        if run is not None:
            if prefix is not None:
                raise ValueError("pass the decision's prefix or its run, not both")
            prefix = self.live_prefix(run)
        spec = self.card.spec
        feats, error = self._extract(prefix if prefix is not None else {})
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
                "deferred": bool(error is not None or (tau is not None and proba[best] < tau)),
                "error": error,
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
            "deferred": bool(error is not None or (tau is not None and conf < tau)),
            "error": error,
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
        ``head.json`` (with the adapter and its prefix settings, and the redaction function's import path,
        when known) and, when present, the evaluation and discovery reports."""
        from tabagent.adapters import describe_adapter

        out = Path(path)
        self.agent.save(out / "model")
        if self.extractor_code:
            (out / "extract.py").write_text(self.extractor_code)
        meta = {
            "format_version": HEAD_FORMAT,
            "target": self.target,
            "gate_threshold": self.gate_threshold,
            "has_extractor": bool(self.extractor_code),
            "adapter": describe_adapter(self.adapter),
            "redact": _object_path(self.redact),
        }
        (out / "head.json").write_text(json.dumps(meta, indent=1))
        if self.report is not None:
            self.report.save(out)
        if self.discovery is not None:
            (out / "discovery.md").write_text(self.discovery.to_markdown())
        return out

    @classmethod
    def load(
        cls,
        path: str | Path,
        trust_extractor: bool = False,
        trust_code: bool = False,
        *,
        sandbox: Any = None,
        redact: Callable[[dict], dict] | str | None = None,
        adapter: Any = None,
    ) -> TabAgentHead:
        """Load a saved head from a directory or ``hf://OWNER/REPO[@REVISION]`` (``tabagent.hub``).
        ``extract.py`` must match the hash recorded in the card.

        ``trust_extractor=True`` runs the extractor in-process (fast); only do this for an extractor you have
        reviewed. Otherwise it runs in ``sandbox`` (see the class attributes). ``trust_code`` allows card
        features computed by modules outside ``tabagent``. ``adapter`` overrides the adapter recorded in
        ``head.json``. ``redact`` is never imported from ``head.json``: when the head records one and none is
        passed, a warning says so, since live runs then reach the extractor unredacted.
        """
        src = hub.resolve(path)
        meta = json.loads((src / "head.json").read_text())
        agent = TabAgent.load(src / "model", trust_code=trust_code)
        code = (src / "extract.py").read_text() if meta.get("has_extractor") else None
        if code is not None:
            expected = (agent.card_.extractor or {}).get("sha256")
            if expected and hashlib.sha256(code.encode()).hexdigest() != expected:
                raise ValueError("extract.py does not match the card's extractor hash")
        if redact is None and meta.get("redact"):
            warnings.warn(
                f"this head's training trajectories were redacted with {meta['redact']}; pass redact= "
                "(tabagent serve --redact) to scrub live runs the same way",
                UserWarning,
                stacklevel=2,
            )
        return cls(
            agent,
            meta["target"],
            code,
            gate_threshold=meta.get("gate_threshold"),
            trust_extractor=trust_extractor,
            adapter=adapter if adapter is not None else meta.get("adapter"),
            sandbox=sandbox,
            redact=redact,
        )


def import_object(ref: str) -> Any:
    """The object an import path ``"module:attribute"`` names (``"mypkg.privacy:scrub"``, for example)."""
    module, sep, attr = ref.partition(":")
    if not (module and sep and attr):
        raise ValueError(f"expected an import path 'module:function', not {ref!r}")
    import importlib

    obj: Any = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _object_path(obj: Any) -> str | None:
    """``obj``'s import path ``module:qualname`` when it has one, ``obj`` itself when it is a string."""
    if obj is None or isinstance(obj, str):
        return obj
    module, name = getattr(obj, "__module__", None), getattr(obj, "__qualname__", "")
    if not module or not name or "<" in name:  # a lambda or a nested function cannot be imported
        return f"<unnamed {type(obj).__name__}>"
    return f"{module}:{name}"


_KEY_ERROR = re.compile(r"KeyError: ['\"]([^'\"]+)['\"]")


def _explain(error: str, prefix: dict[str, Any]) -> str:
    """``error``, with a hint when the extractor read a top-level field the live prefix lacks."""
    m = _KEY_ERROR.search(error)
    if m and m.group(1) not in prefix:
        return (
            f"{error}. The live prefix has no {m.group(1)!r} field: build it as in training, with the "
            "run-level task fields, through head.live_prefix(run), decide(run=...), ToolFilter(task=...) or "
            'POST /decide {"run": ...}'
        )
    return error


# --------------------------------------------------------------------------- onboarding
# Rough sizes for the dry-run estimate (characters; 4 characters per token). The estimate follows the stages:
# per sampled decision point the analyzer (the clipped prefix as indented JSON, its schema, the target step),
# the extractor (the plans of the point's proposals, retried about every other time) and the validator; then
# the aggregator (40 proposals per call), the three judges and the MetaJudge (25 features per call) on the
# distinct proposals, and the compile step (extractor and validator on the accepted features).
# Calibrated on two real onboardings (τ²-bench ToolShortlist, 10 decision points, Claude Opus 5.5 with the
# extractor on GPT-6 Sol, 2026-10-02): proposals, merging and acceptance rates, and answer lengths per item.
_PROPOSALS = 15  # features an analyzer answer proposes
_NEW_SHARE = 0.8  # share of a further decision point's proposals that no earlier point proposed
_KEPT_SHARE, _ACCEPTED_SHARE = 0.7, 0.35  # of the distinct proposals: aggregated, then accepted
_SPEC_CHARS, _ROW_CHARS, _CODE_CHARS = 600, 4500, 11000  # one feature plan, one MetaJudge row, extractor code
_RETRY_FACTOR = 1.8  # extractor calls per extraction, retries and code-tag re-asks included
_TOKENS_PER_PROPOSAL = {"analyzer": 300, "validator": 170, "aggregator": 270, "judge": 380, "meta": 410}
_CODE_TOKENS = 2900  # an extractor answer (thoughts and code)


def _estimate(
    sample,
    prices: dict | None,
    model: str,
    max_chars: int = 400_000,
    stage_models: dict[str, str] | None = None,
) -> tuple[int, int, float | None]:
    """Estimated (input tokens, output tokens, USD or None without a price) of discovering on ``sample``."""
    text = prompts.text
    n = len(sample)
    unique = _PROPOSALS * (1 + _NEW_SHARE * max(0, n - 1)) if n else 0
    kept, accepted = _KEPT_SHARE * unique, _ACCEPTED_SHARE * _KEPT_SHARE * unique
    schemas = [len(json.dumps(union_schema([dp.prefix]), default=str)) for dp in sample]
    analyzer = sum(
        len(text("feature_analyzer"))
        + len(json.dumps(clip_for_prompt(dp.prefix, max_chars), indent=2, default=str, ensure_ascii=False))
        + schema
        + len(json.dumps(clip_for_prompt(dp.target_step, max_chars // 4), indent=2, default=str))
        for dp, schema in zip(sample, schemas, strict=True)
    )
    plans = _PROPOSALS * _SPEC_CHARS
    extractor = _RETRY_FACTOR * sum(
        len(text("code_extractor")) + s + plans + _CODE_CHARS / 2 for s in schemas
    )
    validator = sum(len(text("code_validator")) + s + plans + _CODE_CHARS for s in schemas)
    schema = max(schemas, default=0)
    whole_run = (
        np.ceil(unique / 40) * len(text("feature_aggregator"))
        + unique * _SPEC_CHARS
        + 3 * (np.ceil(kept / 25) * len(text("judge_relevance")) + kept * _SPEC_CHARS)
        + np.ceil(kept / 25) * len(text("meta_judge"))
        + kept * _ROW_CHARS
    )
    compile_ = _RETRY_FACTOR * (len(text("code_extractor")) + schema + accepted * _SPEC_CHARS) + (
        len(text("code_validator")) + schema + accepted * _SPEC_CHARS + _CODE_CHARS
    )
    per = _TOKENS_PER_PROPOSAL
    whole_run_out = unique * per["aggregator"] + kept * (3 * per["judge"] + per["meta"])
    # (input chars, output tokens) by the stage whose model prices them
    stages = {
        "analyzer": (analyzer, n * _PROPOSALS * per["analyzer"]),
        "extractor": (extractor + compile_, (n + 1) * _RETRY_FACTOR * _CODE_TOKENS),
        "validator": (validator, (n * _PROPOSALS + accepted) * per["validator"]),
        "aggregator": (whole_run, whole_run_out),
    }
    tokens_in = int(sum(c for c, _ in stages.values())) // 4
    tokens_out = int(sum(o for _, o in stages.values()))
    models = stage_models or {}
    cost = 0.0
    for stage, (chars, out) in stages.items():
        name = models.get(stage) or (models.get("judges") if stage == "aggregator" else None) or model
        if not prices or name not in prices:
            return tokens_in, tokens_out, None
        pin, pout = prices[name]
        cost += (chars / 4 * pin + out * pout) / 1e6
    return tokens_in, tokens_out, cost


def _sandbox_from(spec: Any, timeout_s: float) -> SubprocessSandbox | None:
    """The sandbox ``onboard``'s ``sandbox=`` names (None: TabSchema's default ``SubprocessSandbox``)."""
    if spec is None or spec == "subprocess":
        return None
    if isinstance(spec, str) and (spec == "docker" or spec.startswith("docker:")):
        image = spec.partition(":")[2]
        return (
            DockerSandbox(image=image, timeout_s=timeout_s) if image else DockerSandbox(timeout_s=timeout_s)
        )
    if hasattr(spec, "run_batch"):
        return spec
    raise ValueError(f"sandbox: 'subprocess', 'docker', 'docker:<image>' or a sandbox, not {spec!r}")


def _preflight(adapter: Any, dps: list, decision: str, kind: str, probe: int = 50) -> None:
    """Check, before any LLM call, that the data can make a decision table and a cross-validated report:
    enough decision points and task groups, and an adapter that gives the decision's labels and candidates,
    with chosen candidates among the offered ones (on the first ``probe`` decision points)."""
    if len(dps) < 2:
        raise ValueError(f"TabSchema needs at least 2 decision points of {decision!r}, found {len(dps)}")
    groups = {str(dp.group or dp.trajectory_id) for dp in dps}
    if len(groups) < 2:
        raise ValueError(
            f"the {len(dps)} {decision} decision points all come from one task group, and the grouped "
            "cross-validation never splits a task across folds: it needs runs of at least 2 tasks"
        )
    some = dps[:probe]
    labels = [set(adapter.labels(dp) or ()) for dp in some]  # an adapter without labels raises here
    if kind == "single":
        several = next((dp.id for dp, chosen in zip(some, labels, strict=True) if len(chosen) > 1), None)
        if several:
            raise ValueError(
                f"kind='single' needs one label per decision, but {several} has several; use kind='ranking' "
                "or 'multilabel'"
            )
        return
    offered = [{c.id for c in adapter.candidates(dp)} for dp in some]
    if not any(offered):
        raise ValueError(
            f"none of the first {len(some)} {decision} decision points has candidates: the adapter offers "
            "nothing to choose from (pass a tool catalog, or candidates_fn=)"
        )
    if not any(chosen & cands for chosen, cands in zip(labels, offered, strict=True)):
        seen = sorted(set().union(*labels))[:5]
        raise ValueError(
            f"in the first {len(some)} {decision} decision points no chosen candidate ({seen}) is among the "
            f"offered ones ({sorted(set().union(*offered))[:5]}): the catalog and the logged choices must "
            "use the same names"
        )


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
    sandbox: Any = None,
    stage_models: dict[str, str] | None = None,
    fallback_model: str | None = None,
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
    sandbox: where the LLM-written code runs, in discovery, in the decision table and in the returned head:
        ``"subprocess"`` (the default, ``SubprocessSandbox``), ``"docker"`` or ``"docker:<image>"``
        (``DockerSandbox``: no network, read-only filesystem), or a sandbox instance. Not with
        ``trust_extractor``.
    run_dir: the TabSchema run directory itself (default: a new one under ``$TABAGENT_CACHE/runs``). A rerun
        with the same ``run_dir`` resumes from it: answers it already holds cost nothing.
    stage_models: a model per TabSchema stage, e.g. ``{"extractor": "azure/gpt-6-sol"}`` (see
        ``TabSchemaConfig.stage_models``); each needs a price in ``prices`` with a budget.
    fallback_model: asked once, for any stage, when the provider's content filter stops an answer.
    config: a full ``TabSchemaConfig``; it then also sets ``model``, ``n_analyze``, ``max_cost_usd``,
        ``prices``, ``stage_models`` and ``fallback_model``, which must not be passed separately.
    gate_coverage: calibrate the confidence gate so this share of decisions stays with the head (rankings).
    iterations: override the CatBoost iteration count (for quick runs).

    Returns
    -------
    The fitted ``TabAgentHead``, or a ``DryRun`` with ``dry_run=True``.

    Before any LLM call (and in a dry run) the data is checked: some trajectories must be successful, the
    decision must occur at least twice in at least 2 task groups, and the adapter must give its labels and
    candidates; otherwise ``ValueError`` (or the adapter's ``NotImplementedError``) says why.
    """
    given = {
        "model": model,
        "n_analyze": n_analyze,
        "max_cost_usd": max_cost_usd,
        "prices": prices,
        "stage_models": stage_models,
        "fallback_model": fallback_model,
    }
    if config is not None and any(v is not None for v in given.values()):
        clash = sorted(k for k, v in given.items() if v is not None)
        raise ValueError(f"config= already sets {clash}; set them in the TabSchemaConfig instead")
    if gate_coverage is not None:
        if kind == "single":
            raise ValueError("gate_coverage calibrates a ranking gate; it does not apply to kind='single'")
        if not 0 < gate_coverage <= 1:
            raise ValueError("gate_coverage must be in (0, 1]")
    if trust_extractor and sandbox not in (None, "subprocess"):
        raise ValueError(
            "trust_extractor=True runs the extractor in-process, outside any sandbox; drop sandbox= or "
            "trust_extractor="
        )
    if adapter == "auto":
        from tabagent.adapters import GenericJsonAdapter, detect_adapter, detect_format

        if isinstance(logs, (str, Path)):
            adapter = detect_adapter(logs)
        else:  # Trajectory objects: detect the format from the first raw run
            logs = list(logs)
            if not logs:
                raise ValueError("no trajectories given")
            adapter = (detect_format(logs[0].raw) or GenericJsonAdapter)()
    trajs = adapter.load(logs) if isinstance(logs, (str, Path)) else list(logs)
    if redact is not None:  # applied before anything can reach the LLM provider
        trajs = [Trajectory(t.id, redact(t.raw), t.success, t.group, t.source) for t in trajs]
    good = [t for t in trajs if t.success]
    if not good:
        raise ValueError(
            f"no successful trajectories among the {len(trajs)} loaded: TabSchema learns from successful "
            "runs only (check the outcome field the adapter reads)"
        )
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
        try:
            _preflight(adapter, dps, decision, kind)
            problems = []
        except (ValueError, NotImplementedError) as e:
            problems = [str(e)]
        labels = getattr(adapter, "labels", None)
        sample = sample_decision_points(dps, cfg.n_analyze, cfg.sample_seed, labels, cfg.sample_strategy)
        name = cfg.model or (llm.model if llm is not None else DEFAULT_MODEL)
        tin, tout, cost = _estimate(sample, cfg.prices, name, cfg.max_prompt_chars, cfg.stage_models)
        return DryRun(len(trajs), len(good), len(dps), len(sample), tin, tout, cost, comps, name, problems)

    llm = llm or auto_llm(cfg.model)
    box = _sandbox_from(sandbox, cfg.timeout_s)
    schema = TabSchema(llm, cfg, adapter, sandbox=box)  # checks the budget can be enforced
    _preflight(adapter, dps, decision, kind)  # and the data, still before any LLM call
    run = Path(run_dir) if run_dir else None
    schema.store = RunStore(run.parent, run.name) if run else RunStore()
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
        adapter=adapter,  # head.json records it, so live prefixes are built as in training
        sandbox=sandbox,
        redact=redact,  # and the redaction function's import path
    )
    if gate_coverage is not None:
        head.calibrate_gate(rep.cv, gate_coverage)
    if out is not None:
        head.save(out)
        logger.info("saved the head to %s", out)
    return head


def load(
    path: str | Path, trust_extractor: bool = False, trust_code: bool = False, **head_options: Any
) -> TabAgentHead | TabAgent:
    """Load a saved head: a ``TabAgentHead`` for a directory written by ``onboard`` (it has ``head.json``),
    else the ``TabAgent`` artifact written by ``TabAgent.save``.

    ``path`` is a local directory or a Hugging Face Hub repository, ``hf://OWNER/REPO[@REVISION]``, which is
    downloaded first (``pip install "tabagent[hub]"``; see ``tabagent.hub``) and then loaded and verified
    exactly as a local directory.

    A ``TabAgentHead`` decides from a trajectory prefix (``decide(prefix, candidates, k=...)``); a bare
    ``TabAgent`` decides from candidate rows (``decide(rows, k=...)``). ``trust_extractor``, ``trust_code``
    and the ``head_options`` (``sandbox``, ``redact``, ``adapter``) are described in
    :meth:`TabAgentHead.load`; a ``TabAgent`` has no extractor and takes no ``head_options``.
    """
    p = hub.resolve(path)
    if (p / "head.json").exists():
        return TabAgentHead.load(p, trust_extractor, trust_code, **head_options)
    if head_options:
        raise TypeError(f"{sorted(head_options)} apply to a head saved by onboard; {path} is a TabAgent")
    return TabAgent.load(p, trust_code=trust_code)
