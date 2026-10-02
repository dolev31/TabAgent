"""Cross-validation with TabSchema discovery inside every training fold (paper Sec. 4.3).

The paper fits "every data-dependent stage ... inside the training portion of each fold", and that "covers
the TabSchema loop (feature proposal, aggregation, judging, and extraction-code construction)": only the
resulting extractors are applied to the held-out fold. ``onboard`` discovers the feature card once, on every
onboarding trajectory, and its report cross-validates the head on that card. Its folds keep the head's
training rows and the fitted statistics apart, but every held-out task was among the trajectories the card
was discovered from, so that estimate is quick (one discovery) and can be optimistic.

:func:`cross_validate_discovery` estimates the whole pipeline instead. For every fold it runs
``TabSchema.discover`` on the training fold's trajectories only, builds the training and held-out rows with
that fold's extractor, selects the training recipe on the training rows (as ``onboard`` does), fits, and
scores the held-out decisions; ``tabagent.cross_validate`` does the splitting, the per-seed refits and the
scoring, so the metrics and intervals are the usual ``CVResult`` ones. Discovery runs once per fold (seeds
reuse it) in its own run directory, so a rerun with the same ``run_dir`` resumes from the cached LLM answers
and pays nothing again. An LLM budget (``config.max_cost_usd``) covers all folds together.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tabagent.defaults import DEFAULT_SEED, cache_root
from tabagent.evaluation.cv import CVResult, assign_folds, cross_validate
from tabagent.table import DecisionSpec, DecisionTable

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryCVResult(CVResult):
    """A ``CVResult`` whose feature cards were discovered inside each training fold.

    ``discoveries`` holds each fold's ``DiscoveryResult`` (card, extractor, report), ``recipes`` the training
    recipe chosen on each training fold, ``cost_usd`` the LLM cost of every fold's discovery together (None
    when a call's model had no price) and ``run_dir`` the directory of the per-fold runs.
    """

    discoveries: dict = field(default_factory=dict, repr=False)
    recipes: dict = field(default_factory=dict)
    cost_usd: float | None = None
    run_dir: str = ""


def _group(t) -> str:
    """The cross-validation group of a trajectory, as ``rows_to_table`` names it."""
    return str(t.group or t.id)


class _FoldDiscovery:
    """Per-fold discovery, extraction and recipe selection, cached by the set of training groups."""

    def __init__(self, trajs, dps, target: str, kind: str, adapter, llm, config, base_dir: Path, **options):
        from tabagent.tabschema.orchestrator import DECISION

        self.trajs, self.target, self.kind, self.adapter, self.llm = trajs, target, kind, adapter, llm
        self.config, self.base_dir, self.decision_col = config, base_dir, DECISION
        self.dps: dict[str, Any] = {dp.id: dp for dp in dps}
        self.trust_extractor, self.sandbox = options["trust_extractor"], options["sandbox"]
        self.search, self.head_kw = options["search"], options["head_kw"]
        self.runs: dict[frozenset, tuple[Any, Any]] = {}  # training groups -> (TabSchema, DiscoveryResult)
        self.recipes: dict[frozenset, Any] = {}
        self.features: dict[tuple[frozenset, str], dict | None] = {}
        self.spent, self.priced = 0.0, True
        self._lock = threading.Lock()

    def discover_all(self, trains: list[list], workers: int) -> None:
        """Discover on every fold's training groups, ``workers`` folds at a time. With a budget, each
        fold gets an equal share of what is left, so concurrent folds cannot overspend it together."""
        from tabagent.tabschema.agents import in_parallel

        todo = [t for t in trains if frozenset(map(str, t)) not in self.runs]
        share = None
        if self.config.max_cost_usd is not None and todo:
            share = (self.config.max_cost_usd - self.spent) / len(todo)
        in_parallel(lambda t: self.discover(t, budget=share), todo, workers)

    def discover(self, train_groups: Iterable, budget: float | None = None) -> frozenset:
        from tabagent.tabschema.llm import BudgetExceeded
        from tabagent.tabschema.orchestrator import TabSchema
        from tabagent.tabschema.runstore import RunStore, digest

        key = frozenset(map(str, train_groups))
        if key in self.runs:
            return key
        cfg = self.config
        if budget is not None:  # a share of the budget, set by discover_all for concurrent folds
            cfg = dataclasses.replace(cfg, max_cost_usd=budget)
        elif cfg.max_cost_usd is not None:  # one budget for all folds: each fold gets what is left
            remaining = cfg.max_cost_usd - self.spent
            if remaining <= 0:
                raise BudgetExceeded(
                    f"in-fold discovery cost ${self.spent:.2f} reached max_cost_usd={cfg.max_cost_usd}"
                )
            cfg = dataclasses.replace(cfg, max_cost_usd=remaining)
        if cfg.usefulness and not cfg.prune_harmful:  # a per-fold report nobody reads: only when it prunes
            cfg = dataclasses.replace(cfg, usefulness=False)
        store = RunStore(self.base_dir, f"fold-{digest(sorted(key))[:12]}")
        schema = TabSchema(self.llm, cfg, self.adapter, store=store, sandbox=self.sandbox)
        train = [t for t in self.trajs if _group(t) in key]
        logger.info("in-fold discovery: %d training trajectories (run directory %s)", len(train), store.dir)
        try:
            result = schema.discover(train, self.target, self.kind)
        finally:
            with self._lock:
                self.spent += schema.meter.total_usd
                self.priced = self.priced and schema.meter.priced
        self.runs[key] = (schema, result)
        return key

    def table(self, key: frozenset, table: DecisionTable) -> DecisionTable:
        """``table`` (a slice of the skeleton rows) with the fold extractor's columns, filled as
        ``rows_to_table`` fills them, under the fold card's spec."""
        from tabagent.tabschema.compile import ExtractorRunner

        schema, result = self.runs[key]
        ids = list(dict.fromkeys(table.df[self.decision_col]))
        todo = [d for d in ids if (key, d) not in self.features]
        if todo:
            runner = ExtractorRunner(result.extractor.code, self.trust_extractor, schema.sandbox)
            for d, f in zip(todo, runner.run([self.dps[d].prefix for d in todo]), strict=True):
                self.features[(key, d)] = f
        cols = [f.name for f in result.card.features if f.source == "extractor" and f.name not in table.df]
        feats = [self.features[(key, d)] or {} for d in table.df[self.decision_col]]
        extra = pd.DataFrame({c: [f.get(c) for f in feats] for c in cols}, index=table.df.index)
        return DecisionTable(pd.concat([table.df, extra], axis=1), result.card.spec, table.has_labels)

    def agent(self, key: frozenset, train: DecisionTable, seed: int):
        from tabagent import auto
        from tabagent.agent import TabAgent
        from tabagent.tabhead import TabHead

        card = self.runs[key][1].card
        if self.search is None:
            head = TabHead(card.spec.kind, preset="paper", seed=seed, **self.head_kw)
            return TabAgent(card, head=head, seed=seed)
        if key not in self.recipes:  # selected once per fold, on the training rows only
            self.recipes[key] = auto.select(train, card, search=self.search, **self.head_kw)[0]
        return auto.build_agent(card, self.recipes[key], seed=seed, **self.head_kw)


class _InFoldModel:
    """What ``cross_validate`` fits per fold and seed: discovery on the training groups, then a TabAgent on
    the rows that fold's extractor produces."""

    def __init__(self, ctx: _FoldDiscovery, seed: int = 0):
        self.ctx, self.seed = ctx, seed

    def clone(self, seed: int | None = None) -> _InFoldModel:
        return _InFoldModel(self.ctx, self.seed if seed is None else seed)

    def fit(self, table: DecisionTable) -> _InFoldModel:
        self.key_ = self.ctx.discover(table.groups)
        train = self.ctx.table(self.key_, table)
        self.agent_ = self.ctx.agent(self.key_, train, self.seed).fit(train)
        return self

    def score(self, table: DecisionTable):
        return self.agent_.score(self.ctx.table(self.key_, table))


def cross_validate_discovery(
    trajectories,
    target: str,
    *,
    adapter: Any = "auto",
    llm: Any = None,
    config: Any = None,
    kind: str = "ranking",
    folds: Mapping | None = None,
    n_splits: int = 5,
    fold_seed: int = DEFAULT_SEED,
    seeds: Iterable[int] = (0,),
    search: str | None = "fast",
    iterations: int | None = None,
    run_dir: str | Path | None = None,
    trust_extractor: bool = False,
    sandbox: Any = None,
    redact: Callable[[dict], dict] | None = None,
    fold_workers: int = 3,
) -> DiscoveryCVResult:
    """Grouped cross-validation of the whole onboarding pipeline, with TabSchema discovery run inside every
    training fold (see the module docstring).

    Parameters
    ----------
    trajectories: a file or directory of agent logs, or a list of ``Trajectory``.
    target: the decision component to replace (as ``onboard``'s ``decision``).
    adapter: the log adapter, or ``"auto"`` to detect it from the files.
    llm, config: the TabSchema LLM (default ``auto_llm(config.model)``, from environment keys) and
        ``TabSchemaConfig``. ``config.n_analyze`` applies to each fold's discovery, so the LLM cost is about
        ``n_splits`` times one onboarding; ``config.max_cost_usd`` (with ``config.prices``) caps all folds
        together.
    kind: ``ranking``, ``multilabel`` or ``single``.
    folds, n_splits, fold_seed: as for ``cross_validate``; the defaults give the folds of ``onboard``'s
        report, so the two estimates are comparable.
    seeds: head seeds (``onboard``'s report uses one); discovery and recipe selection run once per fold.
    search: the recipe search of ``tabagent.auto.select`` (``fast`` or ``full``) run on each training fold,
        as ``onboard`` does on all data; None fits the paper's CatBoost preset without TabSynth.
    iterations: override the CatBoost iteration count (for quick runs).
    run_dir: the directory of the per-fold run directories (default a new one under
        ``$TABAGENT_CACHE/runs``); a rerun with the same directory and folds makes no new LLM calls.
    trust_extractor: run each fold's extractor in-process instead of in the sandbox.
    sandbox: the sandbox for discovery and extraction (default ``SubprocessSandbox``; ``DockerSandbox()``
        when the logs or the code are untrusted).
    redact: applied to every raw trajectory before anything can reach the LLM provider.
    fold_workers: folds whose discovery runs at the same time (each makes up to ``config.workers`` LLM calls
        at once). With ``config.max_cost_usd``, each concurrent fold gets an equal share of the budget.

    Returns
    -------
    DiscoveryCVResult: out-of-fold scores with the usual metric helpers (``summary``, ``ci``, ``compare``),
    each fold's discovery, the recipe chosen per fold and the total LLM cost.
    """
    from tabagent.tabschema.card import FeatureCard
    from tabagent.tabschema.llm import auto_llm
    from tabagent.tabschema.orchestrator import (
        CANDIDATE,
        DECISION,
        GROUP,
        LABEL,
        TabSchemaConfig,
        rows_to_table,
    )
    from tabagent.tabschema.trajectory import Trajectory

    if adapter == "auto":
        from tabagent.adapters import detect_adapter

        adapter = detect_adapter(trajectories)
    trajs = adapter.load(trajectories) if isinstance(trajectories, (str, Path)) else list(trajectories)
    if redact is not None:  # applied before anything can reach the LLM provider
        trajs = [Trajectory(t.id, redact(t.raw), t.success, t.group, t.source) for t in trajs]
    config = config or TabSchemaConfig()
    llm = llm or auto_llm(config.model)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_dir = Path(run_dir) if run_dir else cache_root() / "runs" / f"{stamp}-infold"

    good = [t for t in trajs if t.success]
    dps = [dp for t in good for dp in adapter.decision_points(t, target)]
    if not dps:
        raise ValueError(f"no occurrences of {target!r} in the successful trajectories")
    if kind == "single":
        spec = DecisionSpec("single", GROUP, "label", decision=(DECISION,))
    else:
        spec = DecisionSpec(kind, GROUP, LABEL, CANDIDATE, decision=(DECISION,))
    # The rows without extractor features: decision ids, candidates and labels, as onboard's table has them.
    skeleton = rows_to_table(FeatureCard("skeleton", spec, []), dps, [None] * len(dps), adapter)

    ctx = _FoldDiscovery(
        trajs,
        dps,
        target,
        kind,
        adapter,
        llm,
        config,
        base_dir,
        trust_extractor=trust_extractor,
        sandbox=sandbox,
        search=search,
        head_kw={} if iterations is None else {"iterations": iterations},
    )
    # The folds cross_validate uses; their discoveries run first, fold_workers at a time, and are reused.
    fold_map = dict(folds) if folds is not None else assign_folds(skeleton.groups, n_splits, fold_seed)
    groups = list(dict.fromkeys(skeleton.groups))
    fold_ids = sorted({fold_map.get(g, fold_map.get(str(g))) for g in groups})
    trains = [[g for g in groups if fold_map.get(g, fold_map.get(str(g))) != f] for f in fold_ids]
    ctx.discover_all(trains, max(1, fold_workers))
    res = cross_validate(_InFoldModel(ctx), skeleton, folds=fold_map, seeds=seeds)
    groups = set(map(str, skeleton.groups))
    fold_of = {str(g): f for g, f in res.folds.items() if str(g) in groups}
    by_fold = {
        f: frozenset(g for g, other in fold_of.items() if other != f) for f in sorted(set(fold_of.values()))
    }
    return DiscoveryCVResult(
        res.table,
        res.oof,
        res.folds,
        discoveries={f: ctx.runs[k][1] for f, k in by_fold.items() if k in ctx.runs},
        recipes={f: ctx.recipes[k].label() for f, k in by_fold.items() if k in ctx.recipes},
        cost_usd=ctx.spent if ctx.priced else None,
        run_dir=str(base_dir),
    )
