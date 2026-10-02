"""Data-driven usefulness of the extractor's features: the release's addition to TabSchema, not in the paper.

The judges score a feature from its description; whether it helps the head is an empirical question.
In the spirit of CAAFE (Hollmann et al., 2023, which keeps an LLM-proposed feature only when it
improves cross-validated performance), every extractor feature is scored by drop-column importance on
the onboarding rows alone:

1. the decision table is built from the compile step's extractor outputs (no new extractor run), on at
   most ``usefulness_max_decisions`` decision points sampled one per task first, and with fewer of them
   when the rows (decision points x candidates) would exceed ``usefulness_max_rows``, which bounds the
   runtime (at least ``MIN_DECISIONS`` are kept);
2. a cheap head (CatBoost with ``ITERATIONS`` iterations) is cross-validated with ``FOLDS`` task-grouped
   folds on the card's features, without the embedding similarity (slow to compute, and not what is being
   measured). The same is repeated with each extractor feature removed, on the same folds;
3. a feature's contribution is the paired, task-level difference in the decision's primary metric (P@R
   for rankings, F1 for multi-label, accuracy for single-class decisions) between the head with it and
   the head without it, with a 95% bootstrap interval over tasks (the paper's protocol) and a bootstrap
   p-value, Holm-adjusted over the features (``tabagent.evaluation.stats``). A feature is **useful** when the
   adjusted p-value is below 0.05 and the difference positive, **harmful** when it is below 0.05 and the
   difference negative (removing it helps), and **neutral** otherwise.

The report is informational by default (one seed). ``TabSchemaConfig(prune_harmful=True)`` repeats the
estimate with a second seed (other folds and model seed) and drops the LLM-written features that are harmful
under both, but never all of them. Pruning on the onboarding rows makes the onboarding report's
cross-validation optimistic, since the same rows chose the features; the report says so, and
``tabagent.cross_validate_discovery`` (which prunes inside each training fold) gives the honest estimate.
Only the trajectories given to discovery are used, so the estimate never sees held-out data. In-fold
discovery runs it only when it prunes. Its cost is (features + 1) x ``FOLDS`` small fits per seed, on at
most ``usefulness_max_rows`` rows: on public τ²-bench logs (15 features, capped to 210 decision points of
about 14 candidates, 2,988 rows) it took 32 seconds on a laptop CPU with one seed, and a 195-candidate
decision took 28 seconds instead of four minutes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd

from tabagent.defaults import DEFAULT_SEED
from tabagent.table import DecisionTable
from tabagent.tabschema.card import FeatureCard

logger = logging.getLogger(__name__)

#: Decision points the estimate uses at most (sampled one per task first), the rows (decision points x
#: candidates) it fits on at most, and the fewest decision points it keeps under the row cap.
DEFAULT_MAX_DECISIONS, DEFAULT_MAX_ROWS, MIN_DECISIONS = 400, 3000, 30
#: Task-grouped folds and CatBoost iterations of the estimate.
FOLDS, ITERATIONS = 3, 50
#: Holm-adjusted p-value below which a feature is useful or harmful.
ALPHA = 0.05


def _per_group(
    card: FeatureCard, table: DecisionTable, metric: str, n_splits: int, iterations: int, seed: int
) -> pd.Series:
    from tabagent.agent import TabAgent
    from tabagent.evaluation.cv import cross_validate
    from tabagent.tabhead import TabHead

    head = TabHead(table.spec.kind, iterations=iterations, seed=seed)
    res = cross_validate(
        TabAgent(card, head=head, seed=seed), table, n_splits=n_splits, fold_seed=seed, seeds=(seed,)
    )
    return res.per_group(metric)


def feature_usefulness(
    table: DecisionTable,
    card_for: Callable[[frozenset[str]], FeatureCard],
    columns: Iterable[str],
    *,
    seed: int = DEFAULT_SEED,
    seeds: int = 1,
    n_splits: int = FOLDS,
    iterations: int = ITERATIONS,
) -> dict[str, Any]:
    """Drop-column usefulness of ``columns`` on ``table`` (see the module docstring). ``card_for(dropped)``
    returns the card without the features in ``dropped``. With ``seeds`` > 1 the estimate is repeated with
    other folds and model seeds, and a feature is useful or harmful only when every seed says so. Returns a
    JSON-able record: the metric, the base score and per feature its contribution ``delta`` (base minus
    without it, averaged over seeds), the first seed's 95% interval ``ci``, the largest Holm-adjusted p-value
    ``p_holm``, each seed's verdict and the ``verdict``."""
    from tabagent.auto import PRIMARY_METRIC
    from tabagent.evaluation.stats import bootstrap_pvalue, holm, paired_bootstrap

    t0 = time.monotonic()
    columns = list(columns)
    metric = PRIMARY_METRIC[table.spec.kind]
    k = min(n_splits, len(table.groups))
    rec: dict[str, Any] = {
        "metric": metric,
        "folds": k,
        "iterations": iterations,
        "seeds": seeds,
        "decisions": int(table.decision_ids.nunique()),
        "rows": len(table),
        "tasks": len(table.groups),
        "features": {},
    }
    if k < 2:
        rec["skipped"] = "fewer than 2 tasks: a task-grouped cross-validation needs at least 2"
        return rec
    runs = []
    for s in range(seed, seed + seeds):
        base = _per_group(card_for(frozenset()), table, metric, k, iterations, s)
        diffs, cis = [], []
        for col in columns:
            a, b = base.align(
                _per_group(card_for(frozenset({col})), table, metric, k, iterations, s), join="inner"
            )
            d = a.to_numpy(float) - b.to_numpy(float)
            diffs.append(d)
            cis.append(paired_bootstrap(a.to_numpy(), b.to_numpy(), seed=s))
        p = holm([bootstrap_pvalue(d, seed=s) for d in diffs]) if diffs else []
        verdicts = [
            ("useful" if np.nanmean(d) > 0 else "harmful") if q < ALPHA else "neutral"
            for d, q in zip(diffs, p, strict=True)
        ]
        runs.append((float(base.mean()), diffs, cis, list(p), verdicts))
    rec["base"] = round(runs[0][0], 4)
    for j, col in enumerate(columns):
        per_seed = [r[4][j] for r in runs]
        delta = float(np.mean([np.nanmean(r[1][j]) for r in runs]))
        ci = runs[0][2][j]
        rec["features"][col] = {
            "delta": round(delta, 4),
            "ci": [round(ci.low, 4), round(ci.high, 4)],
            "p_holm": round(max(float(r[3][j]) for r in runs), 4),
            "verdicts": per_seed,
            "verdict": per_seed[0] if len(set(per_seed)) == 1 else "neutral",
        }
        logger.info(
            "usefulness: %s %s %+.4f (Holm p %.3f)",
            col,
            rec["features"][col]["verdict"],
            delta,
            rec["features"][col]["p_holm"],
        )
    rec["seconds"] = round(time.monotonic() - t0, 1)
    return rec
