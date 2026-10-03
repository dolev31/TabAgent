"""Data-driven usefulness of the extractor's features: the release's addition to TabSchema, not in the paper.

The judges score a feature from its description; whether it helps the head is an empirical question.
In the spirit of CAAFE (Hollmann et al., 2023, which keeps an LLM-proposed feature only when it
improves cross-validated performance), every extractor feature is scored by drop-feature importance on
the onboarding rows alone. A feature is what the plan proposed: the output columns that implement one
planned feature (a multi-hot of the tools called so far, say, one column per tool) are dropped together,
since the other columns of a feature would make up for any one of them:

1. the decision table is built from the compile step's extractor outputs (no new extractor run), on at
   most ``usefulness_max_decisions`` decision points sampled one per task first, and with fewer of them
   when the rows (decision points x candidates) would exceed ``usefulness_max_rows``, which bounds the
   runtime (at least ``MIN_DECISIONS`` are kept);
2. a cheap head (CatBoost with ``ITERATIONS`` iterations) is cross-validated with ``FOLDS`` task-grouped
   folds on the card's features, without the embedding similarity (slow to compute, and not what is being
   measured). The same is repeated with each extractor feature removed, on the same folds;
3. a feature's contribution is the paired, task-level difference in the decision's primary metric (P@R
   for rankings, F1 for multi-label, accuracy for single-class decisions) between the head with it and
   the head without it (and, in one more row, without every LLM-written feature at once: whether discovery
   helped at all), with a 95% bootstrap interval over tasks (the paper's protocol) and a paired
   t-test p-value over tasks, Holm-adjusted over the features (``tabagent.evaluation.stats``). A feature is
   **useful** when the adjusted p-value is below 0.05 and the difference positive, **harmful** when it is
   below 0.05 and the difference negative (removing it helps), and **neutral** otherwise. The t-test, not a
   bootstrap p-value, because a bootstrap p-value cannot fall below 1 / (resamples + 1): over a hundred
   features, Holm's correction would then call every feature neutral, however large its effect. The 95%
   interval is not corrected for the number of features, so a feature can be neutral although its interval
   excludes zero.

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
from collections.abc import Callable, Iterable, Mapping
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
#: The report's row for every LLM-written feature dropped at once: whether discovery helped at all. One
#: feature among many overlapping ones rarely moves the head; the block does. A single question, so its
#: p-value is not Holm-adjusted with the features'; never pruned.
ALL_LLM = "(all LLM-written features)"


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


def feature_groups(compiled: Any) -> dict[str, list[str]]:
    """A compiled extractor's output columns grouped by the planned feature each implements
    (``compiled.specs[column]["feature_name"]``); a column without a plan is its own feature. With more
    than one LLM-written feature, ``ALL_LLM`` groups them all."""
    groups: dict[str, list[str]] = {}
    for col in compiled.features:
        name = str((compiled.specs.get(col) or {}).get("feature_name") or col)
        groups.setdefault(name, []).append(col)
    llm = list(getattr(compiled, "llm_columns", []) or [])
    if len({n for n, cols in groups.items() if set(cols) & set(llm)}) > 1:
        groups[ALL_LLM] = llm
    return groups


def paired_pvalue(diffs: np.ndarray) -> float:
    """Two-sided paired t-test p-value of H0: mean difference = 0 (1.0 when the differences are all equal)."""
    from scipy import stats

    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2 or np.allclose(d, d[0]):
        return 1.0 if len(d) < 2 or d[0] == 0 else 0.0
    return float(stats.ttest_1samp(d, 0.0).pvalue)


def feature_usefulness(
    table: DecisionTable,
    card_for: Callable[[frozenset[str]], FeatureCard],
    columns: Iterable[str] | Mapping[str, Iterable[str]],
    *,
    seed: int = DEFAULT_SEED,
    seeds: int = 1,
    n_splits: int = FOLDS,
    iterations: int = ITERATIONS,
) -> dict[str, Any]:
    """Drop-feature usefulness on ``table`` (see the module docstring). ``columns`` lists the columns to
    score one by one, or maps each feature to the columns that implement it, which are dropped together.
    ``card_for(dropped)`` returns the card without the columns in ``dropped``. With ``seeds`` > 1 the estimate
    is repeated with other folds and model seeds, and a feature is useful or harmful only when every seed
    says so. Returns a JSON-able record: the metric, the base score and per feature its ``columns``, its
    contribution ``delta`` (base minus without it, averaged over seeds), the first seed's 95% interval
    ``ci``, the largest Holm-adjusted p-value ``p_holm``, each seed's verdict and the ``verdict``."""
    from tabagent.auto import PRIMARY_METRIC
    from tabagent.evaluation.stats import holm, paired_bootstrap

    t0 = time.monotonic()
    if isinstance(columns, Mapping):
        groups = {name: list(cols) for name, cols in columns.items()}
    else:
        groups = {c: [c] for c in columns}
    names = list(groups)
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
        for name in names:
            a, b = base.align(
                _per_group(card_for(frozenset(groups[name])), table, metric, k, iterations, s), join="inner"
            )
            d = a.to_numpy(float) - b.to_numpy(float)
            diffs.append(d)
            cis.append(paired_bootstrap(a.to_numpy(), b.to_numpy(), seed=s))
        raw = [paired_pvalue(d) for d in diffs]
        # Holm over the features; the block of every LLM-written feature is one question of its own.
        family = [j for j, n in enumerate(names) if n != ALL_LLM]
        p = list(raw)
        for j, q in zip(family, holm([raw[j] for j in family]) if family else [], strict=True):
            p[j] = float(q)
        verdicts = [
            ("useful" if np.nanmean(d) > 0 else "harmful") if q < ALPHA else "neutral"
            for d, q in zip(diffs, p, strict=True)
        ]
        runs.append((float(base.mean()), diffs, cis, list(p), verdicts))
    rec["base"] = round(runs[0][0], 4)
    for j, col in enumerate(names):
        per_seed = [r[4][j] for r in runs]
        delta = float(np.mean([np.nanmean(r[1][j]) for r in runs]))
        ci = runs[0][2][j]
        rec["features"][col] = {
            "columns": groups[col],
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
