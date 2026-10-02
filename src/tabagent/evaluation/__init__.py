"""Evaluation protocol: paper metrics, task-grouped cross-validation, and bootstrap intervals."""

from tabagent.evaluation.cv import (
    DEFAULT_SEED,
    PAPER_SEEDS,
    CVResult,
    assign_folds,
    cross_validate,
    load_folds,
    paper_strata,
    save_folds,
    stable_uid,
)
from tabagent.evaluation.discovery import DiscoveryCVResult, cross_validate_discovery
from tabagent.evaluation.metrics import normalized, per_decision
from tabagent.evaluation.stats import Interval, bootstrap_ci, holm, paired_bootstrap

__all__ = [
    "DEFAULT_SEED",
    "PAPER_SEEDS",
    "CVResult",
    "DiscoveryCVResult",
    "Interval",
    "assign_folds",
    "bootstrap_ci",
    "cross_validate",
    "cross_validate_discovery",
    "holm",
    "load_folds",
    "normalized",
    "paired_bootstrap",
    "paper_strata",
    "per_decision",
    "save_folds",
    "stable_uid",
]
