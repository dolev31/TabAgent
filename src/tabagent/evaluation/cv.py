"""Task-grouped cross-validation (paper Sec. 4.3; App. M).

Five folds at the task level: every row, synthetic row and feature statistic derived from a task stays
in one fold, and every data-dependent stage is fitted inside the training portion of each fold. Folds can
be stratified by application or domain and by deciles of |G(t)|. Models run over the paper's three seeds
(``PAPER_SEEDS``) and task-level scores are averaged over seeds before any interval is computed.

``cross_validate`` is the only place in the library that decides what is training data: for every fold
and seed it clones the model, fits it on the training groups, and scores the held-out groups through a
label-free view of the table.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from tabagent.defaults import DEFAULT_SEED, PAPER_SEEDS
from tabagent.evaluation import metrics as M
from tabagent.evaluation import stats as S
from tabagent.table import DecisionTable

logger = logging.getLogger(__name__)


def stable_uid(prefix: str, *parts: object) -> str:
    """Deterministic id: ``<prefix>_`` and the first 24 hex digits of the SHA-256 of the parts joined by
    U+001F. The dataset builders key tasks and trajectories by these ids, as do the pinned fold files."""
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


# --------------------------------------------------------------------------- folds
def assign_folds(
    groups: Iterable[Hashable],
    n_splits: int = 5,
    seed: int = DEFAULT_SEED,
    strata: Mapping[Hashable, Hashable] | None = None,
) -> dict:
    """Map each unique group to a fold in ``range(n_splits)``.

    The unique groups are sorted (so folds do not depend on row order), shuffled with ``default_rng(seed)``
    and dealt round-robin. With ``strata`` the groups of each stratum are shuffled and dealt round-robin,
    continuing the deal across strata, so each fold gets a near-equal share of every stratum (the paper's
    stratification by application and |G(t)| decile).
    """
    uniq = sorted(dict.fromkeys(groups), key=str)
    rng = np.random.default_rng(seed)
    if strata is None:
        rng.shuffle(uniq)
        return {g: i % n_splits for i, g in enumerate(uniq)}
    by_stratum: dict[Hashable, list] = {}
    for g in uniq:
        by_stratum.setdefault(strata[g], []).append(g)
    out, offset = {}, 0
    for key in sorted(by_stratum, key=str):
        members = by_stratum[key]
        rng.shuffle(members)
        for i, g in enumerate(members):
            out[g] = (offset + i) % n_splits
        offset += len(members)
    return out


def paper_strata(table: DecisionTable, by: str | None = None, size_deciles: bool = True) -> dict:
    """Composite stratum per group: the ``by`` column (application or domain) and the decile of |G(t)|."""
    s = table.spec
    df = table.df
    key = pd.Series("", index=pd.Index(table.groups, name=s.group), dtype=object)
    col = by or s.strata
    if col:
        key = key + df.groupby(s.group, sort=False)[col].first().astype(str).reindex(key.index)
    if size_deciles and s.kind != "single":
        size = df.groupby(s.group, sort=False)[s.label].sum().reindex(key.index)
        n_bins = min(10, size.nunique())
        dec = pd.qcut(size.rank(method="first"), n_bins, labels=False) if n_bins > 1 else 0
        key = key + "|" + pd.Series(dec, index=key.index).astype(str)
    return key.to_dict()


def save_folds(path: str | Path, folds: Mapping, group_col: str) -> None:
    """Write a fold file: ``{"group_col": ..., "by_group": {group: fold}}`` (see :func:`load_folds`)."""
    by_group = {str(g): int(f) for g, f in folds.items()}
    Path(path).write_text(
        json.dumps({"group_col": group_col, "by_group": by_group}, indent=0, sort_keys=True)
    )


def load_folds(path: str | Path) -> dict:
    """Read a fold file written by :func:`save_folds` (the format of the bundled paper folds)."""
    return {g: int(f) for g, f in json.loads(Path(path).read_text())["by_group"].items()}


# --------------------------------------------------------------------------- protocol
class Model(Protocol):
    """What ``cross_validate`` needs from a model (``TabAgent`` implements it)."""

    def clone(self, seed: int) -> Model: ...

    def fit(self, table: DecisionTable) -> Model: ...

    def score(self, table: DecisionTable) -> np.ndarray:
        """Positive-class probability per row; for single-class decisions, the predicted class."""
        ...


@dataclass
class CVResult:
    """Out-of-fold scores for every seed, plus the metric and interval helpers of the paper protocol."""

    table: DecisionTable
    oof: pd.DataFrame  # one column per seed, aligned with table.df
    folds: dict = field(repr=False)

    @property
    def seeds(self) -> list:
        """The model seeds, one ``oof`` column each."""
        return list(self.oof.columns)

    def _groups_of(self, decisions: pd.Index) -> pd.Series:
        lookup = pd.Series(self.table.df[self.table.spec.group].to_numpy(), index=self.table.decision_ids)
        return lookup[~lookup.index.duplicated()].reindex(decisions)

    def per_decision(self, metric: str, **kw) -> pd.Series:
        """Metric per decision, averaged over seeds (paper: task-level scores averaged over seeds)."""
        per_seed = [M.per_decision(self.table, self.oof[s].to_numpy(), metric, **kw) for s in self.seeds]
        return pd.concat(per_seed, axis=1).mean(axis=1).rename(metric)

    def per_group(self, metric: str, **kw) -> pd.Series:
        """Metric per task: the mean over the task's decisions (one value when a task is one decision)."""
        d = self.per_decision(metric, **kw)
        return d.groupby(self._groups_of(d.index).to_numpy()).mean().rename(metric)

    def ci(self, metric: str, level: str = "group", **kw) -> S.Interval:
        """Bootstrap interval of the metric's mean over tasks (``level="group"``) or decisions; ``kw`` go to
        :func:`tabagent.evaluation.stats.bootstrap_ci`."""
        values = self.per_group(metric) if level == "group" else self.per_decision(metric)
        return S.bootstrap_ci(values.to_numpy(), **kw)

    def summary(self, metrics: Iterable[str] = ("p@r",), level: str = "group", **kw) -> pd.DataFrame:
        """One row per metric: estimate, 95% interval bounds and the number of resampled units."""
        rows = []
        for m in metrics:
            iv = self.ci(m, level=level, **kw)
            rows.append({"metric": m, "value": iv.estimate, "ci_low": iv.low, "ci_high": iv.high, "n": iv.n})
        return pd.DataFrame(rows).set_index("metric")

    def compare(self, other: CVResult, metric: str, level: str = "group", **kw) -> S.Interval:
        """Paired bootstrap CI of (self - other) on the units both results cover."""
        a = self.per_group(metric) if level == "group" else self.per_decision(metric)
        b = other.per_group(metric) if level == "group" else other.per_decision(metric)
        a, b = a.align(b, join="inner")
        return S.paired_bootstrap(a.to_numpy(), b.to_numpy(), **kw)


def cross_validate(
    model: Model,
    table: DecisionTable,
    *,
    folds: Mapping | None = None,
    n_splits: int = 5,
    fold_seed: int = DEFAULT_SEED,
    seeds: Iterable[int] = PAPER_SEEDS,
    stratify: bool | str = False,
) -> CVResult:
    """Grouped K-fold out-of-fold scoring of ``model`` on ``table`` (paper Sec. 4.3; App. M).

    For every seed and fold the model is cloned with that seed, fitted on the training groups and scored on
    a label-free view of the held-out groups.

    Parameters
    ----------
    model: anything with ``clone(seed)``, ``fit(table)`` and ``score(table)``, such as a ``TabAgent``.
    table: the labeled decision table.
    folds: a pinned ``{group: fold}`` map (use the bundled paper folds to reproduce published numbers);
        otherwise folds are assigned with ``fold_seed``.
    n_splits: number of folds when ``folds`` is not given.
    fold_seed: seed of the fold assignment.
    seeds: model seeds; the default is the paper's three seeds. Seeds never change the folds.
    stratify: ``True`` stratifies by ``spec.strata`` and |G(t)| deciles; a column name overrides the
        stratum column.

    Returns
    -------
    CVResult: out-of-fold scores (one column per seed) with metric and interval helpers.
    """
    s = table.spec
    if folds is None:
        strata = paper_strata(table, by=stratify if isinstance(stratify, str) else None) if stratify else None
        folds = assign_folds(table.groups, n_splits, fold_seed, strata)
    fold_of = table.df[s.group].map(lambda g: folds.get(g, folds.get(str(g))))
    if fold_of.isna().any():
        raise ValueError(f"{int(fold_of.isna().sum())} rows belong to groups without a fold")
    fold_of = fold_of.astype(int).to_numpy()

    single = s.kind == "single"
    oof = pd.DataFrame(index=table.df.index)
    seeds, fold_ids = list(seeds), sorted(set(fold_of))
    for seed in seeds:
        col = np.empty(len(table), dtype=object if single else float)
        for i, f in enumerate(fold_ids, 1):
            logger.info("cross-validation: seed %s, fold %d/%d", seed, i, len(fold_ids))
            test_mask = fold_of == f
            train_groups = pd.unique(table.df.loc[~test_mask, s.group])
            test_groups = pd.unique(table.df.loc[test_mask, s.group])
            fitted = model.clone(seed).fit(table.subset(train_groups))
            test = table.subset(test_groups)
            scored = np.asarray(fitted.score(test.drop_labels()))
            # subset() preserves row order, so the scores map back onto the held-out rows directly.
            col[np.flatnonzero(test_mask)] = scored
        oof[seed] = col
    return CVResult(table=table, oof=oof, folds=dict(folds))
