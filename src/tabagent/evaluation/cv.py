"""Task-grouped cross-validation (paper Sec. 4.3, L443-445; App. E, L2292).

Five folds at the task level: every row, synthetic row and feature statistic derived from a task stays
in one fold, and every data-dependent stage is fitted inside the training portion of each fold. Folds can
be stratified by application or domain and by deciles of |G(t)|. Stochastic methods run over three seeds
and task-level scores are averaged over seeds before any interval is computed.

``cross_validate`` is the only place in the library that decides what is training data: for every fold
and seed it clones the model, fits it on the training groups, and scores the held-out groups through a
label-free view of the table.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from tabagent.evaluation import metrics as M
from tabagent.evaluation import stats as S
from tabagent.table import DecisionTable

PAPER_SEEDS = (20260720, 20260726, 20260727)
FOLD_SEED = 20260720


def stable_uid(prefix: str, *parts: object) -> str:
    """Deterministic id from parts (identical to the trajectory-corpus builder)."""
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


# --------------------------------------------------------------------------- folds
def assign_folds(
    groups: Iterable[Hashable],
    n_splits: int = 5,
    seed: int = FOLD_SEED,
    strata: Mapping[Hashable, Hashable] | None = None,
    order: str = "sorted",
) -> dict:
    """Map each unique group to a fold in ``range(n_splits)``.

    Without ``strata`` the unique groups are shuffled with ``default_rng(seed)`` and dealt round-robin;
    with ``order="appearance"`` this is identical to the pre-release ``groupkfold``. ``order="sorted"``
    (default) sorts the groups first, so folds do not depend on row order. With ``strata`` the groups of
    each stratum are shuffled and dealt round-robin, continuing the deal across strata, so each fold gets
    a near-equal share of every stratum (the paper's stratification by application and |G(t)| decile).
    """
    uniq = list(dict.fromkeys(groups))
    if order == "sorted":
        uniq = sorted(uniq, key=str)
    elif order != "appearance":
        raise ValueError("order must be 'sorted' or 'appearance'")
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
    by_group = {str(g): int(f) for g, f in folds.items()}
    Path(path).write_text(
        json.dumps({"group_col": group_col, "by_group": by_group}, indent=0, sort_keys=True)
    )


def load_folds(path: str | Path) -> dict:
    """Read a fold file (``{"group_col": ..., "by_group": {group: fold}}``, the pre-release format)."""
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
        values = self.per_group(metric) if level == "group" else self.per_decision(metric)
        return S.bootstrap_ci(values.to_numpy(), **kw)

    def summary(self, metrics: Iterable[str] = ("p@r",), level: str = "group", **kw) -> pd.DataFrame:
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
    fold_seed: int = FOLD_SEED,
    seeds: Iterable[int] = (0, 1, 2),
    stratify: bool | str = False,
) -> CVResult:
    """Grouped K-fold out-of-fold scoring.

    ``folds`` pins a group->fold map (use this to reproduce published numbers); otherwise folds are
    assigned with ``fold_seed``, optionally stratified (``stratify=True`` uses ``spec.strata`` and |G(t)|
    deciles; a column name overrides the stratum column). ``seeds`` seed the model, not the folds.
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
    for seed in seeds:
        col = np.empty(len(table), dtype=object if single else float)
        for f in sorted(set(fold_of)):
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
