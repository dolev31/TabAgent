"""The decision table: one row per (decision context, candidate) pair.

Every closed-set decision TabAgent replaces is reduced to the same pointwise layout (paper Sec. 3.3): a row
pairs the state of the agent at one decision point with one legal candidate, its columns are
behavior-grounded features, and its label says whether the successful trajectory chose that candidate.
``DecisionSpec`` names the columns that play a structural role; everything else is a feature candidate.

Roles
  group      task id. The unit of cross-validation: rows of one group never straddle folds.
  decision   columns identifying one decision (one candidate slate). Defaults to ``(group,)``.
  candidate  the candidate id (tool, skill, element, model, app). Absent for single-class decisions.
  label      binary relevance (ranking, multilabel, policy) or the class (single).
  reward     realized reward of choosing this row's candidate (policy decisions; never a feature).
  eval       other post-decision columns kept for scoring only (never features).
  strata     optional column used to stratify folds (for example the application or domain).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import pandas as pd

Kind = Literal["ranking", "multilabel", "single", "policy"]
KINDS: tuple[str, ...] = ("ranking", "multilabel", "single", "policy")


class LeakageError(RuntimeError):
    """Raised when post-decision information could reach the features or a held-out fold."""


@dataclass(frozen=True)
class DecisionSpec:
    """The structural columns of a decision table (see the module docstring for each role).

    ``kind`` is ``ranking``, ``multilabel``, ``single`` or ``policy``. Single-class decisions have no
    candidate column; every other kind needs one, and policies also need a ``reward`` column.
    """

    kind: Kind
    group: str
    label: str
    candidate: str | None = None
    decision: tuple[str, ...] = ()
    reward: str | None = None
    eval: tuple[str, ...] = ()
    strata: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if (self.kind == "single") != (self.candidate is None):
            raise ValueError("single-class decisions have no candidate column; all other kinds need one")
        if self.kind == "policy" and self.reward is None:
            raise ValueError("policy decisions need a reward column (the realized reward of each arm)")
        object.__setattr__(self, "decision", tuple(self.decision))
        object.__setattr__(self, "eval", tuple(self.eval))

    @property
    def decision_cols(self) -> tuple[str, ...]:
        """The columns identifying one decision: ``decision``, or the group when it is empty."""
        return self.decision or (self.group,)

    @property
    def outcome_cols(self) -> tuple[str, ...]:
        """Columns that carry the outcome of a decision. They are never features."""
        return tuple(c for c in (self.label, self.reward, *self.eval) if c)

    @property
    def structural_cols(self) -> tuple[str, ...]:
        """Every column with a structural role (ids, candidate, strata and outcomes)."""
        cols = [self.group, *self.decision_cols, self.candidate, self.strata, *self.outcome_cols]
        return tuple(dict.fromkeys(c for c in cols if c))


@dataclass
class DecisionTable:
    """A validated pointwise table plus its ``DecisionSpec``.

    ``has_labels`` is False for views produced by :meth:`drop_labels`, which is what a model sees at
    scoring time: the outcome columns are physically absent.
    """

    df: pd.DataFrame
    spec: DecisionSpec
    has_labels: bool = True
    _decision_ids: pd.Series | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.df = self.df.reset_index(drop=True)
        self._validate()

    # ------------------------------------------------------------------ validation
    def _validate(self) -> None:
        s, df = self.spec, self.df
        required = [s.group, *s.decision_cols] + ([s.candidate] if s.candidate else [])
        if self.has_labels:
            required += list(s.outcome_cols)
        if s.strata:
            required.append(s.strata)
        missing = [c for c in dict.fromkeys(required) if c not in df.columns]
        if missing:
            raise KeyError(f"decision table is missing columns {missing}")
        if not self.has_labels:
            present = [c for c in s.outcome_cols if c in df.columns]
            if present:
                raise LeakageError(f"label-free view still contains outcome columns {present}")
        if df[s.group].isna().any():
            raise ValueError(f"group column {s.group!r} has missing values")

        per_decision = df.groupby(list(s.decision_cols), sort=False)[s.group].nunique()
        if (per_decision > 1).any():
            raise ValueError("a decision spans several groups; decision ids must nest within groups")
        if s.candidate:
            dup = df.duplicated([*s.decision_cols, s.candidate])
            if dup.any():
                raise ValueError(f"{int(dup.sum())} duplicate (decision, candidate) rows")
        if self.has_labels and s.kind != "single":
            vals = set(pd.unique(df[s.label].dropna()))
            if not vals <= {0, 1}:
                raise ValueError(f"label {s.label!r} must be binary for {s.kind} decisions, got {vals}")

    # ------------------------------------------------------------------ accessors
    def __len__(self) -> int:
        return len(self.df)

    @property
    def groups(self) -> np.ndarray:
        """Unique group ids in order of first appearance."""
        return pd.unique(self.df[self.spec.group])

    @property
    def decision_ids(self) -> pd.Series:
        """One string id per row identifying its decision (slate)."""
        if self._decision_ids is None:
            cols = list(self.spec.decision_cols)
            key = self.df[cols[0]].astype(str)
            for c in cols[1:]:
                key = key + "\x1f" + self.df[c].astype(str)
            self._decision_ids = key.rename("decision_id")
        return self._decision_ids

    @property
    def labels(self) -> np.ndarray:
        """The label column (raises ``LeakageError`` on a label-free view)."""
        if not self.has_labels:
            raise LeakageError("labels were dropped from this view")
        return self.df[self.spec.label].to_numpy()

    def feature_frame(self, columns: Iterable[str]) -> pd.DataFrame:
        """Select feature columns, refusing any outcome column."""
        cols = list(columns)
        bad = [c for c in cols if c in self.spec.outcome_cols]
        if bad:
            raise LeakageError(f"outcome columns cannot be features: {bad}")
        return self.df[cols]

    # ------------------------------------------------------------------ views
    def subset(self, groups: Iterable) -> DecisionTable:
        """The rows of the given groups, in their original order."""
        keep = self.df[self.spec.group].isin(set(groups))
        return DecisionTable(self.df[keep], self.spec, self.has_labels)

    def drop_labels(self) -> DecisionTable:
        """A label-free view: the outcome columns are removed (what a model sees at scoring time)."""
        cols = [c for c in self.spec.outcome_cols if c in self.df.columns]
        return DecisionTable(self.df.drop(columns=cols), self.spec, has_labels=False)

    def with_df(self, df: pd.DataFrame) -> DecisionTable:
        """The same spec over another frame (validated again)."""
        return DecisionTable(df, self.spec, self.has_labels)

    def with_spec(self, **changes) -> DecisionTable:
        """The same rows under a spec with ``changes`` applied."""
        return DecisionTable(self.df, replace(self.spec, **changes), self.has_labels)
