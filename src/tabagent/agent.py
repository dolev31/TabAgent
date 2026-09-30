"""TabAgent: TabSchema card + TabSynth + TabHead as one estimator.

``TabAgent.fit`` is the whole training recipe of the paper, applied to whatever rows it is given:

1. fit the card's dependency statistics on the real training rows (paper Sec. 4.3);
2. build the feature frame;
3. add TabSynth paraphrase rows, if a synthesizer is set (training rows only, label-preserving; App. H,
   Algorithm 1);
4. fit TabHead on real + synthetic rows.

``score`` and ``decide`` never synthesize and accept label-free tables. ``cross_validate`` (in
``tabagent.evaluation``) clones the agent for every fold and seed, so every data-dependent stage is fitted
inside the training portion of each fold. The card may also be a callable ``DecisionTable -> FeatureCard``,
called on each training table (for example to select features per fold); the columns its features read
must already be in the table.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabagent.tabhead import TabHead, confidence, decode
from tabagent.table import DecisionTable, LeakageError
from tabagent.tabschema.card import FeatureCard

CardLike = FeatureCard | Callable[[DecisionTable], FeatureCard]

_DECIDE_USAGE = (
    "TabAgent.decide(table, k=...) scores a table of candidate rows: a DecisionTable, or a DataFrame with "
    "one row per (decision, candidate) holding the card's columns. decide(prefix, candidates, k=...) is the "
    "API of a head saved by tabagent.onboard (a TabAgentHead), which bundles the extractor that turns a "
    "trajectory prefix into features; a TabAgent saved with TabAgent.save or taken from an sklearn "
    "estimator has no extractor, so build its candidate rows and call decide(rows, k=...)."
)


def _label_copy_check(X: pd.DataFrame, y: np.ndarray) -> None:
    """Tripwire: refuse a feature column that reproduces the label exactly (or its complement)."""
    if len(np.unique(y)) != 2:
        return
    yb = pd.Series(pd.factorize(pd.Series(y))[0], index=X.index)  # labels may be 0/1 or two class names
    for c in X.columns:
        col = X[c]
        if col.nunique(dropna=False) != 2:
            continue
        codes = pd.Series(pd.factorize(col)[0], index=X.index)
        if (codes == yb).all() or (codes == 1 - yb).all():
            raise LeakageError(f"feature {c!r} reproduces the label exactly; check the card for leakage")


class TabAgent:
    """A feature card, an optional TabSynth and a TabHead, trained and applied as one model.

    Parameters
    ----------
    card: the ``FeatureCard`` (or a ``DecisionTable -> FeatureCard`` callable, called on each training table).
    head: the ``TabHead`` to fit; default ``TabHead(card.spec.kind)`` with the paper's CatBoost preset.
    synth: a ``TabSynth`` (or anything with its ``augment`` method) applied to the training rows, or None.
    seed: the head's seed (``cross_validate`` replaces it per seed through ``clone``).

    After ``fit``: ``card_`` (the fitted card), ``head_`` (the fitted head) and ``n_synthetic_``.
    """

    def __init__(self, card: CardLike, head: TabHead | None = None, synth: Any = None, seed: int = 0):
        self.card = card
        self.head = head
        self.synth = synth
        self.seed = seed
        self.card_: FeatureCard | None = None
        self.head_: TabHead | None = None
        self.n_synthetic_: int = 0

    # ------------------------------------------------------------------ estimator protocol
    def clone(self, seed: int | None = None) -> TabAgent:
        """An unfitted copy with the same card, head settings and synthesizer (and ``seed``, when given)."""
        card = self.card.clone() if isinstance(self.card, FeatureCard) else self.card
        return TabAgent(
            card, copy.deepcopy(self.head), copy.deepcopy(self.synth), self.seed if seed is None else seed
        )

    def fit(self, table: DecisionTable) -> TabAgent:
        """Fit the card's statistics, synthesize (if set) and fit the head on the labeled ``table``."""
        card = self.card.clone() if isinstance(self.card, FeatureCard) else self.card(table)
        card.fit(table)
        X = card.transform(table)
        y = table.labels
        _label_copy_check(X, y)
        self.n_synthetic_ = 0
        if self.synth is not None:
            X_syn, y_syn = self.synth.augment(X, y, table, card)
            self.n_synthetic_ = len(X_syn)
            if len(X_syn):
                X = pd.concat([X, X_syn], ignore_index=True)
                y = np.concatenate([y, y_syn])
        head = (self.head or TabHead(card.spec.kind)).clone(seed=self.seed)
        self.head_ = head.fit(X, y, card.types)
        self.card_ = card
        return self

    def _check_fitted(self) -> None:
        if self.head_ is None:
            raise RuntimeError("TabAgent is not fitted")

    def as_table(self, data: DecisionTable | pd.DataFrame) -> DecisionTable:
        """``data`` as a label-free ``DecisionTable`` under the fitted card's spec.

        A DataFrame holds one row per (decision, candidate). Missing group and decision columns are filled
        with one constant id, so a frame of one decision's candidates needs no id columns; outcome columns,
        if present, are dropped.
        """
        self._check_fitted()
        if isinstance(data, DecisionTable):
            return data
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"expected a DecisionTable or a DataFrame, got {type(data).__name__}. {_DECIDE_USAGE}"
            )
        spec = self.card_.spec
        df = data.drop(columns=[c for c in spec.outcome_cols if c in data.columns])
        for col in dict.fromkeys((spec.group, *spec.decision_cols)):
            if col not in df.columns:
                df = df.assign(**{col: "decision"})
        return DecisionTable(df, spec, has_labels=False)

    def features(self, data: DecisionTable | pd.DataFrame) -> pd.DataFrame:
        """The fitted card's feature frame for ``data`` (a table or a DataFrame of candidate rows)."""
        table = self.as_table(data)
        return self.card_.transform(table)

    def score(self, data: DecisionTable | pd.DataFrame) -> np.ndarray:
        """p(candidate is chosen) per row; the predicted class per row for single-class decisions."""
        X = self.features(data)
        return self.head_.score(X)

    # ------------------------------------------------------------------ decisions
    def decide(
        self,
        data: DecisionTable | pd.DataFrame,
        *args: Any,
        k: int | None = None,
        defer_below: float | None = None,
        fallback: Callable[[pd.DataFrame], Any] | None = None,
        confidence_method: str = "largest_gap",
    ) -> pd.DataFrame:
        """Decode a decision for every slate in ``data``.

        Parameters
        ----------
        data: a ``DecisionTable`` or a DataFrame with one row per (decision, candidate) (see ``as_table``).
            Further positional arguments raise a ``TypeError`` that explains the difference from
            ``TabAgentHead.decide(prefix, candidates, k=...)``.
        k: keep the top ``k`` candidates of each ranking (keyword-only).
        defer_below: mark decisions whose confidence is below this value ``deferred`` (the confidence-gated
            fallback, App. N.3). For rankings the confidence is ``tabhead.confidence(method=
            confidence_method)``; for single-class decisions it is the predicted class's probability.
        fallback: called with the deferred decisions' rows (typically the original LLM component); its
            return value is stored in ``result.attrs["fallback"]``.

        Returns
        -------
        One row per selected (decision, candidate) with ``score`` and ``rank`` (ties ordered by candidate
        id), or one row per decision with ``choice`` for single-class decisions; with ``defer_below``, also
        ``confidence`` and ``deferred``.
        """
        if args:
            raise TypeError(_DECIDE_USAGE)
        table = self.as_table(data)
        spec = self.card_.spec
        ids = table.decision_ids.to_numpy()
        if spec.kind == "single":
            X = self.card_.transform(table)
            proba = self.head_.predict_proba(X)
            classes = np.asarray(self.head_.classes_, dtype=object)
            out = decode("single", classes[proba.argmax(axis=1)], ids)
            conf = pd.Series(proba.max(axis=1), index=ids)
        else:
            scores = self.score(table)
            cands = table.df[spec.candidate].to_numpy()
            out = decode(spec.kind, scores, ids, cands, k=k, threshold=self.head_.threshold)
            conf = None if defer_below is None else confidence(scores, ids, cands, method=confidence_method)
        if defer_below is not None:
            out["confidence"] = out["decision"].map(conf.groupby(level=0).first())
            out["deferred"] = out["confidence"] < defer_below
            if fallback is not None:
                deferred = set(out.loc[out["deferred"], "decision"])
                if deferred:
                    rows = table.df[table.decision_ids.isin(deferred).to_numpy()]
                    out.attrs["fallback"] = fallback(rows)
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path, metrics: dict | None = None) -> Path:
        """Write a portable, pickle-free artifact directory (see ``tabagent.artifact``); ``metrics`` are
        recorded in its manifest."""
        from tabagent import artifact

        return artifact.save(self, path, metrics)

    @classmethod
    def load(cls, path: str | Path, verify: bool = True, trust_code: bool = False) -> TabAgent:
        """Load an artifact written by ``save``. ``verify`` checks every file against the manifest's SHA-256;
        ``trust_code`` allows card features computed by modules outside ``tabagent`` (``python:`` sources)."""
        from tabagent import artifact

        return artifact.load(path, verify, trust_code=trust_code)
