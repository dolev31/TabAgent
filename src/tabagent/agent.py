"""TabAgent: TabSchema card + TabSynth + TabHead as one estimator.

``TabAgent.fit`` is the whole training recipe of the paper, applied to whatever rows it is given:

1. fit the card's dependency statistics on the real training rows (L443);
2. build the feature frame;
3. add TabSynth paraphrase rows, if a synthesizer is set (training only, label-preserving, L2076);
4. fit TabHead on real + synthetic rows.

``score`` and ``decide`` never synthesize and accept label-free tables. ``cross_validate`` (in
``tabagent.evaluation``) clones the agent for every fold and seed, so every data-dependent stage is fitted
inside the training portion of each fold. The card may be a callable ``DecisionTable -> FeatureCard``: it is
then re-derived on each training fold, which is how "the TabSchema loop runs in-fold" is executed.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from tabagent.tabhead import TabHead, confidence, decode
from tabagent.table import DecisionTable, LeakageError
from tabagent.tabschema.card import FeatureCard

CardLike = FeatureCard | Callable[[DecisionTable], FeatureCard]


def _label_copy_check(X: pd.DataFrame, y: np.ndarray) -> None:
    """Tripwire: refuse a feature column that reproduces the label exactly (or its complement)."""
    if len(np.unique(y)) != 2:
        return
    yb = pd.Series(y, index=X.index).astype(int)
    for c in X.columns:
        col = X[c]
        if col.nunique(dropna=False) != 2:
            continue
        codes = pd.Series(pd.factorize(col)[0], index=X.index)
        if (codes == yb).all() or (codes == 1 - yb).all():
            raise LeakageError(f"feature {c!r} reproduces the label exactly; check the card for leakage")


class TabAgent:
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
        card = self.card.clone() if isinstance(self.card, FeatureCard) else self.card
        return TabAgent(
            card, copy.deepcopy(self.head), copy.deepcopy(self.synth), self.seed if seed is None else seed
        )

    def fit(self, table: DecisionTable) -> TabAgent:
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

    def features(self, table: DecisionTable) -> pd.DataFrame:
        self._check_fitted()
        return self.card_.transform(table)

    def score(self, table: DecisionTable) -> np.ndarray:
        """p(candidate is chosen) per row; the predicted class per row for single-class decisions."""
        X = self.features(table)
        return self.head_.score(X)

    # ------------------------------------------------------------------ decisions
    def decide(
        self,
        table: DecisionTable,
        k: int | None = None,
        *,
        defer_below: float | None = None,
        fallback: Callable[[pd.DataFrame], Any] | None = None,
        confidence_method: str = "largest_gap",
    ) -> pd.DataFrame:
        """Decode decisions for every slate in ``table``.

        With ``defer_below``, decisions whose confidence (App. D gate) is below the threshold are marked
        ``deferred``; if a ``fallback`` callable is given (typically the original LLM component), it is
        called with the deferred decisions' rows and its return value is stored in ``fallback``.
        """
        spec = self.card_.spec if self.card_ else table.spec
        scores = self.score(table)
        ids = table.decision_ids.to_numpy()
        cands = table.df[spec.candidate].to_numpy() if spec.candidate else None
        out = decode(spec.kind, scores, ids, cands, k=k, threshold=self.head_.threshold)
        if defer_below is not None and spec.kind != "single":
            conf = confidence(scores, ids, cands, method=confidence_method)
            out["confidence"] = out["decision"].map(conf)
            out["deferred"] = out["confidence"] < defer_below
            if fallback is not None:
                deferred = set(out.loc[out["deferred"], "decision"])
                if deferred:
                    rows = table.df[table.decision_ids.isin(deferred).to_numpy()]
                    out.attrs["fallback"] = fallback(rows)
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, path, metrics: dict | None = None):
        """Write a portable, pickle-free artifact directory (see ``tabagent.artifact``)."""
        from tabagent import artifact

        return artifact.save(self, path, metrics)

    @classmethod
    def load(cls, path, verify: bool = True) -> TabAgent:
        from tabagent import artifact

        return artifact.load(path, verify)
