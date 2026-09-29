"""TabHead: the single-pass tabular classifier that scores every candidate of a decision (paper Sec. 3.3).

TabHead scores each (context, candidate) row over the TabSchema feature card. The head is CatBoost in
every main-paper result (L321); other estimators (for the head-invariance ablation, L647/L2251) plug in
through ``estimator=``. Three decision patterns share one pointwise reduction (L323), plus the policy
family (L1181):

  ranking     score every (x, y) pair, sort by p(y|x), return the top-K (ties broken by candidate id)
  multilabel  binary relevance, p(y|x) >= threshold, backing off to the top-1 candidate if none passes
  single      a softmax over the label set; the prediction is the argmax class
  policy      pick the arm with the highest predicted success, optionally penalized by a cost prior

``confidence`` implements the gate of App. D (L2347): the margin between the R-th and (R+1)-th ranked
scores, or the cutoff-free largest gap. Low-confidence decisions may be deferred to the original LLM.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

FeatureType = Literal["text", "num", "cat"]

#: CatBoost settings. ``paper`` is the configuration stated in the paper (L384: CatBoost defaults,
#: 1000 iterations, depth 6, L2 leaf regularization 3.0, automatically selected learning rate).
#: ``tmlr-runs`` is the configuration of the scripts that produced the reported numbers.
PRESETS: dict[str, dict[str, Any]] = {
    "paper": {"iterations": 1000, "depth": 6, "l2_leaf_reg": 3.0},
    "tmlr-runs": {"iterations": 300, "depth": 6, "learning_rate": 0.08},
}
CATBOOST_RUNTIME = {"thread_count": 4, "verbose": 0, "allow_writing_files": False}

TFIDF_MAX_FEATURES = 400


class Featurizer:
    """Turn declared feature columns into a numeric matrix, fitted on training rows only.

    text  TF-IDF (unigrams and bigrams, ``max_features`` terms, ``min_df=2``); skipped if the training
          vocabulary is empty
    num   numeric, missing values imputed with the training median
    cat   integer codes in order of first appearance in training; unseen values map to -1

    Types are declared (from the feature card), never inferred per fold, so a column cannot change its
    representation between folds.
    """

    def __init__(self, types: Mapping[str, FeatureType], text_max_features: int = TFIDF_MAX_FEATURES):
        bad = {c: t for c, t in types.items() if t not in ("text", "num", "cat")}
        if bad:
            raise ValueError(f"unknown feature types {bad}")
        self.types = dict(types)
        self.text_max_features = text_max_features
        self.parts: list[tuple[str, str, Any]] = []

    def fit(self, X: pd.DataFrame) -> Featurizer:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.parts = []
        for col, kind in self.types.items():
            s = X[col]
            if kind == "text":
                vec = TfidfVectorizer(max_features=self.text_max_features, ngram_range=(1, 2), min_df=2)
                try:
                    vec.fit(s.fillna("").astype(str))
                except ValueError:  # empty vocabulary on this training fold
                    continue
                self.parts.append((col, "text", vec))
            elif kind == "num":
                self.parts.append((col, "num", float(pd.to_numeric(s, errors="coerce").median())))
            else:
                self.parts.append((col, "cat", {v: i for i, v in enumerate(s.astype(str).unique())}))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        blocks = []
        for col, kind, state in self.parts:
            s = X[col]
            if kind == "text":
                blocks.append(state.transform(s.fillna("").astype(str)).toarray())
            elif kind == "num":
                blocks.append(
                    pd.to_numeric(s, errors="coerce").fillna(state).to_numpy(dtype=float).reshape(-1, 1)
                )
            else:
                codes = s.astype(str).map(lambda v, m=state: m.get(v, -1))
                blocks.append(codes.to_numpy(dtype=float).reshape(-1, 1))
        return np.hstack(blocks) if blocks else np.zeros((len(X), 1))


def catboost_estimator(preset: str, kind: str, seed: int, **overrides) -> Any:
    from catboost import CatBoostClassifier

    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {list(PRESETS)}")
    params = {**PRESETS[preset], **CATBOOST_RUNTIME, "random_seed": seed, **overrides}
    params.setdefault("loss_function", "MultiClass" if kind == "single" else "Logloss")
    return CatBoostClassifier(**params)


class TabHead:
    """The TabAgent head. ``fit``/``score`` operate on feature frames; decoding is done per decision.

    Parameters
    ----------
    kind: the decision pattern (``ranking``, ``multilabel``, ``single``, ``policy``); sets loss and decoder.
    preset: CatBoost configuration (see ``PRESETS``); ignored when ``estimator`` is given.
    estimator: ``seed -> classifier`` factory for alternative heads (fit/predict_proba/classes_).
    featurize: set False for heads that consume raw columns themselves (for example TabSTAR).
    """

    def __init__(
        self,
        kind: str = "ranking",
        preset: str = "paper",
        estimator: Callable[[int], Any] | None = None,
        featurize: bool = True,
        text_max_features: int = TFIDF_MAX_FEATURES,
        threshold: float = 0.5,
        seed: int = 0,
        **catboost_overrides,
    ):
        self.kind = kind
        self.preset = preset
        self.estimator = estimator
        self.featurize = featurize
        self.text_max_features = text_max_features
        self.threshold = threshold
        self.seed = seed
        self.catboost_overrides = catboost_overrides
        self.featurizer_: Featurizer | None = None
        self.model_: Any = None
        self.columns_: list[str] = []

    def clone(self, seed: int | None = None) -> TabHead:
        new = copy.copy(self)
        new.featurizer_, new.model_, new.columns_ = None, None, []
        if seed is not None:
            new.seed = seed
        return new

    def _new_model(self) -> Any:
        if self.estimator is not None:
            return self.estimator(self.seed)
        return catboost_estimator(self.preset, self.kind, self.seed, **self.catboost_overrides)

    def fit(self, X: pd.DataFrame, y, types: Mapping[str, FeatureType]) -> TabHead:
        self.columns_ = list(types)
        y = np.asarray(y)
        if self.kind != "single":
            y = y.astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("TabHead needs at least two label values in the training data")
        if self.featurize:
            self.featurizer_ = Featurizer(types, self.text_max_features).fit(X[self.columns_])
            Z = self.featurizer_.transform(X[self.columns_])
        else:
            Z = X[self.columns_].reset_index(drop=True)
        self.model_ = self._new_model()
        self.model_.fit(Z, y)
        return self

    def _matrix(self, X: pd.DataFrame):
        if self.model_ is None:
            raise RuntimeError("TabHead is not fitted")
        if self.featurize:
            return self.featurizer_.transform(X[self.columns_])
        return X[self.columns_].reset_index(drop=True)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._matrix(X)))

    @property
    def classes_(self) -> list:
        return list(np.asarray(self.model_.classes_).ravel())

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """p(positive) per row, or the predicted class for single-class decisions."""
        proba = self.predict_proba(X)
        classes = self.classes_
        if self.kind == "single":
            return np.asarray(classes, dtype=object)[proba.argmax(axis=1)]
        return proba[:, classes.index(1)]


# --------------------------------------------------------------------------- decoding
def _order(decision_ids, candidates, scores) -> pd.DataFrame:
    f = pd.DataFrame(
        {
            "decision": np.asarray(decision_ids),
            "candidate": np.asarray(candidates).astype(str),
            "score": scores,
        }
    )
    idx = np.lexsort((f["candidate"].to_numpy(), -f["score"].to_numpy(float), f["decision"].to_numpy()))
    f = f.iloc[idx].reset_index(drop=True)
    f["rank"] = f.groupby("decision", sort=False).cumcount()
    return f


def decode(
    kind: str,
    scores: np.ndarray,
    decision_ids,
    candidates=None,
    *,
    k: int | None = None,
    threshold: float = 0.5,
    cost=None,
    cost_lambda: float = 0.0,
) -> pd.DataFrame:
    """Turn row scores into decisions. Returns one row per selected (decision, candidate) with its score
    and rank; for ``single`` the ``choice`` column holds the predicted class of each decision row."""
    if kind == "single":
        return pd.DataFrame(
            {"decision": np.asarray(decision_ids), "choice": np.asarray(scores, dtype=object)}
        )
    s = np.asarray(scores, dtype=float)
    if kind == "policy" and cost is not None and cost_lambda:
        s = s - cost_lambda * np.asarray(cost, dtype=float)
    f = _order(decision_ids, candidates, s)
    if kind == "ranking":
        return f if k is None else f[f["rank"] < k].reset_index(drop=True)
    if kind == "policy":
        return f[f["rank"] == 0].reset_index(drop=True)
    if kind == "multilabel":
        keep = f["score"] >= threshold
        empty = keep.groupby(f["decision"]).transform("sum") == 0
        return f[keep | (empty & (f["rank"] == 0))].reset_index(drop=True)
    raise ValueError(f"unknown kind {kind!r}")


# --------------------------------------------------------------------------- confidence gate
def confidence(scores, decision_ids, candidates, method: str = "largest_gap", r=None) -> pd.Series:
    """Per-decision confidence of a ranking (App. D, L2347).

    margin_at_r  score(R-th) - score((R+1)-th), where R = |G(t)| (needs ``r``: a mapping or Series from
                 decision id to R; an oracle quantity, used for analysis)
    largest_gap  the largest drop between consecutive ranked scores (cutoff-free, deployable)
    """
    f = _order(decision_ids, candidates, np.asarray(scores, dtype=float))
    out = {}
    for d, g in f.groupby("decision", sort=True):
        s = g["score"].to_numpy()
        if len(s) < 2:
            out[d] = float("inf")
            continue
        if method == "largest_gap":
            out[d] = float(np.max(s[:-1] - s[1:]))
        elif method == "margin_at_r":
            if r is None:
                raise ValueError("margin_at_r needs the relevant-set size per decision")
            big_r = int(r[d])
            out[d] = float(s[big_r - 1] - s[big_r]) if 0 < big_r < len(s) else float("inf")
        else:
            raise ValueError("method must be 'largest_gap' or 'margin_at_r'")
    return pd.Series(out, name="confidence")


def risk_coverage(confidence_values, correct) -> pd.DataFrame:
    """Selective quality when deferring the least-confident decisions (Fig. gate): for each coverage level
    (fraction kept), the mean correctness of the kept decisions."""
    c = np.asarray(confidence_values, dtype=float)
    y = np.asarray(correct, dtype=float)
    order = np.argsort(-c, kind="stable")
    kept = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    return pd.DataFrame({"coverage": np.arange(1, len(y) + 1) / len(y), "selective_quality": kept})
