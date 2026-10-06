"""TabHead: the single-pass tabular classifier that scores every candidate of a decision (paper Sec. 3.3).

TabHead scores each (context, candidate) row over the TabSchema feature card. The head is CatBoost in
every main-paper result; other estimators (the head-invariance ablation, Fig. 3 and App. K.3) plug in
through ``estimator=``. Three decision patterns share one pointwise reduction (Sec. 3.3), plus the policy
family (Sec. 5; App. B.7):

  ranking     score every (x, y) pair, sort by p(y|x), return the top-K
  multilabel  binary relevance, p(y|x) >= threshold, backing off to the top-1 candidate if none passes
  single      a softmax over the label set; the prediction is the argmax class
  policy      pick the arm with the highest predicted success, optionally penalized by a cost prior

``decode`` turns scores into decisions. A live decision needs one answer, so tied scores are ordered by
candidate id; the paper's evaluation instead averages over tie orders (``tabagent.evaluation.metrics``).

``confidence`` implements the confidence-gated fallback (App. N.3, Fig. 14): the margin between the R-th
and (R+1)-th ranked scores, or the cutoff-free largest gap. Low-confidence decisions may be deferred to the
original LLM.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from tabagent.table import KINDS

FeatureType = Literal["text", "num", "cat"]

#: CatBoost settings. ``paper`` is the configuration the paper states and every reported number uses:
#: 300 boosting iterations, learning rate 0.08, depth 6, CatBoost's defaults otherwise (including L2 leaf
#: regularization 3.0). ``catboost-default`` is CatBoost's own schedule (1000 iterations, automatically
#: selected learning rate), a second option for the automatic recipe search.
PRESETS: dict[str, dict[str, Any]] = {
    "paper": {"iterations": 300, "depth": 6, "learning_rate": 0.08},
    "catboost-default": {"iterations": 1000, "depth": 6, "l2_leaf_reg": 3.0},
}
CATBOOST_RUNTIME = {"thread_count": 4, "verbose": 0, "allow_writing_files": False}

TFIDF_MAX_FEATURES = 400
#: CatBoost models are fitted one at a time per process. In-fold discovery runs folds in threads, and each
#: fold's usefulness estimate fits CatBoost; concurrent fits from several Python threads crashed natively in
#: CI. Each fit already uses ``CATBOOST_RUNTIME["thread_count"]`` threads, so little is lost.
_CATBOOST_FIT = threading.Lock()
#: The LSA text representation: TF-IDF over ``LSA_TERMS`` terms (unigrams and bigrams) of a column's distinct
#: texts, reduced by truncated SVD to ``LSA_DIMS`` dense dimensions. ``TabHead(text_repr="auto")`` uses it
#: for cards with more than ``LSA_AUTO_TEXT_COLUMNS`` text columns (an extractor can return a text in several
#: forms), where per-column TF-IDF blocks would make the head's matrix tens of thousands of columns wide;
#: cards with fewer text columns, the paper's among them, keep TF-IDF. LSA keeps 64 dimensions per column
#: instead of 400; in one internal API-shortlisting run with many text columns (no interval) it improved
#: ranking in less than half the time.
LSA_AUTO_TEXT_COLUMNS, LSA_DIMS, LSA_TERMS = 6, 64, 4000
TEXT_REPRS = ("auto", "tfidf", "lsa")


def resolve_text_repr(text_repr: str, n_text: int) -> str:
    """``tfidf`` or ``lsa``: ``auto`` picks LSA above ``LSA_AUTO_TEXT_COLUMNS`` text columns."""
    if text_repr not in TEXT_REPRS:
        raise ValueError(f"text_repr must be one of {TEXT_REPRS}, not {text_repr!r}")
    if text_repr != "auto":
        return text_repr
    return "lsa" if n_text > LSA_AUTO_TEXT_COLUMNS else "tfidf"


class Featurizer:
    """Turn declared feature columns into a numeric matrix, fitted on training rows only.

    text  TF-IDF (unigrams and bigrams, ``max_features`` terms, ``min_df=2``), or with ``text_repr="lsa"``
          TF-IDF over the column's distinct texts reduced by truncated SVD to ``LSA_DIMS`` dimensions;
          skipped if the training vocabulary is empty
    num   numeric, missing values imputed with the training median
    cat   integer codes in order of first appearance in training; unseen values map to -1

    Types are declared (from the feature card), never inferred per fold, so a column cannot change its
    representation between folds.
    """

    def __init__(
        self,
        types: Mapping[str, FeatureType],
        text_max_features: int = TFIDF_MAX_FEATURES,
        text_repr: str = "tfidf",
    ):
        bad = {c: t for c, t in types.items() if t not in ("text", "num", "cat")}
        if bad:
            raise ValueError(f"unknown feature types {bad}")
        if text_repr not in ("tfidf", "lsa"):
            raise ValueError(f"text_repr must be 'tfidf' or 'lsa', not {text_repr!r}")
        self.types = dict(types)
        self.text_max_features = text_max_features
        self.text_repr = text_repr
        self.parts: list[tuple[str, str, Any]] = []

    def fit(self, X: pd.DataFrame) -> Featurizer:
        """Learn vocabularies, medians and category codes from the training rows ``X``."""
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.parts = []
        for col, kind in self.types.items():
            s = X[col]
            if kind == "text" and self.text_repr == "lsa":
                texts = pd.unique(s.fillna("").astype(str))  # a decision's text repeats on its candidate rows
                # min_df=1: among distinct texts, a term in one of them still covers a whole decision's rows,
                # and a short code-like column has each value exactly once
                vec = TfidfVectorizer(max_features=LSA_TERMS, ngram_range=(1, 2), min_df=1)
                try:
                    m = vec.fit_transform(texts)
                except ValueError:  # empty vocabulary on this training fold
                    continue
                k = min(LSA_DIMS, m.shape[1] - 1, m.shape[0] - 1)
                if k < 1:
                    self.parts.append((col, "text", vec))  # too few texts or terms to reduce: TF-IDF
                    continue
                self.parts.append((col, "lsa", (vec, TruncatedSVD(k, random_state=0).fit(m))))
            elif kind == "text":
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
        if not self.parts:
            raise ValueError(
                f"no usable feature among {list(self.types)}: every declared column was dropped (text "
                "columns need terms that occur in at least two training rows)"
            )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """The numeric matrix of ``X`` under the training statistics, one block of columns per part."""
        blocks: list[np.ndarray | None] = [None] * len(self.parts)
        # A live decision has few rows and many columns, so per-column pandas calls would dominate its
        # latency: numeric and category columns are converted in blocks here. Every value comes out as the
        # per-column code below would make it; columns this cannot handle exactly (numbers written as text)
        # go through that code.
        dtypes = X.dtypes
        num = [i for i, (col, kind, _) in enumerate(self.parts) if kind == "num"]
        typed = [i for i in num if dtypes[self.parts[i][0]].kind in "biuf"]
        if typed:
            block = X[[self.parts[i][0] for i in typed]].to_numpy(dtype=float, na_value=np.nan)
            medians = np.array([self.parts[i][2] for i in typed], dtype=float)
            block = np.where(np.isnan(block), medians, block)
            for j, i in enumerate(typed):
                blocks[i] = block[:, j : j + 1]
        untyped = [i for i in num if blocks[i] is None]
        if untyped:
            values = X[[self.parts[i][0] for i in untyped]].to_numpy(dtype=object)
            for j, i in enumerate(untyped):
                column = _plain_numbers(values[:, j])
                if column is not None:
                    blocks[i] = np.where(np.isnan(column), self.parts[i][2], column).reshape(-1, 1)
        cat = [i for i, (_, kind, _) in enumerate(self.parts) if kind == "cat"]
        if cat:
            labels = X[[self.parts[i][0] for i in cat]].astype(str).to_numpy()
            for j, i in enumerate(cat):
                codes = self.parts[i][2]
                blocks[i] = np.array([codes.get(v, -1) for v in labels[:, j]], dtype=float).reshape(-1, 1)
        for i, (col, kind, state) in enumerate(self.parts):
            if blocks[i] is not None:
                continue
            s = X[col]
            if kind == "text":
                # Vectorize each distinct text once: a decision's context text repeats on every candidate row.
                text = s.fillna("").astype(str)
                codes, uniques = pd.factorize(text)
                blocks[i] = state.transform(uniques).toarray()[codes]
            elif kind == "lsa":
                vec, svd = state
                codes, uniques = pd.factorize(s.fillna("").astype(str))
                blocks[i] = svd.transform(vec.transform(uniques))[codes]
            else:  # a numeric column holding text: pandas' own coercion
                blocks[i] = (
                    pd.to_numeric(s, errors="coerce").fillna(state).to_numpy(dtype=float).reshape(-1, 1)
                )
        return np.hstack(blocks)


_NUMBERS = (bool, int, float, np.bool_, np.integer, np.floating)


def _plain_numbers(column: np.ndarray) -> np.ndarray | None:
    """``column`` (objects) as floats with NaN for missing values, when every value is a number, a bool or
    missing, which ``pd.to_numeric`` would convert to exactly these floats; None otherwise."""
    out = np.empty(len(column), dtype=float)
    for k, v in enumerate(column):
        if v is None or v is pd.NA:
            out[k] = np.nan
        elif isinstance(v, _NUMBERS):
            try:
                out[k] = float(v)
            except OverflowError:  # an integer beyond float range: pandas decides
                return None
        else:
            return None
    return out


def catboost_estimator(preset: str, kind: str, seed: int, **overrides) -> Any:
    """A ``CatBoostClassifier`` with the ``preset`` settings, 4 threads and ``random_seed=seed``.

    ``overrides`` replace any CatBoost parameter. The loss is ``MultiClass`` for single-class decisions and
    ``Logloss`` otherwise.
    """
    from catboost import CatBoostClassifier

    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {list(PRESETS)}")
    params = {**PRESETS[preset], **CATBOOST_RUNTIME, "random_seed": seed, **overrides}
    params.setdefault("loss_function", "MultiClass" if kind == "single" else "Logloss")
    return CatBoostClassifier(**params)


class TabHead:
    """The TabAgent head (paper Sec. 3.3): a classifier over the feature card's columns.

    ``fit`` and ``score`` operate on feature frames (the card's output); ``decode`` turns scores into
    decisions.

    Parameters
    ----------
    kind: the decision pattern (``ranking``, ``multilabel``, ``single``, ``policy``); sets loss and decoder.
    preset: CatBoost configuration (see ``PRESETS``; ``paper`` is the paper's, Sec. 4.2). Not used with
        ``estimator``.
    estimator: ``seed -> classifier`` factory for alternative heads (``fit``/``predict_proba``/``classes_``).
    featurize: set False for heads that consume raw columns themselves (for example TabSTAR).
    text_max_features: TF-IDF vocabulary size per text column.
    text_repr: ``tfidf``, ``lsa`` (TF-IDF reduced by SVD, see ``LSA_DIMS``) or ``auto`` (LSA above
        ``LSA_AUTO_TEXT_COLUMNS`` text columns); the fitted choice is ``text_repr_``.
    threshold: the multi-label decision threshold (App. J).
    seed: the model seed.
    catboost_overrides: CatBoost parameters that replace the preset's (not allowed with ``estimator``).
    """

    def __init__(
        self,
        kind: str = "ranking",
        preset: str = "paper",
        estimator: Callable[[int], Any] | None = None,
        featurize: bool = True,
        text_max_features: int = TFIDF_MAX_FEATURES,
        text_repr: str = "auto",
        threshold: float = 0.5,
        seed: int = 0,
        **catboost_overrides,
    ):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if estimator is not None and catboost_overrides:
            raise ValueError(
                f"CatBoost parameters {sorted(catboost_overrides)} have no effect with estimator=; "
                "configure the estimator factory instead"
            )
        self.kind = kind
        self.preset = preset
        self.estimator = estimator
        self.featurize = featurize
        self.text_max_features = text_max_features
        resolve_text_repr(text_repr, 0)  # validate
        self.text_repr = text_repr
        self.text_repr_: str | None = None
        self.threshold = threshold
        self.seed = seed
        self.catboost_overrides = catboost_overrides
        self.featurizer_: Featurizer | None = None
        self.model_: Any = None
        self.columns_: list[str] = []

    def clone(self, seed: int | None = None) -> TabHead:
        """An unfitted copy with the same settings (and ``seed``, when given)."""
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
        """Fit on the feature frame ``X`` with labels ``y``; ``types`` declares each column's type."""
        self.columns_ = list(types)
        y = np.asarray(y)
        if self.kind != "single":
            y = y.astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("TabHead needs at least two label values in the training data")
        if self.featurize:
            n_text = sum(t == "text" for t in types.values())
            self.text_repr_ = resolve_text_repr(self.text_repr, n_text)
            self.featurizer_ = Featurizer(types, self.text_max_features, self.text_repr_).fit(
                X[self.columns_]
            )
            Z = self.featurizer_.transform(X[self.columns_])
        else:
            Z = X[self.columns_].reset_index(drop=True)
        self.model_ = self._new_model()
        if self.estimator is None:  # CatBoost
            with _CATBOOST_FIT:
                self.model_.fit(Z, y)
        else:
            self.model_.fit(Z, y)
        return self

    def _matrix(self, X: pd.DataFrame):
        if self.model_ is None:
            raise RuntimeError("TabHead is not fitted")
        if self.featurize:
            return self.featurizer_.transform(X[self.columns_])
        return X[self.columns_].reset_index(drop=True)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class probabilities per row, columns in ``classes_`` order."""
        return np.asarray(self.model_.predict_proba(self._matrix(X)))

    @property
    def classes_(self) -> list:
        """The class labels of the fitted model (0 and 1, or the single-class labels)."""
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
    """Turn row scores into decisions.

    Returns one row per selected (decision, candidate) with its ``score`` and 0-based ``rank``; tied scores
    are ordered by candidate id, so a decision always has one answer. ``k`` truncates rankings; ``threshold``
    is the multi-label cutoff; ``cost`` and ``cost_lambda`` penalize policy arms. For ``single`` the
    ``choice`` column holds the predicted class of each decision row.
    """
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
    """Per-decision confidence of a ranking (App. N.3, "Confidence-gated fallback").

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
    """Selective quality when deferring the least-confident decisions (App. N.3, Fig. 14): for each coverage
    level (fraction kept), the mean correctness of the kept decisions."""
    c = np.asarray(confidence_values, dtype=float)
    y = np.asarray(correct, dtype=float)
    order = np.argsort(-c, kind="stable")
    kept = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    return pd.DataFrame({"coverage": np.arange(1, len(y) + 1) / len(y), "selective_quality": kept})
