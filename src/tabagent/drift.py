"""Feature drift: the training distribution of a head's inputs, and how far live decisions have moved from it.

A head stays accurate while live decisions look like the training ones. The first sign that they no longer do
(the agent's prompts changed, the log format moved, the extractor now fails on a field, new tools appeared) is
a shift in the head's input features, long before labels are available to measure quality.

``feature_stats`` summarizes each feature of the head's input frame (the card's columns, computed on the real
training rows, before TabSynth): its null rate, and

* ``num``: quantiles, mean and standard deviation;
* ``cat``: the top categories with their frequencies, the set of training categories, and the Good-Turing
  estimate of the share of new values to expect (the share of values whose category occurs in a single
  training cluster);
* ``text``: length statistics (texts are compared by length, not content).

Each feature also gets decile bins with their training counts for the population stability index (PSI). A
feature that is constant within every training decision (a property of the decision's context, such as the
extractor's columns) is counted once per decision; a feature that varies across a decision's candidates is
counted once per row. Values are not independent: a decision's candidates share its context, and a task's
decisions (several runs of one task, several steps of one run) share theirs. ``feature_stats`` therefore
measures, per feature, the design effect of that clustering (by the training table's task groups): how much
larger the variance of the bin shares is than under independent sampling. ``TabAgent.fit`` computes the
statistics and the saved artifact keeps them in ``feature_stats.json``.

``DriftMonitor`` keeps the most recent live decisions in a rolling window and compares them with the training
statistics, per feature:

* ``psi``: the population stability index over the training bins (numeric values, text lengths, or the top
  categories plus one bucket for every other value), ``sum((a - e) * ln(a / e))`` with the training share
  ``e`` and the live share ``a`` of each bin (both smoothed with half a count per bin). A common reading:
  below 0.1 no meaningful shift, 0.1 to 0.25 a moderate one, above 0.25 a major one;
* ``null_rate_shift``: the live null rate minus the training null rate (an extractor that starts failing
  shows here first);
* ``unseen_share``: the share of live categorical values never seen in training (new tools, new labels),
  compared with the share the training data itself predicts.

Small or clustered samples move these statistics on their own. Between two samples of one distribution,
the PSI is close to ``c * X`` with ``X`` chi-square with ``B - 1`` degrees of freedom over ``B`` bins and
``c = deff * (1/n_train + 1/n_live)``. A feature drifts only when a statistic passes both a practical
threshold (``psi_threshold``, ``null_shift_threshold``, ``unseen_threshold``) and the ``1 - a`` quantile of
its no-drift distribution, where ``a`` is ``alpha`` split over all the tests of all the features (Sidak). The
chance that a window with no drift reports any drifting feature is thus at most about ``alpha``. Nothing
drifts before the window holds ``min_decisions`` decisions, nor for a head fitted on fewer than
``min_train_decisions`` decisions.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import math
import queue
import threading
import time
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

STATS_VERSION = 2
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
DECILES = tuple(i / 10 for i in range(1, 10))
TOP_K = 20  # categories with their frequencies (PSI buckets: these and one for every other value)
MAX_CATEGORIES = 1000  # training categories kept to recognize unseen values
EPS = 1e-4  # floor of a bin share in ``psi``
SMOOTHING = 0.5  # pseudo-count per bin in the monitor's PSI
MIN_BIN_COUNT = 5  # training values a PSI bin needs; smaller bins are merged

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- math
def psi(expected, actual, eps: float = EPS) -> float:
    """The population stability index of two distributions over the same bins (shares that sum to 1):
    ``sum((a - e) * ln(a / e))``, each share floored at ``eps`` so that an empty bin stays finite."""
    e = np.maximum(np.asarray(expected, dtype=float), eps)
    a = np.maximum(np.asarray(actual, dtype=float), eps)
    if e.shape != a.shape:
        raise ValueError(f"expected and actual need the same bins ({e.shape} != {a.shape})")
    return float(np.sum((a - e) * np.log(a / e)))


def smoothed(counts, pseudo: float = SMOOTHING) -> np.ndarray:
    """Bin shares from counts with ``pseudo`` added to every bin."""
    c = np.asarray(counts, dtype=float) + pseudo
    return c / c.sum()


def bin_counts(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """How many ``values`` fall in each bin ``(-inf, e0], (e0, e1], ..., (e_last, inf)``."""
    values = np.asarray(values, dtype=float)
    idx = np.searchsorted(np.asarray(edges, dtype=float), values, side="left")
    return np.bincount(idx, minlength=len(edges) + 1).astype(float)


def bin_shares(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """The share of ``values`` in each bin ``(-inf, e0], (e0, e1], ..., (e_last, inf)``."""
    counts = bin_counts(values, edges)
    return counts / counts.sum() if counts.sum() else counts


def chi2_quantile(p: float, df: int) -> float:
    """The ``p`` quantile of a chi-square distribution with ``df`` degrees of freedom: exact for 1 and 2,
    the Wilson-Hilferty approximation above (within a few percent in the upper tail)."""
    if df < 1:
        return 0.0
    if df == 1:
        return NormalDist().inv_cdf((1 + p) / 2) ** 2
    if df == 2:
        return -2.0 * math.log(1 - p)
    z = NormalDist().inv_cdf(p)
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def sampling_psi(expected, n_train: float, n_live: float, deff: float = 1.0) -> float:
    """The PSI that sampling alone gives, on average, between ``n_train`` and ``n_live`` values of one
    distribution: ``deff * (B - 1) * (1 / n_train + 1 / n_live)`` for the ``B`` bins that hold training
    values (the PSI is close to a chi-square statistic for small shifts); ``deff`` is the design effect of
    clustered values."""
    bins = int(np.sum(np.asarray(expected, dtype=float) > 0))
    if bins < 2 or n_train <= 0 or n_live <= 0:
        return 0.0
    return deff * (bins - 1) * (1 / n_train + 1 / n_live)


def design_effect(bins: np.ndarray, clusters: np.ndarray, n_bins: int) -> float:
    """How much clustering inflates the variance of bin shares: ``sum_b Var(p_b) / p_b`` with the variance of
    a ratio estimator over clusters, divided by its value for independent values, ``(B - 1) / N``. At least 1;
    ``N`` (one effective value) when all values share one cluster."""
    n = len(bins)
    if n < 2:
        return 1.0
    codes, uniques = pd.factorize(pd.Series(clusters))
    k = len(uniques)
    if k < 2:
        return float(n)
    counts = np.zeros((k, n_bins))
    np.add.at(counts, (codes, np.asarray(bins, dtype=int)), 1.0)
    m = counts.sum(axis=1)
    p = counts.sum(axis=0) / n
    var = k / (k - 1) * ((counts - np.outer(m, p)) ** 2).sum(axis=0) / n**2
    present = p > 0
    srs = (present.sum() - 1) / n
    if srs <= 0:
        return 1.0
    return float(max(1.0, (var[present] / p[present]).sum() / srs))


def _finite(x: float | None) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


# --------------------------------------------------------------------------- column encodings
def _null_mask(values: np.ndarray) -> np.ndarray:
    return pd.isna(pd.Series(values, dtype=object)).to_numpy()


def _as_numbers(values: np.ndarray) -> np.ndarray:
    """Floats, NaN where a value is missing or not numeric (as TabHead's featurizer reads ``num``)."""
    s = pd.Series(values)
    if s.dtype == object:
        s = s.map(lambda v: float(v) if isinstance(v, (bool, np.bool_)) else v)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _as_lengths(values: np.ndarray) -> np.ndarray:
    """Text lengths, -1 where the value is missing."""
    null = _null_mask(values)
    out = np.full(len(values), -1.0)
    if (~null).any():
        out[~null] = pd.Series(values[~null], dtype=object).astype(str).str.len().to_numpy(dtype=float)
    return out


def _as_categories(values: np.ndarray) -> list[str | None]:
    null = _null_mask(values)
    return [None if n else str(v) for v, n in zip(values, null, strict=True)]


_SMALL = 512  # a live decision's values are encoded in plain Python, which is faster than pandas there


def _is_null(v: Any) -> bool:
    return v is None or v is pd.NA or (isinstance(v, (float, np.floating)) and math.isnan(v))


def _number(v: Any) -> float:
    if _is_null(v):
        return math.nan
    try:
        return float(v)  # bools, numbers and numeric strings, as pandas.to_numeric reads them
    except (TypeError, ValueError):
        return math.nan


def encode(values, ftype: str):
    """The compact form the monitor keeps of a feature's values: floats (NaN = missing) for ``num``, lengths
    (-1 = missing) for ``text``, strings (None = missing) for ``cat``."""
    values = np.asarray(values, dtype=object) if not isinstance(values, np.ndarray) else values
    if len(values) <= _SMALL:
        if ftype == "num":
            return np.array([_number(v) for v in values], dtype=float)
        if ftype == "text":
            return np.array([-1.0 if _is_null(v) else float(len(str(v))) for v in values], dtype=float)
        return [None if _is_null(v) else str(v) for v in values]
    if ftype == "num":
        return _as_numbers(values)
    if ftype == "text":
        return _as_lengths(values)
    return _as_categories(values)


def _present(encoded, ftype: str):
    """The null mask and the present values of an encoded column."""
    if ftype == "cat":
        null = np.array([v is None for v in encoded], dtype=bool)
        return null, [v for v in encoded if v is not None]
    x = np.asarray(encoded, dtype=float)
    null = np.isnan(x) if ftype == "num" else x < 0
    return null, x[~null & np.isfinite(x)]


# --------------------------------------------------------------------------- training statistics
def _summary(x: np.ndarray) -> dict[str, Any]:
    if not len(x):
        return {"mean": None, "std": None, "quantiles": {}}
    qs = np.quantile(x, QUANTILES)
    return {
        "mean": _finite(float(np.mean(x))),
        "std": _finite(float(np.std(x))),
        "quantiles": {
            f"p{round(q * 100):02d}": _finite(float(v)) for q, v in zip(QUANTILES, qs, strict=True)
        },
    }


def _stats_of(encoded, ftype: str, clusters: np.ndarray) -> dict[str, Any]:
    null, present = _present(encoded, ftype)
    pclusters = clusters[~null]
    out: dict[str, Any] = {
        "n": len(null),
        "null_rate": float(null.mean()) if len(null) else 0.0,
        "n_clusters": int(len(np.unique(pclusters))) if len(pclusters) else 0,
        "deff_null": design_effect(null.astype(int), clusters, 2),
    }
    if ftype in ("num", "text"):
        edges = [float(v) for v in np.unique(np.quantile(present, DECILES))] if len(present) else []
        if ftype == "num":
            out.update(_summary(present))
        else:
            out["length"] = _summary(present)
            out["empty_rate"] = float((present == 0).mean()) if len(present) else 0.0
        counts = bin_counts(present, edges) if len(present) else np.zeros(0)
        out["edges"] = edges
        out["counts"] = [int(c) for c in counts]
        out["freq"] = [round(float(c / counts.sum()), 6) for c in counts] if len(present) else []
        idx = np.searchsorted(np.asarray(edges, dtype=float), present, side="left")
        out["deff"] = design_effect(idx, pclusters, len(edges) + 1) if len(present) else 1.0
        return out
    n = len(present)
    counter = collections.Counter(present)
    top = counter.most_common(TOP_K)
    index = {v: i for i, (v, _) in enumerate(top)}
    bins = np.array([index.get(v, len(top)) for v in present], dtype=int)
    clusters_per = pd.Series(pclusters).groupby(pd.Series(present, dtype=object)).nunique() if n else None
    single = set(clusters_per.index[clusters_per == 1]) if n else set()
    out.update(
        {
            "n_categories": len(counter),
            "top": [[v, round(c / n, 6)] for v, c in top],
            "top_counts": [int(c) for _, c in top],
            "other_count": int(n - sum(c for _, c in top)),
            "other_freq": round(1 - sum(c for _, c in top) / n, 6) if n else 0.0,
            "categories": [v for v, _ in counter.most_common(MAX_CATEGORIES)],
            "complete": len(counter) <= MAX_CATEGORIES,
            # Good-Turing: the share of new values to expect, from categories seen in one cluster only
            "unseen_expected": round(sum(counter[v] for v in single) / n, 6) if n else 0.0,
            "deff": design_effect(bins, pclusters, len(top) + 1) if n else 1.0,
        }
    )
    return out


def feature_stats(
    X: pd.DataFrame,
    types: dict[str, str],
    decision_ids=None,
    group_ids=None,
    max_decisions: int = 20_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Training statistics of the feature frame ``X`` (see the module docstring); ``types`` maps each column
    to ``num``, ``cat`` or ``text``, ``decision_ids`` gives each row's decision (None: one decision per row)
    and ``group_ids`` its cluster of related decisions, such as its task (None: each decision its own). At
    most ``max_decisions`` decisions, sampled with ``seed``, are summarized."""
    ids = np.arange(len(X)) if decision_ids is None else np.asarray(decision_ids)
    codes, uniques = pd.factorize(pd.Series(ids))
    groups = codes if group_ids is None else pd.factorize(pd.Series(np.asarray(group_ids)))[0]
    n_dec = len(uniques)
    if n_dec > max_decisions:
        keep = np.zeros(n_dec, dtype=bool)
        keep[np.random.default_rng(seed).choice(n_dec, max_decisions, replace=False)] = True
        rows = keep[codes]
        X, codes, groups = X.loc[rows], codes[rows], groups[rows]
    first = ~pd.Series(codes).duplicated().to_numpy()
    features = {}
    for col, ftype in types.items():
        if col not in X.columns:
            continue
        values = X[col].to_numpy()
        encoded = encode(values, ftype)
        key = pd.Series(encoded) if ftype == "num" else pd.Series(values, dtype=object).astype(str)
        varies = bool((key.groupby(codes).nunique(dropna=False) > 1).any())
        level = "candidate" if varies else "decision"
        clusters = groups
        if level == "decision":
            if ftype == "cat":
                encoded = [v for v, f in zip(encoded, first, strict=True) if f]
            else:
                encoded = encoded[first]
            clusters = groups[first]
        features[col] = {"type": ftype, "level": level, **_stats_of(encoded, ftype, clusters)}
    return {
        "version": STATS_VERSION,
        "n_decisions": int(min(n_dec, max_decisions)),
        "n_groups": int(len(np.unique(groups))),
        "n_rows": int(len(X)),
        "features": features,
    }


# --------------------------------------------------------------------------- live monitoring
def _expected_counts(stat: dict[str, Any]) -> np.ndarray:
    """A feature's training counts per bin: the decile bins, or the top categories plus one bin for every
    other value (older statistics without counts: shares times ``n``)."""
    if stat["type"] == "cat":
        if "top_counts" in stat:
            return np.array([*stat["top_counts"], stat.get("other_count", 0)], dtype=float)
        n = stat["n"] * (1 - stat["null_rate"])
        return np.array([f for _, f in stat.get("top", [])] + [stat.get("other_freq", 0.0)]) * n
    if "counts" in stat:
        return np.asarray(stat["counts"], dtype=float)
    return np.asarray(stat.get("freq", []), dtype=float) * stat["n"] * (1 - stat["null_rate"])


def bin_groups(stat: dict[str, Any], min_count: int = MIN_BIN_COUNT) -> np.ndarray:
    """The PSI bin of each training bin: bins with fewer than ``min_count`` training values are merged (a
    chi-square comparison needs a few expected values per bin). Categories join the bin of every other value;
    numeric and length bins join their neighbours."""
    counts = _expected_counts(stat)
    if stat["type"] == "cat":
        groups = np.arange(len(counts))
        rare = counts[:-1] < min_count
        groups[:-1][rare] = len(counts) - 1  # the 'other' bin
        _, groups = np.unique(groups, return_inverse=True)
        return groups
    groups, current, filled = np.zeros(len(counts), dtype=int), 0, 0.0
    for i, c in enumerate(counts):
        if filled >= min_count:
            current, filled = current + 1, 0.0
        groups[i], filled = current, filled + c
    if current and filled < min_count:  # a small last group joins the previous one
        groups[groups == current] = current - 1
    return groups


def cat_codes(stat: dict[str, Any], values) -> np.ndarray:
    """A categorical feature's live values as codes: -1 missing, ``i`` the i-th top category, ``T`` (the
    number of top categories) any other category seen in training, ``T + 1`` a category never seen in
    training."""
    top = {v: i for i, (v, _) in enumerate(stat.get("top", []))}
    seen = set(stat.get("categories", []))
    t = len(top)
    return np.array(
        [-1 if v is None else top.get(v, t if v in seen else t + 1) for v in values], dtype=np.int32
    )


def _upper_deff(deff: float, clusters: int, level: float = 0.9) -> float:
    """An upper confidence bound on a design effect estimated from ``clusters`` clusters (its variance
    estimate has about ``clusters - 1`` degrees of freedom)."""
    if clusters < 3:
        return deff * 10.0
    return deff * (clusters - 1) / chi2_quantile(1 - level, clusters - 1)


def count_tests(stat: dict[str, Any]) -> int:
    """How many drift tests a feature gets: the PSI (when it has at least two bins), the null rate, and for
    a categorical feature with a complete training category set, the unseen share."""
    groups = bin_groups(stat)
    return int(len(np.unique(groups)) >= 2) + 1 + int(stat["type"] == "cat" and bool(stat.get("complete")))


def compare(stat: dict[str, Any], encoded, alpha: float | None = None) -> dict[str, Any]:
    """One feature's live values (``encode``d; categorical ones may also be ``cat_codes``) against its
    training statistics: ``n``, ``psi`` (None without live values or bins), ``psi_sampling`` (its expected
    value under no drift, ``sampling_psi``), ``null_rate``, ``null_rate_shift``, ``unseen_share`` and
    ``unseen_expected`` (categorical features with a complete training category set; None otherwise). With
    ``alpha`` (the level of one test), also the limits a no-drift sample exceeds with probability at most
    ``alpha``: ``psi_limit``, ``null_rate_limit`` and ``unseen_limit`` (an excess over
    ``unseen_expected``)."""
    ftype = stat["type"]
    counts = _expected_counts(stat)
    groups = bin_groups(stat)
    n_groups = len(np.unique(groups)) if len(groups) else 0
    if ftype == "cat":
        codes = encoded if isinstance(encoded, np.ndarray) and encoded.dtype.kind == "i" else None
        codes = cat_codes(stat, encoded) if codes is None else codes
        null = codes < 0
        t = len(stat.get("top", []))
        live_bins = np.minimum(codes[~null], t)  # an unseen category falls in the 'other' bin
        n_present = int(len(live_bins))
    else:
        null, present = _present(encoded, ftype)
        live_bins = np.searchsorted(np.asarray(stat.get("edges", []), dtype=float), present, side="left")
        n_present = int(len(present))
    n_live = int(len(null))
    n_train = stat["n"]
    n_train_present = max(1.0, n_train * (1 - stat["null_rate"]))
    deff = float(stat.get("deff", 1.0))
    deff_hi = _upper_deff(deff, int(stat.get("n_clusters", n_train_present)))
    out: dict[str, Any] = {"n": n_live, "psi": None, "psi_sampling": None}
    out.update(unseen_share=None, unseen_expected=None)
    if n_present and n_groups >= 2:
        expected = np.bincount(groups, weights=counts, minlength=n_groups)
        live = np.bincount(groups[live_bins], minlength=n_groups)
        out["psi"] = psi(smoothed(expected), smoothed(live))
        c = 1 / n_train_present + 1 / n_present
        out["psi_sampling"] = deff * c * (n_groups - 1)
        if alpha is not None:
            out["psi_limit"] = deff_hi * c * chi2_quantile(1 - alpha, n_groups - 1)
    if ftype == "cat" and n_present and stat.get("complete"):
        t = len(stat.get("top", []))
        out["unseen_share"] = float((codes[~null] == t + 1).mean())
        u0 = float(stat.get("unseen_expected", 0.0))
        out["unseen_expected"] = u0
        if alpha is not None:
            spread = max(u0, 1 / n_train_present) * (1 - u0) * deff_hi * (1 / n_present + 1 / n_train_present)
            out["unseen_limit"] = NormalDist().inv_cdf(1 - alpha) * math.sqrt(spread)
    out["null_rate"] = float(null.mean()) if n_live else None
    out["null_rate_shift"] = None if out["null_rate"] is None else out["null_rate"] - stat["null_rate"]
    if alpha is not None and n_live:
        pooled = (stat["null_rate"] * n_train + out["null_rate"] * n_live) / (n_train + n_live)
        deff_null = _upper_deff(float(stat.get("deff_null", 1.0)), int(stat.get("n_clusters", n_train)))
        spread = pooled * (1 - pooled) * deff_null * (1 / n_train + 1 / n_live)
        out["null_rate_limit"] = NormalDist().inv_cdf(1 - alpha / 2) * math.sqrt(spread)
    return out


class DriftMonitor:
    """Compare a rolling window of live decisions with a head's training statistics (module docstring).

    Parameters
    ----------
    stats: ``feature_stats`` output (``agent.feature_stats_``, saved as ``feature_stats.json``).
    window: the number of most recent decisions compared.
    min_decisions: below this many decisions in the window, nothing is reported as drifting.
    psi_threshold, null_shift_threshold, unseen_threshold: the practical size of a shift worth reporting:
        a feature drifts when its PSI, the absolute change of its null rate, or its share of unseen
        categories beyond the expected share exceeds both this and the no-drift limit of the test.
    alpha: the chance, per evaluation of a window without drift, of reporting any drifting feature (split
        over every test of every feature).
    min_train_decisions: a head fitted on fewer decisions reports no drift (status ``insufficient_training``).
    check_every: every this many decisions the window is re-evaluated: a warning is logged for each feature
        that starts drifting (and an info message when it stops).
    background: ``observe`` only queues the decision, and a background thread encodes it and re-evaluates
        the window, off the request path; False does that work in ``observe``.
    """

    def __init__(
        self,
        stats: dict[str, Any],
        window: int = 500,
        min_decisions: int = 100,
        psi_threshold: float = 0.2,
        null_shift_threshold: float = 0.1,
        unseen_threshold: float = 0.2,
        alpha: float = 0.01,
        min_train_decisions: int = 30,
        check_every: int = 50,
        background: bool = True,
    ):
        if not stats or not stats.get("features"):
            raise ValueError("no training statistics: re-save the head with this version of tabagent")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.stats = stats
        self.window, self.min_decisions = max(1, window), max(1, min_decisions)
        self.psi_threshold, self.null_shift_threshold = psi_threshold, null_shift_threshold
        self.unseen_threshold, self.alpha = unseen_threshold, alpha
        self.min_train_decisions = min_train_decisions
        self.check_every = max(1, check_every)
        total = sum(count_tests(s) for s in self.features.values())
        self.test_alpha = 1 - (1 - alpha) ** (1 / max(1, total))  # Sidak over every test
        self._decisions: collections.deque[dict[str, Any]] = collections.deque(maxlen=self.window)
        self._lock = threading.Lock()
        self._ingest_lock = threading.Lock()
        self._seen = 0
        self._dropped = 0
        self._cached: tuple[int, dict[str, Any]] | None = None
        self._drifting: set[str] = set()
        self._background = background
        self._queue: queue.Queue = queue.Queue(maxsize=10_000)
        self._thread: threading.Thread | None = None
        self._stopped = False

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        return self.stats["features"]

    # ------------------------------------------------------------------ input
    def observe(self, frame: pd.DataFrame) -> None:
        """Add one live decision: its feature frame (the head's input, one row per candidate). In the
        background mode this only queues the frame; a full queue drops it (counted in ``dropped``)."""
        if not self._background:
            self._ingest(frame)
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return
        if self._thread is None or not self._thread.is_alive():
            with self._lock:
                if not self._stopped and (self._thread is None or not self._thread.is_alive()):
                    self._thread = threading.Thread(target=self._consume, daemon=True, name="tabagent-drift")
                    self._thread.start()

    def close(self) -> None:
        """Stop the background thread (pending decisions are still counted by the next ``report``)."""
        self._stopped = True
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)

    def _consume(self) -> None:
        while not self._stopped:
            frame = self._queue.get()
            try:
                if frame is None:
                    return
                self._ingest(frame)
            finally:
                self._queue.task_done()

    def _drain(self, wait_s: float = 2.0) -> None:
        """Wait (up to ``wait_s``) until the background thread has ingested every queued decision, in order;
        without a running thread, ingest them here."""
        end = time.monotonic() + wait_s
        while self._queue.unfinished_tasks:
            thread = self._thread
            if thread is None or not thread.is_alive() or self._stopped:
                try:
                    frame = self._queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    if frame is not None:
                        self._ingest(frame)
                finally:
                    self._queue.task_done()
            elif time.monotonic() > end:
                return
            else:
                time.sleep(0.001)

    def _ingest(self, frame: pd.DataFrame) -> None:
        record = {}
        for col, stat in self.features.items():
            if col not in frame.columns or not len(frame):
                continue
            values = frame[col].to_numpy()
            if stat["level"] == "decision":
                values = values[:1]
            encoded = encode(values, stat["type"])
            record[col] = cat_codes(stat, encoded) if stat["type"] == "cat" else encoded
        with self._ingest_lock:
            with self._lock:
                self._decisions.append(record)
                self._seen += 1
                check = self._seen % self.check_every == 0
            if check:
                self._check()

    # ------------------------------------------------------------------ output
    def report(self) -> dict[str, Any]:
        """The window's drift: ``status`` (``warming`` below ``min_decisions``, ``insufficient_training``
        for a head fitted on too few decisions, then ``ok`` or ``drift``), ``window`` (decisions in it),
        ``observed`` (decisions seen in total), ``dropped``, the thresholds, the drifting features, and per
        feature its ``compare`` values with ``drifting`` and the ``reasons``."""
        self._drain()
        return self._evaluate()

    def latest(self) -> dict[str, Any]:
        """The report of the latest evaluation (the background thread re-evaluates every ``check_every``
        decisions), without evaluating the window again; ``report`` when there is none yet. Cheap enough
        for every metrics scrape."""
        with self._lock:
            cached = self._cached
        return cached[1] if cached is not None else self.report()

    def _evaluate(self) -> dict[str, Any]:
        with self._lock:
            seen, decisions = self._seen, list(self._decisions)
            if self._cached is not None and self._cached[0] == seen:
                return self._cached[1]
        evaluate = len(decisions) >= self.min_decisions
        enough_training = self.stats.get("n_decisions", 0) >= self.min_train_decisions
        per_feature = {}
        for col, stat in self.features.items():
            values = [d[col] for d in decisions if col in d]
            empty = np.array([], dtype=np.int32 if stat["type"] == "cat" else float)
            encoded = np.concatenate(values) if values else empty
            cmp = compare(stat, encoded, self.test_alpha)
            reasons = []
            if evaluate and enough_training:
                if cmp["psi"] is not None and cmp["psi"] > max(self.psi_threshold, cmp.get("psi_limit", 0.0)):
                    reasons.append("psi")
                shift = cmp["null_rate_shift"]
                limit = max(self.null_shift_threshold, cmp.get("null_rate_limit", 0.0))
                if shift is not None and abs(shift) > limit:
                    reasons.append("null_rate")
                if cmp["unseen_share"] is not None:
                    excess = cmp["unseen_share"] - cmp["unseen_expected"]
                    if excess > max(self.unseen_threshold, cmp.get("unseen_limit", 0.0)):
                        reasons.append("unseen_categories")
            per_feature[col] = {
                "type": stat["type"],
                "level": stat["level"],
                **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in cmp.items()},
                "train_null_rate": stat["null_rate"],
                "drifting": bool(reasons),
                "reasons": reasons,
            }
        drifting = sorted(c for c, f in per_feature.items() if f["drifting"])
        if not enough_training:
            status = "insufficient_training"
        elif not evaluate:
            status = "warming"
        else:
            status = "drift" if drifting else "ok"
        report = {
            "status": status,
            "window": len(decisions),
            "observed": seen,
            "dropped": self._dropped,
            "min_decisions": self.min_decisions,
            "train_decisions": self.stats.get("n_decisions"),
            "thresholds": {
                "psi": self.psi_threshold,
                "null_rate_shift": self.null_shift_threshold,
                "unseen_share": self.unseen_threshold,
                "alpha": self.alpha,
                "test_alpha": round(self.test_alpha, 8),
            },
            "drifting": drifting,
            "features": per_feature,
        }
        with self._lock:
            self._cached = (seen, report)
        return report

    def _check(self) -> None:
        report = self._evaluate()
        now = set(report["drifting"])
        with self._lock:
            started, stopped = now - self._drifting, self._drifting - now
            self._drifting = now
        for col in sorted(started):
            f = report["features"][col]
            logger.warning(
                "feature drift: %s (%s) over the last %d decisions: "
                "psi=%s null_rate_shift=%s unseen_share=%s",
                col,
                ", ".join(f["reasons"]),
                report["window"],
                f["psi"],
                f["null_rate_shift"],
                f["unseen_share"],
                extra={
                    "fields": {
                        "event": "feature_drift",
                        "feature": col,
                        "reasons": f["reasons"],
                        "psi": f["psi"],
                        "null_rate_shift": f["null_rate_shift"],
                        "unseen_share": f["unseen_share"],
                        "window": report["window"],
                    }
                },
            )
        for col in sorted(stopped):
            logger.info(
                "feature drift ended: %s",
                col,
                extra={
                    "fields": {"event": "feature_drift_ended", "feature": col, "window": report["window"]}
                },
            )
