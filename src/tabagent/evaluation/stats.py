"""Confidence intervals and multiple-comparison control (paper App. E, L1804-1851 and L2290-2309).

Protocol: the resampling unit is the task (or decision group); an A-versus-B comparison is a paired
task-level bootstrap with 2,000 resamples of the difference; an improvement is claimed only when the 95%
interval excludes zero; significance within a metric across applications is Holm-Bonferroni controlled.

``method="percentile"`` reproduces the pre-release scripts exactly (a loop of ``rng.integers(0, n, n)``
draws from ``numpy.random.default_rng(seed)``); ``method="BCa"`` uses ``scipy.stats.bootstrap``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 20260720


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    method: str
    n: int
    confidence: float = 0.95

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.low:.4f}, {self.high:.4f}] ({self.method}, n={self.n})"


def _percentile_draws(values: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(values)
    # One call per resample, as in the original scripts, so the random stream is reproduced exactly.
    return np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_resamples)])


def bootstrap_ci(
    values,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
    method: str = "percentile",
) -> Interval:
    """CI of the mean of per-unit values (one value per task or decision)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        raise ValueError("need at least two units for a bootstrap interval")
    alpha = (1 - confidence) / 2
    if method == "percentile":
        draws = _percentile_draws(v, n_resamples, seed)
        q = round(100 * alpha, 10)  # exactly 2.5 for a 95% interval, as in the original scripts
        lo, hi = np.percentile(draws, [q, 100 - q])
    elif method == "BCa":
        from scipy.stats import bootstrap

        res = bootstrap(
            (v,),
            np.mean,
            n_resamples=n_resamples,
            confidence_level=confidence,
            method="BCa",
            random_state=np.random.default_rng(seed),
        )
        lo, hi = res.confidence_interval
    else:
        raise ValueError("method must be 'percentile' or 'BCa'")
    return Interval(float(v.mean()), float(lo), float(hi), method, len(v), confidence)


def paired_bootstrap(a, b, **kw) -> Interval:
    """CI of mean(a - b) over paired units (the paper's paired task-level bootstrap)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired vectors must have the same length")
    keep = ~(np.isnan(a) | np.isnan(b))
    return bootstrap_ci(a[keep] - b[keep], **kw)


def bootstrap_pvalue(diffs, *, n_resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED) -> float:
    """Two-sided bootstrap p-value for H0: mean difference = 0 (resampling the centred differences)."""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    observed = d.mean()
    null = _percentile_draws(d - observed, n_resamples, seed)
    return float(min(1.0, (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_resamples + 1)))


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (monotone step-down), in the input order."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted
