import numpy as np
import pandas as pd
import pytest

from tabagent.evaluation import cv, stats
from tabagent.table import DecisionSpec, DecisionTable


# --------------------------------------------------------------------------- stats
def _legacy_paired_ci(a, b, seed):
    """Verbatim logic of the pre-release paired bootstrap (percentile, 2,000 resamples)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    deltas = [(a[idx] - b[idx]).mean() for idx in (rng.integers(0, n, n) for _ in range(2000))]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return lo, hi


def test_percentile_bootstrap_reproduces_legacy_exactly():
    rng = np.random.default_rng(3)
    a, b = rng.random(87), rng.random(87)
    iv = stats.paired_bootstrap(a, b, seed=20260720)
    lo, hi = _legacy_paired_ci(a, b, 20260720)
    assert (iv.low, iv.high) == (lo, hi)
    assert iv.estimate == pytest.approx((a - b).mean())


def test_bca_interval_brackets_the_mean():
    v = np.random.default_rng(0).normal(0.7, 0.1, 200)
    iv = stats.bootstrap_ci(v, method="BCa")
    assert iv.low < v.mean() < iv.high and iv.method == "BCa"


def test_holm_step_down():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_pvalue_small_for_clear_effect():
    assert stats.bootstrap_pvalue(np.full(50, 0.1) + np.random.default_rng(0).normal(0, 0.01, 50)) < 0.01


# --------------------------------------------------------------------------- folds
def _legacy_groupkfold(groups, n_splits=5, seed=0):
    uniq = list(dict.fromkeys(groups))
    np.random.default_rng(seed).shuffle(uniq)
    return {g: i % n_splits for i, g in enumerate(uniq)}


def test_assign_folds_matches_legacy_in_appearance_order():
    groups = [f"task{i}" for i in np.random.default_rng(0).permutation(137)]
    assert cv.assign_folds(groups, 5, 20260720, order="appearance") == _legacy_groupkfold(groups, 5, 20260720)


def test_sorted_folds_do_not_depend_on_row_order():
    g = [f"t{i}" for i in range(50)]
    assert cv.assign_folds(g) == cv.assign_folds(list(reversed(g)))


def test_stratified_folds_balance_each_stratum():
    groups = [f"t{i}" for i in range(100)]
    strata = {g: ("amazon" if i < 60 else "spotify") for i, g in enumerate(groups)}
    folds = cv.assign_folds(groups, 5, 7, strata=strata)
    counts = pd.crosstab(pd.Series(strata), pd.Series(folds))
    assert counts.loc["amazon"].tolist() == [12] * 5 and counts.loc["spotify"].tolist() == [8] * 5


def test_fold_file_roundtrip(tmp_path):
    folds = {"a": 0, "b": 3}
    cv.save_folds(tmp_path / "f.json", folds, "task")
    assert cv.load_folds(tmp_path / "f.json") == folds


def test_stable_uid():
    assert cv.stable_uid("task", "tau2", 7) == cv.stable_uid("task", "tau2", "7")
    assert cv.stable_uid("task", "a").startswith("task_") and len(cv.stable_uid("x", "a")) == 26


# --------------------------------------------------------------------------- cross_validate
SPEC = DecisionSpec(kind="ranking", group="task", candidate="tool", label="used")


def _table(n_tasks=25):
    rng = np.random.default_rng(0)
    rows = [
        {"task": f"t{t}", "tool": f"x{i}", "used": int(rng.random() < 0.3), "signal": rng.random()}
        for t in range(n_tasks)
        for i in range(6)
    ]
    return DecisionTable(pd.DataFrame(rows), SPEC)


class Spy:
    """Scores rows by the 'signal' column; records every table it is fit on or asked to score."""

    fitted: list = []
    scored: list = []

    def __init__(self, seed=0):
        self.seed = seed

    def clone(self, seed):
        return Spy(seed)

    def fit(self, table):
        Spy.fitted.append(set(table.groups))
        return self

    def score(self, table):
        Spy.scored.append((set(table.groups), table.has_labels, list(table.df.columns)))
        return table.df["signal"].to_numpy() + 0.001 * self.seed


def test_cross_validate_never_trains_on_held_out_groups():
    Spy.fitted, Spy.scored = [], []
    t = _table()
    res = cv.cross_validate(Spy(), t, seeds=(0, 1))
    assert len(Spy.fitted) == 10  # 5 folds x 2 seeds
    for train, (test, has_labels, cols) in zip(Spy.fitted, Spy.scored, strict=True):
        assert not (train & test)
        assert not has_labels and "used" not in cols
    assert res.oof.shape == (len(t), 2) and not res.oof.isna().any().any()


def test_cv_result_metrics_and_compare():
    t = _table()
    res = cv.cross_validate(Spy(), t, seeds=(0, 1, 2))
    per_dec = res.per_decision("p@r")
    assert per_dec.between(0, 1).all()
    s = res.summary(["p@r", "recall@3"])
    assert set(s.index) == {"p@r", "recall@3"} and (s.ci_low <= s.value).all()
    same = res.compare(res, "p@r")
    assert same.estimate == 0 and same.low == 0 and same.high == 0


def test_pinned_folds_are_respected():
    t = _table(10)
    folds = {g: (0 if i < 5 else 1) for i, g in enumerate(t.groups)}
    Spy.fitted = []
    cv.cross_validate(Spy(), t, folds=folds, seeds=(0,))
    assert len(Spy.fitted) == 2 and all(len(f) == 5 for f in Spy.fitted)
