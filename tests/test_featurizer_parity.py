"""The block conversions of ``Featurizer.transform`` give exactly the values of the per-column code."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabagent.tabhead import Featurizer


def reference(f: Featurizer, X: pd.DataFrame) -> np.ndarray:
    """``Featurizer.transform`` as it was: one pandas conversion per column."""
    blocks = []
    for col, kind, state in f.parts:
        s = X[col]
        if kind == "text":
            codes, uniques = pd.factorize(s.fillna("").astype(str))
            blocks.append(state.transform(uniques).toarray()[codes])
        elif kind == "num":
            blocks.append(
                pd.to_numeric(s, errors="coerce").fillna(state).to_numpy(dtype=float).reshape(-1, 1)
            )
        else:
            codes = s.astype(str).map(lambda v, m=state: m.get(v, -1))
            blocks.append(codes.to_numpy(dtype=float).reshape(-1, 1))
    return np.hstack(blocks)


COLUMNS = {
    "floats": [1.5, None, -2.0, 3.25, np.nan, 0.0],
    "ints": [1, 2, 3, 4, 5, 6],
    "bools": [True, False, True, True, False, True],
    "bools_missing": [True, None, False, None, True, None],
    "ints_missing": [1, None, 3, None, 5, 7],
    "mixed_numbers": [True, 2, 3.5, None, np.int64(4), np.float32(0.5)],
    "numeric_text": ["1", "2.5", None, "x", "1e3", " 7 "],
    "huge_int": [2**80, 1, None, 3, 4, 5],
    "pd_na": [pd.NA, 1, 2, pd.NA, 3, 4],
    "all_missing": [None] * 6,
    "cat_str": ["a", "b", None, "a", "c", "b"],
    "cat_mixed": [1, "1", True, None, 1.0, "x"],
    "text": ["book a flight", "cancel the order", None, "book a hotel", "cancel", "book a flight"],
}
TYPES = {c: "num" for c in COLUMNS} | {"cat_str": "cat", "cat_mixed": "cat", "text": "text"}


def frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(v, dtype=getattr(v, "dtype", object)) for c, v in values.items()})


@pytest.mark.parametrize("seed", range(5))
def test_block_conversions_match_the_per_column_code(seed):
    rng = np.random.default_rng(seed)
    train = frame(COLUMNS)
    f = Featurizer(TYPES).fit(train)
    order = rng.permutation(6)
    live = frame(
        {c: [v[i] for i in order] if not hasattr(v, "dtype") else v[order] for c, v in COLUMNS.items()}
    )
    for X in (train, live, live.iloc[:1], live.astype(object)):
        np.testing.assert_array_equal(f.transform(X), reference(f, X))


def test_nullable_dtypes():
    # The per-column code raised here (``fillna`` of an Int64 column with a fractional median).
    train = pd.DataFrame(
        {
            "i": pd.array([1, None, 3, 4, None, 6], dtype="Int64"),
            "b": pd.array([True, None, False, True, None, False], dtype="boolean"),
        }
    )
    f = Featurizer({"i": "num", "b": "num"}).fit(train)
    assert f.transform(train).tolist() == [[1, 1], [3.5, 0.5], [3, 0], [4, 1], [3.5, 0.5], [6, 0]]


def test_unseen_categories_and_float_frames():
    f = Featurizer({"n": "num", "c": "cat"}).fit(pd.DataFrame({"n": [1.0, 2.0, None], "c": ["a", "b", "a"]}))
    X = pd.DataFrame({"n": [None, 5.0], "c": ["z", "b"]})
    np.testing.assert_array_equal(f.transform(X), reference(f, X))
    assert f.transform(X)[0].tolist() == [1.5, -1.0]
