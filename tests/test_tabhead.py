"""TabHead follows paper Sec. 3.3 (L320-323), L384 (CatBoost configuration) and App. D (L2347, gate)."""

import numpy as np
import pandas as pd
import pytest

from tabagent.tabhead import (
    PRESETS,
    Featurizer,
    TabHead,
    catboost_estimator,
    confidence,
    decode,
    risk_coverage,
)


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    words = np.array(["book flight", "cancel order", "reset router", "refund item", "change seat"])
    text = words[rng.integers(0, len(words), n)]
    X = pd.DataFrame(
        {
            "intent": text,
            "tool": rng.choice(["book", "cancel", "reset", "refund", "seat"], n),
            "prior": rng.random(n),
        }
    )
    y = (X["intent"].str.split().str[0].str[:4] == X["tool"].str[:4]).astype(int).to_numpy()
    return X, y


TYPES = {"intent": "text", "tool": "cat", "prior": "num"}


def test_featurizer_uses_declared_types_and_train_statistics():
    train = pd.DataFrame(
        {"t": ["alpha beta", "alpha gamma", "beta gamma"], "c": ["x", "y", "x"], "n": [1.0, None, 3.0]}
    )
    test = pd.DataFrame({"t": ["alpha zeta"], "c": ["unseen"], "n": [None]})
    fz = Featurizer({"t": "text", "c": "cat", "n": "num"}).fit(train)
    Z = fz.transform(test)
    vocab = fz.parts[0][2].vocabulary_
    assert "zeta" not in vocab and "alpha" in vocab  # TF-IDF vocabulary comes from training rows only
    assert Z[0, -2] == -1  # unseen category
    assert Z[0, -1] == 2.0  # training median
    with pytest.raises(ValueError):
        Featurizer({"t": "embedding"})


def test_featurizer_skips_text_column_with_empty_vocabulary():
    train = pd.DataFrame({"t": ["only", "words", "once"], "n": [1, 2, 3]})
    fz = Featurizer({"t": "text", "n": "num"}).fit(train)  # min_df=2 -> empty vocabulary
    assert [p[0] for p in fz.parts] == ["n"]


def test_presets_match_paper_and_runs_L384():
    paper = catboost_estimator("paper", "ranking", 0).get_params()
    assert paper["iterations"] == 1000 and paper["depth"] == 6 and paper["l2_leaf_reg"] == 3.0
    assert "learning_rate" not in paper  # automatically selected
    assert paper["loss_function"] == "Logloss"
    runs = catboost_estimator("tmlr-runs", "ranking", 7).get_params()
    assert runs["iterations"] == 300 and runs["learning_rate"] == 0.08 and runs["random_seed"] == 7
    assert catboost_estimator("paper", "single", 0).get_params()["loss_function"] == "MultiClass"
    assert set(PRESETS) == {"paper", "tmlr-runs"}


def test_fit_score_learns_signal_and_is_seed_deterministic():
    X, y = _frame()
    h = TabHead("ranking", preset="tmlr-runs").fit(X, y, TYPES)
    p = h.score(X)
    assert p.shape == (len(X),) and ((p > 0.5) == y).mean() > 0.9
    p2 = TabHead("ranking", preset="tmlr-runs").fit(X, y, TYPES).score(X)
    assert np.array_equal(p, p2)
    assert h.clone(seed=3).model_ is None


def test_single_class_returns_labels():
    X, _ = _frame()
    y = X["intent"].str.split().str[0].to_numpy()  # the class is determined by the intent
    h = TabHead("single", preset="tmlr-runs", iterations=50).fit(X, y, {"intent": "text", "prior": "num"})
    pred = h.score(X)
    assert set(pred) <= set(y) and (pred == y).mean() > 0.5


def test_custom_estimator_plugs_in():
    from sklearn.linear_model import LogisticRegression

    X, y = _frame()
    h = TabHead("ranking", estimator=lambda seed: LogisticRegression(max_iter=500, random_state=seed))
    assert h.fit(X, y, TYPES).score(X).shape == (len(X),)


def test_decode_ranking_ties_by_candidate_L448():
    out = decode("ranking", np.array([0.5, 0.5, 0.9]), ["d"] * 3, ["b", "a", "c"], k=2)
    assert out["candidate"].tolist() == ["c", "a"]


def test_decode_multilabel_backoff_L2168():
    out = decode("multilabel", np.array([0.2, 0.4, 0.9, 0.6]), ["d1", "d1", "d2", "d2"], ["a", "b", "a", "b"])
    assert out[out.decision == "d1"]["candidate"].tolist() == ["b"]
    assert out[out.decision == "d2"]["candidate"].tolist() == ["a", "b"]


def test_decode_policy_with_cost_penalty():
    s, cost = np.array([0.8, 0.7]), np.array([1.0, 0.0])
    assert decode("policy", s, ["d", "d"], ["big", "small"])["candidate"].tolist() == ["big"]
    out = decode("policy", s, ["d", "d"], ["big", "small"], cost=cost, cost_lambda=0.5)
    assert out["candidate"].tolist() == ["small"]


def test_confidence_margin_and_largest_gap_L2347():
    s = np.array([0.9, 0.8, 0.3, 0.2])
    ids, cands = ["d"] * 4, list("abcd")
    assert confidence(s, ids, cands, "largest_gap")["d"] == pytest.approx(0.5)
    assert confidence(s, ids, cands, "margin_at_r", r={"d": 2})["d"] == pytest.approx(0.5)
    assert confidence(s, ids, cands, "margin_at_r", r={"d": 1})["d"] == pytest.approx(0.1)


def test_risk_coverage_defers_least_confident_first():
    rc = risk_coverage([0.9, 0.1, 0.5], [1, 0, 1])
    assert rc["selective_quality"].tolist() == pytest.approx([1.0, 1.0, 2 / 3])


def test_featurizer_dedups_text_without_changing_output():
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = pd.DataFrame({"t": ["alpha beta", "alpha gamma", "alpha beta", None, "beta gamma"] * 3})
    fz = Featurizer({"t": "text"}).fit(X)
    ref = TfidfVectorizer(max_features=400, ngram_range=(1, 2), min_df=2).fit(X["t"].fillna("").astype(str))
    assert np.array_equal(fz.transform(X), ref.transform(X["t"].fillna("").astype(str)).toarray())
