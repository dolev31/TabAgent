"""TabHead follows paper Sec. 3.3, Sec. 4.2 (CatBoost configuration) and App. N.3 (confidence gate)."""

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


def test_paper_preset_is_the_configuration_the_paper_states():
    paper = catboost_estimator("paper", "ranking", 7).get_params()
    assert (paper["iterations"], paper["learning_rate"], paper["depth"]) == (300, 0.08, 6)
    assert paper["loss_function"] == "Logloss" and paper["random_seed"] == 7
    assert catboost_estimator("paper", "single", 0).get_params()["loss_function"] == "MultiClass"
    default = catboost_estimator("catboost-default", "ranking", 0).get_params()
    assert default["iterations"] == 1000 and default["l2_leaf_reg"] == 3.0
    assert "learning_rate" not in default  # automatically selected
    assert set(PRESETS) == {"paper", "catboost-default"} and TabHead().preset == "paper"


def test_fit_score_learns_signal_and_is_seed_deterministic():
    X, y = _frame()
    h = TabHead("ranking", preset="paper").fit(X, y, TYPES)
    p = h.score(X)
    assert p.shape == (len(X),) and ((p > 0.5) == y).mean() > 0.9
    p2 = TabHead("ranking", preset="paper").fit(X, y, TYPES).score(X)
    assert np.array_equal(p, p2)
    assert h.clone(seed=3).model_ is None


def test_single_class_returns_labels():
    X, _ = _frame()
    y = X["intent"].str.split().str[0].to_numpy()  # the class is determined by the intent
    h = TabHead("single", preset="paper", iterations=50).fit(X, y, {"intent": "text", "prior": "num"})
    pred = h.score(X)
    assert set(pred) <= set(y) and (pred == y).mean() > 0.5


def test_custom_estimator_plugs_in():
    from sklearn.linear_model import LogisticRegression

    X, y = _frame()
    h = TabHead("ranking", estimator=lambda seed: LogisticRegression(max_iter=500, random_state=seed))
    assert h.fit(X, y, TYPES).score(X).shape == (len(X),)


def test_conflicting_or_unknown_settings_are_rejected():
    from sklearn.linear_model import LogisticRegression

    with pytest.raises(ValueError, match="estimator"):
        TabHead("ranking", estimator=lambda s: LogisticRegression(), iterations=50)
    with pytest.raises(ValueError, match="kind"):
        TabHead("rank")


def test_no_usable_feature_is_an_error_not_a_constant_column():
    X = pd.DataFrame({"t": ["only", "words", "once", "here"]})  # min_df=2 leaves no vocabulary
    with pytest.raises(ValueError, match="no usable feature"):
        TabHead("ranking", iterations=10).fit(X, np.array([0, 1, 0, 1]), {"t": "text"})


def test_decode_orders_tied_scores_by_candidate_id():
    out = decode("ranking", np.array([0.5, 0.5, 0.9]), ["d"] * 3, ["b", "a", "c"], k=2)
    assert out["candidate"].tolist() == ["c", "a"]


def test_decode_multilabel_backoff_appJ():
    out = decode("multilabel", np.array([0.2, 0.4, 0.9, 0.6]), ["d1", "d1", "d2", "d2"], ["a", "b", "a", "b"])
    assert out[out.decision == "d1"]["candidate"].tolist() == ["b"]
    assert out[out.decision == "d2"]["candidate"].tolist() == ["a", "b"]


def test_decode_policy_with_cost_penalty():
    s, cost = np.array([0.8, 0.7]), np.array([1.0, 0.0])
    assert decode("policy", s, ["d", "d"], ["big", "small"])["candidate"].tolist() == ["big"]
    out = decode("policy", s, ["d", "d"], ["big", "small"], cost=cost, cost_lambda=0.5)
    assert out["candidate"].tolist() == ["small"]


def test_confidence_margin_and_largest_gap_appN3():
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


def test_many_text_columns_get_a_dense_lsa_representation():
    import numpy as np
    import pandas as pd

    from tabagent.tabhead import LSA_AUTO_TEXT_COLUMNS, LSA_DIMS, Featurizer, TabHead, resolve_text_repr

    words = [f"w{i}" for i in range(2000)]
    rows = [" ".join(words[(i * 7 + j) % 2000] for j in range(40)) for i in range(300)]
    assert resolve_text_repr("auto", LSA_AUTO_TEXT_COLUMNS) == "tfidf"  # the paper's cards: unchanged
    assert resolve_text_repr("auto", LSA_AUTO_TEXT_COLUMNS + 1) == "lsa"
    cols = {f"t{i}": "text" for i in range(10)}
    X = pd.DataFrame({c: rows for c in cols})
    lsa = Featurizer(cols, text_repr="lsa").fit(X)
    Z = lsa.transform(X)
    assert Z.shape == (300, 10 * LSA_DIMS) and np.isfinite(Z).all()
    np.testing.assert_allclose(lsa.transform(X.iloc[:5]), Z[:5])  # rows are encoded independently
    head = TabHead("ranking")
    y = (np.arange(300) % 7 == 0).astype(int)
    head.fit(X.assign(n=np.arange(300)), y, {**cols, "n": "num"})
    assert head.text_repr_ == "lsa"


def test_lsa_keeps_short_code_like_text_columns():
    import pandas as pd

    from tabagent.tabhead import Featurizer

    codes = [f"C{i % 4}" for i in range(80)]  # four distinct values, each on many rows
    fz = Featurizer({"code": "text"}, text_repr="lsa").fit(pd.DataFrame({"code": codes}))
    Z = fz.transform(pd.DataFrame({"code": codes}))
    assert Z.shape[0] == 80 and Z.shape[1] >= 1 and len({tuple(r) for r in Z.round(6)}) == 4


def test_catboost_fits_from_threads_are_serialized_and_identical():
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import pandas as pd

    import tabagent.tabhead as th

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = (X.a + 0.3 * rng.normal(size=400) > 0).astype(int)
    types = {"a": "num", "b": "num"}
    active, peak = [0], [0]
    real = th._CATBOOST_FIT

    class Counting:
        def __enter__(self):
            real.acquire()
            active[0] += 1
            peak[0] = max(peak[0], active[0])

        def __exit__(self, *exc):
            active[0] -= 1
            real.release()

    th._CATBOOST_FIT = Counting()
    try:
        with ThreadPoolExecutor(4) as pool:
            heads = list(pool.map(lambda _: th.TabHead("ranking", iterations=30).fit(X, y, types), range(4)))
    finally:
        th._CATBOOST_FIT = real
    assert peak[0] == 1
    scores = [h.score(X) for h in heads]
    assert all(np.array_equal(scores[0], s) for s in scores[1:])
