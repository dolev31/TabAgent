"""Bring-your-own-dataset API: sklearn estimators, auto card, nested recipe selection, pointwise helper."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_score

import tabagent
from tabagent import TabAgentClassifier, TabAgentRanker, TabAgentRouter, pointwise
from tabagent.auto import infer_card, infer_type

from conftest import make_table


def _xy(n=60):
    df = make_table(n).df
    return df.drop(columns=["used"]), df["used"].to_numpy()


def test_ranker_end_to_end_with_auto_selection():
    X, y = _xy()
    r = TabAgentRanker(group_col="task", candidate_col="tool", pair_embeddings=False, iterations=60).fit(X, y)
    assert set(r.feature_types_) >= {"intent", "__candidate__", "tool_desc", "candidate_prior"}
    assert r.card_.synth_text == ["intent"]  # context text only; tool descriptions are candidate-side
    assert not r.selection_.empty and "p@r" in r.selection_.columns
    top = r.rank(X, k=2)
    assert (top.groupby("decision").size() == 2).all()
    assert r.score(X, y) > 0.9
    assert r.predict_proba(X).shape == (len(X), 2) and set(np.unique(r.predict(X))) <= {0, 1}


def test_ranker_is_sklearn_compatible():
    X, y = _xy(45)
    est = TabAgentRanker(
        group_col="task", candidate_col="tool", synth=0, prior=True, pair_embeddings=False, iterations=40
    )
    assert clone(est).get_params() == est.get_params()
    scores = cross_val_score(est, X, y, groups=X["task"], cv=GroupKFold(3), scoring=tabagent.scorer("p@r"))
    assert scores.shape == (3,) and (scores > 0.7).all()


def test_preset_conflicts_with_full_search():
    X, y = _xy(12)
    with pytest.raises(ValueError, match="preset"):
        TabAgentRanker(group_col="task", candidate_col="tool", search="full", preset="paper").fit(X, y)


def test_ranker_requires_groups_and_candidates():
    X, y = _xy(12)
    with pytest.raises(ValueError, match="groups"):
        TabAgentRanker(candidate_col="tool").fit(X, y)
    with pytest.raises(ValueError, match="candidate_col"):
        TabAgentRanker(group_col="task").fit(X, y)


def test_classifier_single_class():
    X, _ = _xy(60)
    X = X.drop_duplicates("task").reset_index(drop=True)[["intent"]]
    y = np.where(
        X.intent.str.contains("flight"), "travel", np.where(X.intent.str.contains("email"), "mail", "other")
    )
    clf = TabAgentClassifier(synth=0, iterations=80).fit(X, y)
    assert set(clf.classes_) == {"travel", "mail", "other"}
    assert (clf.predict(X) == y).mean() > 0.9
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 3)
    scores = clf.decision_function(X)  # class scores, not labels
    assert scores.dtype.kind == "f" and np.array_equal(scores, proba)
    assert (np.asarray(clf.classes_)[scores.argmax(axis=1)] == clf.predict(X)).all()


def test_router_picks_best_arm():
    rows = []
    rng = np.random.default_rng(0)
    for t in range(40):
        math = t % 2 == 0
        text = ("solve the equation " if math else "write a short poem about ") + str(rng.integers(100))
        rows += [
            {"task": t, "query": text, "model": m, "ok": int((m == "math-model") == math)}
            for m in ("math-model", "poet")
        ]
    df = pd.DataFrame(rows)
    X, y = df.drop(columns=["ok"]), df["ok"].to_numpy()
    router = TabAgentRouter(
        group_col="task", candidate_col="model", synth=0, prior=False, pair_embeddings=False, iterations=60
    ).fit(X, y)
    assert router.score(X, y) > 0.9
    assert len(router.route(X)) == 40


def test_infer_type_rules():
    assert infer_type(pd.Series([1, 2, 3])) == "num"
    assert infer_type(pd.Series(["a", "b", "a"])) == "cat"
    assert (
        infer_type(pd.Series(["a much longer free text description that exceeds forty characters"] * 3))
        == "text"
    )


def test_pointwise_expands_decisions():
    decisions = pd.DataFrame({"task": [1, 2], "intent": ["book", "mail"], "tools_used": [["a", "b"], ["c"]]})
    catalog = pd.DataFrame({"tool": ["a", "b", "c"], "desc": ["x", "y", "z"]})
    out = pointwise(decisions, chosen="tools_used", catalog=catalog, candidate_col="tool")
    assert len(out) == 6 and out.groupby("task")["chosen"].sum().tolist() == [2, 1]
    assert "desc" in out.columns


def test_auto_card_marks_candidate_side_text_as_schema():
    t = make_table(10)
    card = infer_card(t.df, t.spec, ["intent", "tool", "tool_desc"], pair_embeddings=False)
    fam = {f.name: f.family for f in card.features}
    assert fam["intent"] == "state" and fam["tool_desc"] == "schema" and fam["tool"] == "schema"
