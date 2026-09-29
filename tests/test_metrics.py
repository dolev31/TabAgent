"""Metrics follow the paper's definitions (tabagent.tex line numbers in each test name/docstring)."""

import numpy as np
import pandas as pd
import pytest

from tabagent.evaluation import metrics as M
from tabagent.table import DecisionSpec, DecisionTable

RANK = DecisionSpec(kind="ranking", group="task", candidate="tool", label="used")


def _one(scores, used=(1, 0, 1, 0), tools=("a", "b", "c", "d")):
    df = pd.DataFrame({"task": "t", "tool": list(tools), "used": list(used)})
    return DecisionTable(df, RANK), np.asarray(scores, dtype=float)


def test_p_at_r_and_recall_at_k_L451_L462():
    t, s = _one([0.9, 0.8, 0.7, 0.1])  # G = {a, c}; ranking a, b, c, d
    assert M.p_at_r(t, s).iloc[0] == pytest.approx(0.5)
    assert M.recall_at_k(t, s, 1).iloc[0] == pytest.approx(0.5)
    assert M.recall_at_k(t, s, 3).iloc[0] == pytest.approx(1.0)
    assert M.hit_at_k(t, s, 1).iloc[0] == 1.0


def test_ties_broken_by_candidate_id_L448():
    t, s = _one([0.5, 0.5, 0.5, 0.5], used=(0, 1, 0, 0))  # all tied -> order a, b, c, d
    assert M.recall_at_k(t, s, 1).iloc[0] == 0.0
    assert M.recall_at_k(t, s, 2).iloc[0] == 1.0


def test_p_at_r_equals_recall_at_r_L462():
    rng = np.random.default_rng(0)
    rows = []
    for d in range(30):
        n = int(rng.integers(5, 15))
        used = (rng.random(n) < 0.3).astype(int)
        used[0] = 1
        rows += [{"task": f"t{d}", "tool": f"x{i}", "used": int(u)} for i, u in enumerate(used)]
    t = DecisionTable(pd.DataFrame(rows), RANK)
    s = rng.random(len(t))
    par = M.p_at_r(t, s)
    r_of = t.df.groupby("task")["used"].sum()
    for dec, v in par.items():
        assert v == pytest.approx(M.recall_at_k(t.subset([dec]), s[t.df.task == dec], int(r_of[dec])).iloc[0])


def test_decisions_without_relevant_candidates_are_skipped():
    t, s = _one([0.9, 0.8, 0.7, 0.1], used=(0, 0, 0, 0))
    assert M.p_at_r(t, s).empty


def test_chance_p_at_r_is_r_over_n():
    t, _ = _one([0, 0, 0, 0])
    assert M.chance_p_at_r(t).iloc[0] == pytest.approx(2 / 4)


def test_route_at_1_is_reward_of_chosen_arm_L1181():
    spec = DecisionSpec(kind="policy", group="task", candidate="model", label="ok", reward="r")
    df = pd.DataFrame({"task": "t", "model": ["m1", "m2", "m3"], "ok": [1, 0, 1], "r": [0.2, 0.9, 0.6]})
    t = DecisionTable(df, spec)
    assert M.route_at_1(t, np.array([0.1, 0.3, 0.8])).iloc[0] == pytest.approx(0.6)
    assert M.chance_route(t).iloc[0] == pytest.approx((0.2 + 0.9 + 0.6) / 3)
    assert M.oracle_route(t).iloc[0] == pytest.approx(0.9)


def test_multilabel_threshold_and_backoff_L2168():
    spec = DecisionSpec(kind="multilabel", group="task", candidate="app", label="needed")
    df = pd.DataFrame({"task": ["t1"] * 3 + ["t2"] * 3, "app": list("abc") * 2, "needed": [1, 0, 0, 0, 1, 1]})
    t = DecisionTable(df, spec)
    s = np.array([0.2, 0.1, 0.3, 0.9, 0.6, 0.1])  # t1: nothing >= .5 -> back off to top-1 ('c')
    pred = M.multilabel_predict(t, s)
    assert set(pred[(pred.decision == "t1") & (pred.pred == 1)].candidate) == {"c"}
    prf = M.multilabel_prf(t, s)
    assert prf.loc["t1", "precision"] == 0.0 and prf.loc["t1", "recall"] == 0.0
    assert prf.loc["t2", "precision"] == pytest.approx(0.5)  # predicted {a, b}, relevant {b, c}
    assert prf.loc["t2", "recall"] == pytest.approx(0.5)


def test_single_class_accuracy_and_macro_f1_L2267():
    spec = DecisionSpec(kind="single", group="task", label="op")
    df = pd.DataFrame({"task": ["t1", "t2", "t3", "t4"], "op": ["CLICK", "TYPE", "CLICK", "SELECT"]})
    t = DecisionTable(df, spec)
    pred = np.array(["CLICK", "TYPE", "TYPE", "SELECT"], dtype=object)
    assert M.accuracy(t, pred).mean() == pytest.approx(0.75)
    assert M.macro_f1(t, pred) == pytest.approx(np.mean([2 / 3, 2 / 3, 1.0]))


def test_normalized_headroom():
    assert M.normalized(0.68, 0.43, 0.94) == pytest.approx((0.68 - 0.43) / (0.94 - 0.43))


def test_dispatch():
    t, s = _one([0.9, 0.8, 0.7, 0.1])
    assert M.per_decision(t, s, "recall@3").iloc[0] == 1.0
    assert M.per_decision(t, s, "P@R").iloc[0] == 0.5
    with pytest.raises(ValueError):
        M.per_decision(t, s, "ndcg")


def test_legacy_tie_break_matches_pre_release_sort():
    """The legacy option reproduces `groupby(unit)` + `sort_values(ascending=False)` exactly."""
    rng = np.random.default_rng(1)
    rows = [{"task": f"t{d}", "tool": f"x{i}", "used": int(i < 2)} for d in range(20) for i in range(8)]
    t = DecisionTable(pd.DataFrame(rows), RANK)
    s = np.round(rng.random(len(t)), 1)  # many ties
    d = t.df.assign(_p=s)
    legacy = []
    for _, gr in d.groupby("task"):
        lab = gr.sort_values("_p", ascending=False)["used"].to_numpy()
        legacy.append(lab[: int(lab.sum())].sum() / lab.sum())
    assert M.p_at_r(t, s, tie_break="legacy").to_numpy() == pytest.approx(np.array(legacy))
