"""Figure and efficiency experiments (Fig. 2, Fig. 3, Fig. 4, cost appendix) on tiny fabricated tables.

No model download: the embedding features use the hashing encoder, and TabSTAR is replaced by a fake class.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import heads as H
from experiments.efficiency import cost, latency, pareto
from experiments.figures import head_ablation as HA
from experiments.figures import overload as OV
from experiments.figures import overload_llm as OL
from experiments.heads.tabstar import TabSTARHead
from tabagent import TabAgent, TabHead
from tabagent.evaluation.cv import assign_folds

from conftest import make_card, make_table

RAW = {"query": ("intent",), "passage": ("tool", "tool_desc")}


@pytest.fixture(scope="module")
def toy():
    t = make_table(30)
    return t, make_card("hashing"), assign_folds(t.groups, 3)


# --------------------------------------------------------------------------- heads
def test_cpu_heads_available_and_fit_through_tabhead():
    assert {"catboost", "logreg", "random_forest", "histgbm", "mlp"} <= set(H.available())
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.random(120), "b": rng.random(120)})
    y = (X["a"] > 0.5).astype(int).to_numpy()
    for name in ("logreg", "random_forest", "catboost"):
        head = H.make_head(name, "ranking").fit(X, y, {"a": "num", "b": "num"})
        s = head.score(X)
        assert s.shape == (120,) and s[y == 1].mean() > s[y == 0].mean()
    assert H.factory("histgbm")(3).random_state == 3
    assert H.factory("catboost")(0).get_params()["iterations"] == 300


def test_unknown_or_missing_head(monkeypatch):
    with pytest.raises(KeyError):
        H.is_available("nope")
    monkeypatch.setitem(H.HEADS, "xgboost", (H.HEADS["xgboost"][0], True, "no_such_package_xyz"))
    assert "xgboost" not in H.available()
    with pytest.raises(ImportError):
        H.make_head("xgboost")


class FakeTabSTAR:
    """Stands in for TabSTARClassifier: records its arguments; collapses on the first ``collapses`` fits."""

    instances: list = []
    collapses = 1

    def __init__(
        self,
        lora_lr=1e-3,
        lora_r=32,
        max_epochs=50,
        patience=5,
        random_state=None,
        verbose=False,
        is_paper_version=False,
        output_dir=None,
    ):
        self.kw = dict(
            lora_lr=lora_lr,
            lora_r=lora_r,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state,
            is_paper_version=is_paper_version,
            output_dir=output_dir,
        )
        self.collapsed = len(FakeTabSTAR.instances) < FakeTabSTAR.collapses
        FakeTabSTAR.instances.append(self)

    def fit(self, X, y):
        assert Path(self.kw["output_dir"]).is_dir()
        self.cols = list(X.columns)

    def predict_proba(self, X):
        n = len(X)
        p = np.full(n, 0.3) if self.collapsed else np.linspace(0.1, 0.9, n)
        return np.column_stack([1 - p, p])


def _fake(collapses):
    FakeTabSTAR.instances, FakeTabSTAR.collapses = [], collapses
    return FakeTabSTAR


def test_tabstar_collapse_guard_retries_once_with_lower_lr():
    X = pd.DataFrame({"t": [f"row {i}" for i in range(50)], "n": np.arange(50.0)})
    y = np.arange(50) % 2
    head = TabSTARHead(seed=7, classifier_cls=_fake(1), small_data_rows=0).fit(X, y)
    first, second = FakeTabSTAR.instances
    assert head.attempts_ == 2 and not head.collapsed_
    assert first.kw["lora_lr"] == 1e-3 and first.kw["lora_r"] == 16 and first.kw["max_epochs"] == 50
    assert second.kw["lora_lr"] == 2e-4 and second.kw["is_paper_version"]
    assert not Path(first.kw["output_dir"]).exists()  # disposable checkpoint dirs are removed
    assert head.predict_proba(X).shape == (50, 2)

    head = TabSTARHead(classifier_cls=_fake(2)).fit(X, y)  # 50 rows < 1000: small-card schedule
    assert head.attempts_ == 2 and head.collapsed_  # kept and recorded
    assert FakeTabSTAR.instances[0].kw["max_epochs"] == 15 and FakeTabSTAR.instances[0].kw["patience"] == 6


def test_tabstar_head_reads_raw_card_columns(toy):
    t, card, _ = toy
    _fake(0)
    raw = HA.raw_text_card(card, ["intent", "tool", "tool_desc"])
    head = TabHead(
        "ranking", estimator=lambda s: TabSTARHead(seed=s, classifier_cls=FakeTabSTAR), featurize=False
    )
    agent = TabAgent(raw, head).fit(t)
    assert FakeTabSTAR.instances[-1].cols == ["intent", "tool", "tool_desc"]
    assert agent.score(t.drop_labels()).shape == (len(t),)


# --------------------------------------------------------------------------- Fig. 3 head ablation
def test_raw_embedding_model_centers_on_training_rows(toy):
    from tabagent.tabschema.fitted import encoder

    t, _, _ = toy
    train = t.subset(t.groups[:20])
    m = HA.RawEmbeddingModel(H.factory("logreg"), encoder="hashing", **RAW).fit(train)
    enc = encoder("hashing")
    uq = list(dict.fromkeys(train.df["intent"]))
    up = list(dict.fromkeys((train.df["tool"] + HA.RAW_SEP + train.df["tool_desc"]).tolist()))
    mu = np.vstack([enc.encode(uq, "query: "), enc.encode(up, "passage: ")]).mean(0, keepdims=True)
    np.testing.assert_allclose(m.mu_, mu, rtol=1e-6)
    s = m.clone(1).fit(train).score(t.subset(t.groups[20:]).drop_labels())
    assert s.shape == (10 * 8,) and np.all((s >= 0) & (s <= 1))


def test_head_ablation_grid_json_resume_and_plot(toy, tmp_path):
    t, card, folds = toy
    kw = dict(card_objects=[card], table=t, folds=folds, seeds=(0, 1), encoder="hashing", out=tmp_path, **RAW)
    state = HA.run(["logreg"], log=lambda m: None, **kw)
    res = state["results"]
    assert set(res) == {"logreg|nocard", "logreg|card:toy_tools"}
    assert len(res["logreg|nocard"]["seeds"]) == 2
    # the card carries the decision: an additive model on raw embeddings cannot rank within a task
    assert res["logreg|card:toy_tools"]["mean"] > res["logreg|nocard"]["mean"] + 0.3
    saved = json.loads((tmp_path / "head_ablation.json").read_text())
    assert saved["results"] == res and saved["cards"] == ["toy_tools"]
    msgs = []
    HA.run(["logreg"], log=msgs.append, **kw)
    assert all("cached" in m for m in msgs)
    table = HA.comparison(state)
    assert table.loc["logreg", "paper_with"] == HA.PAPER_FIG3["logreg"]["card"][0]
    paths = HA.plot(state, tmp_path / "fig3")
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


# --------------------------------------------------------------------------- Fig. 2 overload
def _frame(pos_scores, neg_scores, pos_ids=None, neg_ids=None):
    rows = []
    for u, (ps, ns) in enumerate(zip(pos_scores, neg_scores, strict=True)):
        pid = pos_ids or [f"p{i}" for i in range(len(ps))]
        nid = neg_ids or [f"n{i:03d}" for i in range(len(ns))]
        rows += [{"unit": f"u{u}", "candidate": c, "y": 1, "score": s} for c, s in zip(pid, ps, strict=True)]
        rows += [{"unit": f"u{u}", "candidate": c, "y": 0, "score": s} for c, s in zip(nid, ns, strict=True)]
    return pd.DataFrame(rows)


def test_simulate_k_perfect_inverted_and_skipped():
    good = _frame([[0.9, 0.8]] * 3, [list(np.linspace(0, 0.5, 20))] * 3)
    out = OV.simulate_k(good, ks=(3, 10, 20, 50), reps=5)
    assert out[3]["p@r"] == 1.0 and out[20]["acc@1"] == 1.0 and out[20]["n_units"] == 3
    assert out[50]["n_units"] == 0 and out[50]["p@r"] is None  # fewer than K negatives
    assert out[10]["chance_p@r"] == round(2 / 12, 4)
    bad = _frame([[0.1]] * 3, [list(np.linspace(0.5, 1, 20))] * 3)
    assert OV.simulate_k(bad, ks=(5,), reps=5)[5]["p@r"] == 0.0


def test_simulate_k_random_is_chance_and_ties_are_scored_in_expectation():
    rng = np.random.default_rng(0)
    rand = _frame([list(rng.random(2)) for _ in range(1000)], [list(rng.random(60)) for _ in range(1000)])
    out = OV.simulate_k(rand, ks=(10,), reps=10)
    assert abs(out[10]["p@r"] - out[10]["chance_p@r"]) < 0.03
    # every candidate tied: the expected P@R and acc@1 are chance, whatever the candidate ids
    for pos_id in ("a", "z"):
        tied = _frame([[0.5]], [[0.5] * 5], pos_ids=[pos_id], neg_ids=[f"b{i}" for i in range(5)])
        res = OV.simulate_k(tied, ks=(3,), reps=3)[3]
        assert res["p@r"] == res["acc@1"] == res["chance_p@r"] == 0.25


def test_expected_hits_splits_the_free_slots_over_the_tied_candidates():
    s, y = np.array([0.9, 0.5, 0.5, 0.1]), np.array([0, 1, 0, 0])
    assert OV.expected_hits(s, y, 1) == 0.0
    assert OV.expected_hits(s, y, 2) == 0.5  # one slot left for two tied candidates, one of them relevant
    assert OV.expected_hits(s, y, 3) == 1.0


def test_power_law_recovers_parameters():
    ks = np.array([3, 5, 10, 20, 50, 100])
    fit = OV.power_law(ks, 0.95 * ks**-0.06)
    assert (
        fit["a"] == pytest.approx(0.95) and fit["b"] == pytest.approx(-0.06) and fit["r2"] == pytest.approx(1)
    )
    assert fit["projection"][400] == pytest.approx(0.95 * 400**-0.06)


def test_overload_tabagent_curve_and_plot(toy, tmp_path):
    t, card, folds = toy
    res = OV.run_tabagent(
        "toy", table=t, card=card, folds=folds, ks=(2, 3, 5), unit="decision", reps=4, log=lambda m: None
    )
    assert set(res["curve"]) == {2, 3, 5} and res["config"]["unit"] == "decision"
    assert all(0 <= v["p@r"] <= 1 for v in res["curve"].values())
    assert res["curve"][2]["p@r"] >= res["curve"][2]["chance_p@r"]
    llm = {"curve": {2: {"p@r": 0.6}, 3: {"p@r": 0.5}, 5: {"p@r": 0.4}}}
    state = {"curves": {"toy": {"tabagent": res, "sft_gen": llm}}}
    paths = OV.plot(state, tmp_path / "fig2")
    assert all(p.exists() for p in paths)
    assert OV.resolve_unit("mind2web/element") == "task_uid"
    assert OV.resolve_unit("skillsbench/skill-select") is None
    assert OV.resolve_unit("mind2web/element", "decision") is None


def test_overload_llm_pure_helpers(toy):
    t, _, _ = toy
    assert OL.parse_rank("3, 1, 3, 99, 2", 4, seed=0)[:3].tolist() == [2, 0, 1]
    assert sorted(OL.parse_rank("none", 5).tolist()) == list(range(5))
    prompt = OL.slate_prompt("book a flight", ["alpha", "beta"], "skills", 80)
    assert "1. alpha\n2. beta" in prompt and "relevant candidate numbers" in prompt
    plan = OL.plan(t, "toy", "sft_gen", "llama8b", ks=(3, 5, 7), unit=None)
    assert plan["units"] == 30 and plan["units_per_k"][7] <= 30 and plan["reps"] == OL.SFT["reps"]
    assert plan["generations"] == OL.SFT["reps"] * sum(plan["units_per_k"].values())
    assert OL.plan(t, "toy", "fewshot", "llama8b", ks=(3,), unit=None)["scored_pairs"] == len(t)
    with pytest.raises(ValueError):
        OL.plan(t, "toy", "bogus", "llama8b", ks=(3,), unit=None)


# --------------------------------------------------------------------------- cost, latency, Pareto
def test_cost_formulas_reproduce_the_paper_table():
    assert cost.CPU["usd_per_vcpu_hour"] == 0.063  # App. C: r6i.xlarge, $0.252 per hour for 4 vCPU
    assert cost.c_cpu(2.682e-3) == pytest.approx(4 * 2.682e-3 / 3600 * 0.063)
    assert cost.c_cpu(2.682e-3) == pytest.approx(1.9e-7, rel=0.02)  # App. F: "approx. $1.9e-7"
    assert cost.c_local(3600) == pytest.approx(0.25 * 1.4 * 0.20 + 0.20)
    rows = {r["method"]: r for r in cost.check_paper_table()}
    assert "Qwen2.5-7B-Instruct" in rows and len(rows) == 6  # the rows of Table 15
    for row in rows.values():
        if row["pricing"] == "api":
            assert row["recomputed_usd"] is None
        elif row["pricing"] == "cpu":  # Table 15 rounds 1.9e-7 up to 2.0e-7
            assert row["recomputed_usd"] <= row["reported_usd"] < 1.1 * row["recomputed_usd"], row
        else:
            assert abs(row["rel_diff"]) < 0.05, row


def test_latency_levels_and_pricing(toy):
    t, card, folds = toy
    ticks = iter(np.arange(0, 1e6) * 1e-3)  # every clock read advances 1 ms
    agent, test, _ = latency.fit_holdout(t, card, folds, holdout_fold=0)
    timing = latency.time_decisions(agent, test, n=4, repeats=2, warmup=1, clock=lambda: next(ticks))
    assert timing["decisions"] == 4 and timing["candidates_per_decision"]["mean"] == 8
    assert timing["levels"]["model"]["calls"] == 8 and timing["levels"]["agent_cold"]["calls"] == 4
    assert all(v["mean_ms"] == pytest.approx(1.0) for v in timing["levels"].values())
    rec = latency.run("toy", table=t, card=card, folds=folds, n=3, repeats=1, warmup=1, log=lambda m: None)
    lv = rec["levels"]["model"]
    assert lv["usd_per_decision"] == pytest.approx(cost.c_cpu(lv["mean_ms"] / 1000, 4))
    assert 0 <= rec["quality_holdout_fold"]["p@r"] <= 1 and rec["threads"] == 4


def test_query_embeddings_are_evicted_before_timing():
    from tabagent.tabschema.fitted import encoder

    enc = encoder("hashing")
    enc.encode(["a task text"], "query: ")
    enc.encode(["a catalog entry"], "passage: ")
    assert latency.evict_query_embeddings() >= 1
    assert ("query: ", "a task text") not in enc.memo and ("passage: ", "a catalog entry") in enc.memo


def test_pareto_front_points_and_plot(tmp_path):
    P = pareto.Point
    a = P("fast-cheap", 2, 1e-7, 0.8, "tabagent")
    b = P("slow-better", 500, 5e-5, 0.85, "finetuned")
    c = P("dominated", 600, 6e-5, 0.5, "fewshot")
    assert pareto.dominates(a, c) and not pareto.dominates(a, b)
    assert pareto.pareto_front([a, b, c]) == [a, b]
    assert pareto.marker_area(1e-7) > pareto.marker_area(1e-3)
    with pytest.raises(ValueError):
        P("bad", 0, 1e-7, 0.5)
    rec = {
        "dataset": "toy",
        "preset": "paper",
        "candidates_per_decision": {"mean": 8},
        "levels": {"model": {"mean_ms": 1.5, "usd_per_decision": 1.7e-9}},
        "quality_holdout_fold": {"p@r": 0.7},
    }
    path = tmp_path / "latency.json"
    path.write_text(json.dumps({"runs": {"toy|paper": rec}}))
    [pt] = pareto.points_from_latency(path)
    assert pt.kind == "tabagent" and pt.latency_ms == 1.5 and pt.quality == 0.7
    paths = pareto.plot([*pareto.PAPER_POINTS, pt], tmp_path / "pareto")
    assert all(p.exists() for p in paths)
