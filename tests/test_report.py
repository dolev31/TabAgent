import numpy as np

from tabagent import TabAgent, TabAgentRanker, TabHead
from tabagent.baselines import bm25
from tabagent.report import evaluate

from conftest import make_card, make_table


def test_bm25_prefers_lexically_matching_candidates():
    t = make_table(12)
    s = bm25(t, "intent", ["tool", "tool_desc"])
    df = t.df.assign(s=s)
    flights = df[df.intent.str.contains("flight")]
    if len(flights):
        top = flights.loc[flights.groupby("task")["s"].idxmax(), "tool"]
        assert set(top) <= {"search_flights", "book_flight"}


def test_evaluate_ranking_report(tmp_path):
    t = make_table(40)
    agent = TabAgent(make_card(), head=TabHead("ranking", preset="tmlr-runs", iterations=40))
    rep = evaluate(agent, t, n_splits=4, llm_cost_per_call=0.05)
    assert {"TabAgent", "chance", "BM25"} <= set(rep.results.index)
    assert rep.latency_ms > 0 and rep.cost_per_decision_usd < 1e-4
    md = rep.to_markdown()
    assert "cheaper" in md and "P@R" not in md.split("Quality")[0]
    out = rep.save(tmp_path)
    assert (out / "report.md").exists() and (out / "report.json").exists()
    assert rep.importance is not None and np.isfinite(rep.importance).all()


def test_estimator_report():
    df = make_table(40).df
    X, y = df.drop(columns=["used"]), df["used"].to_numpy()
    r = TabAgentRanker(
        group_col="task", candidate_col="tool", synth=0, prior=True, pair_embeddings=False, iterations=40
    ).fit(X, y)
    rep = r.report(X, y, n_splits=4)
    assert "TabAgent" in rep.results.index and any("recipe" in n for n in rep.notes)
