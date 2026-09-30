import numpy as np
import pandas as pd
import pytest

from tabagent import DecisionSpec, DecisionTable, Feature, FeatureCard, TabAgent, TabHead, cross_validate

from conftest import make_card, make_table


def fast_head():
    return TabHead("ranking", preset="paper", iterations=60)


def test_fit_score_decide(table, card):
    agent = TabAgent(card, head=fast_head()).fit(table)
    s = agent.score(table.drop_labels())
    assert s.shape == (len(table),)
    top = agent.decide(table.drop_labels(), k=2)
    assert (top.groupby("decision").size() == 2).all()


def test_learns_the_toy_task():
    table = make_table(90)
    res = cross_validate(TabAgent(make_card(), head=fast_head()), table, seeds=(0,))
    assert res.summary(["p@r"]).loc["p@r", "value"] > 0.9


def test_decide_with_deferral_and_fallback(table, card):
    agent = TabAgent(card, head=fast_head()).fit(table)
    calls = []
    out = agent.decide(
        table.drop_labels(), k=2, defer_below=1.0, fallback=lambda rows: calls.append(len(rows))
    )
    assert {"confidence", "deferred"} <= set(out.columns) and out["deferred"].all() and calls


def test_card_factory_is_called_on_training_rows(table, card):
    seen = []

    def factory(train):
        seen.append(set(train.groups))
        return card.clone()

    cross_validate(TabAgent(factory, head=fast_head()), table, seeds=(0,))
    assert len(seen) == 5 and all(len(s) < len(table.groups) for s in seen)


def test_unfitted_agent_raises(table, card):
    with pytest.raises(RuntimeError):
        TabAgent(card).score(table)
    assert np.isfinite(TabAgent(card, head=fast_head()).fit(table).score(table)).all()


def test_decide_accepts_a_frame_of_one_decisions_candidate_rows(table, card):
    agent = TabAgent(card, head=fast_head()).fit(table)
    first = table.groups[0]
    rows = table.df[table.df["task"] == first].drop(columns=["task", "used"])  # no ids, no label
    from_frame = agent.decide(rows, k=2)
    from_table = agent.decide(table.subset([first]).drop_labels(), k=2)
    assert from_frame["candidate"].tolist() == from_table["candidate"].tolist()
    assert np.array_equal(agent.score(rows), agent.score(table.subset([first])))


def test_decide_with_a_prefix_and_candidates_explains_the_two_head_types(table, card):
    agent = TabAgent(card, head=fast_head()).fit(table)
    with pytest.raises(TypeError, match="TabAgentHead"):
        agent.decide({"steps": []}, ["search_flights", "book_flight"], k=2)
    with pytest.raises(TypeError, match="DecisionTable"):
        agent.decide({"steps": []})


def test_single_class_decide_reports_confidence_and_defers():
    spec = DecisionSpec(kind="single", group="task", label="op")
    texts = ["book a flight", "cancel my order", "play a song"] * 10
    df = pd.DataFrame(
        {"task": [f"t{i}" for i in range(30)], "text": texts, "op": [t.split()[0] for t in texts]}
    )
    card = FeatureCard("ops", spec, [Feature("text", "state", "text")])
    agent = TabAgent(card, head=TabHead("single", iterations=30)).fit(DecisionTable(df, spec))
    out = agent.decide(df.drop(columns=["op"]), defer_below=1.01)
    assert set(out.columns) == {"decision", "choice", "confidence", "deferred"}
    assert out["deferred"].all() and out["confidence"].between(0, 1).all()
    assert (out["choice"] == df["op"]).mean() > 0.9
