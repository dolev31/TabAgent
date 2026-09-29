import numpy as np
import pytest

from tabagent import TabAgent, TabHead, cross_validate

from conftest import make_card, make_table


def fast_head():
    return TabHead("ranking", preset="tmlr-runs", iterations=60)


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
