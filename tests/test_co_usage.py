"""The fitted co-usage feature (paper Sec. 4.3 "co-usage counts"): what it computes, that it is fitted on
training groups only, and that it persists (chat logs feeding it: tests/adapters/test_co_usage_chat.py)."""

import numpy as np
import pandas as pd
import pytest

from tabagent import TabAgent, TabHead, cross_validate
from tabagent.evaluation import assign_folds
from tabagent.table import DecisionSpec, DecisionTable
from tabagent.tabschema import Feature, FeatureCard
from tabagent.tabschema.fitted import FITTED, CoUsage, context_items

SPEC = DecisionSpec("ranking", "task", "used", "tool", decision=("decision",))
TOOLS = ["search", "book", "weather", "email"]
# the agent searches first, books after a search, emails after booking, then checks the weather
NEXT = {"": "search", "search": "book", "book": "email", "email": "weather", "weather": "search"}


def _table(n_tasks=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tasks):
        history: list[str] = []
        for step in range(3):
            prev = history[-1] if history else ""
            chosen = NEXT[prev] if rng.random() < 0.9 else str(rng.choice(TOOLS))
            for tool in TOOLS:
                rows.append(
                    {
                        "task": f"t{t:03d}",
                        "decision": f"t{t:03d}#{step}",
                        "tool": tool,
                        "last_call": prev,
                        "used": int(tool == chosen),
                    }
                )
            history.append(chosen)
    return DecisionTable(pd.DataFrame(rows), SPEC)


def _card(**args):
    return FeatureCard(
        "co_usage_toy",
        SPEC,
        [
            Feature("tool", "schema", "cat"),
            Feature("co", "dependency", "num", "fitted:co_usage", {"context": "last_call", **args}),
        ],
    )


def test_rates_are_smoothed_training_counts():
    df = pd.DataFrame(
        {
            "task": ["a", "a", "b", "b", "c", "c"],
            "decision": ["a", "a", "b", "b", "c", "c"],
            "tool": ["x", "y", "x", "y", "x", "y"],
            "last_call": ["u", "u", "u", "u", "", ""],
            "used": [0, 1, 0, 1, 1, 0],
        }
    )
    feat = CoUsage(context="last_call", alpha=1.0).fit(df, SPEC)
    prior_x, prior_y = 1 / 3, 2 / 3
    assert feat.prior_ == pytest.approx({"x": prior_x, "y": prior_y})
    live = pd.DataFrame({"tool": ["x", "y", "x", "z"], "last_call": ["u", "u", "", "u"]})
    got = feat.transform(live, SPEC).to_numpy()
    # n(u, x) = 2 rows, chosen 0; n(u, y) = 2, chosen 2; empty context -> prior; unseen tool -> global mean
    assert got == pytest.approx([(0 + prior_x) / 3, (2 + prior_y) / 3, prior_x, 0.5])
    assert feat.seen_groups_ == {"a", "b", "c"}
    assert "co_usage" in FITTED


def test_context_cells_may_be_lists_strings_or_json():
    assert context_items(["a", "b", "a"]) == ["a", "b"]
    assert context_items("a, b; c|d e") == ["a", "b", "c", "d", "e"]
    assert context_items('["a", "b"]') == ["a", "b"] and context_items("[a, b]") == ["a", "b"]
    assert context_items(None) == [] and context_items("") == [] and context_items(float("nan")) == []
    table = _table(12)
    as_lists = table.df.assign(last_call=table.df["last_call"].map(lambda s: s.split(", ") if s else []))
    a = CoUsage(context="last_call").fit(table.df, SPEC).transform(table.df, SPEC)
    b = CoUsage(context="last_call").fit(as_lists, SPEC).transform(as_lists, SPEC)
    assert np.allclose(a, b)


def test_co_usage_learns_what_follows_what():
    table = _table()
    feat = CoUsage(context="last_call", agg="max").fit(table.df, SPEC)
    live = pd.DataFrame({"tool": TOOLS, "last_call": ["search"] * 4})
    rates = dict(zip(TOOLS, feat.transform(live, SPEC), strict=True))
    assert max(rates, key=rates.get) == "book" and rates["book"] > 0.7
    with pytest.raises(ValueError, match="agg"):
        CoUsage(context="last_call", agg="median").fit(table.df, SPEC)


def test_co_usage_is_fitted_on_training_groups_only():
    table = _table(50)
    folds = assign_folds(table.groups, 5, 7)
    agent = TabAgent(_card(), head=TabHead("ranking", iterations=30))
    base = cross_validate(agent, table, folds=folds, seeds=(0,)).oof[0].to_numpy(float)
    rng = np.random.default_rng(1)
    for f in range(5):
        train = [g for g in table.groups if folds[g] != f]
        fitted = agent.clone().fit(table.subset(train))
        assert fitted.card_._fitted["co"].seen_groups_ == set(train)
        df = table.df.copy()
        test = (df["task"].map(folds) == f).to_numpy()
        df.loc[test, "used"] = rng.permutation(df.loc[test, "used"].to_numpy())
        oof = cross_validate(agent, table.with_df(df), folds=folds, seeds=(0,)).oof[0].to_numpy(float)
        assert np.array_equal(oof[test], base[test]), f"fold {f}: held-out labels reached the feature"


def test_co_usage_survives_save_and_load(tmp_path):
    table = _table(30)
    agent = TabAgent(_card(alpha=2.0), head=TabHead("ranking", iterations=30)).fit(table)
    agent.save(tmp_path / "m")
    again = TabAgent.load(tmp_path / "m")
    live = table.subset(table.groups[:5]).drop_labels()
    assert np.allclose(agent.score(live), again.score(live))
    assert again.card_._fitted["co"].pairs_ == agent.card_._fitted["co"].pairs_
