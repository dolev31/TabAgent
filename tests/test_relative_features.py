"""Within-decision relative features (``card.with_relative``): a feature's percentile rank among its
decision's candidates and its gap to the decision's best, computed from the decision's own rows only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from tabagent import TabAgent, TabHead, TabSynth, auto
from tabagent.table import DecisionSpec, DecisionTable
from tabagent.tabschema.card import (
    Feature,
    FeatureCard,
    has_relative,
    relative_values,
    with_relative,
    without_features,
    without_relative,
)

from conftest import make_card, make_table

SPEC = DecisionSpec("ranking", "task", "chosen", "candidate", decision=("decision_id",))


def test_rank_and_gap_within_each_decision():
    ids = ["d1"] * 4 + ["d2"] * 2 + ["d3"]
    values = [0.2, 0.9, 0.2, None, True, False, 5.0]
    rank = relative_values(values, ids, "relative_rank")
    gap = relative_values(values, ids, "relative_gap")
    np.testing.assert_allclose(rank[:3], [0.5, 1.0, 0.5])  # ties share the average rank, over the 3 present
    assert np.isnan(rank[3]) and np.isnan(gap[3])  # missing stays missing
    np.testing.assert_allclose(gap[:3], [-0.7, 0.0, -0.7])
    np.testing.assert_allclose(rank[4:6], [1.0, 0.5]) and np.testing.assert_allclose(gap[4:6], [0.0, -1.0])
    assert rank[6] == 1.0 and gap[6] == 0.0  # a decision's only candidate is its best
    np.testing.assert_allclose(relative_values(np.array([3, 1]), ["a", "a"], "relative_rank"), [1.0, 0.5])
    with pytest.raises(ValueError, match="unknown relative source"):
        relative_values([1], ["a"], "relative_mean")


def discovery_card() -> FeatureCard:
    return FeatureCard(
        "c",
        SPEC,
        [
            Feature("user_intent", "state", "text", "extractor"),
            Feature("candidate_name", "schema", "cat"),
            Feature("overlap", "schema", "num", "extractor_candidate"),
            Feature("api_verb", "schema", "cat", "extractor_candidate"),
            Feature(
                "context_candidate_lex",
                "schema",
                "num",
                "jaccard",
                {"a": ["user_intent"], "b": ["candidate_name"]},
            ),
            Feature("candidate_prior", "dependency", "num", "fitted:candidate_prior", {"key": "candidate"}),
        ],
        synth_text=["user_intent"],
        extractor={"sha256": "x"},
    )


def test_with_relative_adds_rank_and_gap_for_the_numeric_per_candidate_and_dependency_features():
    card = discovery_card()
    rel = with_relative(card)
    added = [f for f in rel.features if f.source in ("relative_rank", "relative_gap")]
    assert [f.name for f in added] == [
        "overlap_rank_in_decision",
        "overlap_gap_to_best",
        "context_candidate_lex_rank_in_decision",
        "context_candidate_lex_gap_to_best",
        "candidate_prior_rank_in_decision",
        "candidate_prior_gap_to_best",
    ]
    assert all(
        (f.family, f.type, f.provenance) == ("dependency", "num", {"added_by": "tabagent"}) for f in added
    )
    assert with_relative(rel).to_dict() == rel.to_dict()  # idempotent
    assert without_relative(rel).to_dict() == card.to_dict() and not has_relative(card)
    lean = without_features(rel, {"candidate_prior"})
    assert not any("candidate_prior" in f.name for f in lean.features) and lean.synth_text == ["user_intent"]
    single = FeatureCard("s", DecisionSpec("single", "task", "label"), [Feature("n", "state", "num")])
    assert with_relative(single).to_dict() == single.to_dict()  # no candidates, nothing to compare


def test_a_relative_feature_must_follow_the_feature_it_reads():
    rank = Feature("x_rank", "dependency", "num", "relative_rank", {"of": "x"})
    with pytest.raises(ValueError, match="must come earlier in the card"):
        FeatureCard("c", SPEC, [rank, Feature("x", "schema", "num")])
    d = with_relative(discovery_card()).to_dict()
    d["features"] = d["features"][-1:] + d["features"][:-1]  # a saved card edited out of order
    with pytest.raises(ValueError, match="must come earlier"):
        FeatureCard.from_dict(yaml.safe_load(yaml.safe_dump(d)))


def toy() -> tuple[FeatureCard, DecisionTable]:
    card = with_relative(make_card(), of=["lex", "prior", "sim"])
    return card, make_table(40)


def test_values_come_from_the_decision_and_ignore_the_labels():
    card, table = toy()
    X = card.fit(table).transform(table)
    ids = table.decision_ids.to_numpy()
    expected = pd.Series(X["lex"].to_numpy()).groupby(ids).rank(pct=True, method="average").to_numpy()
    np.testing.assert_allclose(X["lex_rank_in_decision"].to_numpy(), expected)
    gaps = X["sim_gap_to_best"].to_numpy()
    assert (gaps <= 0).all() and (pd.Series(gaps).groupby(ids).max() == 0).all()
    df = table.df.copy()
    df["used"] = np.random.default_rng(0).permutation(df["used"].to_numpy())
    relabeled = card.clone().fit(table).transform(table.with_df(df))  # same fit, other labels at transform
    pd.testing.assert_frame_equal(relabeled, X)


def test_tabsynth_variants_are_decisions_of_their_own():
    """Copied variants keep their real decision's values; recomputed ones are ranked among the variant's own
    rows, never mixed with the real rows."""
    card, table = toy()
    card.fit(table)
    X, y = card.transform(table), table.labels
    src = TabSynth(k=1)
    copied, _ = src.augment(X, y, table, card)
    pd.testing.assert_series_equal(
        copied["lex_rank_in_decision"].reset_index(drop=True),
        X["lex_rank_in_decision"].iloc[src.source_rows_].reset_index(drop=True),
    )

    def louder(text, j):  # a paraphrase that adds words some tools' descriptions have: lex changes
        return text + " flight seat booking"

    syn = TabSynth(k=1, recompute=True, paraphrase=louder)
    X_syn, _ = syn.augment(X, y, table, card)
    ids = table.decision_ids.to_numpy()[syn.source_rows_]
    own = pd.Series(X_syn["lex"].to_numpy()).groupby(ids).rank(pct=True, method="average").to_numpy()
    np.testing.assert_allclose(X_syn["lex_rank_in_decision"].to_numpy(), own)
    assert not np.allclose(X_syn["lex"].to_numpy(), X["lex"].to_numpy()[syn.source_rows_])


def test_tabsynth_keeps_the_real_values_of_a_decision_only_partly_paraphrased():
    """With ``where`` or ``max_source_rows``, a decision of which only some rows are paraphrased keeps its
    real rows' rank and gap (computed on the whole decision); a whole decision is ranked among its variant's
    rows."""
    card, table = toy()
    card.fit(table)
    X, y = card.transform(table), table.labels
    ids = table.decision_ids.to_numpy()
    relative = [c for c in X.columns if c.endswith(("_rank_in_decision", "_gap_to_best"))]
    whole = pd.unique(ids)[:10]
    mask = np.isin(ids, whole) | X["tool"].isin(["search_flights", "book_flight", "get_weather"]).to_numpy()

    def louder(text, j):
        return text + " flight seat booking"

    for syn in (
        TabSynth(k=1, recompute=True, paraphrase=louder, where=lambda F: mask),
        TabSynth(k=1, recompute=True, paraphrase=louder, max_source_rows=60, seed=0),
    ):
        X_syn, _ = syn.augment(X, y, table, card)
        src = syn.source_rows_
        counts, chosen = pd.Series(ids).value_counts(), pd.Series(ids[src]).value_counts()
        part = np.array([chosen[d] < counts[d] for d in ids[src]])
        assert part.any()
        np.testing.assert_allclose(
            X_syn[relative].to_numpy(float)[part], X[relative].to_numpy(float)[src][part]
        )
        assert not np.allclose(X_syn["lex"].to_numpy()[part], X["lex"].to_numpy()[src][part])  # recomputed
        if (~part).any():  # whole decisions: ranked among the variant's own rows
            own = pd.Series(X_syn["lex"].to_numpy()[~part]).groupby(ids[src][~part]).rank(pct=True)
            np.testing.assert_allclose(X_syn["lex_rank_in_decision"].to_numpy()[~part], own.to_numpy())


def test_scoring_one_decision_alone_or_in_a_batch_gives_the_same_scores():
    card, table = toy()
    train, test = table.subset(table.groups[:30]), table.subset(table.groups[30:])
    fitted = TabAgent(card, head=TabHead("ranking", iterations=40)).fit(train)
    together = fitted.score(test.drop_labels())
    alone = np.concatenate([fitted.score(test.subset([g]).drop_labels()) for g in test.groups])
    np.testing.assert_allclose(together, alone, atol=1e-12)
    one = test.subset([test.groups[0]]).df.iloc[:1]
    X = fitted.features(one.drop(columns=["used"]))
    assert X["lex_rank_in_decision"].iloc[0] == 1.0 and X["prior_gap_to_best"].iloc[0] == 0.0


def test_they_are_off_by_default_and_select_tries_them_once(monkeypatch):
    card, table = toy()
    assert auto.Recipe(relative=True).label() == "synth=off prior=on paper relative=on"
    assert auto.Recipe(relative=False, context=False).label().endswith("paper relative=off context=off")
    assert auto.Recipe().label() == "synth=off prior=on paper"  # a card without them: the label as before
    flat = auto.build_agent(card, auto.Recipe(relative=False)).card
    assert not has_relative(flat) and has_relative(auto.build_agent(card, auto.Recipe(relative=True)).card)
    no_prior = auto.build_agent(with_relative(discovery_card()), auto.Recipe(prior=False, relative=True)).card
    assert not any(f.name.startswith("candidate_prior") for f in no_prior.features)
    assert "overlap_rank_in_decision" in no_prior.feature_names
    assert all(r.relative is False for r in auto.recipes(card))  # off in the grid
    assert all(r.relative is None for r in auto.recipes(make_card()))

    class Scores:
        def __init__(self, v):
            self.v = v

        def per_decision(self, metric):
            return pd.Series([self.v])

    def search(on: float, off: float):
        seen = []

        def fake_cv(agent, tbl, **kw):
            rel = has_relative(agent.card)
            seen.append(rel)
            return Scores(on if rel else off)

        monkeypatch.setattr(auto, "cross_validate", fake_cv)
        best, rows = auto.select(table, card)
        return best, rows, seen

    best, rows, seen = search(on=0.80, off=0.75)  # strictly better: the challenger joins
    assert best.relative is True and "relative=on" in best.label()
    assert seen.count(True) == 2  # tried once, then once with the other text representation
    assert len(rows) == len(auto.recipes(card)) + 2
    best, rows, seen = search(on=0.75, off=0.75)  # a tie keeps them off
    assert best.relative is False and seen.count(True) == 1
    best, rows, seen = search(on=0.70, off=0.75)
    assert best.relative is False and "relative=off" in best.label()
    few = table.subset(table.groups[:4])  # too few groups to search: off, nothing scored
    best, rows = auto.select(few, card)
    assert best.relative is False and rows["note"].tolist() == ["no search"]
    assert not has_relative(auto.build_agent(card, best).card)


def test_a_discovered_card_gets_them_only_as_the_searchs_challenger(monkeypatch):
    """TabSchema's card never holds them: ``select(relative=True)`` adds them to the winner as a challenger
    (``with_relative``), and only then."""
    plain, table = discovery_card(), make_table(40)
    assert not has_relative(plain)
    assert all(r.relative is None for r in auto.recipes(plain))  # not asked: not searched
    assert all(r.relative is False for r in auto.recipes(plain, relative=True))
    on = auto.build_agent(plain, auto.Recipe(relative=True)).card
    assert on.to_dict() == with_relative(plain).to_dict()
    assert auto.build_agent(plain, auto.Recipe(relative=False)).card.to_dict() == plain.to_dict()
    single = FeatureCard("s", DecisionSpec("single", "task", "label"), [Feature("n", "state", "num")])
    assert not auto.searches_relative(single, relative=True)  # nothing to rank within a decision
    seen = []

    def fake_cv(agent, tbl, **kw):
        seen.append(has_relative(agent.card))
        return _Scores(0.80 if seen[-1] else 0.75)

    monkeypatch.setattr(auto, "cross_validate", fake_cv)
    best, _ = auto.select(table, plain)
    assert best.relative is None and not any(seen)
    best, _ = auto.select(table, plain, relative=True)
    assert best.relative is True and has_relative(auto.build_agent(plain, best).card)


class _Scores:
    def __init__(self, v):
        self.v = v

    def per_decision(self, metric):
        return pd.Series([self.v])
