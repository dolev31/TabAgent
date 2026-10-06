"""Leakage suite: structural guarantees of the protocol (paper Sec. 4.3: every data-dependent stage is
fitted inside the training portion of each fold and never sees held-out tasks)."""

import numpy as np
import pandas as pd
import pytest

from tabagent import LeakageError, TabAgent, TabHead, TabSynth, cross_validate
from tabagent.evaluation import assign_folds
from tabagent.tabschema import Feature, FeatureCard
from tabagent.tabschema.card import with_relative

from conftest import make_card, make_table


def agent(card=None):
    return TabAgent(card or make_card(), head=TabHead("ranking", preset="paper", iterations=60))


def test_scrambling_test_labels_does_not_change_test_scores():
    table = make_table(50)
    folds = assign_folds(table.groups, 5, 1)
    base = cross_validate(agent(), table, folds=folds, seeds=(0,)).oof[0].to_numpy(float)
    rng = np.random.default_rng(0)
    for f in range(5):
        df = table.df.copy()
        test = df["task"].map(folds).to_numpy() == f
        df.loc[test, "used"] = rng.permutation(df.loc[test, "used"].to_numpy())
        oof = cross_validate(agent(), table.with_df(df), folds=folds, seeds=(0,)).oof[0].to_numpy(float)
        assert np.array_equal(oof[test], base[test]), f"fold {f} scores depend on its own labels"


@pytest.mark.parametrize("relative", [False, True])
def test_scoring_is_not_transductive(relative):
    """A decision scores the same alone or with others, its within-decision rank and gap features included
    (they read the decision's own rows only)."""
    table = make_table(40)
    train, test = table.subset(table.groups[:30]), table.subset(table.groups[30:])
    card = with_relative(make_card(), of=["lex", "prior", "sim"]) if relative else make_card()
    fitted = agent(card).fit(train)
    together = fitted.score(test.drop_labels())
    one_by_one = np.concatenate([fitted.score(test.subset([g]).drop_labels()) for g in test.groups])
    assert np.allclose(together, one_by_one, atol=1e-12)


def test_fitted_features_only_see_training_groups():
    table = make_table(50)
    folds = assign_folds(table.groups, 5, 3)
    for f in range(5):
        train_groups = [g for g in table.groups if folds[g] != f]
        fitted = agent().fit(table.subset(train_groups))
        for name, feat in fitted.card_._fitted.items():
            assert feat.seen_groups_ == set(train_groups), name


class RecordingSynth(TabSynth):
    """TabSynth that records, for every call, the groups it was given and the groups it paraphrased."""

    calls: list = []

    def augment(self, X, y, table=None, card=None):
        out = super().augment(X, y, table, card)
        RecordingSynth.calls.append((set(table.groups), set(self.source_groups_)))
        return out


def test_tabsynth_paraphrases_only_training_groups_in_every_fold():
    table = make_table(40)
    folds = assign_folds(table.groups, 4, 5)
    RecordingSynth.calls = []
    synth_agent = TabAgent(
        make_card(), head=TabHead("ranking", preset="paper", iterations=20), synth=RecordingSynth(k=1)
    )
    cross_validate(synth_agent, table, folds=folds, seeds=(0, 1))
    assert len(RecordingSynth.calls) == 4 * 2
    held_out_sets = []
    for train_groups, paraphrased in RecordingSynth.calls:
        held_out = set(table.groups) - train_groups
        assert paraphrased and paraphrased <= train_groups and not paraphrased & held_out
        assert {folds[g] for g in held_out} == {folds[next(iter(held_out))]}  # exactly one held-out fold
        held_out_sets.append(frozenset(held_out))
    assert len(set(held_out_sets)) == 4  # every fold was held out once per seed


def test_label_copy_tripwire():
    table = make_table(30)
    leaky = table.with_df(table.df.assign(was_used=table.df["used"].map({0: "no", 1: "yes"})))
    card = make_card()
    card = FeatureCard(card.name, card.spec, [*card.features, Feature("was_used", "state", "cat")])
    with pytest.raises(LeakageError, match="was_used"):
        TabAgent(card, head=TabHead("ranking", preset="paper", iterations=10)).fit(leaky)


def test_models_never_receive_outcome_columns():
    table = make_table(20)
    blind = table.drop_labels()
    assert "used" not in blind.df.columns
    with pytest.raises(LeakageError):
        _ = blind.labels
    with pytest.raises(LeakageError):
        make_card().fit(blind)


def test_outcome_columns_are_never_features():
    table = make_table(10)
    with pytest.raises(LeakageError):
        table.feature_frame(["intent", "used"])
    assert isinstance(table.feature_frame(["intent"]), pd.DataFrame)
