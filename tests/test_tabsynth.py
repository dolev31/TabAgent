"""TabSynth follows Algorithm 1 (App. G, L2078-2098) and the worked example (L2104)."""

import numpy as np
import pandas as pd
import pytest

from tabagent import TabAgent, TabHead
from tabagent.tabsynth import TabSynth, det_paraphrase_v1

from conftest import make_card, make_table

EXAMPLE = "Show events in Los Angeles and select the one on the Friday this weekend."


def test_worked_example_L2104():
    assert det_paraphrase_v1(EXAMPLE, 1) == "Please handle the following. " + EXAMPLE


def test_paraphraser_is_deterministic_and_matches_the_original_generator():
    text = "Book a flight. Then email the receipt!"
    assert [det_paraphrase_v1(text, j) for j in range(4)] == [
        "Task: Book a flight. Then email the receipt!",
        "Then email the receipt! Book a flight.",
        "User request — Book a flight. Then email the receipt!",
        "Book a flight. Then email the receipt!\n(Please proceed accordingly.)",
    ]
    assert det_paraphrase_v1(None, 0) == "Task: "


def _xy():
    X = pd.DataFrame(
        {"intent": ["find a flight to rome", "cancel my booking"], "tool": ["a", "b"], "p": [0.1, 0.9]}
    )
    return X, np.array([1, 0])


def test_label_preserving_and_features_copied_verbatim_L2076_L2100():
    X, y = _xy()
    Xs, ys = TabSynth(k=2, text_cols=["intent"]).augment(X, y)
    assert len(Xs) == 4 and ys.tolist() == [1, 1, 0, 0]  # row-major: each source row, then its variants
    assert Xs["tool"].tolist() == ["a", "a", "b", "b"] and Xs["p"].tolist() == [0.1, 0.1, 0.9, 0.9]
    assert Xs["intent"].iloc[0] == "Task: find a flight to rome"


def test_duplicates_of_real_rows_are_dropped_L2091():
    X = pd.DataFrame({"intent": ["hello", "Task: hello"], "tool": ["a", "a"]})
    Xs, _ = TabSynth(k=1, text_cols=["intent"]).augment(X, np.array([1, 1]))
    assert Xs["intent"].tolist() == ["Task: Task: hello"]  # the variant of row 0 equals real row 1 -> dropped


def test_unknown_text_column_is_rejected():
    X, y = _xy()
    with pytest.raises(KeyError):
        TabSynth(text_cols=["missing"]).augment(X, y)


def test_no_text_columns_means_no_rows():
    X, y = _xy()
    Xs, ys = TabSynth().augment(X, y)
    assert Xs.empty and ys.size == 0


def test_synth_runs_only_in_fit_and_uses_card_defaults():
    table = make_table(30)
    agent = TabAgent(
        make_card(), head=TabHead("ranking", preset="tmlr-runs", iterations=30), synth=TabSynth(k=2)
    )
    agent.fit(table)
    assert agent.n_synthetic_ == 2 * len(table)
    assert agent.synth.stats_["text_cols"] == ["intent"]
    assert set(agent.synth.stats_["source_groups"]) == set(table.groups)
    before = agent.synth.stats_
    agent.score(table.drop_labels())
    assert agent.synth.stats_ is before  # scoring never synthesizes


def test_max_source_rows_subsample_is_seeded():
    X = pd.DataFrame({"intent": [f"task number {i}" for i in range(50)]})
    y = np.zeros(50)
    a, _ = TabSynth(k=1, text_cols=["intent"], max_source_rows=10, seed=1).augment(X, y)
    b, _ = TabSynth(k=1, text_cols=["intent"], max_source_rows=10, seed=1).augment(X, y)
    assert len(a) == 10 and a.equals(b)


def test_domain_matched_synthesis_L1177():
    X = pd.DataFrame(
        {
            "intent": ["buy shoes now", "book a hotel", "buy a hat"],
            "domain": ["Shopping", "Travel", "Shopping"],
        }
    )
    Xs, ys = TabSynth(k=2, text_cols=["intent"], where={"domain": "Shopping"}).augment(X, np.array([1, 0, 1]))
    assert len(Xs) == 4 and set(Xs["domain"]) == {"Shopping"} and ys.tolist() == [1, 1, 1, 1]
    Xc, _ = TabSynth(k=1, text_cols=["intent"], where=lambda f: f["domain"] == "Travel").augment(
        X, np.zeros(3)
    )
    assert Xc["intent"].tolist() == ["Task: book a hotel"]


def test_source_rows_record_each_variants_real_row():
    table = make_table(n_tasks=6)
    card = make_card()
    card.fit(table)
    X = card.transform(table)
    synth = TabSynth(k=2)
    Xs, _ = synth.augment(X, table.labels, table, card)
    assert len(synth.source_rows_) == len(Xs) == len(synth.source_groups_)
    src = X.iloc[synth.source_rows_].reset_index(drop=True)
    assert (Xs["tool"] == src["tool"]).all()  # each variant copies its source row's candidate
    assert list(synth.source_groups_) == list(table.df["task"].to_numpy()[synth.source_rows_])


def test_max_source_rows_samples_within_where():
    X = pd.DataFrame({"intent": [f"request {i}" for i in range(20)], "domain": ["a", "b"] * 10})
    synth = TabSynth(k=1, text_cols=["intent"], where={"domain": "a"}, max_source_rows=4)
    Xs, _ = synth.augment(X, np.zeros(20, dtype=int))
    assert len(Xs) == 4 and set(Xs["domain"]) == {"a"}
    assert set(X["domain"].iloc[synth.source_rows_]) == {"a"}
