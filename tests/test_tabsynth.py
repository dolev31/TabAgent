"""TabSynth follows Algorithm 1 (App. H) and the worked example (App. H, "Worked example")."""

import numpy as np
import pandas as pd
import pytest

from tabagent import TabAgent, TabHead
from tabagent.tabsynth import TabSynth, det_paraphrase_v1

from conftest import make_card, make_table

EXAMPLE = "Show events in Los Angeles and select the one on the Friday this weekend."


def test_worked_example_appH():
    assert det_paraphrase_v1(EXAMPLE, 1) == "Please handle the following. " + EXAMPLE


def test_paraphraser_is_deterministic_golden_values():
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


def test_label_preserving_and_features_copied_verbatim_appH():
    X, y = _xy()
    Xs, ys = TabSynth(k=2, text_cols=["intent"]).augment(X, y)
    assert len(Xs) == 4 and ys.tolist() == [1, 1, 0, 0]  # row-major: each source row, then its variants
    assert Xs["tool"].tolist() == ["a", "a", "b", "b"] and Xs["p"].tolist() == [0.1, 0.1, 0.9, 0.9]
    assert Xs["intent"].iloc[0] == "Task: find a flight to rome"


def test_duplicates_of_real_rows_are_dropped_alg1():
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
    agent = TabAgent(make_card(), head=TabHead("ranking", preset="paper", iterations=30), synth=TabSynth(k=2))
    agent.fit(table)
    assert agent.n_synthetic_ == 2 * len(table)
    assert agent.synth.stats_["text_cols"] == ["intent"]
    assert set(agent.synth.stats_["source_groups"]) == set(table.groups)
    before = agent.synth.stats_
    agent.score(table.drop_labels())
    assert agent.synth.stats_ is before  # scoring never synthesizes


def test_k_defaults_to_the_cards_synth_k():
    table = make_table(6)
    card = make_card()  # synth_k = 2
    card.fit(table)
    X = card.transform(table)
    synth = TabSynth()
    Xs, _ = synth.augment(X, table.labels, table, card)
    assert synth.k_ == 2 and synth.stats_["k"] == 2 and len(Xs) == 2 * len(X)
    card.synth_k = 0  # a card without a k falls back to DEFAULT_K
    synth.augment(X, table.labels, table, card)
    assert synth.k_ == 4
    explicit = TabSynth(k=1)
    explicit.augment(X, table.labels, table, card)
    assert explicit.k_ == 1


def test_max_source_rows_subsample_is_seeded():
    X = pd.DataFrame({"intent": [f"task number {i}" for i in range(50)]})
    y = np.zeros(50)
    a, _ = TabSynth(k=1, text_cols=["intent"], max_source_rows=10, seed=1).augment(X, y)
    b, _ = TabSynth(k=1, text_cols=["intent"], max_source_rows=10, seed=1).augment(X, y)
    assert len(a) == 10 and a.equals(b)


def test_domain_matched_synthesis_appB6():
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


# --------------------------------------------------------------------------- extended operators (App. H)
REQUEST = (
    "Hi, my name is John Smith, user id john_smith_1234, email john.smith@example.org, phone 415-555-0199. "
    "Please cancel reservation ABC123 and order #W1234567, refund $120.50 to zip 94107."
)


def test_entity_swap_replaces_every_entity_and_is_deterministic():
    from tabagent.tabsynth import swap_entities

    out = swap_entities(REQUEST, 17)
    for old in (
        "John Smith",
        "john_smith_1234",
        "john.smith@example.org",
        "415-555-0199",
        "ABC123",
        "W1234567",
        "$120.50",
        "94107",
    ):
        assert old not in out
    assert out == swap_entities(REQUEST, 17) and out != swap_entities(REQUEST, 18)
    assert out.startswith("Hi, my name is ") and "Please cancel reservation " in out
    assert swap_entities("Nothing to swap here.", 3) == "Nothing to swap here."


def test_inverse_sqrt_frequency_allocation_favors_rare_candidates():
    from tabagent.tabsynth import inverse_sqrt_frequency_allocation

    table = make_table(n_tasks=40)
    budget = inverse_sqrt_frequency_allocation(table, k=4, max_variants=12)
    assert set(budget) == set(table.decision_ids)
    assert all(1 <= b <= 12 for b in budget.values())
    assert abs(np.mean(list(budget.values())) - 4) < 1
    df = table.df.assign(d=table.decision_ids.to_numpy())
    pos = df[df["used"] == 1]
    freq = pos.groupby("tool")["d"].nunique()
    rarest = pos.groupby("d")["tool"].agg(lambda s: freq[s].min())
    assert budget[rarest.idxmin()] >= budget[rarest.idxmax()]


def test_recompute_derives_text_features_from_the_paraphrase():
    table = make_table(n_tasks=8)
    card = make_card()
    card.fit(table)
    X = card.transform(table)
    copied = TabSynth(k=2)
    Xc, _ = copied.augment(X, table.labels, table, card)
    assert np.allclose(Xc["lex"].to_numpy(), X["lex"].to_numpy()[copied.source_rows_])  # copied verbatim
    fresh = TabSynth(k=2, recompute=True)
    Xr, yr = fresh.augment(X, table.labels, table, card)
    first = Xr["intent"].str.startswith("Task: ").to_numpy()  # variant j=0 of each source row
    raw = (
        table.df.iloc[fresh.source_rows_[first]]
        .reset_index(drop=True)
        .assign(intent=Xr["intent"][first].to_numpy())
    )
    expected = card.transform(table.with_df(raw))
    assert first.sum() > 0 and np.allclose(Xr["lex"].to_numpy()[first], expected["lex"].to_numpy())
    assert not np.allclose(Xr["lex"].to_numpy(), X["lex"].to_numpy()[fresh.source_rows_])
    assert yr.tolist() == table.labels[fresh.source_rows_].tolist()  # labels preserved


def test_allocation_sets_variants_per_decision_and_needs_the_table():
    table = make_table(n_tasks=12)
    card = make_card()
    card.fit(table)
    X = card.transform(table)
    synth = TabSynth(k=2, allocation="inverse_sqrt_frequency", entity_swap=True)
    Xs, ys = synth.augment(X, table.labels, table, card)
    assert len(Xs) == len(ys) > 0 and synth.stats_["operators"]["allocation"] == "inverse_sqrt_frequency"
    with pytest.raises(ValueError, match="table"):
        TabSynth(k=2, recompute=True).augment(X, table.labels)
    with pytest.raises(ValueError, match="allocation"):
        TabSynth(allocation="rarest")
