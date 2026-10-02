import numpy as np
import pandas as pd
import pytest

from tabagent.table import DecisionSpec, DecisionTable, LeakageError
from tabagent.tabschema import Feature, FeatureCard
from tabagent.tabschema.fitted import CandidatePrior, EmbeddingCosine, HashingEncoder, KnnOutcome


def test_yaml_roundtrip(tmp_path, card):
    path = card.save(tmp_path)
    assert path.name == "card.yaml"
    again = FeatureCard.load(tmp_path)
    assert again.to_dict() == card.to_dict()
    assert again.types == card.types and again.spec == card.spec


def test_card_rejects_outcome_features_and_unknown_sources(card):
    with pytest.raises(LeakageError):
        FeatureCard("x", card.spec, [Feature("used", "state", "num")])
    with pytest.raises(KeyError):
        FeatureCard("x", card.spec, [Feature("p", "dependency", "num", "fitted:nope")])
    with pytest.raises(ValueError):
        Feature("p", "other", "num")


def test_transform_requires_fit_and_matching_spec(card, table):
    with pytest.raises(RuntimeError):
        card.transform(table)
    card.fit(table)
    X = card.transform(table.drop_labels())
    assert list(X.columns) == card.feature_names and len(X) == len(table)
    other = table.with_spec(label="used", strata="task")
    with pytest.raises(ValueError):
        card.transform(other)


def test_candidate_prior_fill_and_rank():
    spec = DecisionSpec("ranking", "g", "y", "c")
    train = pd.DataFrame({"g": [1, 1, 2, 2], "c": ["a", "b", "a", "b"], "y": [1, 0, 1, 1]})
    test = pd.DataFrame({"g": [3, 3], "c": ["b", "z"]})
    p = CandidatePrior().fit(train, spec)
    assert p.transform(test, spec).tolist() == [0.5, 0.0]
    assert CandidatePrior(fill="global").fit(train, spec).transform(test, spec).tolist() == [0.5, 0.75]
    assert CandidatePrior(stat="rank").fit(train, spec).transform(test, spec).tolist() == [2.0, 3.0]
    assert p.seen_groups_ == {1, 2}


def test_embedding_cosine_is_centered_on_training_rows_only():
    spec = DecisionSpec("ranking", "g", "y", "c")
    train = pd.DataFrame(
        {"g": [1, 1], "c": ["book flight", "play song"], "q": ["book a flight"] * 2, "y": [1, 0]}
    )
    f = EmbeddingCosine(query="q", passage="c", encoder="hashing").fit(train, spec)
    enc = HashingEncoder()
    mu = np.vstack([enc.encode(["book a flight"]), enc.encode(["book flight", "play song"])]).mean(0)
    assert np.allclose(f.mu_[0], mu)
    s = f.transform(train, spec)
    assert s.iloc[0] > s.iloc[1]  # the matching tool is closer


def test_knn_outcome_uses_only_training_neighbours():
    spec = DecisionSpec("policy", "g", "ok", "m", reward="ok")
    rows = []
    for g, text, good in [(1, "math problem", "a"), (2, "math question", "a"), (3, "write a poem", "b")]:
        rows += [{"g": g, "text": text, "m": m, "ok": int(m == good)} for m in "ab"]
    train = pd.DataFrame(rows)
    test = pd.DataFrame([{"g": 9, "text": "math homework", "m": m} for m in "ab"])
    f = KnnOutcome(query="text", k=2, encoder="hashing").fit(train, spec)
    out = f.transform(test, spec)
    assert out.iloc[0] > out.iloc[1]  # the two nearest training tasks favoured model a


def test_bundled_card_lookup_error():
    with pytest.raises(FileNotFoundError):
        FeatureCard.load("does_not_exist")


def test_python_source(tmp_path, monkeypatch):
    mod = tmp_path / "myfeats.py"
    mod.write_text("def shout(df, col):\n    return df[col].str.upper()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = DecisionSpec("single", "g", "y")
    card = FeatureCard("x", spec, [Feature("loud", "state", "cat", "python:myfeats:shout", {"col": "t"})])
    t = DecisionTable(pd.DataFrame({"g": [1], "t": ["hi"], "y": ["a"]}), spec)
    assert card.fit(t).transform(t)["loud"].tolist() == ["HI"]


def test_python_sources_outside_tabagent_need_trust_code(tmp_path, monkeypatch):
    mod = tmp_path / "myfeats.py"
    mod.write_text("def shout(df, col):\n    return df[col].str.upper()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = DecisionSpec("single", "g", "y")
    card = FeatureCard("x", spec, [Feature("loud", "state", "cat", "python:myfeats:shout", {"col": "t"})])
    path = card.save(tmp_path / "card")
    with pytest.raises(PermissionError, match="myfeats"):
        FeatureCard.load(path)
    assert FeatureCard.load(path, trust_code=True).features[0].source == "python:myfeats:shout"
    with pytest.raises(ValueError, match="python:<module>:<function>"):
        FeatureCard.from_dict(
            {
                **card.to_dict(),
                "features": [{"name": "f", "family": "state", "type": "num", "source": "python:nofunction"}],
            }
        )


def test_tabagent_modules_are_always_allowed():
    card = FeatureCard.load("llmrouterbench_model_routing")  # python:tabagent.datasets.llmrouterbench:...
    assert any(f.source.startswith("python:tabagent.") for f in card.features)


def test_card_text_columns():
    from conftest import make_card

    card = make_card()
    assert card.context_text_columns() == ["intent"]
    assert card.candidate_text_columns() == ["tool", "tool_desc"]


def test_cached_encoder_memo_is_bounded_and_thread_safe(monkeypatch):
    """A serving process sees a new context text at almost every decision: the memo keeps the most recently
    used texts only, gives the encoder's own vectors, and survives concurrent requests."""
    from concurrent.futures import ThreadPoolExecutor

    from tabagent.tabschema.fitted import CachedEncoder

    enc = CachedEncoder("hashing")
    monkeypatch.setattr(enc, "max_items", 5)
    for i in range(50):  # one new context per decision, the same candidate text every time
        enc.encode([f"context {i}", "book a flight"], "query: ")
    assert len(enc.memo) <= 5 and ("query: ", "book a flight") in enc.memo
    texts = [f"text {i}" for i in range(40)]
    assert np.allclose(enc.encode(texts, "query: "), HashingEncoder().encode(texts))

    def work(i):
        return enc.encode([f"thread {i} {j}" for j in range(8)] + ["book a flight"], "query: ").shape

    with ThreadPoolExecutor(8) as pool:
        assert set(pool.map(work, range(200))) == {(9, HashingEncoder().dim)}
    assert len(enc.memo) <= 5 + 8 * 8  # bounded, give or take the batches in flight


def test_concurrent_first_use_loads_the_encoder_once(monkeypatch):
    """The first decisions of a fresh HTTP service arrive together: the encoder is loaded by one thread while
    the others wait, instead of every thread importing and loading it at once (which failed with an
    ImportError on transformers' lazy imports, answered as HTTP 500)."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from tabagent.tabschema import fitted

    loads = []

    class Slow(HashingEncoder):
        def __init__(self):
            loads.append(threading.get_ident())
            time.sleep(0.2)  # importing torch and reading the weights
            super().__init__()

    monkeypatch.setitem(fitted._ENCODERS, "slow", Slow)
    monkeypatch.delitem(fitted._ENCODER_CACHE, "slow", raising=False)
    monkeypatch.delitem(fitted._SHARED, "slow", raising=False)
    with ThreadPoolExecutor(8) as pool:
        encs = list(pool.map(lambda _: fitted.encoder("slow"), range(8)))
    assert len(loads) == 1 and all(e is encs[0] for e in encs)
    fitted._SHARED.pop("slow", None)
    fitted._ENCODER_CACHE.pop("slow", None)
