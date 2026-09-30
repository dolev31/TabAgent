import json

import numpy as np
import pytest

from tabagent import TabAgent, TabHead, TabSynth, load

from conftest import make_card, make_table


def test_roundtrip_is_exact_and_pickle_free(tmp_path):
    table = make_table(40)
    agent = TabAgent(make_card(), head=TabHead("ranking", preset="paper", iterations=50), synth=TabSynth(k=2))
    agent.fit(table)
    out = agent.save(tmp_path / "head", metrics={"p@r": 0.9})
    names = {p.name for p in out.iterdir()}
    assert names == {
        "card.yaml",
        "fitted.json",
        "fitted.npz",
        "featurizer.json",
        "model.cbm",
        "manifest.json",
    }
    assert not any(p.suffix in (".pkl", ".pickle") for p in out.iterdir())
    again = load(out)
    blind = table.drop_labels()
    assert np.array_equal(again.score(blind), agent.score(blind))
    assert again.manifest_["metrics"] == {"p@r": 0.9}
    assert again.decide(blind, k=2).equals(agent.decide(blind, k=2))


def test_tampered_artifact_is_rejected(tmp_path):
    agent = TabAgent(make_card(), head=TabHead("ranking", preset="paper", iterations=20)).fit(make_table(20))
    out = agent.save(tmp_path / "h")
    fitted = json.loads((out / "fitted.json").read_text())
    fitted["prior"]["state"]["values"][0] = 123.0
    (out / "fitted.json").write_text(json.dumps(fitted))
    with pytest.raises(ValueError, match="hash"):
        load(out)


def test_non_catboost_heads_are_not_saved(tmp_path):
    from sklearn.linear_model import LogisticRegression

    head = TabHead("ranking", estimator=lambda s: LogisticRegression(max_iter=200))
    agent = TabAgent(make_card(), head=head).fit(make_table(20))
    with pytest.raises(TypeError):
        agent.save(tmp_path / "h")


def test_loading_an_artifact_never_imports_foreign_code_by_default(tmp_path, monkeypatch):
    import pandas as pd

    from tabagent import DecisionSpec, DecisionTable
    from tabagent.tabschema import Feature, FeatureCard

    (tmp_path / "shoutfeats.py").write_text("def shout(df, col):\n    return df[col].str.upper()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = DecisionSpec("single", "g", "y")
    feats = [
        Feature("t", "state", "text"),
        Feature("loud", "state", "cat", "python:shoutfeats:shout", {"col": "t"}),
    ]
    texts = ["book a flight to rome", "play a song", "book a flight to oslo", "play a jazz song"] * 5
    df = pd.DataFrame({"g": range(20), "t": texts, "y": [t[0] for t in texts]})
    agent = TabAgent(FeatureCard("x", spec, feats), head=TabHead("single", iterations=10))
    out = agent.fit(DecisionTable(df, spec)).save(tmp_path / "h")
    with pytest.raises(PermissionError, match="shoutfeats"):
        load(out)
    assert load(out, trust_code=True).score(df.drop(columns=["y"])).shape == (20,)
