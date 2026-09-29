import json

import numpy as np
import pytest

from tabagent import TabAgent, TabHead, TabSynth, load

from conftest import make_card, make_table


def test_roundtrip_is_exact_and_pickle_free(tmp_path):
    table = make_table(40)
    agent = TabAgent(
        make_card(), head=TabHead("ranking", preset="tmlr-runs", iterations=50), synth=TabSynth(k=2)
    )
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
    agent = TabAgent(make_card(), head=TabHead("ranking", preset="tmlr-runs", iterations=20)).fit(
        make_table(20)
    )
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
