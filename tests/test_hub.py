"""Loading saved heads from the Hugging Face Hub (``hf://OWNER/REPO[@REVISION]``).

``huggingface_hub.snapshot_download`` is replaced by a fake that returns a local directory, so these tests
need neither the network nor the ``hub`` extra.
"""

import hashlib
import json
import shutil
import sys
import types

import numpy as np
import pandas as pd
import pytest

import tabagent
from tabagent import DecisionSpec, DecisionTable, TabAgent, TabAgentHead, TabHead, hub
from tabagent.tabschema import Feature, FeatureCard

from conftest import make_card, make_table


@pytest.fixture
def fake_hub(monkeypatch):
    """A ``huggingface_hub`` module whose ``snapshot_download`` serves registered local directories."""
    repos, calls = {}, []

    def snapshot_download(repo_id, revision=None, repo_type=None, **kwargs):
        calls.append({"repo_id": repo_id, "revision": revision, "repo_type": repo_type, **kwargs})
        return str(repos[(repo_id, revision)])

    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return types.SimpleNamespace(repos=repos, calls=calls)


@pytest.fixture
def ranker_repo(tmp_path):
    """A fitted toy ranker saved as a Hub snapshot would hold it (with a model card beside the artifact)."""
    table = make_table(30)
    agent = TabAgent(make_card(), head=TabHead("ranking", preset="paper", iterations=30)).fit(table)
    snap = agent.save(tmp_path / "snapshot")
    (snap / "README.md").write_text("---\nlibrary_name: tabagent\n---\n# toy\n")
    return agent, snap, table.drop_labels()


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("hf://owner/repo", ("owner/repo", None)),
        ("hf://owner/repo@v1.0", ("owner/repo", "v1.0")),
        ("hf://owner/repo@0123abcd", ("owner/repo", "0123abcd")),
        ("hf://owner/repo@refs/pr/1", ("owner/repo", "refs/pr/1")),
    ],
)
def test_hub_uris_are_parsed(uri, expected):
    assert hub.parse(uri) == expected


@pytest.mark.parametrize("uri", ["hf://repo", "hf://owner/", "hf:///repo", "hf://a/b/c", "hf://owner/repo@"])
def test_malformed_hub_uris_are_rejected(uri):
    with pytest.raises(ValueError, match="hf://OWNER/REPO"):
        hub.parse(uri)


def test_load_from_the_hub_downloads_the_snapshot_and_loads_it(fake_hub, ranker_repo):
    agent, snap, blind = ranker_repo
    fake_hub.repos[("owner/toy-head", None)] = snap
    fake_hub.repos[("owner/toy-head", "v1")] = snap
    for loader in (tabagent.load, TabAgent.load):
        again = loader("hf://owner/toy-head")
        assert isinstance(again, TabAgent)
        assert np.array_equal(again.score(blind), agent.score(blind))
    assert tabagent.load("hf://owner/toy-head@v1").decide(blind, k=2).equals(agent.decide(blind, k=2))
    assert [(c["repo_id"], c["revision"], c["repo_type"]) for c in fake_hub.calls] == [
        ("owner/toy-head", None, "model"),
        ("owner/toy-head", None, "model"),
        ("owner/toy-head", "v1", "model"),
    ]


def test_a_tampered_hub_snapshot_is_rejected(fake_hub, ranker_repo, tmp_path):
    _, snap, _ = ranker_repo
    bad = shutil.copytree(snap, tmp_path / "tampered")
    fitted = json.loads((bad / "fitted.json").read_text())
    fitted["prior"]["state"]["values"][0] = 123.0
    (bad / "fitted.json").write_text(json.dumps(fitted))
    fake_hub.repos[("owner/toy-head", None)] = bad
    with pytest.raises(ValueError, match="hash"):
        tabagent.load("hf://owner/toy-head")


def test_trust_code_is_required_from_the_hub_as_from_a_local_path(fake_hub, tmp_path, monkeypatch):
    (tmp_path / "hubfeats.py").write_text("def shout(df, col):\n    return df[col].str.upper()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = DecisionSpec("single", "g", "y")
    feats = [
        Feature("t", "state", "text"),
        Feature("loud", "state", "cat", "python:hubfeats:shout", {"col": "t"}),
    ]
    texts = ["book a flight to rome", "play a song", "book a flight to oslo", "play a jazz song"] * 5
    df = pd.DataFrame({"g": range(20), "t": texts, "y": [t[0] for t in texts]})
    agent = TabAgent(FeatureCard("x", spec, feats), head=TabHead("single", iterations=10))
    fake_hub.repos[("owner/foreign", None)] = agent.fit(DecisionTable(df, spec)).save(tmp_path / "h")
    with pytest.raises(PermissionError, match="hubfeats"):
        tabagent.load("hf://owner/foreign")
    assert tabagent.load("hf://owner/foreign", trust_code=True).score(df.drop(columns=["y"])).shape == (20,)


def _onboarded_head(path):
    """A head as ``onboard`` saves it: the artifact under ``model/``, ``head.json`` and a hashed extractor."""
    spec = DecisionSpec("single", "task", "label", decision=("decision_id",))
    texts = ["book a flight to rome", "cancel my order", "play a jazz song", "book a flight to oslo"] * 8
    df = pd.DataFrame(
        {
            "task": [f"t{i}" for i in range(len(texts))],
            "decision_id": [f"t{i}#0" for i in range(len(texts))],
            "intent": texts,
            "label": [t.split()[0] for t in texts],
        }
    )
    code = "features = {'intent': trajectory['intent']}\n"
    card = FeatureCard("ops", spec, [Feature("intent", "state", "text", "extractor")])
    card.extractor = {"sha256": hashlib.sha256(code.encode()).hexdigest()}
    agent = TabAgent(card, head=TabHead("single", iterations=30)).fit(DecisionTable(df, spec))
    return TabAgentHead(agent, "Classifier", code).save(path)


@pytest.mark.parametrize("loader", [tabagent.load, TabAgentHead.load])
def test_an_onboarded_head_from_the_hub_keeps_the_trust_flags(fake_hub, tmp_path, loader):
    fake_hub.repos[("owner/ops-head", None)] = _onboarded_head(tmp_path / "head")
    sandboxed = loader("hf://owner/ops-head")
    assert isinstance(sandboxed, TabAgentHead) and sandboxed.trust_extractor is False
    trusted = loader("hf://owner/ops-head", trust_extractor=True)
    assert trusted.trust_extractor is True
    assert trusted.decide({"intent": "please book a flight to lima"})["choice"] == "book"


def test_an_extractor_that_does_not_match_the_card_is_rejected_from_the_hub(fake_hub, tmp_path):
    head = _onboarded_head(tmp_path / "head")
    (head / "extract.py").write_text("features = {}\n")
    fake_hub.repos[("owner/ops-head", None)] = head
    with pytest.raises(ValueError, match="hash"):
        tabagent.load("hf://owner/ops-head", trust_extractor=True)


def test_hub_loading_without_the_extra_names_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # the import fails as if not installed
    with pytest.raises(ImportError, match=r"tabagent\[hub\]"):
        tabagent.load("hf://owner/toy-head")


def test_local_paths_never_touch_the_hub(monkeypatch, ranker_repo):
    agent, snap, blind = ranker_repo
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    for path in (snap, str(snap)):
        assert np.array_equal(tabagent.load(path).score(blind), agent.score(blind))
