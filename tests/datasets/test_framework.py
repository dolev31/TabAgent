import hashlib

import pytest

from tabagent import datasets
from tabagent.datasets._core import Source, download


def test_registry_lists_datasets_with_licenses():
    table = datasets.available()
    if not table.empty:
        assert {"kind", "decision", "license", "access", "redistribution", "paper"} <= set(table.columns)
        assert set(table.index) == set(datasets.names())


def test_registry_names_printed_paper_anchors():
    import re

    for name in datasets.names():
        paper = datasets.info(name).paper
        assert paper and not re.search(r"\b(tab|fig|app|sec):", paper), (name, paper)


def test_skill_selection_uses_the_card_behind_the_paper_number():
    from tabagent.tabschema import FeatureCard

    assert datasets.card("skillsbench/skill-select").name == "skillsbench_skill_select_noprior"
    prior = FeatureCard.load("skillsbench_skill_select")  # the variant stays loadable by name
    assert "skill_global_freq" in prior.feature_names
    assert prior.spec == datasets.card("skillsbench/skill-select").spec


def test_access_names_how_the_data_is_obtained():
    assert datasets.info("skillsbench/skill-select").access == "download"
    assert datasets.info("llmrouterbench/tau2").access == "download (accept terms)"
    assert datasets.info("appworld/subagent-routing").access == "local copy (APPWORLD_ROOT)"


def test_unknown_dataset():
    with pytest.raises(KeyError):
        datasets.info("nope/nothing")


def test_source_resolution_prefers_mirror_then_cache(tmp_path, monkeypatch):
    src = Source("demo", "demo", "https://example.org", "rev1", "rel1", "MIT", "redistributable", "demo2026")
    monkeypatch.setenv("TABAGENT_CACHE", str(tmp_path / "cache"))
    with pytest.raises(FileNotFoundError, match="download=True"):
        src.path("a.json")
    cached = tmp_path / "cache" / "sources" / "demo" / "rel1" / "files" / "a.json"
    cached.parent.mkdir(parents=True)
    cached.write_text("{}")
    assert src.path("a.json") == cached
    mirror = tmp_path / "mirror" / "demo" / "rel1" / "files" / "a.json"
    mirror.parent.mkdir(parents=True)
    mirror.write_text("{}")
    monkeypatch.setenv("TABAGENT_SOURCE_MIRROR", str(tmp_path / "mirror"))
    assert src.path("a.json") == mirror


def test_download_verifies_checksum(tmp_path):
    payload = tmp_path / "src.txt"
    payload.write_text("hello")
    url = payload.as_uri()
    good = hashlib.sha256(b"hello").hexdigest()
    assert download(url, tmp_path / "out" / "x.txt", good).read_text() == "hello"
    with pytest.raises(ValueError, match="checksum"):
        download(url, tmp_path / "out" / "y.txt", "0" * 64)


def test_cache_keeps_dotted_names_apart(tmp_path):
    import pandas as pd

    from tabagent.datasets import _read_cache, _write_cache

    a, b = tmp_path / "tau2__gpt-4.1__v2", tmp_path / "tau2__gpt-4.1-mini__v2"
    _write_cache(a, pd.DataFrame({"x": [1]}))
    _write_cache(b, pd.DataFrame({"x": [2, 3]}))
    assert len(_read_cache(a)) == 1 and len(_read_cache(b)) == 2
