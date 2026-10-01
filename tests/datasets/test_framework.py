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


def test_prefetch_fetches_only_missing_files_concurrently(tmp_path, monkeypatch):
    import hashlib
    import threading

    import tabagent.datasets._core as core

    monkeypatch.setenv("TABAGENT_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("TABAGENT_SOURCE_MIRROR", raising=False)
    files = {f"f{i}.txt": f"content {i}".encode() for i in range(6)}
    manifest = {
        rel: {"path": rel, "url": f"https://example.org/{rel}", "sha256": hashlib.sha256(b).hexdigest()}
        for rel, b in files.items()
    }
    src = Source("t", "bench", "https://example.org", "rev", "rel1", "MIT", "redistributable", "cite")
    monkeypatch.setattr(Source, "manifest", property(lambda self: manifest))
    cached = src.local_dir() / "f0.txt"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(files["f0.txt"])
    fetched, threads = [], set()

    def fake_fetch(url, dest, sha256=None):
        fetched.append(url.rsplit("/", 1)[1])
        threads.add(threading.get_ident())
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(files[dest.name])
        return dest

    monkeypatch.setattr(core, "_fetch", fake_fetch)
    src.prefetch(sorted(files), download=True, workers=4)
    assert sorted(fetched) == [f"f{i}.txt" for i in range(1, 6)]  # the cached file is not fetched again
    assert all((src.local_dir() / rel).read_bytes() == b for rel, b in files.items())
    src.prefetch(sorted(files), download=False)  # a no-op without download
    assert len(fetched) == 5
