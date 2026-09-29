import hashlib

import pytest

from tabagent import datasets
from tabagent.datasets._core import Source, download


def test_registry_lists_datasets_with_licenses():
    table = datasets.available()
    if not table.empty:
        assert {"decision", "license", "redistribution", "paper"} <= set(table.columns)


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
