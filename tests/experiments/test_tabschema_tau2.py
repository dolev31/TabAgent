"""experiments/tabschema_tau2.py: the pinned public files, the onboarding commands and the download step
(nothing is downloaded: the cached source is replaced by fabricated files)."""

from __future__ import annotations

from pathlib import Path

from experiments import tabschema_tau2 as T
from tabagent.datasets.tau2 import OFFICIAL, RESULTS_DIR


def test_the_three_files_are_pinned_with_their_checksums():
    assert OFFICIAL.revision == "cf71a8070269883e38a365ffa85f78f46844c1f4"
    for name in T.FILES:
        entry = OFFICIAL.manifest[RESULTS_DIR + name]
        assert len(entry["sha256"]) == 64 and OFFICIAL.revision in entry["url"], name
    assert [f.split("_")[1] for f in T.FILES] == ["airline", "retail", "telecom"]


def test_each_pipeline_version_is_onboarded_with_the_same_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("TABAGENT_CACHE", str(tmp_path / "cache"))
    lines = T.commands(Path("runs"))
    onboard = [line for line in lines if line.startswith("tabagent onboard")]
    assert len(onboard) == 3 and all(T.SETTINGS in line for line in onboard)
    assert all(str(T.logs_dir()) in line for line in onboard)
    for name, rev in T.VERSIONS.items():
        assert f"git checkout {rev} " in "\n".join(lines)
        assert f"--run-dir {Path('runs') / name / 'run'} " in "\n".join(onboard)
    assert "--n-analyze 20 --accept accept" in T.SETTINGS and "extractor=azure/gpt-6-sol" in T.SETTINGS


def test_download_gathers_the_files_in_one_folder(tmp_path, monkeypatch):
    """The files resolve as every dataset file does (here from a local mirror of fabricated files: nothing is
    downloaded) and are linked into one folder for ``tabagent onboard``."""
    monkeypatch.setenv("TABAGENT_CACHE", str(tmp_path / "cache"))
    mirror = tmp_path / "mirror"
    monkeypatch.setenv("TABAGENT_SOURCE_MIRROR", str(mirror))
    files = mirror / OFFICIAL.id / OFFICIAL.release_id / "files" / RESULTS_DIR
    files.mkdir(parents=True)
    for name in T.FILES:
        (files / name).write_text("[]")
    out = T.download()
    assert out == T.logs_dir() and sorted(p.name for p in out.iterdir()) == sorted(T.FILES)
    assert all((out / name).read_text() == "[]" for name in T.FILES)
    assert T.download() == out and len(list(out.iterdir())) == 3  # again: the links are replaced
