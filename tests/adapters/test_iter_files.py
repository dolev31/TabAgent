"""``iter_files`` follows linked log folders, once each, and skips hidden files."""

from __future__ import annotations

import os

import pytest

from tabagent.adapters.common import iter_files


def test_linked_folders_are_read_once_and_cycles_end(tmp_path):
    real = tmp_path / "real"
    (real / "sub").mkdir(parents=True)
    (real / "a.json").write_text("{}")
    (real / "sub" / "b.jsonl").write_text("{}")
    (real / ".hidden.json").write_text("{}")
    (real / "notes.txt").write_text("")
    logs = tmp_path / "logs"
    logs.mkdir()
    try:
        os.symlink(real, logs / "link")
        os.symlink(real, logs / "same_again")  # a second link to the same folder: read once
        os.symlink(logs, real / "sub" / "loop")  # a cycle
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")
    found = iter_files(logs)
    assert sorted(f.name for f in found) == ["a.json", "b.jsonl"]
    assert iter_files(real / "a.json") == [real / "a.json"]
