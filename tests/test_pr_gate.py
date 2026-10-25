"""The contribution gate (scripts/pr_gate.py) on a throwaway repository: it fails a sloppy change for every
reason it names, and passes a documented, typed, tested change with a changelog line."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[1] / "scripts" / "pr_gate.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository with a main branch holding one library module and a changelog, and a gate bound to it."""
    (tmp_path / "src/tabagent").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/tabagent/mod.py").write_text('"""A module."""\n')
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## Unreleased\n\n## 0.1.0\n\n- first\n")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "root")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    _git(tmp_path, "switch", "-q", "-c", "feature")
    spec = importlib.util.spec_from_file_location("pr_gate", GATE)
    gate = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "pr_gate", gate)  # a dataclass resolves its module through sys.modules
    spec.loader.exec_module(gate)
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    return tmp_path, gate


def _commit(repo: Path) -> None:
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "change")


def test_a_sloppy_change_fails_every_gate(repo, capsys):
    root, gate = repo
    (root / "src/tabagent/mod.py").write_text(
        '"""A module."""\n\n\ndef helper(x, y=None):\n    print("debug", x)\n    try:\n        return x / y\n'
        "    except:\n        return None  # TODO make it better\n"
    )
    _commit(root)
    assert gate.main(["--base", "main"]) == 1
    out = capsys.readouterr().out
    for g in ("docs", "types", "tests", "changelog", "hygiene"):
        assert f"[{g}]" in out, g
    assert "print()" in out and "bare `except:`" in out and "TODO without an issue number" in out


def test_a_good_change_passes(repo, capsys):
    root, gate = repo
    (root / "src/tabagent/mod.py").write_text(
        '"""A module."""\n\n\ndef half(x: float) -> float:\n    """Half of ``x``."""\n    return x / 2\n'
    )
    (root / "tests/test_mod.py").write_text("def test_half():\n    assert True\n")
    log = root / "CHANGELOG.md"
    log.write_text(log.read_text().replace("## Unreleased\n\n", "## Unreleased\n\n- `half`: halves.\n"))
    _commit(root)
    assert gate.main(["--base", "main"]) == 0, capsys.readouterr().out


def test_a_waived_gate_is_reported_not_failed(repo, capsys):
    root, gate = repo
    (root / "src/tabagent/mod.py").write_text(
        '"""A module."""\n\n\ndef half(x: float) -> float:\n    """Half of ``x``."""\n    return x / 2\n'
    )
    (root / "tests/test_mod.py").write_text("def test_half():\n    assert True\n")
    _commit(root)  # no changelog line
    assert gate.main(["--base", "main"]) == 1
    assert gate.main(["--base", "main", "--waive", "changelog"]) == 0
    assert "waived changelog" in capsys.readouterr().out


def test_old_code_is_not_held_to_the_new_bar(repo):
    """Only what the branch adds or changes is checked: an untouched undocumented function stays allowed."""
    root, gate = repo
    (root / "src/tabagent/old.py").write_text('"""Old."""\n\n\ndef legacy(a):\n    return a\n')
    _commit(root)
    _git(root, "switch", "-q", "main")
    _git(root, "merge", "-q", "--ff-only", "feature")
    _git(root, "switch", "-q", "-c", "next")
    (root / "src/tabagent/mod.py").write_text('"""A module, edited."""\n')
    _commit(root)
    report = gate.Report()
    gate.check_api("src/tabagent/old.py", set(), report)
    assert report.failures == []


def test_hygiene_reads_comments_not_strings(repo):
    """A `# TODO` or `print(` inside a string is text, not a problem; a real comment without an issue is."""
    root, gate = repo
    (root / "src/tabagent/mod.py").write_text(
        '"""A module."""\n\nNOTE = "# TODO make it better, print(x)"\n'
        "x = 1  # TODO(#12): an issue is linked\ny = 2  # TODO no issue\n"
    )
    report = gate.Report()
    gate.check_hygiene("src/tabagent/mod.py", {1, 2, 3, 4, 5}, report)
    assert [m for _, m in report.failures] == [
        "src/tabagent/mod.py:5: a TODO without an issue number: open an issue and write `# TODO(#123): ...`."
    ]
