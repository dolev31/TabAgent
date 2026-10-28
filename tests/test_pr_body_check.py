"""The pull request description check fails a description below the bar, and only those.

`scripts/check_pr_body.py` is what `.github/workflows/pr-quality.yml` runs on every pull request. A check that
passes everything is a template nobody fills in; one that fails a good pull request teaches contributors to
ignore it. So both directions are tested: the unfilled template fails, a filled one passes, and each rule
fails its own gap."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_pr_body", ROOT / "scripts" / "check_pr_body.py")
check = importlib.util.module_from_spec(_spec)
sys.modules["check_pr_body"] = check
_spec.loader.exec_module(check)

TEMPLATE = (ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
#: the template's checklist, every box ticked
TICKED = "\n".join(
    ln.replace("- [ ]", "- [x]") for ln in TEMPLATE.split("## Checklist", 1)[1].strip().splitlines()
)

FILLED = """\
## Why

`tabagent scan` cannot read OpenTelemetry GenAI traces, so agents that only export OTel spans
cannot be scanned before onboarding. Closes #1.

## What changed

A new adapter, `OTelGenAIAdapter`, groups spans into runs by trace id; `detect_adapter`
recognizes OTel exports.

## How it was tested

`tests/adapters/test_otel.py::test_scan_ranks_otel_components` fails before the change and
passes after it. Ran `make pr-ready`.

## Evidence

No number can move: the adapter only reads logs; no head, metric or default changes.

## Agent use

Written with: Claude Code
Read in full by: @octocat

## Checklist

"""
FILLED += TICKED + "\n"


def _replace(body: str, section: str, text: str) -> str:
    start = body.index(f"## {section}\n")
    end = body.find("\n## ", start + 1)
    end = len(body) if end < 0 else end
    return body[:start] + f"## {section}\n\n{text}\n" + body[end:]


def test_a_filled_description_passes():
    assert check.problems(FILLED) == []


def test_the_unfilled_template_fails_on_every_section():
    found = check.problems(TEMPLATE)
    for name in ("Why", "What changed", "How it was tested", "Evidence"):
        assert f'"## {name}" is empty' in found, name
    assert any("Read in full by" in p for p in found)
    assert any("is not ticked" in p for p in found)


def test_the_checker_and_the_template_agree():
    """Every section the checker requires is a heading of the template, in the same order, and every
    required box is a checklist line of the template."""
    headings = re.findall(r"^##\s+(.+?)\s*$", TEMPLATE, re.M)
    assert headings == list(check.SECTIONS)
    boxes = re.findall(r"^\s*-\s+\[ \]\s+(.+)$", TEMPLATE, re.M)
    for fragment in check.REQUIRED_BOXES:
        assert any(fragment in b for b in boxes), fragment


@pytest.mark.parametrize(
    ("section", "text", "expected"),
    [
        ("Why", "Fixes the scan.", '"## Why" is a few words'),
        ("Why", "TBD", 'says only "TBD"'),
        ("Evidence", "No number can move:", "says no number can move but not why"),
        ("How it was tested", "see diff", 'says only "see diff"'),
        ("Agent use", "Written with: none\nRead in full by:", "to name a person by GitHub @handle"),
        ("Agent use", "Written with:\nRead in full by: @octocat", '"Written with:" filled in'),
    ],
)
def test_each_gap_fails_with_its_own_reason(section, text, expected):
    found = check.problems(_replace(FILLED, section, text))
    assert any(expected in p for p in found), found


def test_an_unticked_or_removed_box_fails():
    unticked = FILLED.replace("- [x] I read every line of the diff", "- [ ] I read every line of the diff")
    assert any("I read every line of the diff" in p and "not ticked" in p for p in check.problems(unticked))
    removed = "\n".join(ln for ln in FILLED.splitlines() if "make pr-ready" not in ln)
    assert any("make pr-ready" in p and "was removed" in p for p in check.problems(removed))


def test_a_real_number_counts_as_evidence():
    body = _replace(
        FILLED,
        "Evidence",
        "SkillsBench P@R 0.796 -> 0.812 [+0.004, +0.028], `python -m experiments.reproduce`",
    )
    assert check.problems(body) == []


def test_main_reads_the_environment(monkeypatch, capsys):
    monkeypatch.setenv("PR_BODY", FILLED)
    assert check.main([]) == 0
    monkeypatch.setenv("PR_BODY", "")
    assert check.main([]) == 1
    assert "::error::" in capsys.readouterr().out


def test_claude_md_mirrors_agents_md():
    """Claude Code reads CLAUDE.md and other agents read AGENTS.md: the two must give the same rules."""
    agents, claude = ROOT / "AGENTS.md", ROOT / "CLAUDE.md"
    if not claude.exists():  # the sdist ships AGENTS.md only
        pytest.skip(reason="no CLAUDE.md (not in the sdist)")
    assert claude.read_text(encoding="utf-8") == agents.read_text(encoding="utf-8")


def test_the_workflow_runs_both_checks_safely():
    """pr-quality runs the description and diff checks, on `pull_request` only (a fork's code gets a read-only
    token), and never expands the description into a shell command."""
    wf = (ROOT / ".github" / "workflows" / "pr-quality.yml").read_text(encoding="utf-8")
    assert "scripts/check_pr_body.py" in wf and "scripts/pr_gate.py" in wf
    triggers = yaml.safe_load(wf)[True]  # YAML 1.1 reads the `on:` key as True
    assert set(triggers) == {"pull_request"}
    assert "PR_BODY: ${{ github.event.pull_request.body }}" in wf
    # the description reaches a step only through `env:`, never inside a `run:` script
    for step in wf.split("- name:")[1:]:
        run = step.split("run:", 1)[1] if "run:" in step else ""
        assert "github.event.pull_request.body" not in run
