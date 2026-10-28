"""What a contributor reads about the standard is what the repository enforces.

The contribution standard lives twice: in prose (CONTRIBUTING.md, AGENTS.md, GOVERNANCE.md, the pull request
template, the issue forms, the Wanted page) and in code (scripts/pr_gate.py, scripts/check_pr_body.py, the
workflows, the Makefile, pyproject.toml and the repository's `main` ruleset). Prose drifts silently: nothing
fails when a document names a file that moved, a number the checks no longer use, or a section that was
renamed. Found while writing these tests:

* the contribution-proposal form linked CONTRIBUTING.md#ai-assisted-contributions after that section was
  renamed "AI coding agents", so the link opened the top of the page.

Each test reads a document and checks what it states against the code: every path it names exists; every
number of the bar is the number the checks use; every suppression it shows passes the check it describes;
every link and anchor resolves; every command and make target exists; every waiver label names a real gate;
and the checks it says are required to merge are jobs that run on pull requests. Tests marked `network` also
compare the documents with GitHub: the live ruleset, the labels and the open issues.
"""

import importlib.util
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = "dolev31/TabAgent"

pytestmark = pytest.mark.skipif(
    not (ROOT / "CONTRIBUTING.md").exists() or not (ROOT / ".github").exists(),
    reason="the contributor docs are not in the sdist",
)

#: The documents that state the standard.
STANDARD = (
    "CONTRIBUTING.md",
    "AGENTS.md",
    "GOVERNANCE.md",
    ".github/pull_request_template.md",
    "docs/project/wanted.md",
)
#: Every surface a contributor is pointed at, for links and anchors.
SURFACES = (
    *STANDARD,
    "README.md",
    "CLAUDE.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "scripts/pr_gate.py",
    "scripts/check_pr_body.py",
    *(str(p.relative_to(ROOT)) for p in sorted((ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.yml"))),
)
#: Files of a saved head or of the reader's own machine: named in the docs, never in the repository.
NOT_IN_THE_REPOSITORY = {
    "extract.py",
    "head.json",
    "manifest.json",
    "card.yaml",
    "model.cbm",
    "report.md",
    "report.json",
    "discovery.md",
    "extractor_tests.json",
    "body.md",
}
#: The site serves these root files as pages under docs/project/ (docs/hooks/project_files.py).
SITE_PAGES = {
    "contributing.md": "CONTRIBUTING.md",
    "governance.md": "GOVERNANCE.md",
    "security.md": "SECURITY.md",
}


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _words(path: str) -> str:
    return " ".join(_text(path).split())


def _code(text: str) -> list[str]:
    """The inline code spans and the fenced code blocks of a Markdown document."""
    fenced = re.findall(r"^```[^\n]*\n(.*?)^```", text, re.M | re.S)
    inline = re.findall(r"`([^`\n]+)`", re.sub(r"^```.*?^```", "", text, flags=re.M | re.S))
    return fenced + inline


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # a dataclass resolves its module through sys.modules
    spec.loader.exec_module(module)
    return module


def _slug(heading: str) -> str:
    """GitHub's anchor for a Markdown heading."""
    heading = re.sub(r"[`*_]", "", heading.strip().lower())
    return re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")


def _anchors(path: str) -> set[str]:
    text = re.sub(r"^```.*?^```", "", _text(path), flags=re.M | re.S)
    return {_slug(h) for h in re.findall(r"^#+\s+(.+?)\s*$", text, re.M)}


def _pr_jobs() -> set[str]:
    """The names GitHub gives the checks of every workflow job that runs on a pull request."""
    names = set()
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        triggers = workflow.get(True) or workflow.get("on")  # YAML 1.1 reads the `on:` key as True
        if "pull_request" not in (triggers if isinstance(triggers, (dict, list)) else [triggers]):
            continue
        for job_id, job in workflow["jobs"].items():
            name = job.get("name", job_id)
            matrix = (job.get("strategy") or {}).get("matrix") or {}
            axes = [v for k, v in matrix.items() if k not in ("include", "exclude")]
            if len(axes) == 1:
                names |= {f"{name} ({value})" for value in axes[0]}
            else:
                names.add(name)
    return names


def _required_to_merge() -> list[str]:
    """The checks CONTRIBUTING.md says the `main` ruleset requires."""
    paragraph = _words("CONTRIBUTING.md").split("**Required to merge.**", 1)[1].split("It also", 1)[0]
    return re.findall(r"`([^`]+)`", paragraph.split("green:", 1)[1])


def _github(path: str):
    """GitHub's REST API, with $GITHUB_TOKEN when it is set. Without one GitHub allows 60 requests an hour; a
    test that hits that limit is skipped with the reason, since the limit says nothing about the documents."""
    headers = {"User-Agent": "tabagent-tests", "Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    request = urllib.request.Request(f"https://api.github.com/{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        if e.code in (403, 429) and e.headers.get("X-RateLimit-Remaining") == "0":
            pytest.skip(reason="GitHub's API rate limit; set GITHUB_TOKEN or rerun later")
        raise


# --- what the documents name exists


def test_every_path_the_standard_names_exists():
    """A backticked path in a document of the standard is a file or folder of the repository (or of the
    package, for a path written from inside it), unless it is a file of a saved head or of the reader's
    machine."""
    search = [ROOT, ROOT / "src" / "tabagent"]
    missing = []
    for doc in STANDARD:
        for token in re.findall(r"`([^`\n]+)`", _text(doc)):
            token = token.strip()
            if not re.fullmatch(r"[\w.-]*\w[\w./-]*", token) or token.startswith("http"):
                continue
            if "/" not in token and not re.search(r"\.(py|md|ya?ml|toml|json|txt)$", token):
                continue
            if token in NOT_IN_THE_REPOSITORY or token == "__init__.py":
                continue
            found = any((base / token).exists() for base in search) or (
                "/" not in token and any(next(base.rglob(token), None) for base in search)
            )
            if not found:
                missing.append(f"{doc}: {token}")
    assert not missing, missing


def test_every_make_target_and_command_the_docs_show_exists():
    """`make X` in a code span is a Makefile target; `tabagent X` is a subcommand of the CLI. The Wanted page
    is left out: it names commands that do not exist yet, by design."""
    from tabagent.cli import build_parser

    targets = set(re.findall(r"^([a-z][\w-]*):", _text("Makefile"), re.M))
    subparsers = next(a for a in build_parser()._actions if a.__class__.__name__ == "_SubParsersAction")
    commands = set(subparsers.choices)
    unknown = []
    for doc in (*STANDARD[:-1], "README.md"):
        for code in _code(_text(doc)):
            unknown += [
                f"{doc}: make {t}" for t in re.findall(r"\bmake ([a-z][\w-]*)", code) if t not in targets
            ]
            unknown += [
                f"{doc}: tabagent {c}"
                for c in re.findall(r"(?:^|[;&|]\s*|\$\s+|uv run\s+)tabagent ([a-z][\w-]*)", code, re.M)
                if c not in commands
            ]
    assert not unknown, unknown


def test_every_relative_link_and_anchor_on_a_contributor_surface_resolves():
    """Relative links resolve against their document's folder; an anchor into CONTRIBUTING.md, GOVERNANCE.md
    and the others (written as a relative link, a GitHub URL or plain text in a form or a script) is a
    heading there."""
    broken = []
    for doc in SURFACES:
        text, base = _text(doc), (ROOT / doc).parent
        links = re.findall(r"\]\(([^)#\s]*)(?:#([^)\s]+))?\)", text) if doc.endswith(".md") else []
        for target, anchor in links:
            if target.startswith(("http", "mailto:")):
                continue
            if target and not doc.startswith("docs/") and not (base / target).exists():
                broken.append(f"{doc}: {target}")
            if anchor and not target and anchor not in _anchors(doc):
                broken.append(f"{doc}: #{anchor}")
        refs = re.findall(
            rf"(?:https://github\.com/{REPO}/blob/main/|(?<![\w/.]))([A-Za-z_]+\.md)#([\w-]+)", text, re.I
        )
        for name, anchor in refs:
            root_file = SITE_PAGES.get(name, name) if doc.startswith("docs/") else name
            if not (ROOT / root_file).exists() or root_file not in {
                "CONTRIBUTING.md",
                "GOVERNANCE.md",
                "SECURITY.md",
                "AGENTS.md",
                "README.md",
            }:
                continue
            if anchor not in _anchors(root_file):
                broken.append(f"{doc}: {name}#{anchor}")
    assert not broken, broken


# --- the numbers are the checks' numbers


def test_the_numbers_of_the_bar_are_the_numbers_the_checks_use():
    gate = _load("pr_gate")
    contributing, agents = _words("CONTRIBUTING.md"), _words("AGENTS.md")
    assert (
        f"At least {gate.COVERAGE_MIN}% of the library lines you add or change run in the tests"
        in contributing
    )
    floor = re.search(r"^fail_under\s*=\s*(\d+(?:\.\d+)?)", _text("pyproject.toml"), re.M).group(1)
    assert f"the whole library stays above {float(floor):g}%" in contributing
    for text in (contributing, agents):
        assert f"{gate.MAX_LINES} changed lines" in text
    assert f"({gate.MAX_KB} KB a file)" in contributing
    assert f'"--maxkb={gate.MAX_KB}"' in _text(".pre-commit-config.yaml")

    ci = yaml.safe_load(_text(".github/workflows/ci.yml"))
    pythons = ci["jobs"]["test"]["strategy"]["matrix"]["python"]
    assert f"CI runs the rest on Python {pythons[0]} to {pythons[-1]}" in contributing
    pyproject = _text("pyproject.toml")
    assert f'requires-python = ">={pythons[0]}"' in pyproject
    assert re.findall(r"Programming Language :: Python :: (3\.\d+)", pyproject) == pythons


def test_the_architecture_caveat_is_the_one_the_reproduction_page_measured():
    """GOVERNANCE.md and AGENTS.md quote the CPU caveat docs/reproducing.md measured."""
    measured = re.search(
        r"up to about ([\d.]+) points between CPU architectures \(x86-64", _words("docs/reproducing.md")
    )
    assert measured
    for doc in ("GOVERNANCE.md", "AGENTS.md"):
        assert f"up to about {measured.group(1)} points between x86-64 and arm64" in _words(doc), doc


# --- the examples pass the checks


def test_the_suppressions_contributing_shows_pass_the_gate_and_their_bare_forms_fail(tmp_path, monkeypatch):
    """Every suppression CONTRIBUTING.md gives as the way to write one, and every fix the gate itself
    suggests, passes the gate's hygiene check; the same suppression without its code or reason fails it."""
    gate = _load("pr_gate")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    bullet = _words("CONTRIBUTING.md").split("**Every suppression says why:**", 1)[1].split("- **", 1)[0]
    shown = re.findall(r"`([^`]+)`", bullet)
    shown += re.findall(r"\(`(#[^`]+)`\)", " ".join(why for _, why, *_ in gate.HYGIENE))
    lines = []
    for example in shown:
        if example.startswith("#"):
            lines.append(f"x = 1  {example}")
        elif example.startswith("TODO("):
            lines.append(f"x = 1  # {example}: the follow-up")
        elif example.startswith("pytest.mark.skip("):
            lines += [f"@{example}", "def test_x():", "    return None"]
    assert len(lines) >= 5, shown

    def failures(code: list[str]) -> list[str]:
        (tmp_path / "src").mkdir(exist_ok=True)
        (tmp_path / "src" / "mod.py").write_text('"""A module."""\n' + "\n".join(code) + "\n")
        report = gate.Report()
        gate.check_hygiene("src/mod.py", set(range(1, len(code) + 2)), report)
        return [message for _, message in report.failures]

    assert failures(lines) == [], failures(lines)
    bare = ["x = 1  # noqa", "x = 1  # type: ignore", "x = 1  # pragma: no cover", "x = 1  # TODO fix it"]
    bare += ["@pytest.mark.skip", "def test_y():", "    return None"]
    assert len(failures(bare)) == 5, failures(bare)


def test_the_agent_fields_agents_md_names_are_the_ones_the_check_reads():
    check = _load("check_pr_body")
    template, agents = _text(".github/pull_request_template.md"), _words("AGENTS.md")
    for field in ("Written with", "Read in full by"):
        assert f"{field}:" in template and f'"{field}:"' in agents
        assert f'"{field}"' in _text("scripts/check_pr_body.py")
    assert "Agent use" in check.SECTIONS and '"Agent use"' in agents


# --- labels, gates and required checks


def test_every_waiver_label_names_a_gate_the_gate_reports():
    gate_source = _text("scripts/pr_gate.py")
    gates = set(re.findall(r'report\.fail\(\s*"(\w+)"', gate_source)) | set(
        re.findall(r'\(\s*"(\w+)",\s*\n?\s*[f]?"', gate_source)
    )
    gates &= {"docs", "types", "tests", "coverage", "changelog", "hygiene", "size"}
    workflow = _text(".github/workflows/pr-quality.yml")
    mapping = dict(re.findall(r"s/\^([\w-]+)\$/(\w+)/p", workflow))
    assert mapping and set(mapping.values()) <= gates, (mapping, gates)
    for label in re.findall(r"maintainer adds `([\w-]+)`", _words("CONTRIBUTING.md")):
        if label != "needs-evidence":  # a label for the reviewer, not a waiver
            assert label in mapping, label


def test_the_checks_required_to_merge_are_jobs_that_run_on_pull_requests():
    required = _required_to_merge()
    assert len(required) >= 5, required
    missing = set(required) - _pr_jobs()
    assert not missing, f"CONTRIBUTING.md requires {sorted(missing)}, which no pull-request job is named"


# --- what GOVERNANCE.md and AGENTS.md promise


def test_the_surfaces_governance_protects_exist():
    """GOVERNANCE.md gives four surfaces a deprecation period; each is a real, versioned surface."""
    from tabagent import TabAgentClassifier, TabAgentRanker, TabAgentRouter, artifact
    from tabagent import api as head_api

    governance = _words("GOVERNANCE.md")
    assert isinstance(artifact.FORMAT_VERSION, int) and isinstance(head_api.HEAD_FORMAT, int)
    assert "`head.json`" in governance and "`manifest.json`" in governance
    serve = _text("src/tabagent/serve.py")
    for route in re.findall(r"`(/[a-z_]+)`", governance):
        assert f'"{route}"' in serve, route
    for estimator in (TabAgentRanker, TabAgentRouter, TabAgentClassifier):
        assert callable(estimator.get_params)


def test_the_extractor_stays_sandboxed_by_default():
    """AGENTS.md: "never switch a default to in-process execution"."""
    import dataclasses
    import inspect

    from tabagent.api import TabAgentHead

    fields = {f.name: f.default for f in dataclasses.fields(TabAgentHead)}
    assert fields["trust_extractor"] is False
    assert inspect.signature(TabAgentHead.load).parameters["trust_extractor"].default is False
    assert "never switch a default to in-process execution" in _words("AGENTS.md")


def test_the_changelog_has_an_unreleased_section_and_the_current_version():
    import tabagent

    headings = re.findall(r"^## (.+)$", _text("CHANGELOG.md"), re.M)
    assert headings[0] == "Unreleased"
    assert any(h.split(" ")[0].rstrip(":") == tabagent.__version__ for h in headings), tabagent.__version__


def test_the_readme_and_the_wanted_page_point_at_the_same_open_items():
    readme = _text("README.md").split("## Contribute", 1)[1].split("\n## ", 1)[0]
    in_readme = set(re.findall(rf"github\.com/{REPO}/issues/(\d+)", readme))
    on_wanted = set(re.findall(rf"github\.com/{REPO}/issues/(\d+)", _text("docs/project/wanted.md")))
    assert in_readme and in_readme <= on_wanted, sorted(in_readme - on_wanted)


# --- against GitHub (network)


@pytest.mark.network
def test_the_live_ruleset_requires_what_contributing_lists():
    rules = _github(f"repos/{REPO}/rules/branches/main")
    kinds = {rule["type"] for rule in rules}
    assert {"pull_request", "required_status_checks", "non_fast_forward", "deletion"} <= kinds
    live = {
        check["context"]
        for rule in rules
        if rule["type"] == "required_status_checks"
        for check in rule["parameters"]["required_status_checks"]
    }
    assert live == set(_required_to_merge())


@pytest.mark.network
def test_the_labels_contributing_names_exist():
    labels = {label["name"] for label in _github(f"repos/{REPO}/labels?per_page=100")}
    named = set(re.findall(r"maintainer (?:may )?adds? `([\w-]+)`", _words("CONTRIBUTING.md")))
    assert named and named <= labels, sorted(named - labels)


@pytest.mark.network
def test_every_issue_the_wanted_page_links_is_open():
    numbers = set(re.findall(rf"github\.com/{REPO}/issues/(\d+)", _text("docs/project/wanted.md")))
    open_now = {str(issue["number"]) for issue in _github(f"repos/{REPO}/issues?state=open&per_page=100")}
    closed = sorted(numbers - open_now, key=int)
    assert not closed, f"the Wanted page links closed issues {closed}; take them off the page"
