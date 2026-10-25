"""The contribution bar: checks that a pull request's own changes must pass, on top of `make check`.

    python scripts/pr_gate.py                    # against origin/main (what CI runs)
    python scripts/pr_gate.py --base upstream/main --coverage coverage.xml

The repository-wide checks (ruff, the test suite on four Python versions, the strict docs build, gitleaks)
hold for every file. These gates hold for what the pull request adds or changes, so the bar rises with each
contribution without asking anyone to rewrite code they did not touch:

  docs        every new or changed public function, method and class in src/tabagent has a docstring
  types       every new or changed public function in src/tabagent annotates its parameters and its return
  tests       a change to src/tabagent comes with a change under tests/
  coverage    the changed lines of src/tabagent are covered by the test suite (--coverage, diff-cover)
  changelog   a change to src/tabagent's behavior adds a line under "Unreleased" in CHANGELOG.md
  hygiene     no print() in the library, no bare `except:`, no TODO/FIXME without an issue number, no new
              file over 500 KB, no notebook outputs, no `# noqa` or `# type: ignore` without a reason after it
  size        the pull request changes at most 800 lines outside tests/ and docs/ (split larger work)

Each failure says what is wrong, where, and how to fix it. A maintainer can waive a gate for one pull request
with the label `gate-waived: <gate>` and a reason in the description; the waiver is visible in the review.
Standard library only, plus git.
"""

from __future__ import annotations

import argparse
import ast
import io
import os
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = "src/tabagent/"
MAX_LINES = 800
MAX_KB = 500
COVERAGE_MIN = 90
#: changes to these paths are not library behavior (no changelog line needed)
NO_CHANGELOG = re.compile(r"^src/tabagent/(cards|datasets/manifests)/")


@dataclass
class Report:
    failures: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, gate: str, message: str) -> None:
        self.failures.append((gate, message))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def changed_files(base: str) -> list[str]:
    """Files added, copied, modified or renamed on this branch since it left ``base``."""
    out = git("diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD")
    return [f for f in out.splitlines() if f]


def added_lines(base: str, path: str) -> set[int]:
    """Line numbers (in the new file) that this branch added or changed in ``path``."""
    lines: set[int] = set()
    for hunk in re.finditer(
        r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", git("diff", "-U0", f"{base}...HEAD", "--", path), re.M
    ):
        start, count = int(hunk.group(1)), int(hunk.group(2) or 1)
        lines.update(range(start, start + count))
    return lines


def _public(name: str) -> bool:
    return not name.startswith("_")


def _defs(tree: ast.Module):
    """(qualified name, node) of every public def and class reachable through public classes."""

    def walk(body, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and _public(node.name):
                yield prefix + node.name, node
                if isinstance(node, ast.ClassDef):
                    yield from walk(node.body, f"{prefix}{node.name}.")

    yield from walk(tree.body, "")


def _touched(node: ast.AST, lines: set[int]) -> bool:
    end = getattr(node, "end_lineno", node.lineno)
    return any(node.lineno <= n <= end for n in lines)


def check_api(path: str, lines: set[int], report: Report) -> None:
    """Docstrings and annotations on the public definitions this branch adds or changes."""
    source = (ROOT / path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:  # ruff reports it too; say where
        report.fail("hygiene", f"{path}:{e.lineno}: does not parse ({e.msg})")
        return
    for name, node in _defs(tree):
        header_end = node.body[0].lineno - 1 if node.body else node.lineno
        header = set(range(node.lineno, header_end + 1))
        if not (lines & header or (_touched(node, lines) and not isinstance(node, ast.ClassDef))):
            continue
        if not ast.get_docstring(node):
            report.fail(
                "docs",
                f"{path}:{node.lineno}: `{name}` is public and has no docstring. Say what it does, "
                "its parameters and what it returns, in the style of the module around it.",
            )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [
                a
                for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                if a.arg not in ("self", "cls")
            ]
            for star in (node.args.vararg, node.args.kwarg):
                if star is not None:
                    args.append(star)
            missing = [a.arg for a in args if a.annotation is None]
            if missing or node.returns is None:
                what = ", ".join(f"`{m}`" for m in missing) + (
                    " and the return" if node.returns is None else ""
                )
                report.fail(
                    "types",
                    f"{path}:{node.lineno}: `{name}` needs type annotations for {what.lstrip(' and')}.",
                )


#: (pattern, why, path prefix it applies to, "code" or "comment": what part of the line the pattern reads)
HYGIENE = [
    (
        re.compile(r"^\s*print\("),
        "print() in the library: use the `tabagent` logger (logging.getLogger(__name__))",
        LIB,
        "code",
    ),
    (
        re.compile(r"^\s*except\s*:"),
        "a bare `except:` also catches KeyboardInterrupt and SystemExit: name the exception",
        "",
        "code",
    ),
    (
        re.compile(r"^#\s*(TODO|FIXME|XXX)\b(?!.*#\d+)"),
        "a TODO without an issue number: open an issue and write `# TODO(#123): ...`",
        "",
        "comment",
    ),
    (
        re.compile(r"^#\s*noqa(?::\s*[A-Z0-9, ]+)?\s*$"),
        "`# noqa` without a reason: add why after it (`# noqa: E402  the import must follow X`)",
        "",
        "comment",
    ),
    (
        re.compile(r"^#\s*type:\s*ignore(\[[^\]]*\])?\s*$"),
        "`# type: ignore` without a reason: add why after it",
        "",
        "comment",
    ),
]


def _comments(source: str) -> dict[int, str]:
    """The comment on each line of a Python source (line number -> the comment, from its ``#``), read with the
    tokenizer, so a ``#`` inside a string is never taken for one."""
    out: dict[int, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                out[tok.start[0]] = tok.string
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass  # unparsable source: check_api reports it
    return out


def check_hygiene(path: str, lines: set[int], report: Report) -> None:
    full = ROOT / path
    if full.stat().st_size > MAX_KB * 1024 and path != "uv.lock":
        report.fail(
            "hygiene",
            f"{path}: {full.stat().st_size // 1024} KB, over the {MAX_KB} KB limit. Large files belong "
            "in a GitHub Release or are downloaded at run time (docs/datasets.md).",
        )
    if path.endswith(".ipynb") and '"outputs": [\n     {' in full.read_text(
        encoding="utf-8", errors="replace"
    ):
        report.fail(
            "hygiene",
            f"{path}: the notebook carries cell outputs. Clear them (Kernel → Restart & Clear Output).",
        )
    if not path.endswith(".py"):
        return
    source = full.read_text(encoding="utf-8", errors="replace")
    comments = _comments(source)
    for n, text in enumerate(source.splitlines(), 1):
        if n not in lines:
            continue
        code = text.split(comments[n], 1)[0] if n in comments else text
        for pattern, why, scope, where in HYGIENE:
            target = comments.get(n, "") if where == "comment" else code
            if path.startswith(scope) and pattern.search(target):
                report.fail("hygiene", f"{path}:{n}: {why}.")


def check_coverage(base: str, xml: str, report: Report) -> None:
    """Changed library lines must be covered by the tests (diff-cover over a coverage.py XML report)."""
    cmd = [
        sys.executable,
        "-m",
        "diff_cover.diff_cover_tool",
        xml,
        f"--compare-branch={base}",
        "--include",
        "src/tabagent/*",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if "No module named" in result.stderr:
        report.fail("coverage", "diff-cover is not installed: `uv sync --group dev` installs it.")
        return
    out = result.stdout
    found = re.search(r"Coverage:\s*([\d.]+)%", out)
    if found is None:  # no library line changed, or none the coverage report knows
        report.notes.append("coverage: no changed library lines to measure")
        return
    pct = float(found.group(1))
    if pct >= COVERAGE_MIN:
        report.notes.append(f"coverage: {pct:.0f}% of the changed library lines run in the tests")
        return
    missing = [ln.strip() for ln in out.splitlines() if "Missing lines" in ln]
    report.fail(
        "coverage",
        f"{pct:.0f}% of the changed library lines run in the tests (at least {COVERAGE_MIN}% needed). "
        "Add tests that reach:\n      " + "\n      ".join(missing[:12]),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--base", default=os.environ.get("GATE_BASE", "origin/main"), help="the branch the PR targets"
    )
    ap.add_argument("--coverage", help="a coverage.py XML report of the test run, for the coverage gate")
    ap.add_argument(
        "--waive", default=os.environ.get("GATE_WAIVE", ""), help="comma-separated gates a maintainer waived"
    )
    args = ap.parse_args(argv)
    waived = {g.strip() for g in args.waive.split(",") if g.strip()}

    report = Report()
    files = changed_files(args.base)
    lib = [f for f in files if f.startswith(LIB) and f.endswith(".py")]
    for path in files:
        lines = added_lines(args.base, path)
        check_hygiene(path, lines, report)
        if path in lib:
            check_api(path, lines, report)

    if lib and not any(f.startswith("tests/") for f in files):
        report.fail(
            "tests",
            f"{len(lib)} library file(s) changed and no test did. Add or update a test under tests/ "
            "that fails without your change.",
        )
    behavior = [f for f in lib if not NO_CHANGELOG.match(f)]
    if behavior:
        diff = git("diff", f"{args.base}...HEAD", "--", "CHANGELOG.md")
        unreleased = git("show", "HEAD:CHANGELOG.md").split("## Unreleased", 1)[-1].split("\n## ", 1)[0]
        if (
            "CHANGELOG.md" not in files
            or not any(ln.startswith("+-") or ln.startswith("+  ") for ln in diff.splitlines())
            or not unreleased.strip()
        ):
            report.fail(
                "changelog",
                'the library changed and CHANGELOG.md has no new line under "## Unreleased". '
                "Add one sentence a user would understand: what changed for them.",
            )

    stat = git("diff", "--numstat", f"{args.base}...HEAD")
    size = sum(
        int(a) + int(d)
        for a, d, f in (ln.split("\t") for ln in stat.splitlines())
        if a != "-" and not f.startswith(("tests/", "docs/")) and f != "uv.lock"
    )
    if size > MAX_LINES:
        report.fail(
            "size",
            f"{size} changed lines outside tests/ and docs/ (limit {MAX_LINES}). Split the work into "
            "pull requests that each do one thing; reviewers can then check each one properly.",
        )
    if args.coverage:
        check_coverage(args.base, args.coverage, report)

    active = [(g, m) for g, m in report.failures if g not in waived]
    print(f"pr-gate: {len(files)} changed file(s) against {args.base}")
    for note in report.notes:
        print(f"  ok    {note}")
    for gate in sorted(waived):
        print(f"  waived {gate} (by a maintainer's label)")
    if not active:
        print("  pass  every gate")
        return 0
    for gate, message in active:
        print(f"  FAIL  [{gate}] {message}")
    print(f"\n{len(active)} problem(s). The bar and the reasons for it: CONTRIBUTING.md#the-bar")
    return 1


if __name__ == "__main__":
    sys.exit(main())
