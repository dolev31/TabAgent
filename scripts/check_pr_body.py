"""Refuse a pull request whose description is below the bar.

    python scripts/check_pr_body.py --file body.md [--labels no-issue]

Run by `.github/workflows/pr-quality.yml` on every pull request that is not a draft, and by hand before you
open one. The bar is written out in CONTRIBUTING.md under "The bar"; this script checks the description, and
`scripts/pr_gate.py` checks the diff, so a reviewer's time goes to the part no machine can check.

Checks the description (from `--file`, `-` for stdin, or `$PR_BODY`):

- every section of `.github/pull_request_template.md` is there and filled in, not left as the template's
  comments or a placeholder such as "TBD";
- "Why" is a reason, not a title, and links the issue it closes (`#12` or an issue URL): a contribution is
  claimed before it is written. A maintainer waives the link for a small fix with the `no-issue` label;
- "How it was tested" names the exact command or test, in backticks;
- "Evidence" is a result, or a line that starts "No number can move:" and gives the reason;
- "Agent use" says what wrote the change and names, by GitHub handle, the person who read every line of it;
- every box in the checklist is ticked.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

GUIDE = "https://github.com/dolev31/TabAgent/blob/main/CONTRIBUTING.md#the-bar"

#: The template's headings, in order. tests/test_pr_body_check.py checks that the two agree.
SECTIONS = ("Why", "What changed", "How it was tested", "Evidence", "Agent use", "Checklist")

#: A fragment of each checklist line that must be there, ticked.
REQUIRED_BOXES = (
    "one logical change",
    "make pr-ready",
    "fails without the change",
    "names its code and gives a reason",
    "CHANGELOG.md",
    "I read every line of the diff",
)

#: What a section holds when nobody filled it in.
PLACEHOLDERS = {"todo", "tbd", "wip", "n/a", "na", "none", "-", "...", "see diff", "see title", "see above"}

#: A "Why" shorter than this is a title, not a reason.
MIN_WHY_WORDS = 8

#: The label with which a maintainer waives the linked issue, for a small fix.
NO_ISSUE = "no-issue"

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.M)
_BOX = re.compile(r"^\s*[-*]\s+\[([ xX])\]\s+(.+)$", re.M)
_HANDLE = re.compile(r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})")
_ISSUE = re.compile(r"(?<![\w/&])#\d+\b|/issues/\d+\b")
_CODE_SPAN = re.compile(r"`[^`\n]+`")


def sections(body: str) -> dict[str, str]:
    """Each ``## `` heading's text, with the template's comments taken out."""
    text = _COMMENT.sub("", body.replace("\r\n", "\n"))
    heads = list(_HEADING.finditer(text))
    out = {}
    for i, head in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out[head.group(1)] = text[head.end() : end].strip()
    return out


def _field(section: str, name: str) -> str | None:
    """What follows ``name:`` on its line, or None when the line is missing."""
    match = re.search(rf"^\s*{re.escape(name)}:[ \t]*(.*)$", section, re.M)
    return match.group(1).strip() if match else None


def _is_placeholder(text: str) -> bool:
    text = text.strip().lower()
    return text in PLACEHOLDERS or text.strip(".!") in PLACEHOLDERS | {""}


def problems(body: str, labels: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """What the description is missing, one sentence each; empty when it meets the bar. ``labels`` are the
    pull request's labels: ``no-issue`` waives the linked issue."""
    found = []
    parts = sections(body or "")
    for name in SECTIONS:
        if name not in parts:
            found.append(f'the description has no "## {name}" section (start from the pull request template)')
        elif not parts[name]:
            found.append(f'"## {name}" is empty')
        elif _is_placeholder(parts[name]):
            found.append(f'"## {name}" says only "{parts[name]}"')

    why = parts.get("Why", "")
    if why and not _is_placeholder(why):
        if len(why.split()) < MIN_WHY_WORDS:
            found.append('"## Why" is a few words; say what problem this solves, and for whom')
        if NO_ISSUE not in labels and not _ISSUE.search(why):
            found.append(
                '"## Why" links no issue; claim one first and link it ("Closes #12"), '
                f"or ask a maintainer to add {NO_ISSUE} for a small fix"
            )

    tested = parts.get("How it was tested", "")
    if tested and not _is_placeholder(tested) and not _CODE_SPAN.search(tested):
        found.append(
            '"## How it was tested" names no command or test; put the exact command you ran, '
            "or the test's name, in backticks"
        )

    evidence = parts.get("Evidence", "")
    if evidence.lower().startswith("no number can move"):
        reason = evidence.split(":", 1)[1].strip() if ":" in evidence else ""
        if len(reason.split()) < 3:
            found.append('"## Evidence" says no number can move but not why')

    agent = parts.get("Agent use", "")
    if agent:
        written = _field(agent, "Written with")
        reader = _field(agent, "Read in full by")
        if not written:
            found.append('"## Agent use" needs "Written with:" filled in (the agent you used, or "none")')
        if reader is None or not _HANDLE.search(reader):
            found.append('"## Agent use" needs "Read in full by:" to name a person by GitHub @handle')

    boxes = _BOX.findall(parts.get("Checklist", ""))
    for fragment in REQUIRED_BOXES:
        match = [ticked for ticked, label in boxes if fragment in label]
        if not match:
            found.append(f'the checklist line with "{fragment}" was removed')
        elif not any(t in "xX" for t in match):
            found.append(f'the checklist line with "{fragment}" is not ticked')
    return found


def main(argv: list[str] | None = None) -> int:
    """Check a pull request description; exit 1 and say what is missing when it is below the bar."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", help="the description, or - for stdin (default: $PR_BODY)")
    parser.add_argument("--labels", default="", help="the pull request's labels, comma-separated")
    args = parser.parse_args(argv)
    if args.file == "-":
        body = sys.stdin.read()
    elif args.file:
        body = Path(args.file).read_text(encoding="utf-8")
    else:
        body = os.environ.get("PR_BODY", "")
    found = problems(body, {label.strip() for label in args.labels.split(",") if label.strip()})
    for p in found:
        print(f"::error::{p}")
    print(f"The bar, and how to meet it: {GUIDE}" if found else "the description meets the bar")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
