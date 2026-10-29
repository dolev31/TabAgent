<!--
Fill in every section. The "pr-quality" check reads this description and fails while a section is empty, still
holds only these comments, or a required box is unticked. The bar, and why it exists:
https://github.com/dolev31/TabAgent/blob/main/CONTRIBUTING.md#the-bar
-->

## Why

<!-- The problem this solves, or the question it answers. Link the issue you claimed ("Closes #12"); a small
fix without one needs a maintainer's `no-issue` label. The diff shows what changed; say why. -->

## What changed

<!-- Two to five lines a reviewer reads before the diff. Name any public function, CLI flag, card field or
endpoint that is new or behaves differently. -->

## How it was tested

<!-- Name the test that fails without this change and passes with it, and the exact command you ran, in
backticks. A docs-only change: say how you checked it (for example `mkdocs build --strict`). -->

## Evidence

<!-- Evidence from the real system, not only from tests with fakes. If this can move a quality, latency or cost
number: the command and its output, on a public dataset, against the current main (for a head: `tabagent.cross_validate` or `experiments/reproduce.py`, with paired intervals). If it
cannot, write one line that starts with "No number can move:" and say why. -->

## Agent use

<!-- AI coding agents are welcome, under the same bar. Say which you used, or "none", and name the person who read
every line of the diff and answers for it. Keep the two lines below and fill them in. -->

Written with:
Read in full by:

## Checklist

- [ ] This PR makes one logical change, and its issue agreed on the approach.
- [ ] `make pr-ready` passes (lint, the strict docs build, the tests with coverage, and the contribution gate).
- [ ] New behavior and every bug fix have a test that fails without the change.
- [ ] Every new `noqa`, `type: ignore` or `pragma: no cover` names its code and gives a reason.
- [ ] `CHANGELOG.md` has a line under Unreleased, or nothing a user sees changed.
- [ ] I read every line of the diff, and I can explain each one.
