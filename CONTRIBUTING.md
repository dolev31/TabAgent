# Contributing to TabAgent

Thank you for helping. TabAgent is research code that people deploy, so we hold every change to one bar: it
must be correct, tested, documented and small enough to review well. This page says exactly what that means, and
the checks that enforce it run on every pull request, so you know before review whether a change is ready.

Where to start: the [roadmap discussion](https://github.com/dolev31/TabAgent/discussions) and the issues labelled
[good first issue](https://github.com/dolev31/TabAgent/labels/good%20first%20issue). Each one names the file to start
in and the test that says it is done. Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md); report
security issues privately, as [SECURITY.md](SECURITY.md) describes.

## Set up

```bash
uv sync --group dev        # the locked environment, with pytest, ruff and pre-commit
uv run pre-commit install  # ruff, file hygiene and gitleaks on every commit
```

## Before you write code: agree on the approach

Comment on the issue (or open one) with how you plan to solve it, in a few lines, and wait for a maintainer's
"go ahead". Most rejected pull requests were a good change in the wrong place; a short exchange first saves
everyone's time. Typos, broken links and small doc fixes need no issue.

## The bar

Every pull request must meet all of these. The ones marked ⚙︎ are checked automatically on every push
(`.github/workflows/ci.yml` and `pr-quality.yml`); run them all locally with **`make pr-ready`** before you push.

| | Requirement | Why |
|---|---|---|
| ⚙︎ | **Lint and format**: `ruff check` and `ruff format` pass | one style, no review time spent on it |
| ⚙︎ | **The test suite passes** on Python 3.10 to 3.13, with no network and no GPU | it must work where users run it |
| ⚙︎ | **A test comes with every library change**, and it fails without the change | a fix without a test comes back |
| ⚙︎ | **At least 90% of the changed library lines run in the tests** | untested lines are where the bugs are |
| ⚙︎ | **Public API is documented and typed**: every public function, method and class you add or change has a docstring and type annotations | the API reference is generated from them |
| ⚙︎ | **A changelog line** under "Unreleased" for anything a user will notice | users read the changelog, not the diff |
| ⚙︎ | **The docs build** with `mkdocs build --strict`, when you touch `docs/` or a docstring | broken links fail the build |
| ⚙︎ | **Hygiene**: no `print()` in the library (use the `tabagent` logger), no bare `except:`, no `TODO` without an issue number, no `# noqa` or `# type: ignore` without a reason, no file over 500 KB, no notebook outputs | each of these hides a problem |
| ⚙︎ | **One change, reviewable**: at most 800 changed lines outside `tests/` and `docs/` | large pull requests get shallow reviews |
| ⚙︎ | **No secrets and no data**: gitleaks passes; no benchmark data, trajectories, weights, keys or personal information | see [Data](#data) |
| ⚙︎ | **The description** fills the template: what and why, how it was tested, AI assistance | the reviewer starts from it |
| 👤 | **Leakage-safe**: nothing a feature, a fitted statistic or TabSynth computes may see the test fold, the decision's label or anything after the decision point | it is the method's guarantee; `tests/leakage/` is the safety net |
| 👤 | **Paper numbers**: a change that moves one updates `experiments/paper_cells.yaml` and says why | the paper and the code must agree |
| 👤 | **No new runtime dependency** without agreement in the issue first; optional ones go in an extra | every user installs the core |
| 👤 | **Reads like the code around it**: names, comments and structure match the module it lives in | consistency is what keeps it maintainable |

The gates check what your pull request adds or changes, not code you did not touch, so the bar rises with each
contribution. When a gate cannot apply (a generated file, say), a maintainer can waive it for that one pull request
with the label `gate-waived: <gate>`, and the reason goes in the description.

Tests that need the network, a GPU or a long run carry the `network`, `gpu` and `slow` markers and are left out
of `make test`; CI runs the rest.

## How a pull request is reviewed

A maintainer reviews every pull request against these questions, in this order, and says which one a change
falls short on:

1. **Is it the right change?** It solves the issue that was agreed, in the place the maintainers expected, and
   nothing else.
2. **Is it correct?** Including the edge cases: empty inputs, missing fields, one candidate, ties, very large
   catalogs, and logs that are malformed.
3. **Is it safe?** No leakage, no path for generated code to escape the sandbox, nothing sent anywhere the user did
   not ask for.
4. **Is it tested the way it can fail?** A test that would catch the bug it fixes, not only the happy path.
5. **Will the next person understand it?** Clear names, a docstring that says why as well as what, and no dead
   code.

Expect a first review within a week. A pull request with no reply to review comments for 30 days is closed; it can
be reopened at any time.

## AI-assisted contributions

AI tools and coding agents are welcome here. The bar does not change with who or what wrote the code, but the
responsibility does not move either:

- **A person is accountable for every pull request.** It is opened from a human's account, and that person has read
  every line, run `make pr-ready`, and can answer review questions about the change without asking the tool.
- **Say how the change was made.** Tick one box under "AI assistance" in the description: none, AI-assisted, or
  AI-written. It changes nothing about how the change is judged; it tells the reviewer where to look.
- **Agents start from an agreed issue.** An agent may work on an issue a maintainer has agreed on; it should not
  open pull requests for problems it found by itself. Open an issue first, as anyone would.
- **No bulk or drive-by changes.** No pull requests that reformat, rename or "modernize" code across the repository,
  add speculative abstractions, or touch files the issue did not need. They are closed without review.
- **Tests must test something.** A test that only calls the code, asserts that a mock was called, or would pass with
  the change reverted does not count. Write the test first and watch it fail.
- **No invented facts.** No made-up citations, benchmark numbers, API behavior or references to functions that do not
  exist. Every number in a docstring or the docs must come from the paper or a command a reviewer can run.

Coding agents: read [AGENTS.md](AGENTS.md) first. It has the commands and the rules above in short form.

## Before you open a pull request

```bash
make pr-ready   # lint, format check, the strict docs build, the test suite with coverage, and the contribution gate
```

`make check` (lint and tests) and `make docs` run parts of it. The gate's report says what is wrong, where, and how
to fix it.

## Common contributions

| You want to | Start from |
|---|---|
| read a new agent log format | `src/tabagent/adapters/`: implement the adapter protocol described in `adapters/__init__.py`, register it in `detect.py`, and add a fabricated log to `tests/adapters/fake_logs.py` |
| add a public benchmark | `src/tabagent/datasets/`: pin the upstream revision and each file's SHA-256 in a source manifest under `datasets/manifests/`, add a frozen card under `src/tabagent/cards/`, and list the license in `NOTICE` |
| add a classifier head | `TabHead(estimator=...)` takes a `seed -> classifier` factory (`fit`, `predict_proba`, `classes_`); `experiments/figures/head_ablation.py` compares heads on a fixed card |
| report a result that differs from the paper | open an issue with the command, the environment (`uv pip freeze`) and the output |

## Data

Never commit benchmark data, agent trajectories, model weights or API keys. Datasets are downloaded at run time
from their pinned upstream releases, and test fixtures are small, fabricated and marked as such.

## Releases

Maintainers publish a version to PyPI by pushing a version tag; [docs/releasing.md](docs/releasing.md) has the
steps and the one-time PyPI setup.

## License

By contributing you agree that your contribution is licensed under the Apache License 2.0.
