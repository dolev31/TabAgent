# Contributing to TabAgent

Thank you for your interest. TabAgent replaces an LLM agent's repeated closed-set decisions with a classifier
trained on the agent's own successful runs, and people deploy it, so the most valuable contributions make that more
accurate, safer, cheaper or easier to adopt: a log format TabAgent can read, a feature primitive, a safer way to
swap a head into an agent, a public benchmark. [GOVERNANCE.md](GOVERNANCE.md) says who decides and how a number is
settled; everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md); report a vulnerability privately,
as [SECURITY.md](SECURITY.md) describes.

Looking for something to work on? The [roadmap](https://github.com/dolev31/TabAgent/discussions/13) lists three
tracks, and every [good first issue](https://github.com/dolev31/TabAgent/labels/good%20first%20issue) names the file
to start in and what "done" means.

## Set up

```bash
git clone https://github.com/dolev31/TabAgent && cd TabAgent
uv sync --group dev        # the locked environment, with pytest, ruff, diff-cover and pre-commit
uv run pre-commit install  # ruff, file hygiene and gitleaks on every commit
make test                  # the CPU suite: no network, no GPU, no API key
```

The default suite is hermetic. Tests that need the network, a GPU or a long run carry the `network`, `gpu` and
`slow` markers and are left out; CI runs the rest on Python 3.10 to 3.13. Before you push, **`make pr-ready`** runs
what a pull request runs: lint and format, the strict docs build, the suite with coverage, and the contribution gate.

## Before you write code: agree on the approach

Comment on the issue (or open one, with the *Contribution proposal* form) with how you plan to solve it, and wait
for a maintainer's go. For a new decision kind, a change to the saved-head format, a metric, or anything over about
400 lines, this is required: the design is agreed before the code is written. Typos, broken links and small doc
fixes need no issue.

## The bar

A person who will maintain your code after you move on reads every pull request, so the bar is the same for
everyone: a first-time contributor, a maintainer, or an AI coding agent working for either. Two checks
(`.github/workflows/pr-quality.yml`) test what a machine can before a reviewer opens the diff, and say exactly what
is missing. Run them yourself first:

```bash
python scripts/check_pr_body.py --file body.md   # the description
make gate                                        # the diff: scripts/pr_gate.py with coverage
```

### Every pull request

- **One logical change, with tests.** A refactor and a feature are two pull requests. At most 800 changed lines
  outside `tests/` and `docs/`; split larger work.
- **The [template](.github/pull_request_template.md) is filled in:** why, what changed, how it was tested, evidence,
  agent use, and the checklist. Explain *why*; the diff already shows what.
- **New behavior and every bug fix come with a test that fails without the change.** A change under
  `src/tabagent/` with nothing changed under `tests/` fails the gate, unless a maintainer adds `no-new-test` to a
  pure refactor. At least 90% of the library lines you add or change run in the tests, and the whole library stays
  above 93% (it was 94.6% when this was set).
- **Public API is documented and typed.** Every public function, method and class you add or change has a docstring
  and type annotations, and a new module under `src/tabagent/` opens with a docstring that says what it is for. The
  [API reference](https://dolev31.github.io/TabAgent/reference/) is generated from them.
- **Anything a user sees gets a line in `CHANGELOG.md` under Unreleased.** A maintainer adds `no-changelog` when
  nothing changed for users.
- **Every suppression says why:** `# noqa: BLE001 - reason`, `# type: ignore[arg-type] - reason`,
  `# pragma: no cover - reason`, `pytest.mark.skip(reason="...")`. An `except` whose body is only `pass` says in a
  comment why the error is safe to ignore. A TODO links its issue, as `TODO(#12)`. No `print()` in the library: use
  the `tabagent` logger. No bare `except:`.
- **No number moves unless that is the point**, and then "Evidence" shows it: the command, its output on a public
  dataset, and the paired interval against main. A change that moves a paper number also updates
  `experiments/paper_cells.yaml`. A maintainer adds `needs-evidence` when a description says a number moves but does
  not show it.
- **Never commit credentials, benchmark data, agent trajectories, model weights or absolute paths** from your
  machine. gitleaks and the size limit (500 KB a file) reject most of them; see [Data](#data).

### The method's guarantees

These are what make TabAgent's numbers mean something. A reviewer checks them on every change that touches them:

- **Leakage safety.** Nothing a feature, a fitted statistic, TabSynth or the recipe search computes may see the test
  fold, the decision's label, or anything after the decision point. `tests/leakage/` fails when it does; that
  failure is the test working.
- **Task-grouped evaluation.** Cross-validation folds group by task; every data-dependent stage is fitted inside the
  training folds.
- **The sandbox.** Generated extractor code runs in the sandbox unless the user reviewed it. A change must not add a
  way for it to reach the network, the filesystem outside its scratch directory, or the environment's secrets.
- **Saved heads keep loading.** A change to `head.json`, `manifest.json` or the card format reads the old version
  (see [GOVERNANCE.md](GOVERNANCE.md#breaking-changes)).

### Done, by kind of change

| Change | Done when |
|---|---|
| Log adapter | It implements the protocol in `src/tabagent/adapters/__init__.py`, is registered in `detect.py` and listed in its `FORMATS` text, and a small fabricated log in `tests/adapters/fake_logs.py` makes `tabagent check-logs` find its decisions, candidates and labels. |
| Feature primitive | It is registered in `FITTED` (`src/tabagent/tabschema/fitted.py`), reads only the training rows and the prefix, and a test in `tests/leakage/test_leakage.py` shows scrambling the test fold's labels leaves it unchanged. |
| Head or estimator | It follows the scikit-learn contract (`clone`, `get_params`, `fit`/`predict_proba`); `experiments/figures/head_ablation.py` compares it on a fixed card with paired intervals. |
| Benchmark | It loads at a pinned revision with each file's SHA-256 in `datasets/manifests/`, its license is in `NOTICE`, a frozen card is under `src/tabagent/cards/`, and its tests run offline. |
| Bug fix | The test that reproduces the bug is in the same pull request and fails on main. |
| Serving or the CLI | The serving tests cover the new path, including a deferral and a timeout; the CLI page of the docs (generated from argparse) builds. |
| Docs | `mkdocs build --strict` passes, and every command shown was run. |

## AI coding agents

Agents are welcome, and the bar does not move for them.

- **A person opens the pull request, reads every line of the diff, and names themselves** under "Read in full by:".
  That person answers review comments, without asking the tool.
- **"Written with:" names the agent, or "none".** Using an agent is not a mark against a pull request; hiding it is.
  A maintainer may add `agent-assisted`, for the record.
- **An agent's confidence is not evidence.** A test that was run is, and so is a command's output.
- **An agent works from an agreed issue.** It does not open pull requests for problems it found by itself, and it
  never makes bulk changes: no reformatting, renaming or "modernizing" across files the issue did not need.
- [AGENTS.md](AGENTS.md) gives an agent this repository's rules. Point yours at it.

## Below the bar

The checks fail and say what is missing; fix the description or the diff and they run again. A pull request a
reviewer still finds below the bar (no reason for the change, untested behavior, a diff its author cannot explain)
is closed with a link to this section. That is a statement about the pull request, not about you: open it again
once it meets the bar. A pull request whose description does not match its diff, or one of many opened at once, is
closed without review. A pull request with no reply to review comments for 30 days is closed; it can be reopened at
any time.

## How a pull request is reviewed

A maintainer reviews every pull request against these questions, in this order, and says which one a change falls
short on:

1. **Is it the right change?** It solves the issue that was agreed, in the place the maintainers expected, and
   nothing else.
2. **Is it correct?** Including the edge cases: empty inputs, missing fields, one candidate, ties, very large
   catalogs, and logs that are malformed.
3. **Is it safe?** The method's guarantees above hold, and nothing is sent anywhere the user did not ask for.
4. **Is it tested the way it can fail?** A test that would catch the bug it fixes, not only the happy path.
5. **Will the next person understand it?** Clear names, a docstring that says why as well as what, and no dead code.

Expect a first review within a week.

## Disagreeing about a number

Show a failing test or a command anyone can run, and the argument is over, whoever wrote the code. For a number in
the paper or the docs that does not reproduce, open a
[result dispute](https://github.com/dolev31/TabAgent/issues/new?template=result_dispute.yml). See
[GOVERNANCE.md](GOVERNANCE.md).

## Data

Never commit benchmark data, agent trajectories, model weights or API keys. Datasets are downloaded at run time
from their pinned upstream releases, and test fixtures are small, fabricated and marked as such.

## Releases

Maintainers publish a version to PyPI by pushing a version tag; [docs/releasing.md](docs/releasing.md) has the
steps and the one-time PyPI setup.

## License

Contributions are accepted under the [Apache License 2.0](LICENSE). Do not paste in code you did not write or that
carries an incompatible license; TabAgent vendors no third-party source.
