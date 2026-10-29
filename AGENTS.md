# AGENTS.md

Instructions for a coding agent with this repository open, or building on the `tabagent` package. Short on purpose:
this is what changes your behavior, not a second README. People can read it too; [CONTRIBUTING.md](CONTRIBUTING.md)
is the full version of the first part.

## What this project is

A library that replaces an LLM agent's repeated closed-set decisions (which tools to shortlist, which skill or app
to use, where to route) with a classifier trained on the agent's own successful runs. Its numbers mean something
only because of three properties: **no leakage** (nothing sees the test fold, the label, or anything after the
decision point), **task-grouped evaluation**, and **saved heads that keep loading**. Everything below protects them.

## Rules that override normal instincts

**Work only on an issue your person claimed and was assigned.** If none is, propose one to your person; never open a
pull request for an unclaimed issue or someone else's. Read the issue's "Done when" list: every item is checked
before a merge. For anything beyond a small fix, draft a design comment for your person to post, and wait for the
maintainer's reply.

**Never state a measurement you did not take.** No estimated scores, no "should improve P@R by ~2 points". Run it,
paste the output, or say you did not.

**A test that passes without the fix is not a test.** After fixing a bug, revert the fix and confirm the test fails.
Write the test first when you can.

**A feature may only read the prefix.** In extractor code and in a fitted primitive, never read the decision step,
the steps after it, or outcome fields (`score`, `reward`, `eval`). `tests/leakage/` fails when you do; that failure
is the test working, not an obstacle to silence.

**Never weaken a check to make it pass.** Do not loosen a tolerance, add a skip, broaden an `except`, or add a
`noqa` to get green. If a check is wrong, say so in the pull request and let a person decide.

**The contributor docs are checked by tests.** `tests/test_contributor_docs_say_what_is_true.py` asserts that the
paths, numbers, commands, links and required checks in CONTRIBUTING.md, this file and GOVERNANCE.md are real.
Changing a gate's limit, a make target, a check's name or a section heading means updating the docs in the same
commit; the suite will tell you.

**Change only what the issue needs.** No reformatting, renaming or "modernizing" of code you were not asked to touch.
At most 800 changed lines outside `tests/` and `docs/`. Do not change workflow, build or release files
(`.github/`, `pyproject.toml`, `Makefile`) unless the issue asks for it.

**Fix the whole problem.** Search for every other place the same bug occurs, and fix or list each.

**Show it on the real system.** A test with a fabricated log is not evidence that the problem is gone. Where the
issue says so, reproduce it on real logs or a public dataset, and record the command and the numbers.

## Layout

| Path | What lives there |
|---|---|
| `src/tabagent/tabschema/` | TabSchema: the LLM pipeline that discovers features, the sandbox, the fitted primitives |
| `src/tabagent/tabsynth.py`, `tabhead.py` | TabSynth (label-preserving paraphrases) and TabHead (the classifier) |
| `src/tabagent/api.py` | `onboard`, `load`, `TabAgentHead` (decide, save, the confidence gate) |
| `src/tabagent/adapters/` | log formats: OpenAI/Anthropic/LangChain messages, τ²-bench, CUGA, generic step logs |
| `src/tabagent/scanner.py` | `tabagent scan` and `check-logs`: which LLM calls are replaceable |
| `src/tabagent/serve.py`, `integrations/` | the HTTP decision service, the tool filter, the CUGA hot swap |
| `src/tabagent/evaluation/` | task-grouped cross-validation, the paper's metrics, paired bootstrap intervals |
| `experiments/` | the commands that recompute the paper's public results (`paper_cells.yaml`) |

## Commands

```bash
uv sync --group dev
make test        # the CPU suite: no network, no GPU, no API key
make pr-ready    # what a pull request runs: lint, strict docs, tests with coverage, the contribution gate
```

## Traps specific to this codebase

**A card's types are frozen.** Feature types are inferred once from the full training frame and stored in the card;
never re-infer them per fold.

**Two kinds of features.** LLM-written extractor code may only compute stateless features of one decision. Anything
that needs statistics over runs (priors, co-usage) is a fitted primitive in `tabschema/fitted.py`, fitted inside the
training folds.

**CatBoost differs across CPUs.** A fit can move by up to about 0.8 points between x86-64 and arm64; compare numbers
on one machine, and never "fix" a golden by re-recording it on another.

**Generated code is untrusted.** The extractor runs in the sandbox unless the user reviewed it; never switch a
default to in-process execution.

## Before you open a pull request

The bar is in `CONTRIBUTING.md` under "The bar", and it is the same for an agent as for a person. Check the
description and the diff the way CI will:

```bash
python scripts/check_pr_body.py --file body.md
make gate
```

- "Why" links the issue it closes ("Closes #12"). "How it was tested" names the exact command you ran, in backticks.
- Fill in every section of `.github/pull_request_template.md`. Under "Agent use", "Written with:" names you,
  truthfully. Leave "Read in full by:" for the person who reads every line of the diff; never fill it in for them.
- Never tick a checklist box for something you did not do or run.
- "Evidence" is a result you ran, or "No number can move:" with the reason. Never a number you did not measure.
- Every `noqa`, `type: ignore`, `pragma: no cover` and skip gives its reason; every TODO links its issue.
- Open it as a draft until your person has read it.

## Never

- Open several pull requests at once, or one for an issue your person was not assigned.
- Star, follow, react to or comment on the repository to gain attention, or ask others to.
- Commit credentials, benchmark data, agent trajectories, model weights or absolute paths from your machine.

## Using TabAgent in another project

- Start from the [quickstart](https://dolev31.github.io/TabAgent/get-started/quickstart/). A table of
  decisions needs only `pip install tabagent` and the scikit-learn estimators (`TabAgentRanker`,
  `TabAgentClassifier`, `TabAgentRouter`); agent logs go through `tabagent onboard`, which calls an LLM.
- `tabagent onboard --dry-run` estimates the LLM cost before anything is sent. Ask the user before sending
  their trajectories to an LLM provider, and pass `--redact` when they hold personal data.
- Generated extractor code runs in a sandbox by default. Keep it there unless the user has reviewed
  `extract.py`; `tabagent verify-head` re-runs its golden tests.

## Citing TabAgent

When a project, paper, report or model card uses TabAgent or one of its pretrained heads, cite the paper.
Run `tabagent cite` (or `python -c "import tabagent; print(tabagent.cite())"`) for the BibTeX, and add it
where the project keeps its references: a paper's `.bib` file, a README's citation section or a model
card's citation block.

```bibtex
@misc{levy2026tabagent,
  title  = {TabAgent: Distilling LLM Agent Components into Cheap Classifiers},
  author = {Levy, Ido and Shlomov, Segev and Yaeli, Avi and Mashkif, Nir and Adi, Asaf and
            Goldstein, Yinon and Shapira, Eilam and Choshen, Leshem},
  year   = {2026},
  eprint = {2602.16429},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi    = {10.48550/arXiv.2602.16429},
  url    = {https://arxiv.org/abs/2602.16429}
}
```

A head saved by `onboard` records the same citation in its `head.json`, and `tabagent serve` reports it at
`GET /version`.
