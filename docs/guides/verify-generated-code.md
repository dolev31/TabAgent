---
title: "Verify the generated code"
description: "The four value checks the release adds to TabSchema, the golden tests shipped with a head, and how to review the extractor before you serve it."
lead: "TabSchema writes the extraction code. These checks, and your own review, are what make it safe to run."
kind: Guide
needs: "an onboarded head"
---

# Verify the generated code

TabSchema's own checks (paper App. D.1) review the extraction *code*. The deterministic checks refuse forbidden
calls, unknown fields and outcome fields, and the CodeValidator, an LLM, reads the code and a few of its
outputs. Code can pass both and still compute wrong values: a regex for a reservation code that matches any
six-letter uppercase word, or a set joined in an order that changes from one run to the next. The release adds
four checks of the values. **None of them is in the paper.** Each records what it found in `report.md` (and in
`discovery.md` in a saved head), and each can be turned off in `TabSchemaConfig`. On the command line,
`tabagent onboard` takes `--no-verify-extractor`, `--no-verify-adapter`, `--no-audit` and `--prune-harmful`.
Reading the clock (`datetime.now()`, `date.today()`, through any `import` alias; a `.now()` method of the
code's own objects is fine) or calling `hash()`/`id()` is refused like a forbidden call (the CodeExtractor
retries with the reason): a feature must be a function of the prefix alone.

**1. Compile-time verification** (`verify=True`, `verify_samples=20`). After the quality gates, the compiled
extractor runs again in the sandbox:

| check | what it does | when it fails |
|---|---|---|
| determinism | runs the code on **every** onboarding decision point again, in fresh interpreters and in a shuffled order, and compares with the compile step's run | the column is dropped: it cannot be reproduced at serving time. Typical causes are a set joined in iteration order (Python seeds string hashing per process), a counter kept across calls, and the clock. |
| outcome invariance | adds the adapter's outcome fields (`reward`, `success`, `evaluation`, ...) to 20 sampled prefixes, once with success values and once with failure values | the column is dropped. No decision-time prefix holds an outcome, so a column that reacts to one would leak the label wherever a prefix carried it: code that reads every top-level field, or the length of `json.dumps(trajectory)`. |
| future invariance (`verify_adapter=True`) | before any LLM call (and in `--dry-run`), changes each sampled run *after* the decision (every later step and the decision step itself perturbed: every string, tool names, arguments and ids included, and every number; the outcome fields changed) and, separately, cuts the run right after the decision; a fresh copy of the adapter rebuilds the prefix, which must stay the same | **onboarding stops** with an error naming the adapter and the first path that changed: the adapter puts post-decision data into prefixes (a cut one step too late, say), and no column can be trusted |
| robustness | runs the code on the shortest onboarding prefixes, on a prefix with no steps, and on prefixes without one task field or one optional step field | reported, not dropped. Raising is the extractor's contract, and a head defers a decision whose extraction fails. |

The samples come from the coverage sampler: new tasks first, then uncovered positions in the run (first,
middle, last decision), prefix lengths and labels, so a decision that recurs in a run is checked beyond its
first occurrence. The future check is a check of the adapter, and it changes the *run*, never the prefix: by
contract every step an extractor sees precedes the decision, so steps appended to a prefix would look like a
longer history and flag legitimate history features (the last tool called, the number of earlier calls). The
perturbation keeps only the step-name key and the `role`/`type` keys, so the adapter still finds the decision;
a fallback keeps the structural keys too, for formats whose ids decide how a message is read. The raw decision
step is located directly for step logs, and for chat logs (whose built-in adapters describe the decision with
a synthetic `target_step`, not a raw message) as the shortest run prefix in which the adapter finds the
decision. The built-in adapters never trigger it; when no decision point can be checked (no runs, an adapter
that cannot read changed runs), the report says *skipped*, not passed. Fields a log writes at the top of a run
after it ends (a final answer, a turn count) do not change with the steps, so this check cannot see them: keep
them out of prefixes with `task_keys` (the adapters' default whitelist does). An adapter whose prefixes
legitimately hold part of the decision step itself, written before the decision (its prompt, say), names those
keys in a `decision_input_keys` attribute, which the check leaves unchanged in the decision step.
`verify_adapter=False` (`--no-verify-adapter`) skips only this check, and `verify=False` skips it with the
others; both say so in a warning and in the report. Columns that read outcome fields are
dropped rather than tolerated by stripping those fields before extraction: adapters already keep outcomes out
of prefixes, and a column that reads unknown top-level fields would change with any field a live run carries.
The appended `context_text` and `last_step_text` ignore outcome fields. The checks of the extractor make three
sandbox runs: one on the sample, one over every decision point (an interpreter per 50), and one for the variants
and edge cases. An extractor with a per-candidate function
([Per-candidate features](../concepts/per-candidate-features.md)) is checked on each decision's candidates too:
determinism with the decision points and each one's candidates shuffled (a value that depends on the candidates
scored before it is dropped), outcome invariance per (decision, candidate), and robustness on three candidates
per edge prefix.

**2. The OutputAuditor** (`audit=True`, `audit_samples=12`, `audit_max_incorrect=0.25`). An LLM stage (role
`auditor`; `stage_models={"auditor": ...}` routes it) reads 12 coverage-sampled decision points that the
FeatureAnalyzer did not see. When fewer are left, it reads them all and then analyzed ones, and the report
counts those. For each point it gets the clipped prefix, between data delimiters it is told never to follow
as instructions, and, for every LLM-written feature, its plan (description, extraction source, computation)
and the value the extractor produced. Like the FeatureAnalyzer's, these prefixes go to your LLM provider
(after `redact=`). It answers `correct`, `incorrect` or `cannot_tell`, with a one-line reason; an answer that
repeats the template or hedges counts as `cannot_tell`, which never counts. A feature is dropped when it was
judged on at least 3 points, at least 2 of them incorrect, **and** the one-sided 95% Wilson lower bound of its
incorrect rate exceeds 25% (on 12 judged points, that takes 6 incorrect), so a few judge errors do not drop a
correct feature. The audit never empties a card: when it would drop every LLM-written feature, it drops none
and says so prominently. Every verdict is in `audit.json` in the run directory. The prompt
(`prompts/output_auditor.txt`) is the release's, not the paper's, and its hash is pinned in the tests and
recorded in every card. The calls are metered, budgeted, cached in the run directory and counted by
`--dry-run`. A content filter or an unusable answer loses only that decision point. When no point gets a
usable answer, or the budget is spent, the audit is skipped with a note and every feature is kept.

**3. Usefulness** (`usefulness=True`, `prune_harmful=False`, `usefulness_max_decisions=400`,
`usefulness_max_rows=3000`). In the spirit of CAAFE (Hollmann et al., 2023), each extractor feature is scored
by drop-feature importance: the columns that implement one planned feature (one per tool in a multi-hot, say)
are dropped together, since the others would make up for any one of them. A cheap head (CatBoost with 50
iterations) is cross-validated on the onboarding rows with 3 task-grouped folds, once with every feature and once
without each feature, on the same folds; one more row drops every LLM-written feature at once, which answers
whether discovery helped at all. The report gives the change in the decision's primary metric (P@R for
rankings, F1 for multi-label, accuracy for single-class decisions) with a 95% bootstrap interval over tasks and
a paired t-test p-value over tasks, Holm-adjusted over the features (the all-features row is a question of its
own, not adjusted). A feature is *useful* (or *harmful*) when that p-value is below 0.05 and the change positive
(or negative), and *neutral* otherwise. Expect most single features of a large card to be neutral: dozens of
overlapping features make up for any one of them, and the correction for testing all of them is strict, so a
feature can be neutral although its uncorrected interval excludes zero. On the τ²-bench next-tool-call card
(64 features, 208 columns), every LLM-written feature together added P@R +0.043 [+0.013, +0.078] in this
estimate. The estimate is a report by default. `prune_harmful=True` repeats it with a
second seed and drops the LLM-written features that are harmful under both, but never all of them; since the
same rows then chose the features, the report notes that the onboarding cross-validation is optimistic
(`cross_validate_discovery`, which prunes inside each training fold, is not). The estimate uses only the
trajectories discovery is given; in-fold discovery runs it only when it prunes. It fits on at most 400 decision
points and at most 3,000 rows (decision points x candidates; at least 30 decision points are kept), which bounds
its runtime to about half a minute on a laptop CPU for a dozen features (2 s per feature, roughly), and it
leaves out the embedding similarity, which is slow and not what it measures. Results are in
`usefulness.json`.

**4. Golden tests shipped with the extractor** (`goldens=8`). The compile step records up to 8
coverage-sampled snapshots of onboarding prefixes with the features the extractor computes on them, plus the
determinism and outcome-invariance checks and the (masked) record of the compile-time checks, in
`extractor_tests.json` next to `extract.py`. `head.save` writes the file, and `tabagent.load` keeps it
(`head.extractor_tests`; a broken file is ignored with a warning and never stops loading). Before deploying a
head, or in CI:

```bash
tabagent verify-head ./my_head                    # exit status 1 when a check fails
tabagent verify-head ./my_head --require-goldens  # fail a head without golden prefixes (goldens=0)
tabagent verify-head hf://OWNER/REPO --sandbox docker
```

```python
report = tabagent.verify_head("./my_head")  # also hf://OWNER/REPO
assert report.passed, report
```

It checks that the model artifact matches its manifest (which must list every file), that the card pins the
hash of `extract.py` and the code matches it, and that the tests cover every extractor feature of the card; it
then re-runs the goldens in the sandbox twice (the second time in reverse order, and with outcome fields
added). A head without golden prefixes (`goldens=0`, or saved without the file) passes with a warning unless
`--require-goldens`; a head that recorded fewer goldens than requested because snapshots made the extractor
raise (masking or clipping removed what the code reads) fails, and the onboarding report says so. An invalid tests file, or a path that is not a head, is an error (exit status 2). For a head
from the Hub it warns that the subprocess sandbox is not a security boundary: use `--sandbox docker` for code
you do not trust. `tabagent.load` checks the hash too and fails closed: a head whose card pins no hash for
`extract.py` (saved by an earlier version) is refused unless you pass `allow_unpinned=True` (`tabagent serve
--allow-unpinned`, `ToolFilter(..., allow_unpinned=True)`) after reviewing the code; it then loads with a
warning, and saving the head again pins the hash. These hashes detect modified or corrupted files; they are not
a signature.

The snapshots are trajectory text. They are taken after `redact=`, and the release masks what it can recognize:
API keys and tokens (OpenAI, Anthropic, Stripe, Hugging Face, GitHub, Slack, AWS, Google, JWTs, bearer tokens,
private-key blocks), credential pairs (`password=...`, `"api_key": "..."`, and any string under a key that
names a credential; never a number, and never a token count such as `total_tokens`), URL user info and
credential query parameters, the local part of e-mail addresses (`masked@<domain>`) and home directories
(`/Users/<name>`, `/home/<name>`, `/root`, `C:\Users\<name>`, UNC paths, `~name`), in values and in keys.
Strings are clipped to 2,000 characters, and a snapshot that cannot be brought under 20,000 characters (middle
steps, then the largest lists, then shorter strings) is not recorded.
Everything else the checks report (examples of differing values, error lines, the auditor's reasons and the
dropped reasons) is masked the same way. Masking is a safety net, not a guarantee: **a saved head contains
snippets of its onboarding trajectories**, so share it as you would share them, or set `goldens=0`. For a
decision whose extractor defines the per-candidate function, each snapshot also records three of the
decision's candidates with their catalog entries (`candidate["spec"]`): check what your catalog holds before
you upload a head to the Hub.

**What the checks cost.** One onboarding on public τ²-bench logs (the GPT-4.1 agent's runs, `ToolShortlist`,
604 decision points, 14 LLM-written features; Claude Opus 5.5, with GPT-6 Sol writing the code) cost $2.92 in
total, of which the audit's 12 calls were about $0.39 (13%) and 40 s. The verification took 1.4 s (three
sandbox runs on 670 prefixes, every decision point included), the usefulness estimate 32 s (210 decision
points, 2,988 rows), and `verify-head` 2 to 3 s. Every rebuilt prefix matched (no leak), no column was
nondeterministic, and every feature was kept. The auditor found two substring bugs, below the drop bound and
recorded in `audit.json`: an intent category read "reservation" as "book" on 2 of 12 points, and a
flight-change flag fired on "exchange" (it contains "change") on 1 of 12.

## Review the extractor

The saved head contains `extract.py`, the code TabSchema compiled, and its golden tests
(`extractor_tests.json`; `tabagent verify-head` re-runs them). For a ranking or multi-label decision it may
define `extract_candidate_features(trajectory_data, candidate)` next to the paper's entry point (`head.json`
then records `"candidate_function": true`); `head.candidate_features_for(prefix, candidates)` shows what it
computes for a live decision. `compile.json` in the run directory lists the kept columns per decision and per
candidate, and the accepted features that got no column, with the reason (merged into another feature,
declined by the CodeExtractor, or not produced). Read the code before serving with
`--trust-extractor` (in-process); without that flag it runs in the sandbox, by default in warm worker
processes that start once and then add well under a millisecond per decision (see
[Deployment](production-runbook.md#the-warm-extractor-pool)). All modes give the same features: in-process, the code
sees a JSON copy of the prefix, as in the sandbox, and cannot change the caller's. Build live prefixes with the
adapter's `live_prefix` (chat logs, CUGA and generic JSON), which
builds them exactly as in training.
