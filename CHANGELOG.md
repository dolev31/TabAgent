# Changelog

Changes that affect users, newest first. The GitHub release notes list every pull request.

## Unreleased

- **The paper, one click from the home page.** The hero's paper pill slides in a panel with the title, the authors,
  links to arXiv, the PDF and the citation, and the paper itself. Esc or a click outside closes it; Ctrl/Cmd-click
  still opens arXiv in a new tab.
- **A published contribution bar.** `CONTRIBUTING.md` sets what every pull request must meet, how it is reviewed, and
  the rules for AI-assisted and agent-written changes. `make pr-ready` runs every check locally, and the
  `pr-quality` workflow enforces them on each pull request: docstrings and types on changed public API, a test with
  every library change, 90% of the changed lines covered, a changelog line, and a size limit.
- **A podcast episode.** The landing page plays *Distilling LLM Agent Decisions into Cheap Classifiers*, a
  23-minute audio overview of the paper (AI-generated voices, NotebookLM), with chapters and a transcript page; the
  README links it. The audio is an asset of the media release, like the explainer video.

## 0.1.1 (2026-10-22)

- **Citation at hand.** `tabagent.cite()` and `tabagent cite` print the paper's BibTeX (or APA, or one line of
  text). A head saved by `onboard` records the citation in its `head.json`, `tabagent serve` reports it at
  `GET /version`, and `tabagent onboard` mentions it in one line, on stderr, when it saves a head. Nothing is
  printed at import time. `AGENTS.md` tells coding agents how to work in the repository and how to cite it.
- **An archived, citable release.** Each GitHub release is archived on Zenodo with a DOI; `.zenodo.json`
  gives the record its authors, license and a link to the paper.
- **Documentation site restructured.** Top-level sections (Get started, Guides, Concepts, API reference, Reproduce
  the paper, Changelog, Project), a quickstart, an API reference generated from the docstrings and a command-line
  reference generated from the CLI. The old pages (`onboarding`, `deployment`, `components`, `bring_your_own_data`)
  were split into guides and concept pages with no change to their content; their addresses redirect.

## 0.1.0: first public release (2026-10-21)

The code of the paper *TabAgent: Distilling LLM Agent Components into Cheap Classifiers*
([arXiv:2602.16429](https://arxiv.org/abs/2602.16429)), as the `tabagent` package (Python 3.10 to 3.13,
Apache-2.0).

- **The paper's components**, under the paper's names: TabSchema (LLM-driven feature discovery from successful
  trajectories, with a sandboxed, reviewed extractor), TabSynth (label-preserving paraphrase rows) and TabHead
  (a CatBoost head that scores every candidate of a decision in one pass), with task-grouped cross-validation,
  the paper's metrics and paired bootstrap intervals.
- **Onboarding from agent logs**: `tabagent.onboard` and `tabagent onboard`, with log adapters for OpenAI,
  Anthropic and LangChain tool-call messages, τ²-bench results, CUGA trajectories and generic JSON; `tabagent
  scan`, `components` and `check-logs` before any LLM call; a dry-run cost estimate, a budget, redaction and
  resumable run directories.
- **scikit-learn estimators**: `TabAgentRanker`, `TabAgentClassifier` and `TabAgentRouter`, which pick their
  recipe by nested validation; `tabagent.scorer` and `tabagent.pointwise`.
- **Deployment**: saved heads whose extractor is pinned by hash, `tabagent verify-head`, `tabagent serve` (an
  HTTP service with warm sandboxed workers, readiness, Prometheus metrics and drift monitoring), the OpenAI
  tool filter, the CUGA hot-swap helpers, and `tabagent.load("hf://...")` for heads on the Hugging Face Hub.
- **Reproduction**: loaders for the public benchmarks at pinned, SHA-256-verified revisions (τ²-bench,
  SkillsBench, Mind2Web, LLMRouterBench as an opt-in download, AppWorld sub-agent routing through its official
  CLI), their frozen or hand-authored feature cards and the paper's folds and seeds, and `experiments/` for the
  paper's tables and figures ([docs/reproducing.md](docs/reproducing.md)).

The sections below record what changed during the release's development, after the paper's experiments.

### Per-candidate features

- **The FeatureAnalyzer is told when the decision chooses among candidates.** Its prompt gets an "Additional
  rules" section (appended; the paper's prompt is unchanged): each row is one candidate of one decision, and
  features of a candidate, or of how the trajectory relates to one, may be proposed, computed per candidate
  from its fields, with up to three of the decision's candidates, neutral examples and a label-safety rule.
  Without it, logs whose steps do not list the candidates (most chat tool-calling agents) gave no
  per-candidate proposals, so the compile step dropped the per-candidate columns the CodeExtractor wrote as
  outside the accepted plan. The FeatureAggregator, judge and MetaJudge prompts get a note that such features
  are obtainable and judged for generality like any other. The example candidates of every prompt (the
  analyzer's, the CodeExtractor's, the CodeValidator's) are picked from the run's seed, the decision point and
  the candidates themselves, never from the adapter's order, so an adapter that lists the chosen candidate
  first does not make it an example, and the points of a fixed catalog show different ones. The dry-run
  estimate counts the new sections. **Rerunning a ranking or multi-label onboarding in an existing run
  directory asks the analyzer (and the aggregator and the judges) again**, since their prompts changed; a
  single-class decision's stored answers are replayed as before.
- **A public measurement of the per-candidate pipeline** (`experiments/tabschema_tau2.py`; concepts/per-candidate-features.md,
  "Evidence"). On the public τ²-bench GPT-4.1 logs (next tool call as a ranking decision; 3,226 decisions),
  one onboarding per pipeline version with Claude Opus 5.5 (GPT-6 Sol writing the code): without the
  FeatureAnalyzer's section the card had no per-candidate feature; with it, 19 of 119 proposed features of a
  candidate were accepted and compiled, and removing them lowered P@R by 0.026 [0.014, 0.039], Recall@3 by
  0.016 [0.007, 0.025] and Recall@5 by 0.013 [0.006, 0.021] (paired bootstrap 95% over tasks, 5-fold grouped
  CV, seeds 0 to 2). Against the previous pipeline's card: Recall@3 +0.020 [+0.012, +0.029], Recall@5 +0.012
  [+0.006, +0.020], P@R +0.010 [−0.003, +0.022]. The positional guard did not trigger there. One onboarding
  per version, so the differences include an LLM's run-to-run variation.
- **An extractor may define `extract_candidate_features(trajectory_data, candidate)`** next to the paper's
  entry point, for ranking and multi-label decisions: features of one (decision, candidate) row, such as the
  request's words in the candidate's description. `candidate` is `{"id", "name", "description", "args",
  "spec"}`, built from the candidate alone, so never a label. The CodeExtractor, CodeValidator and
  OutputAuditor prompts get an "Additional rules" section on it (appended; the paper's prompts are
  unchanged), and its columns go through the same checks, gates, verification, golden tests, audit and
  usefulness estimate, over the (decision, candidate) rows. They enter the card as `schema` features with
  `source: extractor_candidate`. The sandbox, the warm workers and the in-process runner run both functions in
  one call per decision, the prefix parsed once; a decision's time limit grows with its candidates
  (`timeout_s * (1 + candidates / 50)`), and a candidate whose call fails loses only its own values.
  `decide`, `tabagent serve`, the CUGA hot swap and the tool filter score candidates exactly as training did.
  `head.json` records `"candidate_function": true`. Extractors without the function, and saved heads, behave
  as before. On by default; see the public measurement above.
- **Candidates carry their catalog entry.** `Candidate.spec` holds the entry the decision offered (a tool's
  JSON schema; a CUGA API's path, method, parameters and response schemas), and the per-candidate function
  reads it as `candidate["spec"]`. The CUGA adapter (its runs and its `catalog=`), the hot swap, and `decide`
  or the service given raw CUGA API entries (recognized by their `api_name`) leave out the `canary_string`
  CUGA writes anew in every run, at an entry's top level only (`parse_catalog(volatile_keys=...)`); other
  catalogs are kept whole. The adapters, the CUGA hot swap, the tool filter, `decide` and the service read it the same way; a
  `RemoteHead` now sends `Candidate` objects as records the service reads back to the same candidates. A saved
  head's golden tests record three candidates per snapshot with their catalog entries: check them before you
  upload a head to the Hub.
- **Within-decision relative features.** `card.with_relative` gives a ranking or multi-label card's numeric
  per-candidate features, context-candidate similarities, candidate prior and co-usage their percentile rank
  among the decision's candidates and their gap to its best (card sources `relative_rank` and
  `relative_gap`; `without_relative` removes them), computed from the decision's own rows. They are opt-in by
  validation: a discovered card never holds them; the recipe search (`auto.select(relative=True)`, from
  `TabSchemaConfig.relative_features`, `--no-relative-features` to skip) fits the winner without them and
  adds them only when the winner with them scores strictly better (`Recipe.relative`), and `head.json`
  records the choice (`relative_features`). On the public bundled cards the effect is neutral: P@R +0.004
  [−0.002, +0.010] on τ²-bench GPT-4.1 tool shortlisting and −0.011 [−0.036, +0.016] on SkillsBench skill
  selection (paired bootstrap 95% intervals, `experiments/relative_features.py`); on the τ²-bench TabSchema
  card above it was mixed, P@R +0.009 [−0.001, +0.019] and Recall@3 −0.008 [−0.018, −0.001]. Cards without
  them behave as before.
- **Positional columns are moved to the per-candidate function.** When the decision has candidates, prefix
  columns that index them by position (`candidate_3_overlap`) are reported, and the repair round asks for the
  feature in the per-candidate function. An indexed enumeration (no history word in its names) is positional
  when one of its columns' texts name the candidates in order, or, with a word for a candidate in its names,
  when its size follows the candidates' count across decisions of different sizes (columns emitted only where
  set included); a constant count, or the first few of a longer enumeration, needs the texts. The columns are
  dropped once the per-candidate function computes the feature, and kept with a compile note otherwise
  (`TabSchemaConfig.positional_guard`, `--no-positional-guard`). On by default; it did not trigger in the
  public τ²-bench onboarding, so it is not yet tested on public data.
- **One form per text feature.** The CodeExtractor is asked to emit a tokenized, bag-of-words or n-gram feature
  once, as the plain text, under the feature's name (unless another feature already emits that text), and
  compile drops a column that restates another column of the same feature in a shorter derived form on every
  row (`drop_redundant_text`); the longer form, plain text, and an ordered sequence next to a set are kept.
- **Accepted features without a column are asked for.** The compile step's repair round also gives the
  CodeExtractor the plans of the accepted features its code does not produce, to add each one or to list it in
  `NOT_EXTRACTED` as merged into another feature or declined, with the reason. `compile.json` and the report
  count them by reason and list the per-decision and per-candidate columns separately.
- **`last_step_text`**: a second fixed function appended to every extractor, the text of the prefix's last step
  (`TabSchemaConfig.last_step_text`, `--no-last-step-text`): the step that handed control to the decision,
  which the clipped context text can dilute among long payloads. On by default; on the public τ²-bench
  onboarding it was neutral to slightly positive (P@R +0.006 [−0.004, +0.015], Recall@3 +0.003 [−0.001,
  +0.008], Recall@5 −0.000 [−0.004, +0.003], paired over tasks).

### TabSchema in production

- **Provider errors are retried.** A rate limit (honoring `Retry-After`), a dropped connection, a server error or
  an empty answer is retried by the stage with exponential backoff (`TabSchemaConfig.llm_retries`, 5;
  `--llm-retries`); a provider that stays down stops the run with `LLMUnavailable`, resumable. A timeout is
  retried once, then handled like a truncated answer; the request timeout scales with the output limit
  (`request_timeout_s`, `--request-timeout`). Every attempt is in `llm_calls.jsonl`. A prompt the provider's
  filter refuses goes to the fallback model like a filtered answer; a prompt too long for the model is clipped
  harder (analysis, audit) or split (whole-run stages). Other provider errors (an invalid key, say) stop the
  run at any stage.
- **Unusable answers degrade instead of failing the run.** A FeatureAggregator, judge or MetaJudge batch whose
  answer is unparseable, cut at the output limit, filtered or of the wrong shape is asked again in halves; what
  still fails is recorded and reported. Runaway answers are bounded (60 proposals per analyzed point, 2,000
  characters per plan field).
- **Resume and budget stops are exact.** Truncated and filtered answers are kept in the run directory, and only
  the answers that caused a failure are dropped from it, so a resumed run never pays twice and produces the
  card of an uninterrupted run. The budget stops cleanly at every stage, the audit included (it used to skip the
  audit). Cache entries are locked across processes: two runs on one directory pay once for each answer.
- **`run.json` and `tabagent runs show RUN_DIR`**: the run's anatomy as it progresses, stage by stage (status,
  timing, progress with time left, produced, dropped, failed, LLM calls, tokens and cost), and drops by reason;
  `tabagent runs list`. The CLI exits with status 3 for a resumable stop.
- **The dry-run estimate is per stage** and renders the FeatureAnalyzer prompts of the sampled points, counts
  tokens at the model's measured characters per token (2.3 for Claude), and includes the audit and the
  compile step's repair round (`tabagent.tabschema.estimate`); within 30% of the actual cost on the runs it was
  calibrated on.
- **The sandbox fails one trajectory at a time**: a trajectory that hangs in C code, blows up memory or crashes
  its interpreter fails alone, and a fresh interpreter continues. Prints are capped, NaN and infinities become
  missing values, and interpreters run four at a time. Compile bounds its memory on wide outputs over many
  decision points (100,000 decision points: about 65 s).
- **Duplicate text columns are dropped at compile**: a text column equal to another kept one (the context text
  included) word for word, number for number and in order on every onboarding decision point, such as a text and
  the list of its tokens (`drop_redundant_text`).
- **Config checks.** Every `TabSchemaConfig` field is validated (a typo in `accept` used to compile nothing after
  paying for the whole run); `TabSchemaConfig.from_dict`, `tabagent onboard --config FILE`, and flags for
  `accept`, the sample strategy, workers, timeouts, retries, the audit sample and every check.
- **Ctrl+C starts no new LLM call** and waits (up to 2 s) only for the answers being written; a temporary file
  left by a killed process, or older than a day, is removed when a run directory is opened.

### Agent logs

- Every adapter reads files through one reader (`tabagent.adapters.logfiles`), and reports what it skipped
  or repaired in its `warnings` (the onboarding report, the dry run, `tabagent scan`):
  - **encodings:** UTF-8 with or without a BOM, UTF-16 and UTF-32; a file that is not UTF-8 is read in the
    locale's encoding or cp1252, with a warning naming it, and only bytes nothing reads become U+FFFD, with
    a count;
  - **damaged JSON:** trailing commas, bad JSONL lines (counted, with their place), JSON Lines in a `.json`
    file and a pretty-printed document in a `.jsonl` file;
  - **other damage:** values nested deeper than 200 levels, broken surrogate pairs;
  - **files:** linked folders, any-case suffixes, unique run ids.
- `tabagent check-logs LOGS` (`tabagent.check_logs`) diagnoses a log folder before an onboarding, with no
  LLM call. It reports the files, the runs and outcomes and, for each decision: candidates, labels missing
  from the candidates (with near-miss names), train/serve prefix parity and the post-decision data check.
  It exits with status 1 when a problem would stop an onboarding.
- From 256 MB of logs, runs are read again from their files when needed instead of being held in memory.
  Every read checks a fingerprint of exactly the bytes the run came from, so a touched file or an appended
  JSON Lines file is fine, and a changed run stops with `LogChangedError`. `lazy=` on `load` and
  `onboard`, and `--no-lazy` on the CLI, choose. Loading a 1.6 GB folder of agent logs peaks at 0.8 GB instead
  of 2.7 GB. The decision points of one run share their steps, so memory grows linearly with run length:
  1,500 turns took 598 MB and now take 20 MB.
- `redact=` keeps lazy runs lazy (`Trajectory.with_raw`).
- Adapter fixes:
  - CUGA labels read `call_api` as Python: keywords in any order, names assigned one literal resolved,
    calls in comments and strings no longer counted.
  - Chat runs with tool calls as a string or a single object, Responses `custom_tool_call` and `mcp_call`,
    and a conversation written one message per line are read.
  - A list of runs is no longer lost to one run without messages.
  - A run's `tools` string is never opened as a file path.
  - Tool descriptions no longer depend on file order.
  - A `{tool: argument schema}` catalog is read.
  - Malformed steps, prompts and τ²-bench records no longer crash the adapters.

### Serving

- `tabagent serve` runs the extractor in warm sandbox workers by default (`--sandbox warm`): persistent
  `python -I -S` interpreters with the one-shot sandbox's isolation, so the sandbox adds about 0.1 ms to a
  decision instead of an interpreter start-up. `--sandbox subprocess` keeps one interpreter per decision.
- The service has readiness (`/ready`), versioning (`/version`), Prometheus metrics (`/metrics`), feature drift
  monitoring (`/drift`), batch decisions (`/decide_batch`), structured JSON logs without request content, and
  a request timeout that defers instead of failing. See the production runbook in `docs/guides/production-runbook.md`.
- `TabAgent.fit` records the training distribution of the head's inputs (`model/feature_stats.json`), which
  drift monitoring compares live decisions with.

### Upgrading from development builds

- **Older extractors may be refused at serve time.** A sandbox chosen by name now re-applies the
  security-relevant static checks of the compile step to `extract.py` before running it. The checks are the
  allowed standard-library imports, the forbidden builtins, and no interpreter internals or frame
  introspection. A head saved by an earlier version whose `extract.py` fails them, for example by importing
  `time`, `os` or `signal`, now answers `/ready` with 503 and defers every decision to the LLM. Onboard it
  again, or review `extract.py` and serve it with `--allow-unchecked-extractor`
  (`TabAgentHead.load(..., allow_unchecked_extractor=True)`). The opt-out keeps the extractor in the sandbox
  and logs a warning.
- **Status codes.** A failure of the head other than a `RequestError` is a 500 (a plain `ValueError` from a
  custom head used to be a 400). A `k` below 1 is a 400. A decision past its deadline lists every candidate id
  in input order, with empty `scores`.
- **Drift needs refitted heads.** Heads fitted before this version have no training statistics, and their
  `/drift` reports `unavailable`.
- **Outcome values.** These are read as success, in any case: `true`, a positive finite number or numeric
  string, and the words `yes`, `y`, `pass`, `passed`, `success`, `succeeded`, `successful`, `solved`,
  `resolved` and `ok`.
  - The words `false`, `no`, `n`, `fail`, `failed`, `failure`, `error`, `errored`, `unsolved`,
    `unresolved`, `incorrect`, `wrong`, `timeout`, `none` and `null` mean failure.
  - Any other value means failure and is reported: another word, an empty string, `NaN`, an infinity, an
    object or a list. Chat logs used to count an object or a list as a success. Pass `success=` to read
    such values.
  - A CUGA `score` is read the same way, so `"true"` and `"PASS"` now mean success. A missing or unreadable
    score is a reported failure.
- **`Trajectory`** is still a frozen dataclass: `dataclasses.fields`, `asdict` and `replace` work. `raw`
  may now be read lazily from the run's file, each access returning a fresh copy. `replace` on a lazy
  trajectory returns an in-memory one; use `Trajectory.with_raw` to keep it lazy. The repr no longer prints
  the log.
- **The CUGA and generic adapters read `.jsonl` files and lists of runs** in a `.json` file. They also follow
  linked folders.
- **Folders with several log formats.** Detection reads the format most files hold; the generic adapter
  used to read whichever came first. Runs of other formats are skipped with a warning that names them. An
  adapter given explicitly does the same, and never raises. `on_other_formats="error"` (CLI
  `--on-other-formats error`) raises `MixedFormatsError` instead. Chat messages without an assistant turn
  (prompt templates, few-shot files) are not runs.
- **`adapter.load(path)` raises `ValueError`** when it reads files but finds no run in them, saying what it
  skipped. It used to return an empty list.
- **Prefixes of one run share their step objects.** Treat `DecisionPoint.prefix` as read-only, and copy it
  before changing it. `success=` and `labels_fn` functions get a copy of the run.
