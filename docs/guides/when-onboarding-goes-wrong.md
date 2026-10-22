---
title: "When onboarding goes wrong"
description: "What onboarding checks before any LLM call, and how it behaves when a decision point, a provider, a timeout or the compiled extractor fails."
lead: "The checks that stop a run early and the failure modes during discovery, with what to change."
kind: Guide
needs: "a failing or stopped onboarding"
---

# When onboarding goes wrong

Before any LLM call, onboarding checks the data and stops with a message that says what to change: no
successful trajectories, fewer than 2 decision points, decision points of a single task, an adapter that
gives no labels or candidates (for generic JSON, `candidates_fn=` and `labels_fn=`), no chosen candidate among
the offered ones (the catalog and the logged choices use different names), or several labels on a
`kind="single"` decision. A dry run lists these problems instead of stopping.

During discovery:

* **The analyzed sample.** The FeatureAnalyzer reads `n_analyze` decision points (20 by default), so they
  decide which features can be found. Each pick goes to a task not sampled yet and, among those, to the
  decision point that adds the most the sample has not covered: a source (log file), a position in the run
  (first, middle or last decision), a prefix-length quartile and the decision's labels. The run directory
  records the sample (`sample.json`) and its coverage (`sample_coverage.json`, also a line of the report);
  `TabSchemaConfig(sample_strategy="tasks")` restores the earlier sample, the first decision of each task.
  Measured on the next-tool-call decision of the public τ²-bench GPT-4.1 logs (Claude Opus 5.5, with GPT-6 Sol
  writing the code; P@R of the head in grouped 5-fold CV, the same folds for every card):

  | analyzed sample | discovery runs | P@R, full card | P@R, LLM features only | LLM cost per run |
  |---|---|---|---|---|
  | 10, first decision of each task | 4 | 0.812 (0.799–0.818) | 0.778 (0.758–0.794) | $4.0 |
  | 10, coverage | 4 | 0.820 (0.810–0.831) | 0.793 (0.761–0.820) | $6.3 |
  | 20, coverage (the default) | 3 | 0.836 (0.825–0.842) | 0.851 (0.848–0.853) | $11.5 |

  Ten points left the features to chance: two runs on the same logs differed by up to 6 points. Twenty points
  cover all four positions in the run and most of the 35 tools, and three runs agreed within half a point.
  Analyzing 40 points in one run did not improve on 20 (one run, 0.837, $21). A smaller `n_analyze` lowers the
  cost (the dry run estimates it); expect a less stable card.
* **A decision point that fails** (an unusable or truncated answer, a content filter, a prompt the model
  refuses as too long even after its prefix is clipped to half and a quarter of the budget, code that never
  runs) is skipped, recorded in `analysis.json` with its error, and counted in the report and `run.json`; the
  run goes on with the others. When every point fails, the error names the first cause.
* **Provider errors.** A rate limit (429), a dropped or reset connection, a server error (5xx) or an empty
  answer (a proxy's reply with no choices) is retried by the stage that made the call, with exponential backoff
  (2, 4, 8, ... seconds, at most 60, with jitter, never under 1 second) or what the provider's `Retry-After`
  asks, up to `TabSchemaConfig.llm_retries` times (5; `--llm-retries`). Every attempt is a line of
  `llm_calls.jsonl` (`event: retry`). A provider that stays down stops the run with `LLMUnavailable`,
  resumable like a budget stop (a failed attempt is usually not billed; one that timed out may have been). A
  provider that asks to wait more than 5 minutes, or a 429 for an exhausted quota (`insufficient_quota`), stops
  the run at once. Any other provider error (an invalid key, a model that does not exist, a bad request) is not
  retried and stops the run at any stage, with the run directory named in the error.
* **Timeouts.** One request may take `TabSchemaConfig.request_timeout_s` seconds (`--request-timeout`); by
  default `max(300, 120 + max_tokens / 40)`, room for the stage's longest answer at 40 tokens per second: 520 s
  for the per-decision-point stages (16,000 tokens), 920 s for the whole-run stages (32,000). A request that
  times out is retried once; a second timeout is handled like an answer cut at the output limit (the batch is
  split, the decision point skipped), since a long answer would time out every time.
* **Unusable answers of the whole-run stages.** When the FeatureAggregator, a judge or the MetaJudge answers a
  batch unusably (unparseable, without the expected list, cut at the output limit, timed out twice, filtered,
  or refused as too long), each half of the batch is asked again, down to quarters. A feature that still has no usable answer
  is aggregated as proposed, scored without that judge, or reported as without a MetaJudge decision, and the
  report says so; when no call of the stage gives a usable answer, the run stops. An answer that keeps more
  features than its batch proposed, or answers for names it was not given, is trimmed to what it was asked.
* **Runaway answers.** One FeatureAnalyzer answer contributes at most 60 proposals, and a feature plan's text
  fields are clipped to 2,000 characters, so one runaway answer cannot inflate every later prompt.
* **Extraction code** gets the first attempt and three retries; each retry shows every earlier error. An
  answer without code counts as a failed attempt (code in a Markdown fence, or in `<code>` tags that the
  thoughts also mention, is recovered). The code fails only the trajectory on which it raises, calls `exit()`,
  runs past `timeout_s` or returns something other than a flat dict of JSON scalars, and also only the
  trajectory on which it hangs out of reach of its alarm (a regular expression that backtracks for minutes: its
  interpreter is killed 2 seconds past `timeout_s`), blows up memory, or crashes its interpreter: a fresh
  interpreter carries on with the next trajectory (and an interpreter stopped by its chunk's wall-clock limit
  while its trajectories are slow but within their own limits simply continues in a new one). What it prints is capped at 1 MB per interpreter. A value
  that is NaN or infinite becomes missing (`None`), by the fixed code appended to every extractor, so training
  and serving agree. In the compile step at most `1 - compile_min_success` of the decision points may fail;
  those rows get empty features, as they do at serving time.
* **Very long traces** reach the FeatureAnalyzer with their longest strings clipped once their JSON passes
  `TabSchemaConfig.max_prompt_chars` (400,000 characters); the extraction code always runs on the full prefix.
* **The MetaJudge** decides only on the features of its batch, as each judge does; a decision on another name is
  ignored and reported under `dropped`. Accepted and conditionally accepted features are compiled by default:
  `TabSchemaConfig(accept=("accept",))` keeps only accepted ones. On the next-tool-call decision of the
  τ²-bench GPT-4.1 logs, the MetaJudge marked 41% of features conditional, and compiling them raised head P@R
  from 0.761 to 0.827 (10 analyzed decision points) and from 0.840 to 0.869 (20 points), LLM features only,
  with the same folds; compiling the rejected features as well added under 0.004. A run in which no feature is
  accepted stops with an error that says so.
* **The compiled extractor** keeps only columns that implement an accepted feature (the feature itself, a
  sub-key of it, or it under a group prefix), that do not reuse a column name of the decision table (`task`,
  `decision_id`, `candidate`, `chosen`, `candidate_name`, ...), and that pass the quality gates; every other
  column is listed under `dropped` with the reason. A presence flag (`True` where present, missing elsewhere)
  is kept. An unusable CodeValidator answer leaves the deterministic checks in charge, with a note. A column
  that lists earlier candidates (the tools called before the decision, say) becomes the context of the card's
  fitted co-usage statistic (`fitted:co_usage`).
* **Nested outputs and wide features.** Code that returns a nested dict is flattened into `parent_child`
  columns by the CodeExtractor prompt's own `flatten_json`, which TabAgent appends to every extractor (so it
  runs the same at serving) instead of failing the attempt. Code that returns a dict per row, such as word
  counts or one entry per entity, becomes one column per key: compile keeps at most
  `TabSchemaConfig.max_columns_per_feature` (64) columns of one planned feature and
  `max_extractor_columns` (512) in all, those that vary most across decision points, and the report says
  what was trimmed. Without the caps, word-count features can expand into tens of thousands of columns.
* **Duplicate text columns.** A compiled text or categorical column that duplicates another kept one word for
  word on every onboarding decision point (the same lowercased `\w+` tokens, numbers and order; case,
  punctuation, brackets and quotes aside) is dropped as `redundant with <column>`: a text and the list of its
  tokens are one column, while a bag of words with counts, n-grams, the same words in another order, a
  different number, a negation, or a difference on a single decision point keep both. The appended context
  text counts too (an LLM column equal to it is dropped). Of duplicates the co-usage context is kept first,
  then the context text, then a categorical form over a text one, then by name
  (`TabSchemaConfig(drop_redundant_text=False)` turns it off).
* **Repairing flagged features.** When the CodeValidator flags accepted features in the compiled code (a field
  the trajectories lack, or code that does not follow the plan), the CodeExtractor is asked once more with
  the validator's objections in the paper's retry prompt; the new code is kept only if it runs and fewer
  features are flagged (`TabSchemaConfig(compile_repair=False)` turns this off). In one internal
  app-selection run, the best-rated feature had been dropped because its code counted generic words as app
  names.
* **Bounded prompts for wide extractors.** The CodeValidator sees at most 150 keys of each sample output, and
  the output audit judges at most 48 columns per decision point, spread over the planned features (set
  values first), so a wide extractor neither fills a prompt nor truncates an answer.
* **The compile record.** `compile.json` in the run directory holds every code attempt's error, every kept
  column (its type and the planned feature it implements), every column's quality statistics, and every
  dropped column with the reason.
* **The context text.** Every compiled extractor ends with a fixed function, appended by TabAgent and not
  written by the LLM, that adds `context_text`: the prefix's task-level text fields and the texts of its steps
  (system prompts skipped), keeping the start and the latest part of a long history. It gives every card a
  context text for the context-candidate similarity features, TabSynth's paraphrases and the BM25 baseline,
  even when the LLM's features are all flags and counts. Being part of `extract.py`, it runs identically at
  training and at serving. `TabSchemaConfig(context_text=False)` leaves it out.

The sandbox isolates accidents, not adversaries. It runs the code in a fresh interpreter without environment
variables (no API keys), with CPU-time and file-size limits, a memory limit (`mem_mb`: an address-space limit
on Linux, with room for the chunk's input; on macOS the interpreter watches its own peak memory over its peak
with the input loaded), at most 50 trajectories and 32 MB of input per interpreter, and `timeout_s` per
trajectory; the deterministic checks refuse code that opens files, imports modules beyond a
small standard-library list, or reaches for interpreter internals (dunder attributes, and the frames of
generators, coroutines and tracebacks: `gi_frame`, `f_back`, `f_locals`, ...). Use `DockerSandbox` (no
network, read-only filesystem, 1 GB) for code or traces you do not trust: `tabagent.onboard(...,
sandbox="docker")` runs discovery, the decision table and the returned head in it.
