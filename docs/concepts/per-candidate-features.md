---
title: "Per-candidate features"
description: "How TabSchema relates a decision's context to each candidate: the per-candidate extractor function, within-decision relative features, dependency features and discovery inside each fold."
lead: "A feature that relates the context to a candidate has nowhere to go in a function of the prefix alone. This page is how the release gives it one, and the public evidence."
kind: Concept
---

# Per-candidate features

A ranking or multi-label decision table has one row per candidate of a decision. The paper's extractor is one
function of the prefix, so a feature that relates the context to a candidate (the words of the request in the
candidate's description, the candidate named in an earlier thought, how well a model's or a skill's
description fits the task, an API whose HTTP method matches the request's verb) has nowhere to go: the
CodeValidator rejects code that reads the candidate (it is not in the prefix), and code that enumerates a
catalog found in the prefix emits positional columns (`candidate_3_overlap`) that are the same on every row
of a decision and do not describe the candidate the row scores. In one internal API-shortlisting run (no
interval), most of the kept columns were positional, the judges' best-rated features were per-candidate ideas,
and computing them per candidate improved ranking.

### Evidence

The per-candidate prompt sections (the FeatureAnalyzer's, the CodeExtractor's and the
CodeValidator's, and the note to the aggregator and the judges) and the positional guard below are on by
default. One public measurement onboards with an LLM: the GPT-4.1 agent's logs of τ²-bench (sierra-research/
tau2-bench at `cf71a80`, the airline, retail and telecom `default` result files, four trials per task), its
next tool call as a ranking decision (`NextToolCall`): 1,112 runs, 606 successful, 3,226 decisions, 43,093
(decision, tool) rows. Each of three pipeline versions onboarded once, with Claude Opus 5.5 for every stage but
the CodeExtractor (GPT-6 Sol), 20 analyzed decision points and only accepted features compiled: the pipeline
before the per-candidate function (the previous card), PR #47 without the FeatureAnalyzer's section (run 1),
and with it (run 2). The cards were evaluated on identical rows and folds: 5-fold task-grouped
cross-validation, head seeds 0 to 2, the CatBoost paper preset, no TabSynth; 95% bootstrap intervals over
decisions, and paired bootstrap 95% intervals of differences over tasks (`experiments/tabschema_tau2.py`
prints the onboarding commands and, from the three run directories, reproduces the numbers below).

| card | features | P@R | Recall@3 | Recall@5 |
|---|---|---|---|---|
| previous pipeline | 215 | 0.837 [0.825, 0.850] | 0.939 [0.931, 0.946] | 0.967 [0.961, 0.972] |
| run 1 (no per-candidate features) | 217 | 0.847 [0.834, 0.858] | 0.947 [0.939, 0.954] | 0.968 [0.961, 0.973] |
| run 2 (19 per-candidate features) | 85 | 0.848 [0.837, 0.860] | 0.956 [0.949, 0.962] | 0.976 [0.971, 0.981] |
| run 2 without its per-candidate features | 66 | 0.828 [0.816, 0.840] | 0.945 [0.937, 0.952] | 0.966 [0.960, 0.972] |
| run 2 with relative features | 113 | 0.858 [0.847, 0.870] | 0.955 [0.948, 0.961] | 0.977 [0.972, 0.981] |

| paired difference over tasks | P@R | Recall@3 | Recall@5 |
|---|---|---|---|
| run 2 − previous pipeline | +0.010 [−0.003, +0.022] | +0.020 [+0.012, +0.029] | +0.012 [+0.006, +0.020] |
| run 2 − run 1 | +0.006 [−0.008, +0.021] | +0.014 [+0.006, +0.024] | +0.012 [+0.005, +0.021] |
| run 2 − run 2 without its per-candidate features | +0.026 [+0.014, +0.039] | +0.016 [+0.007, +0.025] | +0.013 [+0.006, +0.021] |
| run 2 with relative features − run 2 | +0.009 [−0.001, +0.019] | −0.008 [−0.018, −0.001] | −0.001 [−0.006, +0.003] |
| run 1 − run 1 without `last_step_text` | +0.006 [−0.004, +0.015] | +0.003 [−0.001, +0.008] | −0.000 [−0.004, +0.003] |

* The per-candidate features helped on all three metrics: removing them from run 2's card lowered P@R,
  Recall@3 and Recall@5, each difference outside its interval. Run 2's card, with 85 features against 215
  and 217, ranked better than both earlier cards on Recall@3 and Recall@5; its P@R gain over them is within
  its interval.
* The FeatureAnalyzer's section was necessary. These chat logs do not print the tool catalog in any step, so
  in run 1 the analyzer proposed no feature of a candidate; the CodeExtractor still wrote per-candidate
  features, and the compile step dropped them as outside the accepted plan. With the section, the analyzer
  proposed 119 features of a candidate over the 20 decision points, the MetaJudge accepted 19, and all 19
  compiled.
* The positional guard did not trigger: neither run's compile step reported a positional column, so this run
  does not test it.
* The relative features were mixed: P@R +0.009, within its interval, and Recall@3 −0.008, just outside it.
  This is why the recipe search adds them only when its inner validation scores strictly better.
* `last_step_text` was neutral to slightly positive, every difference within its interval.

This is one onboarding per pipeline version: an onboarding asks an LLM, so a card, and the numbers it gives,
vary with the model and from run to run, and the differences between versions include that variation; the
evaluation of given run directories is deterministic. The internal run above, whose logs list the API catalog
in the shortlister's own step, is the only other measurement.

The extractor may therefore define a second function:

```python
def extract_candidate_features(trajectory_data: dict, candidate: dict) -> dict: ...
```

* `candidate` is `{"id", "name", "description", "args", "spec"}`: the candidate as the decision offers it,
  at training and at serving alike. The first four are the values of its row's `candidate_*` columns;
  `spec` is its entry in the offered catalog (`Candidate.spec`), as a dict, or None when the catalog has
  none: a tool's JSON schema (OpenAI, Anthropic, MCP), or a CUGA API's `app_name`, `api_name`, `path`,
  `method`, `description`, `parameters` (with `required`) and `response_schemas`, without the `canary_string`
  CUGA writes anew in every run (the CUGA adapter, its `catalog=`, the hot swap, and `decide` or the service
  given raw CUGA entries, recognized by their `api_name`, leave it out of an entry's top level). The
  dict is built from the candidate alone (`trajectory.candidate_dict`), so it holds no label: the entry is
  what the decision was offered, not what it chose. As defense in depth, the deterministic checks refuse any
  other top-level key (`candidate["chosen"]`) and allow any path under `candidate["spec"]`, whose shape
  depends on the catalog; the prefix rules (known fields, no outcome fields) apply to both functions.
* The FeatureAnalyzer prompt gets an "Additional rules" section when the decision has candidates, since a
  chat agent's logs rarely list the tools a decision chooses among, and the paper's prompt shows only the
  prefix and the target step: each row is one candidate of one decision; besides features of the trajectory,
  features of a candidate, or of how the trajectory relates to one, may be proposed, computed once per
  candidate from the trajectory and the candidate's fields (`id`, `name`, `description`, `args`, `spec`); up
  to three of the decision's candidates, shown as to the CodeExtractor; neutral examples (how many times the
  candidate was already called, whether the latest user message mentions its name or description words,
  whether its required arguments appear in the conversation, how well a model's, skill's or application's
  description fits the request); and that every feature must be computable before the decision and never
  encode which candidate was chosen (the target step is shown only to explain the decision). The
  FeatureAggregator, judge and MetaJudge prompts get a one-paragraph note under the same heading: such a
  feature is obtainable, since a candidate's fields are known before the decision for every candidate of
  every decision, and its generality is judged as any other feature's (one that hard-codes particular
  candidates is still specific). Without candidates every prompt is the paper's alone, so a run directory's stored
  answers stay valid for it; a ranking or multi-label decision's analyzer, aggregator and judges are asked
  again.
* The CodeExtractor and CodeValidator prompts get an "Additional rules" section when the decision has
  candidates: the code may define the function; up to three of the decision's candidates as it receives them
  (each `spec` fit to 2,000 characters of JSON; in every prompt, the example candidates are picked from the
  run's seed, the decision point's id and the candidates themselves, never from the adapter's order or a
  label, so an adapter that lists the chosen candidate first does not make it an example, and the points of
  a fixed catalog show different ones); a feature about a candidate belongs only there, with examples
  for model, skill, application and tool decisions and, when the catalog describes HTTP APIs, for method and
  parameters; a feature about the history of earlier calls belongs in the first function; the first function
  never enumerates the candidates or indexes them by position; and what depends on the prefix alone is
  computed once, in a helper whose global cache holds one prefix's values.
* The code runs once per decision: the prefix is parsed once, the first function runs, then the second once
  per candidate in the same namespace, in one sandbox, warm-worker or in-process call. A candidate whose call
  raises gets no values (its traceback recorded once per distinct error); the prefix's features and the other
  candidates keep theirs. A decision's time limit grows with its candidates, `timeout_s * (1 + candidates /
  50)`, so a 400-API catalog gets 9 times the time of one prefix, never past a serving deadline
  (`tabagent serve --request-timeout`). A fabricated 400-candidate decision with a 68,000-character prefix took
  2.3 ms in-process, 2.7 ms in a warm worker and 51 ms in a fresh interpreter (start-up included) on a laptop;
  with 1 KB catalog entries, 5.6 and 6.6 ms, for a 0.55 MB sandbox payload (a full chunk of 50 such decisions,
  28 MB, fits the 32 MiB one interpreter takes at once).
* Its columns go through the compile step's gates over the (decision, candidate) rows: success rate, null
  rate, constant columns, types, the column caps, the duplicate-text filter, verification (determinism with
  the decisions and each decision's candidates shuffled, outcome invariance, robustness; future invariance
  needs no extractor run, since both functions read only the prefix), golden tests with three candidates per
  snapshot, the OutputAuditor (three candidates per sampled point, chosen from the seed and the decision,
  never by label) and the usefulness estimate. A name the first function or the decision table already uses
  is dropped. In the card they are `schema` features with `source: extractor_candidate`; TabSynth copies them
  verbatim, and they are not the candidate's own text for the context-candidate similarities or BM25.
* **The positional guard** (`TabSchemaConfig(positional_guard=True)`, `--no-positional-guard` turns it off).
  Prefix columns with an index form enumerations (`candidate_3_overlap` and `candidate_3_name` are both in
  `candidate_`); a name with a history word (`prev`, `last`, `history`, `ago`, `recent`, `step`, ...) is in
  none. An enumeration is positional, all its columns, when one of its columns' texts name the decision's
  candidates in their order, or, when its names hold a word for a candidate (`candidate`, `api`, `tool`,
  `app`, `application`, `model`, `skill`, `agent`, `subagent`, `element`, `doc`, ...; `available_api_2_path`
  counts), when its size follows the number of candidates across decision points of different sizes: as many
  indices as candidates, or, for columns emitted only where a value is set, indices within the candidates that
  reach the largest decision's last one. With the same number of candidates at every point, a size cannot tell
  a fixed-size list from the catalog, so only the texts count; the same holds for the first few of a longer
  enumeration. A fixed-size list (`item_1_price` to `item_3_price`, `tools_used_1` to `tools_used_3`) or a
  list of earlier calls (`prev_tool_1`, `tool_1_ago`) is not positional. The repair round asks the
  CodeExtractor to compute a positional feature in the per-candidate function; its positional columns are
  dropped once that function computes the feature, and kept, with a compile note, until it does. It did not
  trigger in the public τ²-bench onboarding (Evidence, above), so it is not yet tested on public data.
* `TabAgentHead.decide`, `tabagent serve`, the CUGA hot swap and the tool filter pass the decision's
  candidates to the extractor, which scores them exactly as the training rows were scored: `decide` reads
  `Candidate` objects, tool schemas and `{"id", "name", "description", "args", "spec"}` records alike, and a
  `RemoteHead` sends a `Candidate` as such a record, its catalog entry included. An extractor without the
  second function, and every head saved before it existed, runs and decides exactly as before.
* A saved head's golden tests (`extractor_tests.json`) record three candidates per snapshot, catalog entries
  included: check what the catalog holds before you upload a head to the Hub.

### Within-decision relative features

The head scores each (decision, candidate) row on its own, so it cannot
see that a candidate has the best overlap among the candidates of its decision. For a ranking or multi-label
decision, `card.with_relative` gives every numeric per-candidate feature, and the context-candidate
similarities (`context_candidate_lex`, `context_candidate_sim`), the candidate prior and co-usage, two
companions: its
percentile rank among the decision's candidates (`<name>_rank_in_decision`, source `relative_rank`; ties share
the average rank) and its gap to the decision's largest value (`<name>_gap_to_best`, source `relative_gap`).
They are computed when the card transforms a table, from the decision's own rows (booleans as 0 and 1, missing
values left missing), so they read no label and no other decision: a decision scores the same alone or in a
batch, `decide` and the service compute them over all of a live decision's candidates as training did, and a
decision's only candidate gets rank 1.0 and gap 0. TabSynth's recomputed variants are ranked among the
variant's own rows (a decision only partly paraphrased keeps the real rows' values).
`tabagent.tabschema.card.with_relative(card)` adds them to an existing card and `without_relative` removes them.
They are opt-in by validation. A discovered card never holds them, so the usefulness estimate, `prune_harmful`
and a `TabAgent` built on the card directly do not use them. The recipe search (`auto.select(...,
relative=True)`, which `onboard` and in-fold discovery pass from `TabSchemaConfig.relative_features`) fits
the winning recipe without them and tries it once with them (`relative=on` in its label), keeping them only
when the inner validation scores strictly better (ties, and too few groups to search, keep them off); the
saved head's `head.json` records the choice (`"relative_features": true` or `false`).
`TabSchemaConfig(relative_features=False)` (`--no-relative-features`) skips the challenger. In one internal
API-shortlisting run (no interval) they improved ranking. On the TabSchema card of the public τ²-bench next
tool call (Evidence, above) they were mixed: P@R +0.009 [−0.001, +0.019], Recall@3 −0.008 [−0.018, −0.001].
On the public bundled cards the effect is neutral, within its interval (`experiments/relative_features.py`;
the bundled cards
with the paper's folds, paper preset, seeds 0 to 2, rank and gap of the cards' overlap and prior features;
differences are paired bootstrap 95% intervals over tasks):

| decision | P@R without | with | difference | Recall@3 difference | Recall@5 difference |
|---|---|---|---|---|---|
| τ²-bench GPT-4.1 tool shortlist | 0.786 | 0.790 | +0.004 [−0.002, +0.010] | +0.003 [−0.002, +0.009] | −0.003 [−0.009, +0.003] |
| SkillsBench skill selection | 0.800 | 0.790 | −0.011 [−0.036, +0.016] | −0.008 [−0.022, +0.005] | −0.001 [−0.016, +0.013] |

### One form per text feature

The head represents text itself (TF-IDF or LSA), so a text feature emitted in
several forms (the text, its tokens, its bag of words, its n-grams) only adds columns. The CodeExtractor's
additional rules ask that a planned feature that is a tokenized, bag-of-words or n-gram form of a text be
emitted once, as the plain text, under the feature's name, unless another planned feature already emits the
same text. The compile step drops a column of one planned feature that restates another of its columns in a
shorter derived form on every row (`drop_redundant_text`): a bag of words or a subset of its tokens as a JSON
list or dict, or its words sorted and de-duplicated. The longer form, plain text next to a structured form, and
an ordered sequence next to a set are always kept, and columns of different features are never compared this
way.

### Accepted features without a column

When the compiled code produces no column for an accepted feature,
the compile step's repair round (one call, with the CodeValidator's objections and the positional features)
gives the CodeExtractor the feature's plan and asks it to add the feature, or to list it in a global
`NOT_EXTRACTED` dict as merged into the feature that already carries the same information, or declined with a
reason. The new code is kept only if it runs and fewer features are missing, flagged or positional, and every
added column goes through the usual gates. `compile.json` (`not_produced`) and the report count these
features by reason (merged into another feature, declined, not produced) and list them; they also list the
kept columns per decision and per candidate separately.

### The last step's text

`context_text` joins every step and keeps the first 1,000 and the last 3,000
characters, so the step that handed control to the decision can be diluted by long payloads. The library
appends a second fixed function, `last_step_text`: the `content`, `text`, `thoughts`, `intent` and `data`
strings of the prefix's last step with text (steps named `system` skipped, outcome fields never read, the
adapter's step list), clipped the same way. It is a context text like `context_text` (TabSynth paraphrases it,
and the context-candidate similarities use it). It is on by default. On the public τ²-bench next tool call
(Evidence, above) it was neutral to slightly positive: P@R +0.006 [−0.004, +0.015], Recall@3 +0.003 [−0.001,
+0.008], Recall@5 −0.000 [−0.004, +0.003], paired over tasks; in one internal API-shortlisting run (no
interval) it improved ranking.
`TabSchemaConfig(last_step_text=False)` (`--no-last-step-text`) leaves it out.

### Dependency features

These are not code the LLM writes: a card references the library's fitted statistics
(`fitted:<name>`), which are fitted on the training rows of each fold. They are the candidate prior
(`candidate_prior`), the context-candidate embedding similarity (`embedding_cosine`), the outcome of each
candidate on the most similar training tasks (`knn_outcome`) and co-usage (`co_usage`): for a column listing the
candidates used before the decision (an extractor feature with the tools the agent called earlier in the run, for
example), how often the training runs chose the candidate after them, smoothed toward its prior.

### Discovery inside each fold

`onboard` discovers the card once, on all onboarding trajectories, so the
cross-validation in its report estimates the head on a fixed card. `tabagent.cross_validate_discovery(logs,
target, ...)` runs `discover` on each training fold's trajectories only, applies that fold's extractor to its
held-out decisions and reports the usual cross-validated metrics, at the LLM cost of one discovery per fold (one
budget covers all folds; each fold's run directory is cached, so a rerun is free).
