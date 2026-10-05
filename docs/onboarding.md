# Onboarding an agent from its logs

```bash
export OPENAI_API_KEY=...                       # or OPENAI_BASE_URL (any OpenAI-compatible server),
                                                # or ANTHROPIC_API_KEY / GEMINI_API_KEY with tabagent[llm]
tabagent scan ./logs                            # which LLM calls are worth replacing (no LLM call)
tabagent components ./logs                      # the decisions you can replace, with counts
tabagent onboard ./logs --decision ToolShortlist --dry-run
tabagent onboard ./logs --decision ToolShortlist --out ./my_head
tabagent onboard ./logs --decision ToolShortlist --out ./my_head \
    --max-cost 50 --price-in "$PRICE_IN" --price-out "$PRICE_OUT"   # a budget, in USD per 1M tokens
```

`tabagent.onboard()` detects the log format, runs TabSchema discovery on the successful runs, builds the
decision table, selects the training recipe by nested validation, evaluates the head with grouped
cross-validation (quality with CIs against BM25 and chance, latency and cost per decision), fits the final head
and saves it.

The recipe search (on the training rows only) chooses TabSynth off, k=2 or k=4, the candidate prior on or off,
and, for a card whose LLM features include other state, whether to keep the generic context text and the
context-candidate similarities. Rich LLM features can make those redundant: on the next-tool-call decision of
the τ²-bench GPT-4.1 logs, a card discovered from 20 decision points scored P@R 0.853 without them against 0.841
with them, while a card from 10 points needed them (0.810 against 0.761). Choosing by nested validation inside
each training fold reached 0.850 and 0.810. They are dropped only when the inner validation is strictly better;
the report names the recipe (`context=off`).

## Find what to replace

`tabagent scan` reads the logs, calls no LLM and sends nothing anywhere. It ranks every component of the agent
that calls an LLM by four questions:

1. **Cost.** Its LLM calls per run and, where the logs record usage, its share of the agent's LLM tokens,
   latency and cost over the scanned runs. Where the logs record none, the report says *not recorded* instead of
   estimating. `--price-in` and `--price-out` (USD per 1M tokens, as for `onboard`) price every call from its
   recorded tokens.
2. **Closed-set.** Whether its outputs come from a fixed, recurring set of candidates (tool names,
   applications, routes, labels), the decision type TabAgent replaces. The evidence is reported: the number of
   choices, how many distinct values they take, the catalog size and the share of choices inside the catalog
   offered to the call. Free-text generation and planning text are reported as not replaceable.
3. **Data.** How many successful runs contain it, next to the paper's reference points: TabHead reaches 90% of
   its full-pool P@R from 118 successful trajectories on AppWorld and from 37 to 51 on the four τ²-bench
   agents. These are the paper's measurements on its benchmarks, not a prediction for your agent.
4. **Next step.** The `tabagent onboard LOGS --decision NAME` command for the top candidates.

```bash
tabagent scan ./logs                                  # plain text
tabagent scan ./logs --format markdown                # or --format json
tabagent scan ./logs --price-in 2 --price-out 8       # price the recorded tokens, USD per 1M
```

```python
import tabagent

report = tabagent.scan("./logs")  # prices=(usd_per_1M_in, usd_per_1M_out) or {model: (in, out)}
print(report.summary())
best = report.candidates[0]  # closed-set components, best first
best.next_steps  # the onboard commands for it
report.to_json()  # or report.to_markdown(), report.to_dict()
```

Closed-set components rank first, by their share of the agent's LLM cost (else tokens, latency or calls,
whichever the logs record first), then components whose outputs the logs do not show, then free text and
outputs that do not recur. A component is closed-set when at least 5 of its calls show what they chose and
either at least 90% of its choices lie inside the catalog offered to the call, or, when the logs record no
catalog, its choices take at least two distinct values, at most one distinct value per two choices, and at least
80% of them recur. A call that both chooses and writes text (a tool call with generated arguments, a planner
that routes and explains) is closed-set in its choice, and the report gives the free text it writes per call,
which a head does not replace.

What each log format records:

| format | LLM calls | choices and catalog | tokens, latency, cost |
|---|---|---|---|
| chat messages | each assistant turn: `NextToolCall` when it calls tools, else `TextReply` | the tools called; the run's `tools` or the catalog you pass | the message's `usage` (OpenAI, Anthropic), LangChain `usage_metadata` or `response_metadata`, `cost`, `latency_ms`, when present |
| τ²-bench results | the agent's turns, not the user simulator's or the scripted greeting | the tools called; `tool_defs` when the file has them | per-message `usage` and `cost`; latency from the per-message timestamps |
| CUGA trajectories | each agent-node step, one per recorded generation | the APIs the shortlister returned, against the APIs offered to it; the categorical fields of other nodes' JSON output; code, answer and chat nodes write free text | not recorded per step (`results.json` has per-task totals) |
| generic JSON | steps with a usage, model, prompt or messages field; if no step has one, every named step, flagged | the short categorical values of the step's `output`, `data`, `result`, ... | the step's `usage`, `cost_usd`, `latency_ms`, ..., when present |

A custom adapter takes part by implementing `llm_calls(trajectory) -> list[LLMCall]`
(`tabagent.adapters.LLMCall`; `tabagent.adapters.read_usage` reads the common usage shapes). Without it, each
decision point of each component counts as one LLM call, with no usage.

### Example: τ²-bench public results

The scan below reads public data: the GPT-4.1 agent's default-mode runs in the airline, retail and telecom
domains, four trials per task, from the official τ²-bench trajectory release
(`data/tau2/results/final` at commit `cf71a80` of
[sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench), the source of the `tau2/<agent>`
datasets). The three result files are in `./tau2-results`.

```text
$ tabagent scan ./tau2-results
Scanned 1,112 runs (606 successful) with Tau2Adapter: 14,872 LLM calls by 2 components.
Ranked by share of the agent's LLM cost, closed-set components first.

#  component     verdict     calls/run  tokens/call  latency/call  cost/call  cost share  successful runs
1  NextToolCall  closed-set       5.75        6,457        2.93 s    $0.0055         40%              604
2  TextReply     free text        7.62        6,737        4.44 s    $0.0062         60%              606

1. NextToolCall: closed-set
   cost      5.75 LLM calls per run; 42% of the agent's LLM tokens, 33% of its LLM latency and 40%
             of its LLM cost (recorded)
   evidence  its 8,076 choices take 35 distinct values and >99% of them recur (no catalog recorded)
             most frequent: get_details_by_id (1,740), get_order_details (1,274), get_user_details
             (612), get_reservation_details (551), get_customer_by_phone (456)
             it also writes 81 characters of free text per call, which a head does not replace
   data      604 successful runs contain it
   next      tabagent onboard ./tau2-results --decision ToolShortlist
             tabagent onboard ./tau2-results --decision NextToolCall

Not replaceable or unclear:
   TextReply (free text): generates free text (443 characters per call); TabAgent replaces
     closed-set choices, not generation

Data reference points from the paper (not a prediction for your agent): TabHead reaches 90% of its
full-pool P@R from 118 successful trajectories on AppWorld and from 37 to 51 on the four τ²-bench
agents.

Notes:
   * only the agent's LLM calls are counted: the user simulator's turns and the scripted greeting
     (no usage, zero cost) are not
   * latency is the time from the previous message to the agent's message: τ²-bench records a
     timestamp per message, not a duration per call
```

Tokens and costs are the ones the release records on each agent message. `ToolShortlist` makes one decision
per run, the tools the run will call (tool shortlisting, the decision the paper evaluates on τ²-bench);
`NextToolCall` makes one per tool-calling turn. Either way the head chooses tools, and the agent's LLM still
writes the arguments and the replies.

## Log formats

| format | decision components | candidates | labels |
|---|---|---|---|
| OpenAI chat logs (`messages` with `tool_calls`, optional `tools`) | `ToolShortlist` (per task), `NextToolCall` (per turn) | the run's `tools`, or a catalog you pass | tools the run called |
| Anthropic Messages API format (`tool_use` blocks) | same | same | same |
| LangChain message dumps | same | same | same |
| τ²-bench results (`simulations`) | `ToolShortlist`, `NextToolCall` | the domain's tools (`tool_defs`), or a catalog you pass | tools the run called |
| CUGA trajectories | `ShortlisterAgent` (per shortlisting), `TaskAnalyzerAgent` (app selection, multi-label); any other node with your own candidates and labels | APIs offered to the shortlister; the apps listed in the analyzer's prompt | APIs the code agent called before the next shortlisting; apps the run called |
| generic JSON (`steps: [{name, ...}]`) | any step name | `candidates_fn=` | `labels_fn=` |

A run counts as successful when its outcome field says so (`success`, `reward`, `score`, the first one
present; `true`, a positive number or a word such as `passed` means success, anything else failure); runs
without an outcome are treated as successful with a warning in the report. CUGA runs succeed when
`score > 0`.

The CUGA adapter builds the two components the paper replaces. `--decision ShortlisterAgent` ranks the APIs
of the app being shortlisted; `--decision TaskAnalyzerAgent` selects the apps a task needs, a multi-label
decision that `onboard` recognizes by itself (`--kind` is only needed to override it). Its labels are the apps
the successful run called; `CugaAdapter(app_labels="offered")` uses the apps the analyzer chose instead, as
the paper's legacy data did (on CUGA AppWorld runs the two agree on nearly all). Point `onboard` at the
folder of a CUGA evaluation (one JSON file per task; `results.json` and `metadata.json` are skipped), or at a
folder of such folders or of links to them.

A decision point's prefix holds the steps before the decision and only the top-level fields that describe
the task: for chat logs and generic JSON `instruction`, `task`, `query`, `question`, `goal`, `intent` and
`domain`; for CUGA `intent` and `dataset_name`. The task id is never in a prefix: it is the cross-validation
group, not a feature. Other top-level fields (a final
answer, run statistics) never reach the LLM or the extractor; the report names the ones left out. Pass
`task_keys=` to the adapter to keep fields that are known before the decision (`task_keys=None` keeps every
field but the outcome keys). Generic JSON directories are read recursively; `results.json` and
`metadata.json` (run summaries) and files that are not valid JSON are skipped.

Candidate text (each tool's name, description and argument hints) is what the head matches the context
against. When your logs record only the tool names (τ²-bench's public results, say), pass the catalog:
`--catalog tools.json` (or `tools=` to the adapter; `catalog=` for CUGA) with OpenAI or Anthropic tool
schemas, MCP tools or a `{name: description}` mapping. One catalog applies to every run; for a catalog per
domain, keep the tool definitions in each log file.

## Privacy and cost

Trajectories are sent to your LLM provider during discovery. Pass `redact=` (a function applied to every
trajectory before anything is sent) to scrub personal data; the trajectories' file names are sent as they are
(the analyzer prompt names its trajectory file), so rename files whose names are personal. `redact` changes
the onboarding trajectories only: `head.decide(prefix, ...)` does not apply it, so apply the same function to
live prefixes when it changes fields the extractor reads. `--dry-run` reports the trajectories, decision
points and an estimate of the tokens before any call, plus any problem that would stop a real run (see
below). `max_cost_usd` (`--max-cost`) stops discovery before a call whose estimated cost (its prompt plus the
stage's mean answer, counting calls already in flight) would take the total past it; it needs the model's
prices, `prices={model: (usd_per_1M_in, usd_per_1M_out)}` (`--price-in`, `--price-out`; `--price
MODEL=IN,OUT` for the other models of a run), and is refused for a model without a price. The report gives
each model's cost and calls. Run directories hold trajectory text and live outside the repository
(`~/.cache/tabagent/runs` by default).

## LLM providers

`tabagent.onboard` builds the LLM from the model name and the environment (`tabagent.tabschema.llm.auto_llm`):

1. `--model provider/name` for a LiteLLM provider (`anthropic/...`, `gemini/...`, `mistral/...`, `bedrock/...`,
   `vertex_ai/...`, `groq/...`, `ollama/...`, ...) goes through LiteLLM (`pip install "tabagent[llm]"`),
   even when `OPENAI_API_KEY` is also set;
2. `AZURE_OPENAI_ENDPOINT` (with `AZURE_OPENAI_API_KEY`, and `OPENAI_API_VERSION`, default `2024-10-21`)
   selects a classic Azure OpenAI resource; the model is the deployment name;
3. `OPENAI_API_KEY`, or `OPENAI_BASE_URL` alone for a server that needs no key (vLLM, Ollama), selects the
   OpenAI-compatible client; on such a server a model name may contain a slash;
4. otherwise the provider whose key is set (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`).

Every stage sends its prompt as the system message followed by a one-line user turn (some providers,
Anthropic among them, reject a request without one). The per-decision-point stages may answer up to 16,000
tokens and the whole-run stages (the FeatureAggregator, the three judges and the MetaJudge) up to 32,000; a
model with a lower output limit is retried at its own limit. An answer that stops at the limit raises
`LLMTruncated` (on one decision point it only skips that point). The whole-run stages see at most 40
(aggregator) or 25 (judges, MetaJudge) features per call, so that no answer has to restate an arbitrarily long
list; up to those sizes every feature is handled in one call, as in the paper.

**A model per stage.** `TabSchemaConfig(stage_models={"extractor": "azure/gpt-6-sol"})` (CLI `--stage-model
extractor=azure/gpt-6-sol`, repeatable; `judges` covers the three judges) runs one stage on another model,
for example a strong code model for the CodeExtractor. A provider's content filter can refuse one model's
answers to one stage outright (the finish reason `content_filter`): such an answer is never retried on the
same model, cached or re-asked; it raises `LLMContentFiltered`, whose message names the stage. Route that
stage to another model, or set `fallback_model` (`--fallback-model`), which is asked once whenever a filter
stops an answer (`llm_calls.jsonl` records the call with `fallback_from`).

## When something goes wrong

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
* **A decision point that fails** (an unusable or truncated answer, a provider error such as a context
  overflow, code that never runs) is skipped, recorded in `analysis.json` with its error, and counted in the
  report; the run goes on with the others. When every point fails, the error names the first cause.
* **Extraction code** gets the first attempt and three retries; each retry shows every earlier error. An
  answer without code counts as a failed attempt. Code may print freely, and fails only the trajectory on
  which it raises, calls `exit()`, runs past `timeout_s` or returns something other than a flat dict of JSON
  scalars. In the compile step at most `1 - compile_min_success` of the decision points may fail; those rows
  get empty features, as they do at serving time.
* **Very long traces** reach the FeatureAnalyzer with their longest strings clipped once their JSON passes
  `TabSchemaConfig.max_prompt_chars` (400,000 characters); the extraction code always runs on the full prefix.
* **The MetaJudge** decides only on features the FeatureAggregator kept; a decision on another name is reported
  under `dropped`. Only `accept` decisions are compiled by default: add `"conditional"` to
  `TabSchemaConfig.accept` to also keep conditionally accepted features. A run in which no feature is
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
  what was trimmed. On CUGA's shortlister, word-count features had expanded into thousands of columns.
* **Repairing flagged features.** When the CodeValidator flags accepted features in the compiled code (a field
  the trajectories lack, or code that does not follow the plan), the CodeExtractor is asked once more with
  the validator's objections in the paper's retry prompt; the new code is kept only if it runs and fewer
  features are flagged (`TabSchemaConfig(compile_repair=False)` turns this off). On CUGA's app selection, the
  best-rated feature had been dropped because its code counted generic words as app names.
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

A run that stops (a spent budget, no accepted feature, an extractor that cannot be compiled, a provider that
stays down) leaves a partial `report.md` and every paid answer in its run directory. Rerun with the same
`run_dir` (`--run-dir`, the run directory itself) to resume: answers the directory holds cost nothing, while
answers that could not be used are asked again. The report and the card count those reused answers at the
cost they had when first paid. Two runs on one run directory at once are safe, but may pay twice for the same
answer. The same trajectories and the same answers give a byte-identical `card.yaml` and `extract.py`; the card
holds no local paths.

The sandbox isolates accidents, not adversaries. It runs the code in a fresh interpreter without environment
variables (no API keys), with CPU-time and file-size limits, a memory limit on Linux (`mem_mb`), at most 50
trajectories per interpreter and `timeout_s` per trajectory; the deterministic checks refuse code that opens
files, imports modules beyond a small standard-library list, or reaches for interpreter internals (dunder
attributes, and the frames of generators, coroutines and tracebacks: `gi_frame`, `f_back`, `f_locals`, ...). Use
`DockerSandbox` (no network, read-only filesystem, 1 GB) for code or traces you do not trust:
`tabagent.onboard(..., sandbox="docker")` runs discovery, the decision table and the returned head in it.

## How the generated code is verified

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
The appended `context_text` ignores outcome fields. The checks of the extractor make three sandbox runs: one on
the sample, one over every decision point (an interpreter per 50), and one for the variants and edge cases.

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
snippets of its onboarding trajectories**, so share it as you would share them, or set `goldens=0`.

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
(`extractor_tests.json`; `tabagent verify-head` re-runs them). Read the code before serving with
`--trust-extractor` (in-process); without that flag it runs in the sandbox, by default in warm worker
processes that start once and then add well under a millisecond per decision (see
[Deployment](deployment.md#the-warm-extractor-pool)). All modes give the same features: in-process, the code
sees a JSON copy of the prefix, as in the sandbox, and cannot change the caller's. Build live prefixes with the
adapter's `live_prefix` (chat logs, CUGA and generic JSON), which
builds them exactly as in training.

## Estimate the whole pipeline, discovery included

`onboard` discovers the feature card once, on all the onboarding trajectories, and then cross-validates the
head on that card. Its folds keep the head's training rows and fitted statistics apart, but every held-out task
was among the trajectories the card was discovered from, so the report estimates the head on a fixed card: a
quick estimate (one discovery) that can be optimistic. The paper's protocol fits the TabSchema loop inside each
training fold too (Sec. 4.3). `tabagent.cross_validate_discovery` does that:

```python
import tabagent
from tabagent.tabschema import TabSchemaConfig

res = tabagent.cross_validate_discovery(
    "./logs",
    "ToolShortlist",
    config=TabSchemaConfig(max_cost_usd=100, prices={"gpt-5.4": (PRICE_IN, PRICE_OUT)}),  # all folds together
    run_dir="./infold_runs",  # one cached run directory per fold: a rerun makes no new LLM calls
)
print(res.summary(["p@r", "recall@9"]))  # the usual CVResult metrics and intervals
res.discoveries[0].card  # the card discovered on fold 0's training tasks
```

For every fold it runs discovery on the training fold's trajectories only, builds the training and held-out
rows with that fold's extractor, selects the training recipe on the training rows (as `onboard` does on all of
them) and scores the held-out decisions. With the same `n_splits`, its folds are those of `onboard`'s report,
so the two estimates can be compared directly. It costs one discovery per fold, about five times the LLM cost
of onboarding with the default five folds; one budget covers all folds. The folds' discoveries run three at a
time by default (`fold_workers=`), each with an equal share of the budget, so the wall time is about that of
two onboardings rather than five. `redact=` works as in `onboard`, and
`sandbox=DockerSandbox()` runs the generated code in Docker. The head you deploy is still the one `onboard` fits
on all the trajectories; this estimates how well that procedure generalizes to new tasks.
