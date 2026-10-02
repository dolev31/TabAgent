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
| CUGA trajectories | any agent node, default `ShortlisterAgent` | APIs offered to the shortlister | APIs the code agent called |
| generic JSON (`steps: [{name, ...}]`) | any step name | `candidates_fn=` | `labels_fn=` |

A run counts as successful when its outcome field says so (`success`, `reward`, `score`, the first one
present; `true`, a positive number or a word such as `passed` means success, anything else failure); runs
without an outcome are treated as successful with a warning in the report. CUGA runs succeed when
`score > 0`.

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
files, imports modules beyond a small standard-library list, or reaches for interpreter internals. Use
`DockerSandbox` (no network, read-only filesystem, 1 GB) for code or traces you do not trust:
`tabagent.onboard(..., sandbox="docker")` runs discovery, the decision table and the returned head in it.

## Review the extractor

The saved head contains `extract.py`, the code TabSchema compiled. Read it before serving with
`--trust-extractor` (in-process, sub-millisecond); without that flag it runs in the sandbox. Both modes give
the same features: in-process, the code sees a JSON copy of the prefix, as in the sandbox, and cannot change
the caller's. Build live prefixes with the adapter's `live_prefix` (chat logs, CUGA and generic JSON), which
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
