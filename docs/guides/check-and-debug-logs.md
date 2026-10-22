---
title: "Check and debug logs"
description: "Find which LLM calls are worth replacing with tabagent scan, read the supported log formats, and diagnose a log folder with check-logs before any LLM call."
lead: "Before you spend on an LLM, find what to replace, see how your logs are read, and fix what would stop an onboarding."
kind: Guide
needs: "the agent's logs; no LLM call"
---

# Check and debug logs

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
| OpenAI chat logs (`messages` with `tool_calls`, legacy `function_call` or Responses API items; optional `tools`) | `ToolShortlist` (per task), `NextToolCall` (per turn) | the run's `tools`, or a catalog you pass | tools the run called |
| Anthropic Messages API format (`tool_use` blocks) | same | same | same |
| LangChain message dumps | same | same | same |
| τ²-bench results (`simulations`) | `ToolShortlist`, `NextToolCall` | the domain's tools (`tool_defs`), or a catalog you pass | tools the run called |
| CUGA trajectories | `ShortlisterAgent` (per shortlisting), `TaskAnalyzerAgent` (app selection, multi-label); any other node with your own candidates and labels | APIs offered to the shortlister; the apps listed in the analyzer's prompt | APIs the code agent called before the next shortlisting; apps the run called |
| generic JSON (`steps: [{name, ...}]`) | any step name | `candidates_fn=` | `labels_fn=` |

A run counts as successful when its outcome field says so (`success`, `reward`, `score`, the first one
present). `true`, a positive finite number (or a numeric string) and the words `yes`, `pass`, `passed`,
`success`, `solved`, `resolved`, `ok` mean success; `false`, zero or a negative number and the words `no`,
`fail`, `failed`, `error`, `unsolved`, `wrong`, `timeout` mean failure, in any case. Any other value (another
word, an empty string, `NaN`, an infinity, an object, a list) cannot be read: the run counts as failed and
the report says how many runs had such a value, with the first ones; pass `success=` (a field name or a
function of the run) to read them. Runs without an outcome are treated as successful, with a warning. CUGA
runs succeed when `score > 0`; a missing or unreadable score counts as a failure, with a warning.

The CUGA adapter builds the two components the paper replaces. `--decision ShortlisterAgent` ranks the APIs
of the app being shortlisted; `--decision TaskAnalyzerAgent` selects the apps a task needs, a multi-label
decision that `onboard` recognizes by itself (`--kind` is only needed to override it). Its labels are the apps
the successful run called; `CugaAdapter(app_labels="offered")` uses the apps the analyzer chose instead, as
the paper's legacy data did (on the successful runs checked, the two almost always agree). Point `onboard` at the
folder of a CUGA evaluation (one JSON file per task; `results.json` and `metadata.json` are skipped), or at a
folder of such folders or of links to them. The APIs a run called are read from its generated code as Python
(`call_api(app_name, api_name)`, keyword arguments in any order, an app or API given as a name the code
assigns one string literal resolved, Markdown fences removed when the code does not parse with them), so a
call in a comment or a string does not count; code that does not parse is searched with a pattern, comments
removed.

A decision point's prefix holds the steps before the decision and only the top-level fields that describe
the task: for chat logs and generic JSON `instruction`, `task`, `query`, `question`, `goal`, `intent` and
`domain`; for CUGA `intent` and `dataset_name`. The task id is never in a prefix: it is the cross-validation
group, not a feature. Other top-level fields (a final
answer, run statistics) never reach the LLM or the extractor; the report names the ones left out. Pass
`task_keys=` to the adapter to keep fields that are known before the decision (`task_keys=None` keeps every
field but the outcome keys). The prefixes of one run share the step objects they have in common, so a run
with many decisions takes memory linear in its length: treat a prefix as read-only, and copy it before
changing it.

Candidate text (each tool's name, description and argument hints) is what the head matches the context
against. When your logs record only the tool names (τ²-bench's public results, say), pass the catalog:
`--catalog tools.json` (or `tools=` to the adapter; `catalog=` for CUGA) with OpenAI or Anthropic tool
schemas, MCP tools, a `{name: description}` mapping or a `{name: argument schema}` mapping. One catalog
applies to every run; for a catalog per domain, keep the tool definitions in each log file (a run's `tools`
field may also be the JSON text of the list; it is never read as a file path). When runs disagree on a
tool's description, the one most runs give is used (ties: the longest), whatever order the files are read
in.

### How log files are read

Every adapter reads files the same way, and reports what it skipped or repaired instead of dropping it
silently: the warnings of `onboard`'s report and dry run, of `tabagent scan`, and `check-logs` below.

* **Files.** A file, or every `.json` and `.jsonl` file below a folder (any case: `.JSON` too), in sorted
  order. Hidden files and folders are skipped; linked folders are followed, each real folder once, and a file
  reached through two links is read once. Files with other extensions are not read (`check-logs` lists them:
  decompress `.gz` files first).
* **Runs.** A `.json` file holds one run or a list of runs; a `.jsonl` file one run per line. A chat
  conversation written one message per line is one run, and so is a bare list of messages. A run's id is its
  path below the folder (`:<index>` for a run in a list or on a line); a second run with the same id (`a.json`
  and `a.jsonl`) gets the file's suffix, so ids never collide.
* **Encodings.** UTF-8 with or without a byte-order mark, UTF-16 and UTF-32 with one. A file that is not UTF-8
  is read in the first of the locale's encoding and cp1252 that reads every byte, with a warning naming it
  (check that the text reads as intended). A UTF-8 file with a few damaged bytes (a character cut at a buffer
  boundary) stays UTF-8, and bytes nothing can read become U+FFFD, with a warning that counts them; so does
  half of a surrogate pair (text cut in the middle of an emoji), which would otherwise fail when the text is
  written to the run directory or sent to the provider.
* **Damaged JSON.** A file that is not valid JSON is skipped with its first error (`run3.json: line 12,
  column 5: Expecting ',' delimiter`); in a `.jsonl` file only the bad lines are skipped, counted, with the
  first ones, and the file is still read one line at a time. Trailing commas are read without them; `NaN`
  and `Infinity` are read as numbers. A `.json` file that holds one value per line is read as JSON Lines,
  and a `.jsonl` file whose first line opens a pretty-printed document (a lone `{` or `[`) as JSON. Only
  `\n` ends a line: a U+2028 inside a string does not split it.
* **Deep values.** Values nested deeper than 200 levels are replaced below that depth by a placeholder, with
  a warning, so that no later stage runs out of recursion; a file nested too deeply to parse at all is
  skipped.
* **One format per folder.** A folder can hold files that are not runs of its format: a prompt template, a
  few-shot or fine-tuning file, a tool catalog, a task list, run summaries. Detection reads the format most
  files hold (CUGA trajectories over other logs with named steps), and runs of another format are skipped with
  a warning that counts them and names the first files. Messages without an agent turn (a template, a
  few-shot example) are not a run. `on_other_formats="error"` (CLI `--on-other-formats error`) stops with
  `MixedFormatsError` instead: one onboarding reads one format, since the head needs the same adapter at
  serving time. An adapter you pass yourself reads its own runs and skips the rest with a warning.
* **Large inputs.** From 256 MB of logs, runs read from a file of their own or from a JSON Lines line are not
  held in memory. Each is kept as its place in the file and a fingerprint (SHA-256) of exactly the bytes it
  was read from, and every read checks them: a touched file, or a JSON Lines file the agent appended to,
  reads the same runs, while a change to a run's own bytes (even one that keeps the file's size and time)
  stops with `LogChangedError`, saying which file changed. Load the logs again once they stop changing; an
  onboarding rerun with the same run directory reuses every LLM answer already paid for. `lazy=False` (CLI
  `--no-lazy`, on `onboard`, `scan`, `components` and `check-logs`) holds every run in memory instead;
  `lazy=True` reads lazily whatever the size. A few parsed runs are kept per load, and go away with its
  trajectories. The results are the same either way. On a 1.6 GB folder of agent logs, the peak
  memory of loading them fell from 2.7 GB to 0.8 GB, of `tabagent scan` from 2.7 GB to 0.9 GB and of
  `onboard --dry-run` from 2.7 GB to 1.2 GB, for about 1.7 times the reading time. `redact=` keeps them
  lazy: it gets a copy of each run, each time the run is read, so it must be deterministic.

### Checking the logs

`tabagent check-logs` reads the logs as `onboard` would, calls no LLM, and says what is wrong, where and how
to fix it; it exits with status 1 when a problem would stop an onboarding:

```bash
tabagent check-logs ./logs                          # every decision the adapter builds
tabagent check-logs ./logs --decision NextToolCall --format json
```

```python
import tabagent

report = tabagent.check_logs("./logs")  # adapter="auto", decision=None
print(report.summary())  # report.ok is False when an onboarding would stop; report.to_json()
```

It reports the files read, skipped or repaired, the runs and how many succeeded, and for each decision its
decision points and tasks, the share with candidates and with labels, the chosen candidates that are not
among the offered ones (with a near-miss name when the catalog spells it differently, in another case or
with an app prefix), whether the prefix rebuilt from each run before its decision equals the training prefix
(on 200 decision points spread evenly, `--sample`, or all of them when there are fewer), and the adapter's
check that no prefix holds post-decision data (a check that could not check any point says so). A log that
changes while it is being read is reported as such. CUGA `call_api` calls whose app or API name is neither a
literal nor a name assigned one literal are counted. On the public τ²-bench GPT-4.1 logs:

```text
$ tabagent check-logs ./tau2-results
Checked ./tau2-results with Tau2Adapter: 3 log files, 1,112 runs (606 successful).
Components in the successful runs (runs containing them): ToolShortlist (604).

ToolShortlist:
   604 decision points in 604 successful runs of 199 tasks
   candidates on 100%, labels on 100% of them; every label is among the offered candidates
   live prefixes rebuilt exactly as in training on 200 of 200 checked
   no post-decision data in prefixes (20 checked)

NextToolCall:
   3,226 decision points in 604 successful runs of 199 tasks
   candidates on 100%, labels on 100% of them; every label is among the offered candidates
   live prefixes rebuilt exactly as in training on 200 of 200 checked
   no post-decision data in prefixes (20 checked)

Next: tabagent onboard ./tau2-results --decision ToolShortlist --dry-run
```

`--dry-run` lists the same reading notes (files skipped or repaired, unreadable outcomes) after its estimate.
