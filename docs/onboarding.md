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
| τ²-bench results (`simulations`) | `ToolShortlist` | the domain's tools | tools the run called |
| CUGA trajectories | any agent node, default `ShortlisterAgent` | APIs offered to the shortlister | APIs the code agent called |
| generic JSON (`steps: [{name, ...}]`) | any step name | `candidates_fn=` | `labels_fn=` |

A run counts as successful when its outcome field says so (`success`, `reward`, `score`); runs without an
outcome are treated as successful with a warning in the report.

## Privacy and cost

Trajectories are sent to your LLM provider during discovery. Pass `redact=` (a function applied to every
trajectory before anything is sent) to scrub personal data. `--dry-run` reports the trajectories, decision
points and an estimate of the tokens before any call. `max_cost_usd` (`--max-cost`) stops discovery when
the LLM cost exceeds it; it needs the model's prices, `prices={model: (usd_per_1M_in, usd_per_1M_out)}`
(`--price-in`, `--price-out`), and is refused for a model without a price. Run directories hold trajectory text and live outside the repository
(`~/.cache/tabagent/runs` by default).

## Review the extractor

The saved head contains `extract.py`, the code TabSchema compiled. Read it before serving with
`--trust-extractor` (in-process, sub-millisecond); without that flag it runs in the sandbox.
