# Onboarding an agent from its logs

```bash
export OPENAI_API_KEY=...                       # or OPENAI_BASE_URL (any OpenAI-compatible server),
                                                # or ANTHROPIC_API_KEY / GEMINI_API_KEY with tabagent[llm]
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
