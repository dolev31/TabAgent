---
title: "Onboard your agent"
description: "The path from an agent's logs to a served decision head: scan, check, estimate the cost, onboard, review and serve."
kind: "Get started"
lead: "Six commands from a folder of agent logs to a head that answers POST /decide."
needs: "the agent's logs and an LLM API key"
---

# Onboard your agent

Onboarding reads your agent's own logs, lets an LLM discover the features that explain its decisions (TabSchema),
trains and evaluates a head, and saves it. The steps below are the whole path; each links to the guide that covers
it in depth.

!!! warning "What leaves your machine"

    During discovery, trajectories are sent to your LLM provider. Only `scan`, `components`, `check-logs`, `--dry-run`
    and `serve` call no LLM. Pass `--redact module:function` to scrub the trajectories first, and read
    [Privacy and cost](../guides/onboard-from-logs.md#privacy-and-cost) before the first real run.

## 1. Find what is worth replacing

```bash
export OPENAI_API_KEY=...                       # the LLM TabSchema will call (not used by the next four commands)
tabagent scan ./logs                            # which LLM calls are worth replacing
tabagent components ./logs                      # the decisions you can replace, with counts
```

`scan` ranks every component of the agent that calls an LLM by its cost, whether its outputs come from a closed set
of candidates, and how many successful runs contain it. It prints the `onboard` command for the top candidates.
[Check and debug logs](../guides/check-and-debug-logs.md) explains the report.

## 2. Check the logs

```bash
tabagent check-logs ./logs                      # what is wrong with the logs, where, and how to fix it
```

It reads the logs as `onboard` would, calls no LLM, and exits with status 1 when a problem would stop an
onboarding. Supported formats: OpenAI chat logs, the Anthropic Messages API format, LangChain message dumps,
τ²-bench results, CUGA trajectories and generic JSON with named steps.

## 3. Estimate the cost

```bash
tabagent onboard ./logs --decision ToolShortlist --dry-run
```

The dry run lists what discovery would process and what it would cost, stage by stage, and the problems that would
stop a real run, before any LLM call. A cost estimate needs the model's prices: pass `--price-in` and
`--price-out` (USD per 1M tokens).

## 4. Onboard

```bash
tabagent onboard ./logs --decision ToolShortlist --out ./my_head \
    --max-cost 50 --price-in "$PRICE_IN" --price-out "$PRICE_OUT"   # a budget, in USD per 1M tokens
```

or from Python:

```python
import tabagent

head = tabagent.onboard("./logs", decision="ToolShortlist", out="./my_head")
print(head.report)  # grouped cross-validation vs BM25 and chance, latency, cost per decision
```

A run that stops (a spent budget, a provider that stays down, Ctrl+C) keeps every paid answer; rerun it with the
same run directory and it resumes at no extra cost. See
[Stopping, resuming and inspecting a run](../guides/onboard-from-logs.md#stopping-resuming-and-inspecting-a-run).

## 5. Review the extractor

TabSchema writes Python that computes the features from a trajectory prefix. Read `my_head/extract.py` before you
serve the head, and re-run its checks:

```bash
tabagent verify-head ./my_head                  # exit status 1 when a check fails
```

[Verify the generated code](../guides/verify-generated-code.md) lists what is checked and what to look for.

## 6. Serve it

```bash
pip install "tabagent[serve]"
tabagent serve ./my_head --port 8077            # POST /decide
```

```python
import tabagent

head = tabagent.load("my_head", trust_extractor=True)  # trust only an extractor you have reviewed
head.decide(prefix, candidates, k=5)  # {"candidates", "scores", "confidence", "deferred", "error"}
```

A decision the head cannot make well comes back `"deferred": true`, and the agent calls its original LLM
component. [Deploy and serve](../guides/deploy-and-serve.md) and the
[production runbook](../guides/production-runbook.md) cover the service;
[Hot-swap into an agent](../guides/hot-swap-into-an-agent.md) puts the head in front of a tool-calling LLM.
