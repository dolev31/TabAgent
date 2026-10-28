---
title: "Wanted"
description: "What TabAgent needs next, and what done means for each item: log formats, a safer hot swap, a richer feature card, and open research questions."
kind: "Project"
lead: "What TabAgent would like to have next, and what done means for each. Every item has an open issue; comment there to claim it."
---

# Wanted

TabAgent learns an agent's closed-set decisions from its own successful runs, so it is only as useful as the agents
it can read, the features it can compute and the care with which it is swapped into a live agent. The items below
are where help makes the biggest difference. Each has an issue with the files to start in.

| If you want to | Start with |
|---|---|
| Make a first pull request | [Good first issues](#good-first-issues) |
| Teach TabAgent a new agent framework | [Find the bottleneck](#find-the-bottleneck) |
| Run TabAgent in production | [A smoother hot swap](#a-smoother-hot-swap) |
| Improve what the classifier sees | [A richer feature card](#a-richer-feature-card) |
| Run a study | [Research questions](#research-questions) |

---

## Good first issues

Small, with the files named and a test to write. Labeled
[good first issue](https://github.com/dolev31/TabAgent/issues?q=is%3Aopen+label%3A%22good+first+issue%22).

- **[Read OpenTelemetry GenAI traces](https://github.com/dolev31/TabAgent/issues/1)** in `tabagent scan`, so any
  agent that exports OTel spans can be scanned before onboarding.
- **[Read LangSmith and Langfuse exports](https://github.com/dolev31/TabAgent/issues/2)**, two of the most common
  places agent logs live.
- **[A LangGraph integration example](https://github.com/dolev31/TabAgent/issues/4)** with the tool filter, runnable
  offline in under a minute.
- **[A candidate-recency feature](https://github.com/dolev31/TabAgent/issues/7)**: steps since each candidate was last
  used, with its leakage test.

---

## Find the bottleneck

[Track #10](https://github.com/dolev31/TabAgent/issues/10). `tabagent scan` ranks every LLM-calling component of an
agent and marks the closed-set ones TabAgent can replace. Every log format it reads is another set of agents it can
help.

- **[A savings estimate in the scan report](https://github.com/dolev31/TabAgent/issues/3)**: per component, the calls
  and the share of tokens or cost that replacing it would remove, and what each estimate rests on.

**Done when:** the adapter is registered in `detect.py`, a small fabricated log in `tests/adapters/fake_logs.py`
makes `tabagent check-logs` find its decisions, candidates and labels, and the scan report reads it.

---

## A smoother hot swap

[Track #11](https://github.com/dolev31/TabAgent/issues/11). Nobody swaps a component they cannot watch.

- **[Shadow mode](https://github.com/dolev31/TabAgent/issues/5)**: the head scores each decision beside the LLM and
  records how often they agree, before it takes over.
- **[A cache-friendly tool list](https://github.com/dolev31/TabAgent/issues/6)**: a shortened list whose prefix stays
  stable, so the provider's prompt cache keeps working. Measured, with paired intervals, in `experiments/live/`.

**Done when:** the serving tests cover the new path, including a deferral and a timeout, and the deployment guide
says when to use it.

---

## A richer feature card

[Track #12](https://github.com/dolev31/TabAgent/issues/12). The paper's central finding is that the representation
does the work: every head reaches useful quality with the feature card and collapses without it.

- **[`tabagent card explain`](https://github.com/dolev31/TabAgent/issues/8)**: a readable report of what a saved head's
  card holds and what the head relies on.
- **[A feature-card benchmark](https://github.com/dolev31/TabAgent/issues/9)**: one command that tests a card change on
  public tasks with paired intervals.

**Done when:** a new primitive is registered in `FITTED`, reads only the training rows and the prefix, and a test in
`tests/leakage/test_leakage.py` shows scrambling the test fold's labels leaves it unchanged.

---

## Research questions

Open questions TabAgent can answer with a measured comparison. Write the plan before the run (the question, what
changes, what is held fixed, and the result that would prove you wrong) with the
[research form](https://github.com/dolev31/TabAgent/issues/new?template=research_proposal.yml). Either answer is worth
publishing. Labeled [research](https://github.com/dolev31/TabAgent/issues?q=is%3Aopen+label%3Aresearch).

- **Does cross-agent transfer hold across architectures?** The paper's transfer holds within one architecture, four
  ReAct agents that differ only in their LLM; it does not test a different scaffold.
- **Does TabSchema need a frontier model?** The paper found open models not yet reliable for TabSchema's structured
  orchestration. Which parts fail, and could a smaller model do any of them?

---

## How to claim an item

1. Comment on its issue with your plan, or use the
   [contribution proposal form](https://github.com/dolev31/TabAgent/issues/new?template=contribution_proposal.yml).
2. After a maintainer's go, open a draft pull request within two weeks, or the claim lapses and someone else can take
   it.
3. Before you ask for review, meet [the bar](contributing.md#the-bar).

Questions go to the [roadmap discussion](https://github.com/dolev31/TabAgent/discussions/13).
