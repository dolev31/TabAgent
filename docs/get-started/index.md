---
title: "Get started"
description: "Install TabAgent, run it on a table in five minutes, onboard your own agent from its logs, and learn the core ideas."
kind: "Get started"
lead: "From pip install to a decision head, in four short pages."
---

# Get started

TabAgent replaces an LLM agent's repeated closed-set decisions (which tools to shortlist, which skill or
application to use, which model to route to) with a cheap classifier trained on the agent's own successful runs.
These four pages take you from nothing installed to a head you can serve.

<div class="grid cards ta-cards" markdown>

-   :material-download-outline: [__Installation__](installation.md)

    `pip install tabagent`, the extras (`embed`, `tabschema`, `llm`, `serve`, `hub`) and the LLM keys.

-   :material-timer-outline: [__Quickstart (5 minutes)__](quickstart.md)

    Rank tools from a table on CPU with no LLM and no network, then reproduce a paper number.

-   :material-robot-outline: [__Onboard your agent__](onboard-your-agent.md)

    Scan the logs, check them, estimate the cost, onboard a decision and serve it.

-   :material-lightbulb-on-outline: [__Core concepts__](core-concepts.md)

    Closed-set decisions, the decision table, the three phases and the confidence gate in one page.

</div>

## Pick your path

<div class="grid cards ta-cards" markdown>

-   :material-table-large: __I have my decisions as a table__

    Use the scikit-learn estimators: [Bring your own table](../guides/bring-your-own-table.md).

-   :material-file-document-outline: __I have an agent's logs__

    Onboard from the logs: [Onboard from logs](../guides/onboard-from-logs.md).

-   :material-flask-outline: __I want to check the paper__

    [Reproduce the paper](../reproducing.md) from public data, with no LLM call.

</div>
