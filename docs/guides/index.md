---
title: "Guides"
description: "Task-oriented guides for TabAgent: onboard from logs, bring your own table, choose and evaluate a head, deploy and serve, hot-swap into an agent, check logs, and reproduce the paper."
kind: "Guide"
lead: "Pick the task you are doing. Each guide is self-contained and links to the concept or reference it relies on."
---

# Guides

## Build a head

<div class="grid cards ta-cards" markdown>

-   :material-file-document-outline: [__Onboard from logs__](onboard-from-logs.md)

    Turn an agent's logs into a head with `tabagent onboard`. Privacy and cost controls, LLM providers, stopping and resuming.

-   :material-table-large: [__Bring your own table__](bring-your-own-table.md)

    The scikit-learn estimators on a decision table you already have. No LLM.

-   :material-scale-balance: [__Choose and evaluate a head__](choose-and-evaluate-a-head.md)

    Metrics per decision kind, task-grouped cross-validation, paired intervals, other classifiers, discovery inside each fold.

</div>

## Check and debug

<div class="grid cards ta-cards" markdown>

-   :material-magnify-scan: [__Check and debug logs__](check-and-debug-logs.md)

    `tabagent scan`, the supported log formats, how files are read, and `tabagent check-logs`.

-   :material-lifebuoy: [__When onboarding goes wrong__](when-onboarding-goes-wrong.md)

    The checks that stop a run early and what to change when discovery fails at a stage.

-   :material-shield-check-outline: [__Verify the generated code__](verify-generated-code.md)

    The four value checks, the golden tests, and how to review `extract.py` before you serve it.

</div>

## Deploy

<div class="grid cards ta-cards" markdown>

-   :material-rocket-launch-outline: [__Deploy and serve__](deploy-and-serve.md)

    Saved heads, live prefixes, the HTTP service, the sandbox, redaction and pretrained heads on the Hub.

-   :material-swap-horizontal: [__Hot-swap into an agent__](hot-swap-into-an-agent.md)

    `ToolFilter` in front of a tool-calling LLM, the CUGA helpers, and whether filtering the tool list pays.

-   :material-server-outline: [__Production runbook__](production-runbook.md)

    Scale, the warm extractor pool, timeouts, metrics and alerts, drift, upgrading and the security model.

</div>

## Reproduce

<div class="grid cards ta-cards" markdown>

-   :material-flask-outline: [__Reproduce the paper__](../reproducing.md)

    Every public result, the command that recomputes it, and the paper's value next to the released code's.

-   :material-database-outline: [__Public datasets__](../datasets.md)

    Each dataset's pinned source, license and the paper result it backs.

</div>
