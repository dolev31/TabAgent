---
title: "Serving and integrations"
description: "API reference: RemoteHead, create_app, ToolFilter, ToolSession and the CUGA helpers."
kind: "Reference"
lead: "Put a head behind HTTP, or in front of a tool-calling LLM."
---

# Serving and integrations

The guides are [Deploy and serve](../guides/deploy-and-serve.md), [Hot-swap into an agent](../guides/hot-swap-into-an-agent.md)
and the [production runbook](../guides/production-runbook.md). The endpoints of the service are listed in the runbook.

## OpenAI-style tools

```python
from tabagent.integrations.openai_tools import ToolFilter

tool_filter = ToolFilter("my_head", k=7)  # or the URL of a `tabagent serve` service
session = tool_filter.session(task={"instruction": goal})  # one per conversation
client.chat.completions.create(model=..., messages=messages, tools=session(messages, tools))
```

::: tabagent.integrations.openai_tools.ToolFilter
    options:
      members:
        - select
        - session
        - __call__

::: tabagent.integrations.openai_tools.ToolSession

## HTTP service

::: tabagent.serve.RemoteHead

::: tabagent.serve.create_app

## CUGA

::: tabagent.integrations.cuga.CugaShortlister
    options:
      members:
        - shortlist

::: tabagent.integrations.cuga.CugaAppMatcher
    options:
      members:
        - match

::: tabagent.integrations.cuga.run_so_far
