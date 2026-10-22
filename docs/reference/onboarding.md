---
title: "Onboarding and heads"
description: "API reference: tabagent.onboard, tabagent.load, TabAgentHead and DryRun."
kind: "Reference"
lead: "Turn agent logs into a head, and load a saved one."
---

# Onboarding and heads

`onboard` is the one-call path from agent logs to a deployable head. `load` reads a saved head back, from a local
directory or from the Hugging Face Hub. A `TabAgentHead` is the fitted model plus the TabSchema extractor that
feeds it: the drop-in replacement for one agent decision.

```python
import tabagent

head = tabagent.onboard("./agent_logs", decision="ToolShortlist")
print(head.report)  # grouped cross-validation vs BM25 and chance, latency, cost per decision
head.save("my_head")
head.decide(prefix, candidates, k=5)  # {"candidates", "scores", "confidence", "deferred", "error"}

head = tabagent.load("my_head", trust_extractor=True)  # trust only an extractor you have reviewed
```

::: tabagent.api.onboard

::: tabagent.api.load

::: tabagent.api.TabAgentHead
    options:
      members:
        - decide
        - live_prefix
        - warm
        - close
        - calibrate_gate
        - save
        - load
        - features_for
        - candidate_features_for

::: tabagent.api.DryRun

::: tabagent.api.RequestError
