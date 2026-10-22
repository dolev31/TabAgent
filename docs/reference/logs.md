---
title: "Logs and adapters"
description: "API reference: scan, check_logs, ScanReport, LogCheck and the log adapters."
kind: "Reference"
lead: "Find what is worth replacing, diagnose the logs, and read them into trajectories."
---

# Logs and adapters

Both `scan` and `check_logs` read the logs only and call no LLM. The guide is
[Check and debug logs](../guides/check-and-debug-logs.md).

```python
import tabagent

print(tabagent.scan("./agent_logs").summary())
check = tabagent.check_logs("./agent_logs", decision="ToolShortlist")
print(check.summary(), check.ok)
```

::: tabagent.scanner.scan

::: tabagent.scanner.ScanReport

::: tabagent.scanner.check_logs

::: tabagent.scanner.LogCheck

::: tabagent.adapters
    options:
      members:
        - detect_adapter
        - OpenAIMessagesAdapter
        - Tau2Adapter
        - CugaAdapter
        - GenericJsonAdapter
        - live_prefix
