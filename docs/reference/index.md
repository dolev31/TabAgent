---
title: "API reference"
description: "The TabAgent Python API and command line, generated from the code's docstrings."
kind: "Reference"
lead: "Signatures, parameters and examples, generated from the docstrings of the installed version."
---

# API reference

Everything below is generated from the package's docstrings, so it matches the code. The names in
`tabagent.__all__` are the stable entry points; they are importable from the top-level package.

```python
import tabagent
from tabagent import TabAgentHead, TabAgentRanker, TabHead, TabSchema, TabSynth
```

<div class="grid cards ta-cards" markdown>

-   :material-robot-outline: [__Onboarding and heads__](onboarding.md)

    `onboard`, `load`, `TabAgentHead`, `DryRun`.

-   :material-table-large: [__scikit-learn estimators__](estimators.md)

    `TabAgentRanker`, `TabAgentClassifier`, `TabAgentRouter`, `pointwise`, `scorer`.

-   :material-table: [__Tables and cards__](tables.md)

    `DecisionTable`, `DecisionSpec`, `FeatureCard`, `Feature`.

-   :material-layers-triple-outline: [__TabSchema, TabSynth, TabHead__](phases.md)

    The paper's three components, and `TabAgent` that joins them.

-   :material-chart-line: [__Evaluation__](evaluation.md)

    `cross_validate`, `cross_validate_discovery`, the metrics and the bootstrap intervals.

-   :material-magnify-scan: [__Logs and adapters__](logs.md)

    `scan`, `check_logs` and the log adapters.

-   :material-server-outline: [__Serving and integrations__](serving.md)

    `RemoteHead`, `ToolFilter`, the CUGA helpers and the HTTP service.

-   :material-database-outline: [__Datasets__](datasets.md)

    The public benchmarks as decision tables.

-   :material-console: [__Command line__](cli.md)

    Every `tabagent` command and flag.

</div>
