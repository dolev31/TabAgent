---
title: "Datasets"
description: "API reference: tabagent.datasets, the public benchmarks as decision tables."
kind: "Reference"
lead: "Each public benchmark rebuilt from its pinned upstream release, with the paper's card and folds."
---

# Datasets

Nothing is redistributed: `load` rebuilds each table from the upstream release at a pinned revision, verified by
SHA-256. Every dataset, its source, license and the paper result it backs is on [Public datasets](../datasets.md).

```python
from tabagent import datasets

datasets.available()  # what can be built, with licenses and paper rows
table = datasets.load("skillsbench/skill-select", download=True)
card = datasets.card("skillsbench/skill-select")  # the card behind the paper's number
folds = datasets.folds("skillsbench/skill-select")  # the paper's task-level fold assignment
```

::: tabagent.datasets
    options:
      members:
        - available
        - names
        - info
        - load
        - card
        - folds
