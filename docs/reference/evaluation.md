---
title: "Evaluation"
description: "API reference: cross_validate, cross_validate_discovery, CVResult, the paper's metrics and the bootstrap intervals."
kind: "Reference"
lead: "Task-grouped cross-validation, the paper's metrics and paired bootstrap intervals."
---

# Evaluation

`cross_validate` is the only code in the library that decides what is training data. See
[Choose and evaluate a head](../guides/choose-and-evaluate-a-head.md) and [Leakage safety](../concepts/leakage-safety.md).

```python
from tabagent import TabAgent, TabHead, cross_validate, datasets

name = "skillsbench/skill-select"
result = cross_validate(
    TabAgent(datasets.card(name), head=TabHead("ranking")),
    datasets.load(name, download=True),
    folds=datasets.folds(name),
)
print(result.summary(["p@r", "recall@9"]))
```

::: tabagent.evaluation.cv.cross_validate

::: tabagent.evaluation.cv.CVResult

::: tabagent.evaluation.discovery.cross_validate_discovery

::: tabagent.evaluation.cv.assign_folds

::: tabagent.evaluation.metrics.per_decision

::: tabagent.evaluation.stats.bootstrap_ci

::: tabagent.evaluation.stats.paired_bootstrap

::: tabagent.evaluation.stats.Interval

::: tabagent.evaluation.stats.holm
