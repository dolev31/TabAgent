---
title: "scikit-learn estimators"
description: "API reference: TabAgentRanker, TabAgentClassifier, TabAgentRouter, pointwise and scorer."
kind: "Reference"
lead: "The full TabAgent recipe behind fit and predict, for a decision table you already have."
---

# scikit-learn estimators

The estimators infer a feature card from your columns, pick TabSynth and the candidate prior by nested grouped
validation on the training data only, and fit a CatBoost TabHead. They follow the scikit-learn contract (`clone`,
`get_params`, `cross_val_score` with `GroupKFold`). The guide is [Bring your own table](../guides/bring-your-own-table.md).

```python
from tabagent import TabAgentRanker

ranker = TabAgentRanker(group_col="task_id", candidate_col="tool")
ranker.fit(X, y)  # X: one row per (decision, candidate); y: 1 if the candidate was chosen
ranker.rank(X_new, k=5)  # top-5 candidates per decision
```

::: tabagent.sklearn.TabAgentRanker
    options:
      inherited_members: true
      members:
        - fit
        - rank
        - predict
        - predict_proba
        - decision_function
        - score
        - report

::: tabagent.sklearn.TabAgentRouter
    options:
      members:
        - route
        - predict
        - score

::: tabagent.sklearn.TabAgentClassifier
    options:
      members:
        - predict
        - decision_function

::: tabagent.sklearn.pointwise

::: tabagent.sklearn.scorer
