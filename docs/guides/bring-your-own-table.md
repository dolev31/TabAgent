---
title: "Bring your own table"
description: "Use TabAgentRanker, TabAgentClassifier and TabAgentRouter, the scikit-learn estimators, on a decision table you already have: automatic card and recipe selection, cross-validation and saving."
kind: "Guide"
lead: "If your decisions are already a table, you get the full TabAgent recipe through the familiar fit and predict API."
needs: "a table with one row per decision, or per (decision, candidate); no LLM"
---

# Bring your own table

If your decisions are already a table, use the scikit-learn estimators. They need no LLM and no log adapters.

## The three estimators

| Estimator | Decision | Main call |
|---|---|---|
| `TabAgentRanker` | ranking (tool, skill, API, GUI-element shortlisting); `predict` also does multi-label decoding (threshold plus backoff) | `rank(X, k=5)` |
| `TabAgentClassifier` | single-label decisions (operation type, task category); a standard scikit-learn classifier | `predict(X)` |
| `TabAgentRouter` | routing (model, sub-agent, skill on or off); `y` is the arm's success, 0 or 1 | `route(X)`; `score` is route@1 |

## 1. Shape the table

Estimators learn from one row per (decision, candidate). If you have one row per decision with the candidates the
agent used, `tabagent.pointwise` expands it:

```python
import tabagent
from tabagent import TabAgentRanker

rows = tabagent.pointwise(decisions, chosen="tools_used", catalog=tools_df, candidate_col="tool")
X, y = rows.drop(columns=["chosen"]), rows["chosen"]
```

`chosen` names a column of lists with the candidates the agent selected. `catalog` is an iterable of candidate ids or
a DataFrame whose `candidate_col` holds them (its other columns, such as descriptions, are joined onto every row).
`candidates` optionally names a column of per-decision candidate lists, when each decision has its own legal set.

## 2. Fit

```python
ranker = TabAgentRanker(group_col="task_id", candidate_col="tool").fit(X, y)
ranker.rank(X_new, k=5)
print(ranker.report(X, y))  # quality vs BM25 and chance, latency and cost per decision
```

`group_col` names the decision group (the task): rows of one group never straddle cross-validation folds.

What `fit` does:

1. **Infers a feature card** from your columns: types; which text is the decision's context and which is the
   candidate's; a context-candidate lexical overlap; with `tabagent[embed]`, an e5 similarity (the encoder is
   downloaded once, about 128 MB, or left out with `pair_embeddings=False`); and an in-fold candidate prior.
2. **Selects the recipe** by nested grouped validation on the training data only: TabSynth off, k=2 or k=4, and the
   prior on or off.
3. **Fits CatBoost** with the paper's configuration.

`ranker.selection_` shows every recipe it tried.

## 3. Cross-validate

The estimators follow the scikit-learn contract (`clone`, `get_params`), so they work in model selection with
grouped folds and `tabagent.scorer`:

```python
from sklearn.model_selection import GroupKFold, cross_val_score

scores = cross_val_score(ranker, X, y, groups=X["task_id"], cv=GroupKFold(4), scoring=tabagent.scorer("p@r"))
```

`tabagent.scorer(metric)` takes any metric name of `tabagent.evaluation.metrics.per_decision`: `p@r`, `recall@K`,
`hit@K`, `route@1`, `accuracy`, `ml_precision`, `ml_recall`, `ml_f1`. See
[Choose and evaluate a head](choose-and-evaluate-a-head.md).

## 4. Save and reload

```python
ranker.agent_.save("my_tool_ranker")  # a portable, pickle-free artifact
agent = tabagent.load("my_tool_ranker")  # a TabAgent
```

A reloaded `TabAgent` has no extractor, so it decides on rows that already carry the columns it was trained on, with
the decision and candidate ids in the columns its spec names (the estimators call them `__group__` and
`__candidate__`). [Deploy and serve](deploy-and-serve.md) shows the call.

!!! tip "A complete, runnable example"

    `examples/02_bring_your_own_table.py` builds a toy log, cross-validates, fits, reports, saves and reloads. It
    runs offline on CPU in about a minute.
