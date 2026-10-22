---
title: "Decision kinds"
description: "One reduction for every closed-set decision: a row is a (decision context, candidate) pair. The four decision kinds, how each is decoded and scored, and the decision table."
lead: "Ranking, multi-label, single-class and policy decisions all become one pointwise table."
kind: Concept
---

# Decision kinds

An agent repeatedly makes decisions whose answer comes from a fixed set: shortlist tools or skills, pick an
application or an operation, route to a model or a sub-agent, activate a skill or not. TabAgent casts all of
them as one pointwise problem (paper Sec. 3): **a row is a (decision context, candidate) pair**, its columns
are behavior-grounded features, and its label says whether the agent's successful run chose that candidate.

| kind | examples | decoding | metric |
|---|---|---|---|
| `ranking` | tool, skill, API, GUI-element shortlisting | top-k by score | P@R (R-precision), Recall@k |
| `multilabel` | application selection | score ≥ threshold, back off to the best one | example-based precision / recall / F1 |
| `single` | operation type, task category | argmax class | accuracy, macro-F1 |
| `policy` | model routing, sub-agent routing, skill gating | argmax predicted success (optionally cost-aware) | route@1 = reward of the chosen arm |

When a method gives several candidates the same score, the metrics order the tied candidates uniformly at
random and report the exact expected value, for every method (paper Sec. 4.3 and App. L); a deployed head
breaks ties by candidate id so that its decisions are deterministic.

## The decision table

`DecisionTable` pairs a DataFrame with a `DecisionSpec` naming the structural columns:

* `group`: the task id. Cross-validation never splits a group across folds.
* `decision`: the columns identifying one decision (one candidate slate); by default the group.
* `candidate`, `label`, and, for policies, `reward`.
* `eval`: post-decision columns kept for scoring only.

Outcome columns (label, reward, eval) can never be features, and models score *label-free* views of the table.
