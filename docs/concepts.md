# Concepts

## One reduction for every closed-set decision

An agent repeatedly makes decisions whose answer comes from a fixed set: shortlist tools or skills, pick an
application or an operation, route to a model or a sub-agent, activate a skill or not. TabAgent casts all of
them as one pointwise problem (paper Sec. 3): **a row is a (decision context, candidate) pair**, its columns
are behavior-grounded features, and its label says whether the agent's successful run chose that candidate.

| kind | examples | decoding | metric |
|---|---|---|---|
| `ranking` | tool, skill, API, GUI-element shortlisting | top-k by score, ties by candidate id | P@R (R-precision), Recall@k |
| `multilabel` | application selection | score ≥ threshold, back off to the best one | example-based precision / recall / F1 |
| `single` | operation type, task category | argmax class | accuracy, macro-F1 |
| `policy` | model routing, sub-agent routing, skill gating | argmax predicted success (optionally cost-aware) | route@1 = reward of the chosen arm |

## The decision table

`DecisionTable` pairs a DataFrame with a `DecisionSpec` naming the structural columns:

* `group`: the task id. Cross-validation never splits a group across folds.
* `decision`: the columns identifying one decision (one candidate slate); by default the group.
* `candidate`, `label`, and, for policies, `reward`.
* `eval`: post-decision columns kept for scoring only.

Outcome columns (label, reward, eval) can never be features, and models score *label-free* views of the table.

## Leakage safety by construction

The paper requires every data-dependent stage to be fitted inside the training portion of each fold (L443).
In TabAgent that is structural, not a convention:

* `cross_validate` is the only code that decides what is training data; for every fold and seed it clones the
  model, fits it on the training groups and scores the held-out groups through a label-free view.
* Dependency statistics (candidate priors, co-usage, embedding centering, k-NN outcomes) are `fitted` card
  features, fitted on real training rows only; each records the groups it saw.
* TabSynth runs only inside `fit`, on training rows.
* TabSchema's extraction code sees only the trajectory *prefix* before the decision, with outcome fields removed.
* A tripwire refuses a feature that reproduces the label.

The test suite proves these properties (scrambling held-out labels does not change held-out scores; scoring is
not transductive; fitted statistics only see training groups).

## The protocol

5-fold task-level cross-validation (the paper's folds are bundled with each public dataset), three seeds
(20260720, 20260726, 20260727), macro average over decisions, 95% bootstrap intervals, and paired task-level
bootstraps with 2,000 resamples for comparisons (an improvement is claimed only when the interval excludes 0).
