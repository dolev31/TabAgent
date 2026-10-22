---
title: "Choose and evaluate a head"
description: "Pick a TabHead preset or plug in another classifier, evaluate it with task-grouped cross-validation, paired bootstrap intervals and the paper's metrics, and estimate the whole pipeline with discovery inside each fold."
kind: "Guide"
lead: "Which metric, which head, and how to tell a real difference from noise."
needs: "a decision table (or an onboarding run); no LLM except for the last section"
---

# Choose and evaluate a head

## Pick the metric for the decision kind

| Kind | Examples | Metric | Name in code |
|---|---|---|---|
| `ranking` | tool, skill, API, GUI-element shortlisting | P@R (R-precision), Recall@k | `p@r`, `recall@5` |
| `multilabel` | application selection | example-based precision, recall and F1 | `ml_precision`, `ml_recall`, `ml_f1` |
| `single` | operation type, task category | accuracy, macro-F1 | `accuracy` |
| `policy` | model routing, sub-agent routing, skill gating | route@1, the reward of the chosen arm | `route@1` |

P@R is the share of a task's ground-truth candidates in the top R of the ranking, where R is the size of the
relevant set; it equals Recall@R and captures completeness. Recall@k is the share recovered in the top k, at a fixed
budget. When a method gives several candidates the same score, the metrics order the tied candidates uniformly at
random and report the exact expected value, for every method (paper Sec. 4.3 and App. L). Details are in
[Decision kinds](../concepts/decision-kinds.md).

## Evaluate with `cross_validate`

`cross_validate` is the only code that decides what is training data. For every fold and seed it clones the model,
fits it on the training groups and scores the held-out groups through a label-free view of the table.

```python
from tabagent import TabAgent, TabHead, cross_validate, datasets
from tabagent.evaluation.cv import PAPER_SEEDS

name = "skillsbench/skill-select"
table = datasets.load(name, download=True)
agent = TabAgent(datasets.card(name), head=TabHead("ranking"))

result = cross_validate(agent, table, folds=datasets.folds(name), seeds=PAPER_SEEDS)
print(result.summary(["p@r", "recall@9"]))  # estimate and 95% bootstrap interval over tasks
```

The protocol is 5-fold task-level cross-validation (the paper's folds are bundled with each public dataset), three
seeds, macro average over tasks (over steps for Mind2Web element selection), 95% bootstrap intervals, and paired
task-level bootstraps with 2,000 resamples for comparisons. For your own table, leave `folds` out and set
`n_splits`, `fold_seed` and `stratify`.

## Compare two heads

A difference is real only when its paired interval excludes zero (paper App. M):

```python
default = cross_validate(
    TabAgent(datasets.card(name), head=TabHead("ranking", preset="catboost-default")),
    table,
    folds=datasets.folds(name),
    seeds=PAPER_SEEDS,
)
diff = result.compare(default, "p@r")  # paired bootstrap of (result - default) over tasks
print(diff, diff.excludes_zero)
```

`Interval.excludes_zero` is the paper's significance rule. Within one metric across several applications, correct for
multiple comparisons with `tabagent.evaluation.holm`.

## Choose a classifier

`TabHead(kind, preset=...)` is CatBoost over a fold-fitted featurizer (TF-IDF for text, codes for categoricals,
medians for numbers). Two presets exist:

| Preset | Configuration |
|---|---|
| `paper` (default) | the one configuration the paper uses for every decision (Sec. 4.2): 300 iterations, learning rate 0.08, depth 6, CatBoost's defaults otherwise |
| `catboost-default` | CatBoost's own schedule (1000 iterations, automatic learning rate) |

Any scikit-learn-style classifier plugs in with `estimator=` (a `seed -> classifier` factory with `fit`,
`predict_proba` and `classes_`), as in the paper's head ablation (Fig. 3, App. K.3):

```python
from sklearn.ensemble import HistGradientBoostingClassifier

head = TabHead("ranking", estimator=lambda seed: HistGradientBoostingClassifier(random_state=seed))
```

The paper's finding is that the representation, not the model, does the work: with the card held fixed, heads from
classical trees to tabular foundation models reach comparable P@R, and every head collapses to P@R 0.10 to 0.21
without the card. `experiments.figures.head_ablation` reproduces the comparison ([Reproduce the
paper](../reproducing.md)).

## Read an onboarding report

`head.report` (and `ranker.report(X, y)`) compares the head with the training-free baselines on the same folds:
TabAgent, chance and BM25, each with a 95% interval, then the per-decision latency and its cost under the paper's
CPU cost model (App. C: 4 threads at $0.063 per vCPU-hour), and the CatBoost feature importance summed over each
feature's columns. Pass `llm_cost_per_call` (CLI `--llm-cost-per-call`) to see the savings against the LLM component
you replace. `report.save(dir)` writes `report.md` and `report.json`.

## Estimate the whole pipeline, discovery included

`onboard` discovers the feature card once, on all the onboarding trajectories, and then cross-validates the
head on that card. Its folds keep the head's training rows and fitted statistics apart, but every held-out task
was among the trajectories the card was discovered from, so the report estimates the head on a fixed card: a
quick estimate (one discovery) that can be optimistic. The paper's protocol fits the TabSchema loop inside each
training fold too (Sec. 4.3). `tabagent.cross_validate_discovery` does that:

```python
import tabagent
from tabagent.tabschema import TabSchemaConfig

res = tabagent.cross_validate_discovery(
    "./logs",
    "ToolShortlist",
    config=TabSchemaConfig(max_cost_usd=100, prices={"gpt-5.4": (PRICE_IN, PRICE_OUT)}),  # all folds together
    run_dir="./infold_runs",  # one cached run directory per fold: a rerun makes no new LLM calls
)
print(res.summary(["p@r", "recall@9"]))  # the usual CVResult metrics and intervals
res.discoveries[0].card  # the card discovered on fold 0's training tasks
```

For every fold it runs discovery on the training fold's trajectories only, builds the training and held-out
rows with that fold's extractor, selects the training recipe on the training rows (as `onboard` does on all of
them) and scores the held-out decisions. With the same `n_splits`, its folds are those of `onboard`'s report,
so the two estimates can be compared directly. It costs one discovery per fold, about five times the LLM cost
of onboarding with the default five folds; one budget covers all folds. The folds' discoveries run three at a
time by default (`fold_workers=`), each with an equal share of the budget, so the wall time is about that of
two onboardings rather than five. `redact=` works as in `onboard`, and
`sandbox=DockerSandbox()` runs the generated code in Docker. The head you deploy is still the one `onboard` fits
on all the trajectories; this estimates how well that procedure generalizes to new tasks.
