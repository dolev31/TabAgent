---
title: "Core concepts"
description: "The ideas behind TabAgent in one page: closed-set decisions, the decision table, the three phases, the feature card and the confidence gate."
kind: "Get started"
lead: "Five ideas, each with a page of its own in the Concepts section."
---

# Core concepts

## A closed-set decision

An agent repeatedly makes decisions whose answer comes from a fixed set: shortlist tools or skills, pick an
application or an operation, route to a model or a sub-agent, activate a skill or not. TabAgent replaces the LLM
call that makes such a decision. It does not generate free text or plan. If a component writes free text, it is not
a candidate; `tabagent scan` says so.

## One row per (context, candidate)

TabAgent casts every closed-set decision as one pointwise problem: **a row is a (decision context, candidate)
pair**, its columns are behavior-grounded features, and its label says whether the agent's successful run chose that
candidate. A head scores every candidate of a decision in one pass.

| Kind | Example | Decoding |
|---|---|---|
| `ranking` | tool, skill, API, GUI-element shortlisting | top-k by score |
| `multilabel` | application selection | score above a threshold, back off to the best one |
| `single` | operation type, task category | argmax class |
| `policy` | model routing, sub-agent routing, skill gating | argmax predicted success |

[Decision kinds](../concepts/decision-kinds.md) has the metrics and the decision table.

## Three phases

<div class="ta-phases" markdown>

<div class="ta-phase ta-phase--schema" markdown>

**TabSchema**

Discovers features. LLM agents analyze the successful trajectories, write and sandbox-test extraction code, and a
judge ensemble keeps the features that matter.
</div>

<div class="ta-phase ta-phase--synth" markdown>

**TabSynth**

Augments the rows. Deterministic, label-preserving paraphrases of the free-text state, where trajectories are
scarce.
</div>

<div class="ta-phase ta-phase--head" markdown>

**TabHead**

Decides in milliseconds. A CatBoost classifier scores every candidate in one pass.
</div>

</div>

Each phase runs once per agent and catalog. At inference the head replaces the LLM call; the rest of the agent is
unchanged. [The three phases](../concepts/three-phases.md) goes through each.

## The feature card

The tabular layout TabAgent learns from, with one row per (context, candidate) pair and one column per feature.
Features come in three families, as in the paper's Fig. 1: schema, state and dependency. Dependency features are
fitted statistics (a candidate's prior, co-usage rates, embedding similarity, k-NN outcomes), fitted on the training
rows only. A card is saved as YAML (`card.yaml`) next to the head.
[Per-candidate features](../concepts/per-candidate-features.md) covers features that relate the context to a
candidate.

## The confidence gate

A head's confidence says how sure it is. A decision below a threshold is deferred: the agent calls the original LLM
instead. [The confidence gate](../concepts/confidence-gate.md) explains how the threshold is calibrated and what
it costs.

## Leakage safety

Every data-dependent stage is fitted inside the training portion of each cross-validation fold, so a reported
number is an estimate for new tasks. [Leakage safety](../concepts/leakage-safety.md) says how, and where the one
exception is.
