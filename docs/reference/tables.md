---
title: "Tables and cards"
description: "API reference: DecisionTable, DecisionSpec, FeatureCard and Feature."
kind: "Reference"
lead: "The pointwise decision table and the feature card that turns it into columns."
---

# Tables and cards

A `DecisionTable` pairs a DataFrame with a `DecisionSpec` that names the structural columns. A `FeatureCard`
declares the features computed from it. See [Decision kinds](../concepts/decision-kinds.md).

::: tabagent.table.DecisionSpec

::: tabagent.table.DecisionTable
    options:
      members:
        - subset
        - drop_labels
        - groups
        - labels
        - decision_ids

::: tabagent.table.LeakageError

::: tabagent.tabschema.card.FeatureCard

::: tabagent.tabschema.card.Feature
