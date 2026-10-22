---
title: "TabSchema, TabSynth, TabHead"
description: "API reference: the paper's three components, TabSchema, TabSynth and TabHead, and TabAgent, which joins them."
kind: "Reference"
lead: "The three phases of the paper under their own names."
---

# TabSchema, TabSynth, TabHead

`TabAgent` joins a feature card, an optional TabSynth and a TabHead into one model that `cross_validate` can clone,
fit and score. The concepts are on [The three phases](../concepts/three-phases.md).

```python
from tabagent import TabAgent, TabHead, TabSynth, datasets

card = datasets.card("skillsbench/skill-select")
agent = TabAgent(card, head=TabHead("ranking"), synth=TabSynth(k=2))
```

::: tabagent.agent.TabAgent
    options:
      members:
        - fit
        - decide
        - score
        - clone
        - save
        - load

::: tabagent.tabhead.TabHead
    options:
      members:
        - fit
        - score
        - decode

::: tabagent.tabhead.confidence

::: tabagent.tabhead.risk_coverage

::: tabagent.tabsynth.TabSynth
    options:
      members:
        - augment

::: tabagent.tabschema.orchestrator.TabSchema
    options:
      members:
        - discover

::: tabagent.tabschema.orchestrator.TabSchemaConfig
    options:
      show_if_no_docstring: false
      members: false

::: tabagent.tabschema.verify.verify_head
