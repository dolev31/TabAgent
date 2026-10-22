---
title: "Quickstart (5 minutes)"
description: "Rank an agent's tools from a decision table with TabAgentRanker on CPU, offline, then reproduce a paper number on SkillsBench."
kind: "Get started"
lead: "Train a tool ranker from a table in under a minute, then check a paper number on a public benchmark."
needs: "pip install tabagent, and nothing else for the first part"
---

# Quickstart (5 minutes)

## 1. Rank tools from a table

If your agent's decisions are already a table, you do not need an LLM, the logs adapters or the network. This
example runs on CPU in about a minute. It builds a toy log of 80 decisions (a request, and the tools the agent
used on its successful run), expands it to one row per (decision, candidate), and trains a ranker.

```python
import pandas as pd
import tabagent
from tabagent import TabAgentRanker

# The tools an agent can call, with a description each.
tools = pd.DataFrame(
    {
        "tool": ["search_flights", "book_flight", "get_weather", "send_email", "read_inbox", "play_song"],
        "tool_description": [
            "search flights between two airports",
            "book a seat on a flight",
            "weather forecast for a city",
            "send an email message",
            "read the latest emails",
            "play a song or playlist",
        ],
    }
)

# One row per decision: the request, and the tools the agent used on its successful run.
requests = [
    ("find a flight to {c} and book it", ["search_flights", "book_flight"]),
    ("what is the weather in {c} this weekend", ["get_weather"]),
    ("reply to the email about my trip to {c}", ["read_inbox", "send_email"]),
    ("play some music from {c}", ["play_song"]),
]
cities = ["rome", "oslo", "lima", "cairo"]
decisions = pd.DataFrame(
    [
        {"task_id": i, "request": text.format(c=cities[i % 4]), "tools_used": used}
        for i, (text, used) in enumerate(requests[i % 4] for i in range(80))
    ]
)

# Expand to one row per (decision, candidate); `chosen` is 1 when the agent used the tool.
rows = tabagent.pointwise(decisions, chosen="tools_used", catalog=tools, candidate_col="tool")  # (1)!
X, y = rows.drop(columns=["chosen"]), rows["chosen"]

ranker = TabAgentRanker(group_col="task_id", candidate_col="tool", pair_embeddings=False)  # (2)!
ranker.fit(X, y)  # (3)!

# A new decision: one row per candidate tool.
new = pd.DataFrame({"task_id": 999, "request": "find a flight to paris and book it"}, index=[0])
new = new.merge(tools, how="cross")
print(ranker.rank(new, k=2))  # (4)!
```

1. `pointwise` turns "one row per decision, with the tools used" into the layout TabAgent learns from: one row per
   (decision, candidate), with a 0/1 label. See [Decision kinds](../concepts/decision-kinds.md).
2. `pair_embeddings=False` keeps the example offline. With `tabagent[embed]` installed, the default adds an e5
   context-candidate similarity feature (the encoder is downloaded once, about 128 MB).
3. `fit` infers a feature card from your columns, then chooses TabSynth (off, k=2 or k=4) and the candidate prior
   by nested grouped validation on the training data only, and fits a CatBoost head. `ranker.selection_` lists
   every recipe it tried.
4. The top-2 candidates per decision, best first.

It prints the two tools the agent would hand the LLM, in rank order:

```text
  decision       candidate     score  rank
0      999     book_flight  0.998117     0
1      999  search_flights  0.985570     1
```

(The scores vary slightly between runs and machines; the ranking does not.)

Next, see how good the head is, with grouped cross-validation against BM25 and chance:

```python
print(ranker.report(X, y, n_splits=4))  # quality with 95% intervals, latency and cost per decision
```

The same flow is in `examples/02_bring_your_own_table.py`. [Bring your own table](../guides/bring-your-own-table.md)
covers real data.

## 2. Reproduce a paper number

The paper reports P@R 80 and Recall@9 91 for skill selection on SkillsBench. This reproduces it with the paper's
feature card, task-level folds and seeds, on a laptop CPU in a few minutes (it downloads 3.5 MB of data and the
128 MB encoder once):

```bash
git clone https://github.com/dolev31/TabAgent && cd TabAgent
pip install -e ".[embed]"
python examples/01_quickstart_skillsbench.py
```

The core of that script:

```python
from tabagent import TabAgent, TabHead, cross_validate, datasets
from tabagent.evaluation.cv import PAPER_SEEDS

NAME = "skillsbench/skill-select"
table = datasets.load(NAME, download=True)  # rebuilt from the pinned upstream release, SHA-256 verified
card = datasets.card(NAME)  # the paper's feature card for this decision
agent = TabAgent(card, head=TabHead("ranking"))  # CatBoost with the paper's configuration

result = cross_validate(agent, table, folds=datasets.folds(NAME), seeds=PAPER_SEEDS)
print(result.summary(["p@r", "recall@5", "recall@9"]))
```

It prints P@R of about 0.80 and Recall@9 of about 0.90, with 95% bootstrap intervals:

```text
             value    ci_low   ci_high   n
metric
p@r       0.795731  0.739982  0.853777  87
recall@5  0.852974  0.800928  0.904767  87
recall@9  0.904899  0.863278  0.945206  87
```

[Reproduce the paper](../reproducing.md) lists every public result with its command, the value the
paper reports and the value the released code produces.

## 3. Your own agent

Everything above started from a table. To start from an agent's logs, with an LLM discovering the features:
[Onboard your agent](onboard-your-agent.md).
