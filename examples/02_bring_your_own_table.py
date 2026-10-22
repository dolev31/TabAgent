"""Bring your own decisions as a table: scikit-learn estimators with automatic card and recipe selection.

The toy data below stands in for your logs: one row per decision with the tools the agent used. Runs offline
on CPU in about a minute.
"""

import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score

import tabagent
from tabagent import DecisionTable, TabAgentRanker

TOOLS = pd.DataFrame(
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
REQUESTS = [
    ("find a flight to {c} and book it", ["search_flights", "book_flight"]),
    ("what is the weather in {c} this weekend", ["get_weather"]),
    ("reply to the email about my trip to {c}", ["read_inbox", "send_email"]),
    ("play some music from {c}", ["play_song"]),
]
CITIES = ["rome", "oslo", "lima", "cairo"]
rng = np.random.default_rng(0)
decisions = pd.DataFrame(
    [
        {"task_id": i, "request": req.format(c=rng.choice(CITIES)), "tools_used": used}
        for i, (req, used) in enumerate(REQUESTS[rng.integers(len(REQUESTS))] for _ in range(80))
    ]
)

# One row per (decision, candidate); `chosen` is 1 when the agent used the tool.
rows = tabagent.pointwise(decisions, chosen="tools_used", catalog=TOOLS, candidate_col="tool")
X, y = rows.drop(columns=["chosen"]), rows["chosen"]

# pair_embeddings=False keeps the example offline; with tabagent[embed] installed, the default adds an e5
# context-candidate similarity feature (the encoder is downloaded once, about 128 MB).
ranker = TabAgentRanker(group_col="task_id", candidate_col="tool", pair_embeddings=False)
print(cross_val_score(ranker, X, y, groups=X["task_id"], cv=GroupKFold(4), scoring=tabagent.scorer("p@r")))

ranker.fit(X, y)
print(ranker.selection_)  # the recipes tried by nested validation, and their scores
print(ranker.report(X, y, n_splits=4))  # quality vs baselines, latency and cost per decision
new = pd.DataFrame({"task_id": 999, "request": "find a flight to paris and book it"}, index=[0])
new = new.merge(TOOLS, how="cross")  # the new decision, one row per candidate tool
print(ranker.rank(new, k=2))

# A portable, pickle-free artifact. Reloaded, it is a TabAgent: it decides on rows that carry the columns it
# was trained on, with the decision and candidate ids in the columns its spec names (the estimators call them
# __group__ and __candidate__). See docs/guides/deploy-and-serve.md.
out = tempfile.mkdtemp(prefix="tabagent_ranker_")
ranker.agent_.save(out)
agent = tabagent.load(out)
spec = agent.card_.spec
rows = new.assign(**{spec.group: new["task_id"], spec.candidate: new["tool"]})
print(agent.decide(DecisionTable(rows, spec, has_labels=False), k=2))
print(f"saved to {out}; try: python examples/04_deploy.py {out}")
