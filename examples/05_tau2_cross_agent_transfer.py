"""Train on one agent's traces, deploy on another's (paper Table 2): τ²-bench, 67-tool catalog."""

from tabagent import TabAgent, TabHead, datasets
from tabagent.evaluation import metrics

train_name, test_name = "tau2/claude-3-7-sonnet-20250219", "tau2/o4-mini-2025-04-16"
train, test = datasets.load(train_name, download=True), datasets.load(test_name, download=True)
folds = datasets.folds(train_name)  # one task-level split shared by all four agents

held_out = {g for g, f in folds.items() if f == 0}
agent = TabAgent(datasets.card(train_name), head=TabHead("ranking"))
agent.fit(train.subset([g for g in train.groups if str(g) not in held_out]))
target = test.subset([g for g in test.groups if str(g) in held_out])
scores = agent.score(target.drop_labels())
print("P@R on the other agent's held-out tasks:", metrics.p_at_r(target, scores).mean())
