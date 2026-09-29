"""Quickstart on public data: SkillsBench skill selection (paper Table 1), CPU, a few minutes.

On first run the table is rebuilt from the pinned SkillsBench release (a few MB, SHA-256 verified).
"""

from tabagent import FeatureCard, TabAgent, TabHead, cross_validate, datasets

NAME = "skillsbench/skill-select"

table = datasets.load(NAME, download=True)
card = FeatureCard.load("skillsbench_skill_select_noprior")  # the paper card is datasets.card(NAME)
agent = TabAgent(card, head=TabHead("ranking"))

result = cross_validate(agent, table, folds=datasets.folds(NAME))  # the paper's task-level folds
print(result.summary(["p@r", "recall@5", "recall@9"]))

agent.fit(table)
one_task = table.subset(table.groups[:1]).drop_labels()
print(agent.decide(one_task, k=5))  # the five skills TabAgent would hand the agent
