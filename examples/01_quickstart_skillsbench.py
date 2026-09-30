"""Quickstart on public data: SkillsBench skill selection (paper Table 1), CPU, a few minutes.

On first run the table is rebuilt from the pinned SkillsBench release (3.5 MB, SHA-256 verified), and the
e5-small-v2 encoder behind the card's similarity feature is downloaded from Hugging Face once (about 128 MB).
"""

from tabagent import TabAgent, TabHead, cross_validate, datasets
from tabagent.evaluation.cv import PAPER_SEEDS

NAME = "skillsbench/skill-select"

table = datasets.load(NAME, download=True)
card = datasets.card(NAME)  # the paper's feature card for this decision
agent = TabAgent(card, head=TabHead("ranking"))  # CatBoost with the paper's configuration

# The paper's task-level folds and seeds.
result = cross_validate(agent, table, folds=datasets.folds(NAME), seeds=PAPER_SEEDS)
print(result.summary(["p@r", "recall@5", "recall@9"]))

# Fit on every task but one, then shortlist skills for the held-out task.
held_out = table.groups[0]
agent.fit(table.subset([g for g in table.groups if g != held_out]))
task = table.subset([held_out])
print(agent.decide(task.drop_labels(), k=5))  # the five skills TabAgent would hand the agent
print("skills in the task's curated bundle:", sorted(task.df.loc[task.labels == 1, "skill_name"]))
