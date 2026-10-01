# Deployment

A saved model is a directory with no pickle: `card.yaml`, fitted statistics (`fitted.json`, `fitted.npz`),
`featurizer.json`, CatBoost's `model.cbm`, and a `manifest.json` with the SHA-256 of every file (a modified
artifact is refused). A head saved by `onboard` adds the compiled extractor `extract.py`. `tabagent.load(path)`
reads both kinds, and the kind decides what a decision takes as input:

| saved by | `tabagent.load` returns | decide from | call |
|---|---|---|---|
| `tabagent.onboard(..., out=...)` or `tabagent onboard --out` | an onboarded head (with `extract.py`, the compiled TabSchema extractor) | a raw trajectory prefix, in the log format it was trained on | `head.decide(prefix, candidates, k=9)` → `{"candidates", "scores", "confidence", "deferred"}` (`"choice"` instead of `"candidates"` for a single-class head) |
| `TabAgent.save` (also `ranker.agent_.save(...)` from the scikit-learn estimators) | a `TabAgent` (no extractor) | rows that already carry the columns it was trained on, one per candidate, with the decision and candidate ids in the columns its spec names | `agent.decide(rows, k=9)`, with `rows` a `DecisionTable` or a DataFrame of one decision's candidate rows → a DataFrame (decision, candidate, score, rank) |

The HTTP service and the tool filter below need an onboarded head, since they receive raw prefixes. Calling
`decide(prefix, candidates, ...)` on a `TabAgent` raises a `TypeError` that says so.

```python
import tabagent

head = tabagent.load("my_head", trust_extractor=True)  # onboarded; trust only an extractor you have reviewed
head.decide(prefix, candidates, k=9)
```

```python
import tabagent
from tabagent import DecisionTable

agent = tabagent.load("my_tool_ranker")  # a TabAgent artifact
spec = agent.card_.spec  # the scikit-learn estimators name the id columns __group__ and __candidate__
rows = new_decision.merge(tools, how="cross")  # one row per candidate, with the training columns
rows = rows.assign(**{spec.group: rows["task_id"], spec.candidate: rows["tool"]})
agent.decide(DecisionTable(rows, spec, has_labels=False), k=5)
```

[`examples/04_deploy.py`](https://github.com/dolev31/TabAgent/blob/main/examples/04_deploy.py) runs either
kind.

**HTTP service** (`pip install "tabagent[serve]"`): `tabagent serve my_head --port 8077`, then
`POST /decide {"prefix": {...}, "candidates": [...], "k": 9}`.

**In front of a tool-calling LLM**: `ToolFilter` sends the model only the shortlisted tools, and the full list
when the head is unsure.

```python
from tabagent.integrations.openai_tools import ToolFilter

tool_filter = ToolFilter("my_head", k=9)
client.chat.completions.create(model=..., messages=messages, tools=tool_filter(messages, tools))
```

**Confidence gate**: `onboard(..., gate_coverage=0.9)` calibrates a threshold so that 90% of decisions stay
with the head and the least confident 10% go back to the LLM (paper App. N.3, "Confidence-gated fallback";
App. O.2 runs the gate in the loop).

## Pretrained heads and the demo

`tabagent.load` also takes a Hugging Face Hub repository: `hf://OWNER/REPO`, or `hf://OWNER/REPO@REVISION` with
a commit, tag or branch (pin one for a reproducible deployment). It needs the hub extra
(`pip install "tabagent[hub]"`), downloads the repository with `huggingface_hub.snapshot_download` into the
Hub's local cache, and then loads it exactly as a local directory: every file is checked against
`manifest.json`, nothing is unpickled, and `trust_code` and `trust_extractor` mean the same.
`TabAgent.load`, `TabAgentHead.load`, `ToolFilter` and `tabagent serve` accept the same URIs. To share a head,
upload the directory that `save` wrote to a model repository (`hf upload OWNER/REPO my_head .`).

<!-- named -->

Two heads trained with the released code on public data are on the Hub. Each is a `TabAgent` artifact (no
extractor), so it decides from candidate rows:

| Hub repository | decision | fitted on | paper, 5-fold task-level CV | released code, same protocol |
|---|---|---|---|---|
| [dolev31/tabagent-skillsbench-skill-selection](https://huggingface.co/dolev31/tabagent-skillsbench-skill-selection) | skill selection over the 195 SkillsBench skills | all 87 SkillsBench tasks | P@R 80, Recall@9 91 | P@R 79.6, Recall@9 90.5 |
| [dolev31/tabagent-tau2-tool-shortlisting](https://huggingface.co/dolev31/tabagent-tau2-tool-shortlisting) | tool shortlisting over the 38 agent tools of τ²-bench airline, retail and telecom | all 326 tau2-sft-final trajectories | P@R 78, Recall@9 97 (+synth head) | P@R 77.8, Recall@9 97.3 |

Both were fitted once on their full table with the paper's settings: the bundled card, CatBoost with 300
iterations, learning rate 0.08 and depth 6, the first of the paper's three seeds, and for τ²-bench the TabSynth
configuration that the paper's inner cross-validation chooses. The quality columns estimate the recipe on
held-out tasks; a checkpoint fitted on all tasks has no held-out score of its own. The model cards give the
data revisions, usage and limitations.

```python
import tabagent
from tabagent import datasets

head = tabagent.load("hf://dolev31/tabagent-skillsbench-skill-selection")  # pip install "tabagent[embed,hub]"
catalog = datasets.load("skillsbench/skill-select", download=True).df
catalog = catalog[["skill_name", "skill_desc"]].drop_duplicates()
head.decide(catalog.assign(task_text="Recover the missing cells of a damaged Excel workbook."), k=5)
```

The [TabAgent skill shortlisting Space](https://huggingface.co/spaces/dolev31/tabagent-skill-shortlisting)
runs the SkillsBench head live: pick a SkillsBench task or type a task description, and it shows the five
skills handed to the agent with their scores, the decision latency measured on the Space's CPU, and, for a
SkillsBench task, the task's curated skill bundle. Both heads and the Space accompany the paper
([arXiv:2602.16429](https://arxiv.org/abs/2602.16429)).

<!-- /named -->
