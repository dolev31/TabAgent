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
