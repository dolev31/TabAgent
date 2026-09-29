# Deployment

A head saved by `onboard` (or `TabAgent.save`) is a directory with no pickle: `card.yaml`, `extract.py`,
fitted statistics (`fitted.json`, `fitted.npz`), `featurizer.json`, CatBoost's `model.cbm`, and a
`manifest.json` with the SHA-256 of every file (a modified artifact is refused).

```python
import tabagent

head = tabagent.load("my_head", trust_extractor=True)
head.decide(prefix, candidates, k=9)  # {"choice", "scores", "confidence", "defer"}
```

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
with the head and the least confident 10% go back to the LLM (paper App. D).
