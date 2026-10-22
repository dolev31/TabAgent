---
title: "Deploy and serve"
description: "Save, load and serve a TabAgent head: the two artifact kinds, live prefixes, the HTTP service, the sandbox, redaction and pretrained heads on the Hugging Face Hub."
lead: "From a saved head to a live decision: in-process, over HTTP, or from the Hugging Face Hub."
kind: Guide
needs: "a saved head"
---

# Deploy and serve

A saved model is a directory with no pickle: `card.yaml`, fitted statistics (`fitted.json`, `fitted.npz`),
`featurizer.json`, CatBoost's `model.cbm`, the training distribution of the head's inputs for drift monitoring
(`feature_stats.json`), and a `manifest.json` with the SHA-256 of every file (a modified artifact is refused). A
head saved by `onboard` adds the compiled extractor `extract.py`, its golden tests `extractor_tests.json`
(snapshots of onboarding prefixes; `tabagent verify-head` re-runs them, see
[How the generated code is verified](verify-generated-code.md)) and `head.json` (the
decision, the gate threshold and, when the head knows them, the log adapter with its prefix settings and the
redaction function's import path). The card pins the SHA-256 of `extract.py`, and loading refuses code that
does not match it, or a card that pins none. For a head saved by an earlier version, `allow_unpinned=True`
(`tabagent serve --allow-unpinned`) overrides the latter with a warning, and saving the head again pins the
hash. `tabagent.load(path)` reads both kinds, and the kind decides what a decision takes as input:

| saved by | `tabagent.load` returns | decide from | call |
|---|---|---|---|
| `tabagent.onboard(..., out=...)` or `tabagent onboard --out` | an onboarded head (with `extract.py`, the compiled TabSchema extractor) | the decision's trajectory prefix, as the log adapter built it in training; or the live run, from which the head builds that prefix | `head.decide(prefix, candidates, k=9)` or `head.decide(run=run, candidates=..., k=9)` → `{"candidates", "scores", "confidence", "deferred", "error"}` (`"choice"` instead of `"candidates"` for a single-class head) |
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

## Live prefixes

An onboarded head scores the prefix its log adapter produced in training (`DecisionPoint.prefix`), not the raw
log. For chat-message logs that is the run-level task fields the logs carry next to the messages (`instruction`,
`domain`, ...; never the task id, which only names the cross-validation group) and the messages normalized to
`{"name", "content", "tool_calls", "tool_name"}` steps; for CUGA and generic step logs, the trajectory so far
without its outcome fields. A live decision must reach the extractor in the same shape:

```python
run = {"instruction": goal, "messages": messages_so_far}  # the logged run's fields, up to the decision
head.live_prefix(run)  # the prefix exactly as in training (tabagent.adapters.live_prefix)
head.decide(run=run, candidates=tools, k=9)  # the same, in one call
```

`candidates` takes the tool schemas as the agent holds them (OpenAI function tools, the Anthropic Messages API
format, MCP tool listings), read exactly as the adapters read a training catalog, so their descriptions and
parameter names reach the head as in training; tool ids or `{"id", "description", "args"}` records work too.

`live_prefix` uses the adapter recorded in `head.json`. `tabagent onboard` records it; for a head onboarded from
Python, set `head.adapter = adapter` (the adapter that read the logs) before `save`, or pass `adapter=` to
`TabAgentHead.load` or `live_prefix`.

### When the extractor fails

When the extractor fails on a prefix (a field the training prefixes had is missing, the extractor raises,
the sandbox cannot start), the head does not crash: the extractor's features are left empty (None, exactly as
for a training decision point it failed on), the decision is `deferred` to the original LLM whatever its
confidence, and `error` says why. When the extractor read a top-level field the prefix lacks, `error` names it
and points to the run-level fields to pass.

### HTTP service

With `pip install "tabagent[serve]"`, run `tabagent serve my_head --port 8077`, then
`POST /decide {"prefix": {...}, "candidates": [...], "k": 9}`, or `{"run": {...}, ...}` to have the head build
the prefix from the run. A failed extraction is answered with `"deferred": true` and its `"error"`; an invalid
request (a ranking head without candidates, both a prefix and a run, a candidate without a name) with status 400.
The [production runbook](production-runbook.md) covers readiness, timeouts, metrics, logs and drift.

### Sandbox

An onboarded head runs its extractor in resource-limited `python -I -S` worker processes with
an empty environment by default (`sandbox="warm"`, a `WarmSandbox`; on POSIX, and a fresh interpreter per
decision on Windows). The workers start once, compile
`extract.py` once and then serve every decision, so the sandbox adds well under a millisecond to a decision
instead of an interpreter start-up (45 to 65 ms on our benchmark machine); `"warm:<N>"` sets the number of workers (default
`min(4, CPUs)`). `sandbox="subprocess"` starts a fresh interpreter per decision, as onboarding does for its
batches. `trust_extractor=True` (`--trust-extractor`) runs the extractor in-process, for an extractor you have
reviewed; `sandbox="docker"` (`tabagent serve --sandbox docker`; also `docker:<image>`, or a sandbox instance)
runs it in Docker with no network and a read-only filesystem. `TabAgentHead.load`, `tabagent.load` and
`ToolFilter` take `sandbox=` (see `SECURITY.md`). A head's warm workers start at its first decision
(`head.warm()` starts them now) and stop with `head.close()`, when the head is garbage-collected, or when the
process exits.

### Redaction

`onboard(..., redact=fn)` (`--redact module:function`) scrubs the trajectories before
discovery. A head applies the same function to live runs only when it is passed again:
`TabAgentHead.load(..., redact=fn)`, `ToolFilter(..., redact=fn)` or `tabagent serve --redact module:function`.
`head.json` records the function's import path when it is known, and loading such a head without `redact=`
warns. A head never imports it by itself, and `decide(prefix, ...)` takes the prefix as given.

### Confidence gate

`onboard(..., gate_coverage=0.9)` calibrates a threshold so that 90% of decisions stay
with the head and the least confident 10% go back to the LLM (paper App. N.3, "Confidence-gated fallback";
App. O.2 runs the gate in the loop).

See [The confidence gate](../concepts/confidence-gate.md) for what `gate_coverage` calibrates.

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
| [dolev31/tabagent-tau2-tool-shortlisting](https://huggingface.co/dolev31/tabagent-tau2-tool-shortlisting) | tool shortlisting over the 38 agent tools of τ²-bench airline, retail and telecom | all 326 tau2-sft-final trajectories | P@R 78, Recall@9 97 (+synth head) | P@R 77.6, Recall@9 97.3 |

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

Both heads accompany the paper ([arXiv:2602.16429](https://arxiv.org/abs/2602.16429)). To try one in the
browser, open the [Colab quickstart](https://colab.research.google.com/github/dolev31/TabAgent/blob/main/notebooks/quickstart.ipynb).

<!-- /named -->
