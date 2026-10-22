---
title: "Hot-swap into an agent"
description: "Replace the LLM call in a live agent: ToolFilter in front of a tool-calling LLM, the CUGA helpers, and whether filtering the tool list pays."
lead: "Send your LLM only the tools the head shortlists, and the full list whenever the head defers. The surrounding agent is unchanged."
kind: Guide
needs: "an onboarded head"
---

# Hot-swap into an agent

`ToolFilter` sends the model only the tools the head shortlists, and the
full list when the head defers (unsure, or its extractor failed). Start a session per conversation: it keeps
every tool the agent already called (`keep_used`) and every tool it once showed (`sticky`), appending newly
shown tools at the end in catalog order, so the list only grows; a `ToolFilter` itself keeps no state between
calls. Pass the run-level fields the training logs carried as `task`, once (`ToolFilter(..., task={"domain":
"airline"})`), per conversation (`session(task=...)`) or per call. The head may also be the URL of a `tabagent
serve` service (`ToolFilter("http://host:8077", ...)`), for an agent whose environment cannot host it.

```python
from tabagent.integrations.openai_tools import ToolFilter

tool_filter = ToolFilter("my_head", k=7)
session = tool_filter.session(task={"instruction": goal})  # one per conversation
client.chat.completions.create(model=..., messages=messages, tools=session(messages, tools))
```

## Does filtering the tool list pay?

Measured on τ²-bench (v1.0.1) with GPT-4.1 as agent and user simulator: 49 held-out airline and retail tasks,
5 trials, paired by task, a `NextToolCall` head trained on the other tasks' runs, top 7 of 14 to 16 tools plus
the tools already used, before every call ([details](https://github.com/dolev31/TabAgent/blob/main/experiments/live/README.md)):

| | pass rate vs. every tool | prompt tokens per call | agent cost per run |
|---|---|---|---|
| re-ranked at every call (`sticky=False`) | −6.9 points [−13.9, −0.4] | −18.5% | +21% |
| `sticky` (3 trials)¹ | +0.0 points [−9.5, +8.8] | −13.8% | +11% |

¹ Measured before `ToolSession`, with the same tools at every call but in catalog order; the session appends
newly shown tools instead, which leaves more of the list's prefix unchanged.

Both cost more than sending every tool, for two reasons. A tool list that changes between calls defeats the
provider's prompt caching (the share of prompt tokens served from the cache fell from 80% to 55%, and to 66%
with `sticky`). And τ²-bench's tool schemas are a small share of its prompts: hiding about half of them saved
620 to 1,050 of about 4,950 prompt tokens per call. Without `sticky`, the agent also lost tasks when a tool it
needed was not shown: it used another tool or planned around the missing one.

So filtering is likely to pay where those two effects reverse; we did not measure these cases:

* large catalogs whose schemas dominate the prompt;
* providers or self-hosted models without prefix caching;
* replacing a separate LLM component outright, as the paper's tool shortlister does: there the head removes
  a whole LLM call rather than shortening one.

## CUGA

Heads onboarded from CUGA trajectories plug back into CUGA where its LLM calls were
(`tabagent.integrations.cuga`; the paper's in-loop hot swap). Both helpers take the run so far as CUGA's
`ActivityTracker` holds it, which is the shape of the trajectory files the heads learned from, and return None
when the head defers, so CUGA keeps its own LLM call:

```python
from tabagent.integrations.cuga import CugaAppMatcher, CugaShortlister, run_so_far

shortlister = CugaShortlister("shortlister_head", k=9)  # a directory, hf://OWNER/REPO, or a serve URL
app_matcher = CugaAppMatcher("app_head")

# cuga/backend/cuga_graph/nodes/api/shortlister_agent/shortlister_agent.py, in ShortlisterAgent.run
result = shortlister.shortlist(run_so_far(tracker), input_variables.api_shortlister_all_filtered_apis)
if result is not None:
    return AIMessage(content=ShortListerOutput(thoughts=[], result=result).model_dump_json())

# cuga/backend/cuga_graph/nodes/task_decomposition_planning/analyze_task.py, in TaskAnalyzer.match_apps
offered = [{"name": a.name, "description": a.description} for a in apps]
names = app_matcher.match(run_so_far(tracker), offered)
if names is not None:
    res = AppMatch(thoughts="", relevant_apps=names)
```

On CUGA's AppWorld runs, the prefix and candidates the helpers build from the tracker equal the training
ones for every shortlisting and app-selection decision. After a swap, CUGA records the
shortlister's step without the LLM's prompts, so later decisions in the same run see steps unlike the training
ones; measure the swapped agent on held-out tasks before relying on it. With a served head
(`tabagent serve`), CUGA needs only the core package.
