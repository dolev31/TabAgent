# Live validation: replay through the serving paths, and TabAgent in the τ²-bench loop

These scripts check an onboarded head the way an agent uses it: at inference time, from the run as the agent
holds it, through every serving path, and inside a real agent loop. Run them from the repository root. They
read agent logs and write their outputs where you point them; keep both outside the repository.

| script | environment | what it does |
|---|---|---|
| `refit.py` | TabAgent | refits a discovered head (its `card.yaml` and `extract.py`, no LLM call) on the train tasks of a seeded task split |
| `replay.py` | TabAgent | replays held-out runs as a scaffold would call the head; parity, latency, deferral, quality, robustness |
| `tau2_loop.py` | τ²-bench (+ `pip install tabagent`) | runs τ²-bench with the agent's tools from a `ToolFilter` session on a served head before every LLM call |
| `loop_report.py` | either | pairs two (or more) τ²-bench arms by task: pass rate, tokens, prompt-cache share, cost, wall time, head latency |

```bash
# 1. a head fitted on the train tasks only (the split is written when missing)
python -m experiments.live.refit --logs TAU2_RESULTS --run-dir TABSCHEMA_RUN --target NextToolCall \
    --split split.json --out head --select
# 2. replay the held-out runs through decide(run=...), POST /decide and ToolFilter
python -m experiments.live.replay --logs TAU2_RESULTS --head head --split split.json --k 7 \
    --robustness 20 --out replay.json
# 3. the head in the loop (τ²-bench installed in its own environment)
tabagent serve head --port 8077 --trust-extractor
python experiments/live/tau2_loop.py --domain retail --split split.json --arm baseline --trials 5 --save-to base
python experiments/live/tau2_loop.py --domain retail --split split.json --arm tabagent --k 7 --trials 5 \
    --head-url http://127.0.0.1:8077 --save-to tab
python -m experiments.live.loop_report --arm baseline=…/base/results.json --arm tabagent=…/tab/results.json
```

## What `replay.py` checks

For each successful held-out run and each assistant turn that calls a tool, the scaffold's state right before
that turn (the run-level fields and the messages so far, in the logs' format) is sent to the head through
`TabAgentHead.decide(run=...)`, the HTTP service with `{"run": ..., "candidates": <the tool schemas>}`, and
`integrations.openai_tools.ToolFilter`. It is compared with the training path for the same decision point
(`adapter.decision_points` prefix, the extractor run as `build_table` runs it, the row scored as in training):
prefix, feature and score parity; every serving path against `decide`; the sandboxed extractor against the
in-process one. Latency is per decision, each path timed with the decision's context not yet embedded (as live).
Quality is whether the tool the agent called is in the top k, alone and with the safety rule that keeps every
tool already used in the conversation. `--robustness N` perturbs N live states and runs each through every
path of the in-process and the sandboxed head.

## Results: τ²-bench, GPT-4.1 agent (2026-10-02)

**Data and head.** The public τ²-bench results of the GPT-4.1 agent (airline, retail, telecom; 1,112 runs, 606
successful; tool definitions in `tool_defs`), split by task, 70/30, seeded and stratified by domain: 195 train
tasks (408 successful runs, 2,139 `NextToolCall` decisions) and 83 held-out tasks (198 successful runs, 1,087
decisions). The head reuses the card and `extract.py` of a TabSchema onboarding on all the logs (Claude Opus 5.5,
GPT-6 Sol for the extractor) and is refitted on the train tasks only (`refit.py --select`: synth k=2, prior off).
That card's discovery analyzed 10 decision points, 5 of them from held-out tasks; a fresh discovery on the train
tasks only ($3.79) gives the same held-out quality (P@R 0.790 vs 0.805, Recall@7 0.974 vs 0.963), so the reuse
does not inflate the numbers below.

### Replay of the held-out runs (no LLM)

**Parity: 100%** on all 1,087 live decisions: the prefix built from the live run equals the training prefix, the
extractor features are equal, the scores equal those of the training-time row (largest difference 0), and the
HTTP service (sent the OpenAI tool schemas as candidates), `ToolFilter` and the sandboxed extractor give the
in-process answer. The same holds for the `ToolShortlist` head (198 decisions).

**Latency per decision** (ms, p50 / p95; laptop CPU shared with other jobs; each path embeds the decision's new
context, as live):

| path | `NextToolCall` head | `ToolShortlist` head |
|---|---|---|
| `decide(run=...)`, extractor in-process | 44 / 83 | 8 / 9 |
| `POST /decide` (test client), extractor in-process | 47 / 86 | 11 / 13 |
| `ToolFilter` | 44 / 78 | 8 / 9 |
| `decide(run=...)`, extractor in the subprocess sandbox | 112 / 151 | 60 / 73 |

The `NextToolCall` card embeds the context with e5-small (`context_candidate_sim`), which is most of its cost; the
sandbox adds an interpreter start (about 60 ms).

**Deferral: 0%.** The extractor never failed on a live prefix, and the head has no gate; a gate calibrated to
keep 90% of the train-task cross-validated decisions would defer 11.1% of the held-out ones.

**Quality of the live decisions** (held-out runs: is the tool the agent called in the shortlist?):

| k | Recall@k | with the safety rule (+ tools already used) | runs with every call covered | tools shown (of 14.3) |
|---|---|---|---|---|
| 3 | 0.927 | 0.948 | 0.76 | 4.5 |
| 5 | 0.948 | 0.962 | 0.81 | 6.1 |
| 7 | 0.963 | 0.972 | 0.86 | 7.9 |
| 9 | 0.983 | 0.988 | 0.93 | 9.7 |

P@R 0.805. By domain, Recall@7 with the safety rule is 0.910 on airline (the weak domain: `calculate`,
`update_reservation_baggages` and `transfer_to_human_agents` are missed), 0.978 on retail and 1.000 on telecom.
The `ToolShortlist` head (one decision per run, at its start) covers every tool of 79% of the held-out runs at
k=9, against 93% for `NextToolCall` with the safety rule, which is why the loop below uses `NextToolCall`.

**Robustness** (20 live states, each perturbed, through all three paths of the in-process and the sandboxed head):
an unknown tool in the history, an unknown candidate, a missing `tools` field, a 40x history with a 200,000
character message, non-ASCII text (emoji, CJK, RTL, NUL), a lone UTF-16 surrogate, empty and null tool results,
block content with unparsable arguments, an empty conversation and a system prompt in the messages are all
answered (at most 224 ms); a missing run-level field (`domain`) is deferred with an error that names it. Nothing
crashes (after the fixes below).

### In the loop

τ²-bench at `cf71a80` (v1.0.1) with GPT-4.1 (Azure, through a LiteLLM proxy) as agent and user simulator,
temperature 0, on the 49 held-out airline and retail tasks, 5 trials each, seed 300, paired by task and trial.
Arm B shows the agent the head's top 7 (held-out Recall@7 0.949 on these two domains, 0.962 with the safety rule)
plus the tools it already used, before every LLM call; the head is served by `tabagent serve` (in-process
extractor). `tau2_loop.py` now takes these lists from the library's `ToolFilter` session on the served head
(`keep_used=True`, `sticky=False` for B). Replaying the 2,577 recorded B decisions through it (a stub head
returns each recorded answer) sends the same tools in the same order at every call.

| | A: all tools | B: TabAgent top 7 | B - A (95% CI over tasks) |
|---|---|---|---|
| pass rate | 0.710 | 0.641 | -0.069 [-0.139, -0.004] |
| airline / retail | 0.560 / 0.776 | 0.520 / 0.694 | -0.040 [-0.20, 0.12] / -0.082 [-0.159, -0.012] |
| agent prompt tokens per call | 4,952 | 4,034 (-18.5%) | -7,950 per run [-10,217, -5,506] |
| share served from the prompt cache | 79.6% | 55.1% | |
| uncached prompt tokens per call | 1,010 | 1,812 | |
| agent LLM cost per run | $0.0467 | $0.0563 (+21%) | +$0.0096 [0.0064, 0.0133] |
| total LLM cost (245 runs, with the user simulator) | $14.27 | $16.68 | |
| wall time per run | 56.7 s | 57.3 s | +0.6 s [-2.9, 4.0] |

The head answered 2,577 decisions at 80 ms p50 and 338 ms p95 (service round trip, up to 10 concurrent
simulations on one server process); 1 decision (0.04%) was deferred, a 500 from a concurrent cold start (fixed in
this change). The agent never called a tool it was not shown.

* **Filtering cost more, not less.** The tool definitions are part of the prompt prefix the provider caches,
  so a tool list that changes between calls breaks the cache: B sends 18% fewer prompt tokens but pays for 79%
  more uncached ones (the tool list changed between consecutive calls in 74% of B's airline calls and 85% of its
  retail calls). Hiding about half the tools saves 620 (airline) to 1,050 (retail) of about 4,950 prompt tokens
  per call: tool definitions are a minority of a τ²-bench prompt, so even a perfect filter saves little here;
  the case for filtering is a large catalog.
* **Success drops by about 7 points**, mostly in retail. When a needed tool is hidden, the agent substitutes
  another one (airline task 12: in 4 of 5 trials `update_reservation_baggages` was never shown and the agent
  booked a new reservation instead; it passed the one trial where the tool was shown, and all 5 baseline trials)
  or plans the conversation without it: half the calls (50-55%) end in a text reply, a turn the head was never
  trained on (its labels are the tool-calling turns), yet the filter shapes it.
* **A cache-friendly variant** (`--sticky`, now `ToolFilter.session(sticky=True)`: a tool once shown stays
  shown; trials 1-3 only) matches the baseline's pass rate (0.708 vs 0.708, difference 0.000 [-0.095, 0.088])
  with 14% fewer prompt tokens per call; its list still grows at 38-54% of the calls, so 65.7% of the prompt is
  cached, and still costs 11% more per run (+$0.0049 [0.0023, 0.0075]). It was measured before the session
  existed, with new tools inserted in catalog order; replayed through the session, all 1,514 decisions send the
  same tools, and 1,326 send them in another order, because the session appends newly shown tools so that the
  list only grows at its end, to keep more of the cached prefix (its cost was not measured again).

### Caveats

* One agent, one benchmark version (newer than the logs: the retail catalog gained `get_item_details`, which the
  head never saw, and some task scenarios and tool docstrings changed), 49 tasks; the pass-rate CI is wide.
* GPT-4.1 at temperature 0 is not deterministic: the same task and seed can pass in one arm and fail in the other
  by chance, which the pairing averages but does not remove.
* Costs are recomputed from each call's tokens at GPT-4.1 list prices ($2 input, $0.50 cached input, $8 output per
  1M tokens); the cache share is whatever the provider served.
* Latencies come from a laptop shared with other jobs; the loop's wall time also depends on the provider's load,
  which differed between runs.
* The 8 retail tasks with natural-language assertions failed evaluation in a first pass (τ²-bench's judge model
  is hard-coded; `tau2_loop.py` now points it at the user simulator's model) and were re-run in all arms.
