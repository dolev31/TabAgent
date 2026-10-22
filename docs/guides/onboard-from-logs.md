---
title: "Onboard from logs"
description: "Turn an agent's own logs into a deployable decision head with tabagent onboard: privacy and cost controls, LLM providers, stopping and resuming a run."
lead: "Read the agent's logs, discover features with TabSchema, train and evaluate a head, and save it, in one call."
kind: Guide
needs: "an LLM API key; the agent's logs"
---

# Onboard from logs

```bash
export OPENAI_API_KEY=...                       # or OPENAI_BASE_URL (any OpenAI-compatible server),
                                                # or ANTHROPIC_API_KEY / GEMINI_API_KEY with tabagent[llm]
tabagent scan ./logs                            # which LLM calls are worth replacing (no LLM call)
tabagent components ./logs                      # the decisions you can replace, with counts
tabagent check-logs ./logs                      # what is wrong with the logs, where, and how to fix it
tabagent onboard ./logs --decision ToolShortlist --dry-run
tabagent onboard ./logs --decision ToolShortlist --out ./my_head
tabagent onboard ./logs --decision ToolShortlist --out ./my_head \
    --max-cost 50 --price-in "$PRICE_IN" --price-out "$PRICE_OUT"   # a budget, in USD per 1M tokens
tabagent runs show ~/.cache/tabagent/runs/<run>  # what each stage produced, dropped, failed and cost
```

`tabagent.onboard()` detects the log format, runs TabSchema discovery on the successful runs, builds the
decision table, selects the training recipe by nested validation, evaluates the head with grouped
cross-validation (quality with CIs against BM25 and chance, latency and cost per decision), fits the final head
and saves it.

The recipe search (on the training rows only) chooses TabSynth off, k=2 or k=4, the candidate prior on or off,
and, for a card whose LLM features include other state, whether to keep the generic context text and the
context-candidate similarities. Rich LLM features can make those redundant: on the next-tool-call decision of
the τ²-bench GPT-4.1 logs, a card discovered from 20 decision points scored P@R 0.853 without them against 0.841
with them, while a card from 10 points needed them (0.810 against 0.761). Choosing by nested validation inside
each training fold reached 0.850 and 0.810. They are dropped only when the inner validation is strictly better;
the report names the recipe (`context=off`).

## Privacy and cost

Trajectories are sent to your LLM provider during discovery. Pass `redact=` (a function applied to every
trajectory before anything is sent) to scrub personal data; the trajectories' file names are sent as they are
(the analyzer prompt names its trajectory file), so rename files whose names are personal. `redact` changes
the onboarding trajectories only: `head.decide(prefix, ...)` does not apply it, so apply the same function to
live prefixes when it changes fields the extractor reads. Run directories hold trajectory text and live outside
the repository (`~/.cache/tabagent/runs` by default).

**Before any call**, `--dry-run` (`onboard(dry_run=True)`) reports the trajectories, the decision points and an
estimate of the tokens and cost of every stage, plus any problem that would stop a real run (see below):

```text
$ tabagent onboard ./tau2-gpt41 --decision NextToolCall --dry-run --model aws/claude-opus-5-5 \
    --stage-model extractor=azure/gpt-6-sol --price aws/claude-opus-5-5=3,15 --price azure/gpt-6-sol=2,10
1112 trajectories (606 successful), 3226 decision points, 20 analyzed by the LLM; ~1,135,735 input / ~628,195
output tokens; est. cost $11.60
  stage             model                       calls       input     output     cost
  analyzer          aws/claude-opus-5-5            20      75,265     90,000    $1.58
  extractor         azure/gpt-6-sol                48     297,295    187,200    $2.47
  validator         aws/claude-opus-5-5            22     249,944     62,350    $1.69
  ...
  meta_judge        aws/claude-opus-5-5             6     163,444     54,600    $1.31
  auditor           aws/claude-opus-5-5            12     158,242     40,320    $1.08
```

The estimate (`tabagent.tabschema.estimate`) renders the FeatureAnalyzer prompts of the very decision points
the run will analyze, measures each point's schema, the schema merged over the onboarding data and the audited
points' clipped prefixes, and counts the CodeExtractor's retries, the compile step's repair round and the
OutputAuditor. What the LLM will do (how many features it proposes and keeps, how long it answers, how often its
code needs a retry) comes from real runs (`estimate.Profile`), and characters become tokens at the model's
measured ratio: 2.3 characters per token for Claude models (billed counts on real runs were 2.33 to 2.74) and
3.9 for others (measured offline with OpenAI's o200k tokenizer, not against billed usage). On four public
τ²-bench runs (next tool call, $6 to $12 each) and two sets of private agent logs with much longer prompts, the
estimate landed within 30% of the actual cost before the audit (6% to 17% low on τ²-bench). The earlier
estimate, which assumed 4 characters per token for every model, was too low on the logs with the longest
prompts and counted the audit at about a tenth of its cost. A model
much more or less verbose than the profile moves the actual cost away from the estimate.

**A budget**, `max_cost_usd` (`--max-cost`), stops discovery before a call whose estimated cost (its prompt at the
model's characters per token, plus the stage's mean answer, counting calls already in flight) would take the
total past it. It needs the model's prices, `prices={model: (usd_per_1M_in, usd_per_1M_out)}` (`--price-in`,
`--price-out`; `--price MODEL=IN,OUT` for the other models of a run), and is refused for a model without a
price. The stop is clean at any stage, the audit included: the run raises `BudgetExceeded` (the CLI exits with
status 3), `run.json` says where it stopped, and rerunning with the same run directory and a larger budget pays
only for what is left. The report gives each model's cost and calls; `tabagent runs show` gives each stage's.

## LLM providers

`tabagent.onboard` builds the LLM from the model name and the environment (`tabagent.tabschema.llm.auto_llm`):

1. `--model provider/name` for a LiteLLM provider (`anthropic/...`, `gemini/...`, `mistral/...`, `bedrock/...`,
   `vertex_ai/...`, `groq/...`, `ollama/...`, ...) goes through LiteLLM (`pip install "tabagent[llm]"`),
   even when `OPENAI_API_KEY` is also set;
2. `AZURE_OPENAI_ENDPOINT` (with `AZURE_OPENAI_API_KEY`, and `OPENAI_API_VERSION`, default `2024-10-21`)
   selects a classic Azure OpenAI resource; the model is the deployment name;
3. `OPENAI_API_KEY`, or `OPENAI_BASE_URL` alone for a server that needs no key (vLLM, Ollama), selects the
   OpenAI-compatible client; on such a server a model name may contain a slash;
4. otherwise the provider whose key is set (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`).

Every stage sends its prompt as the system message followed by a one-line user turn (some providers,
Anthropic among them, reject a request without one). The per-decision-point stages may answer up to 16,000
tokens and the whole-run stages (the FeatureAggregator, the three judges and the MetaJudge) up to 32,000; a
model with a lower output limit is retried at its own limit. An answer that stops at the limit raises
`LLMTruncated` (on one decision point it only skips that point). The whole-run stages see at most 40
(aggregator) or 25 (judges, MetaJudge) features per call, so that no answer has to restate an arbitrarily long
list; up to those sizes every feature is handled in one call, as in the paper.

**A model per stage.** `TabSchemaConfig(stage_models={"extractor": "azure/gpt-6-sol"})` (CLI `--stage-model
extractor=azure/gpt-6-sol`, repeatable; `judges` covers the three judges) runs one stage on another model,
for example a strong code model for the CodeExtractor. A provider's content filter can refuse one model's
answers to one stage outright (the finish reason `content_filter`, or a request refused for its content): such
an answer is never retried on the same model or re-asked; it is kept in the run directory, so a rerun meets the
same filter without paying for it again, and it raises `LLMContentFiltered`, whose message names the stage
(at one decision point it only skips that point). Route that stage to another model, or set `fallback_model`
(`--fallback-model`), which is asked once whenever a filter stops an answer (`llm_calls.jsonl` records the
call with `fallback_from`).

## Stopping, resuming and inspecting a run

A run that stops leaves a partial `report.md`, `run.json` and every paid answer in its run directory, whether it
was *stopped* (a spent budget, a provider that stayed down, Ctrl+C, a killed process: rerun to resume) or
*failed* (no feasible proposal, no accepted feature, an extractor that cannot be compiled, a leaking adapter).
Rerun with the same `run_dir` (`--run-dir`, the run directory itself) to resume:

* every answer the directory holds is reused at no cost, including the truncated and filtered ones, so a resumed
  run makes the same decisions as an uninterrupted one and produces the same `card.yaml` and `extract.py`, at
  the same recorded cost; no answer is paid for twice (the one exception: a call whose answer arrived in the
  instant before the process was killed, before it was written);
* a filtered answer goes to the `fallback_model` on the rerun too, when one has been added since (its answer is
  stored under the fallback model's own key), and an answer cut at the output limit is asked again when the
  model's output limit (`max_tokens`) has been raised since: both are what the error messages advise;
* the answers that caused a *failure* (an unparseable answer that stopped a stage, extraction code that never
  ran) are dropped from the directory, so the rerun asks for them again; a content filter's answer is kept,
  since the same prompt to the same model would be stopped again (route the stage to another model);
* the report and the card count reused answers at the cost they had when first paid.

Every file is written atomically, so a process killed at any moment leaves complete files (the temporary file
it was writing is removed when a run next opens the directory). Ctrl+C starts no new LLM call and waits, up to
2 s, only for the answers being written. Two runs on one run directory at once (two processes) are safe: each
answer is paid once, by whichever asks first, and the other waits for it (file locks, on POSIX). The same
trajectories and the same answers give a byte-identical `card.yaml` and `extract.py`; the card holds no local
paths. The tests drive every stage with a scripted LLM that misbehaves in every way above, interrupt runs
inside every stage (Ctrl+C, `SIGKILL`, the budget, a provider that stays down) and check these guarantees
(`tests/tabschema/test_chaos.py`, `test_resume.py`).

**`run.json`** is the run's machine-readable summary, written as the run progresses: its status (`running`,
`completed`, `stopped`, `failed`) and the error that ended it; every session that worked in the directory; per
stage (adapter check, sample, analyze, aggregate, judge, meta_judge, compile, audit, usefulness, card) its
status, timing, progress with an estimate of the time left, what it produced, dropped and failed, and its LLM
calls, tokens and cost (paid now, and reused); every drop counted by reason; and the provider errors and retries.
`llm_calls.jsonl` records every call and failed attempt with its stage, session and time. At `-v` the CLI logs
each stage and the analysis's progress (`analyzed decision points: 7/20; about 4m 10s left`).

```bash
tabagent runs show ~/.cache/tabagent/runs/20261005T181407Z   # the anatomy, stage by stage
tabagent runs show RUN_DIR --json                              # the same, machine-readable
tabagent runs list                                             # the run directories, newest first
```

`runs show` reads run directories made before `run.json` existed too, rebuilding what it can from the stage
files. The CLI exits with status 3 for a stop a rerun resumes from, 2 for a user error, 0 on success.

**Large onboarding sets.** The compile step runs the extractor on every onboarding decision point, four
interpreters at a time; when the columns times the decision points would not fit 20 million cells in memory (a
dict per row, such as word counts, over many decision points), every column first goes through the cheap gates
on statistics over all decision points (constant, too often empty, flagged by the CodeValidator), presence flags
are decided from those statistics alone, and the other columns are ranked on a seeded random sample of 2,000
points, at most `max(64, 20,000,000 / decision points)` of them reaching the full gates (a compile note says
when that memory cap binds). On 100,000 synthetic decision points, discovery took about 65 seconds on a laptop, with a peak of
about 0.6 GB, or 1 GB when the extractor produced 100,000 distinct columns.

**Configuration.** Every `TabSchemaConfig` field is checked when the config is made, with a message that names
the field (`TabSchemaConfig.accept: unknown value(s) ['accepted']`); `TabSchemaConfig.from_dict` names unknown
keys with the closest field. On the command line, `--config FILE` (JSON or YAML) sets any field, and the flags
override it: `--n-analyze`, `--accept accept,conditional`, `--sample-strategy`, `--workers`, `--timeout`,
`--llm-retries`, `--audit-samples`, `--max-cost`, `--no-audit`, `--no-verify-extractor`, `--no-verify-adapter`,
`--no-usefulness`, `--no-compile-repair`, `--no-context-text`, `--no-last-step-text`, `--no-positional-guard`,
`--no-relative-features`, `--prune-harmful`.
