---
title: "Production runbook"
description: "Run tabagent serve in production: deploy, scale, the warm extractor pool, timeouts and the fallback policy, metrics and alerts, feature drift, upgrading and the security model."
lead: "The service is built to answer fast and to step aside when it should: whatever it cannot decide well comes back deferred."
kind: Guide
needs: "tabagent[serve]"
---

# Production runbook

`tabagent serve` runs one head per process. The head replaces one LLM call of a live agent, so the service
is built to answer fast and to step aside when it should: any decision it cannot make well comes back
`"deferred": true`, and the agent then calls its original LLM component. This section covers deploying,
scaling, timeouts, monitoring, drift and the security model.

## Deploy

```bash
pip install "tabagent[serve]"
tabagent serve ./my_head --host 0.0.0.0 --port 8077 --workers 4 --request-timeout 10
```

`--host 0.0.0.0` is for a container on a private network: the service has no authentication (see the
security model below).

Review `extract.py` before the first deployment of a head (see `SECURITY.md`). Pin what you serve: a
directory under version control or `hf://OWNER/REPO@REVISION`. `GET /version` reports the extractor's and the
model files' SHA-256, so every log line (`head_id`) and dashboard can be traced to one artifact.

| endpoint | answers |
|---|---|
| `POST /decide` | one decision: `{"prefix" or "run", "candidates", "k", "defer_below"}` → `{"candidates", "scores", "confidence", "deferred", "error"}` |
| `POST /decide_batch` | `{"requests": [...]}` (at most 256) → `{"results": [...]}`, one answer per request in order, under one deadline for the batch; an invalid one is `{"status": 400, "detail": ...}` |
| `GET /health` | liveness: 200 while the process answers |
| `GET /ready` | readiness: 200 once the head is loaded, at least one warm extractor worker runs (`min_ready`) and the extractor can run; 503 before, during shutdown, and for an extractor that does not compile or breaks the extraction rules. The body has the `reason` and the pool's state |
| `GET /version` | the tabagent version and the head's manifest: id, decision, card, kind, extractor SHA-256, model SHA-256, creation time, the tabagent version that wrote it |
| `GET /metrics` | Prometheus text format (below) |
| `GET /drift` | the per-feature drift report (below) |
| `GET /card` | the feature card |

Every response carries an `X-Request-ID` header: the client's own, when it sends one, else a new id. Send
the agent's run or step id there to join the service's logs with the agent's.

In Kubernetes, point the liveness probe at `/health` and the readiness probe at `/ready`:

```yaml
livenessProbe: {httpGet: {path: /health, port: 8077}, periodSeconds: 10}
readinessProbe: {httpGet: {path: /ready, port: 8077}, periodSeconds: 2}
terminationGracePeriodSeconds: 40  # above --graceful-timeout (30 s)
```

On SIGTERM (or Ctrl+C) the service stops accepting connections, lets in-flight decisions finish for up to
`--graceful-timeout` seconds, then stops its decision threads and its extractor workers. The process then
exits with the signal's status (uvicorn re-raises it), which process managers treat as a normal stop.

## Scale

One decision runs three stages: building the prefix from a `run` (when the request sends one), the extractor
in a warm worker, and the head's scoring (the card's features, TF-IDF and CatBoost). On the benchmark head
below, the warm extractor takes about 0.1 ms and scoring about 6 ms, so a process is CPU-bound in scoring and
serves on the order of a hundred decisions per second.

- `--workers N` sets the number of warm extractor workers, which is the number of extractions that run at
  once (default `min(4, CPUs)`). Each is a separate interpreter of about 20 MB; on Linux each is capped at
  2 GB of address space and a CPU-time budget.
- The pool needs POSIX (Linux, macOS). On Windows the default is a fresh interpreter per decision.
- Pre-forking servers and `multiprocessing` work: a process forked after the head served decisions drops the
  workers it inherited and starts its own.
- `--threads` (default 8) sets how many decisions are computed at once. Further requests wait for a thread,
  and that wait counts toward the decision's deadline. More threads do not add throughput, because scoring holds
  the interpreter lock most of the time (see the benchmark below).
- Scale out with more processes or replicas behind a load balancer, one head per process: replicas share
  nothing, and each one has its own `/ready`, metrics and drift window. Give each replica about 4 vCPUs for
  scoring plus one per extractor worker that is busy at the same time.

## The warm extractor pool

By default (`--sandbox warm`) a served head runs its extractor in a pool of persistent sandboxed
interpreters (`tabagent.tabschema.warm.WarmSandbox`). Each worker compiles `extract.py` once and runs it on
one prefix per request, in a fresh namespace with its own copy of the builtins, under the same isolation as the
one-shot sandbox used in onboarding. Before any worker starts, the head re-applies the security-relevant
static checks of the compile step to `extract.py` (allowed imports, forbidden builtins, interpreter
internals, frame introspection). An extractor that fails them, or does not compile, is never run: `/ready`
answers 503 with the reason, `tabagent_extractor_broken` is 1, and every decision is deferred
(`extractor_broken`). `--allow-unchecked-extractor` (`allow_unchecked_extractor=True`) skips the check, and
still runs the extractor in the sandbox, with a warning; see [Upgrading](#upgrading).

The workers start when the service starts; `/ready` answers 200 as soon as one runs (`min_ready`) and
`tabagent_extractor_degraded` is 1 while fewer than `--workers` run. A background thread keeps the pool
full. It retries a worker that cannot start with exponential backoff (up to 30 s), counts it in
`tabagent_extractor_start_failures_total` and logs it at ERROR. It also replaces a worker:

- after a timeout: the worker's own alarm stops the code at the request timeout and the worker lives on. A
  worker that ignores the alarm (code that blocks in C, or disabled the alarm) is killed a grace period later
  (1 s plus 10% of the timeout), with its whole process group, and replaced;
- after a crash: the code exited the interpreter, hit a memory or CPU limit, or the worker died otherwise
  (killed from outside, say, even while idle);
- after 5000 requests, or once less than one request's CPU budget is left, to bound whatever state an
  interpreter accumulates.

`tabagent_extractor_restarts_total` counts the replacements after a timeout or a crash, and
`tabagent_extractor_idle_deaths_total` those of workers found dead while idle; each idle death is logged at
WARNING, and repeated deaths slow the replacements down (the same backoff as failed starts), so an
environment that kills workers cannot make the pool spin. A missing or broken interpreter is a start failure:
the decision is deferred (`sandbox_unavailable`), never an exception. The request that hit
the failure is answered `deferred` with the reason in `error`, and the next request goes to a healthy worker.
Workers exit on their own within a second when the serving process dies, even in the middle of a request. `--sandbox subprocess` starts a fresh interpreter per decision instead (45
to 65 ms more per decision on the benchmark machine, nothing persistent), `--trust-extractor` runs a reviewed
extractor in the serving process, and `--sandbox docker` runs each extraction in a container.

## Timeouts and the fallback policy

`--request-timeout T` (default 10 s; 0 turns it off; otherwise a positive number) is the time the extractor
gets per decision. The wait for a free worker counts against it. When it runs out, the decision is still
answered with status 200: the candidates are scored without the extractor's features and the answer is
`"deferred": true` with a `TimeoutError` in `error`. A decision that has no answer `T + 1.5 s + 0.1 T` after
it arrived (for example an in-process extractor that cannot be interrupted, or a queue of requests waiting
for a thread) is answered `"deferred": true` in the response's usual shape: `candidates` lists the request's
candidate ids in their input order (`choice` is null for a single-class head), `scores` is empty and
`confidence` is null. A batch has one such deadline for all its requests: those still unanswered when it
passes come back the same way. Set the client's own timeout above that bound.

The agent's side of the contract:

| the service answers | the agent does |
|---|---|
| 200, `"deferred": false` | use the head's `candidates` (or `choice`) |
| 200, `"deferred": true` | call the original LLM component. The reason is the confidence gate (`error` is null), a failed or timed-out extraction, or the deadline (`error` says which) |
| 400 | fix the request: the head cannot answer it as given (`RequestError`: a ranking head without candidates, both a prefix and a run, a malformed candidate, a run it cannot turn into a prefix, a `k` below 1) |
| 422 | fix the request: its body does not fit the schema (`candidates` not a list, `k` not an integer, ...) |
| 5xx, a connection error or the client's own timeout | call the original LLM component, and alert: the service is unhealthy (`RemoteHead` and `ToolFilter("http://...")` defer on these) |

Deferral never makes the agent worse than it was before the head: it falls back to the component the head
replaced. Watch the deferral rate, because deferred decisions cost what the LLM costs.

## Monitoring and alerts

`GET /metrics` uses the Prometheus text format and has no dependency. When `prometheus_client` is
installed, its default registry is appended (process CPU, memory, open files, garbage collection).

| metric | type | labels |
|---|---|---|
| `tabagent_decisions_total` | counter | `endpoint`, `outcome` = `head` or `deferred` |
| `tabagent_deferrals_total` | counter | `reason` = `low_confidence`, `extractor_error`, `extractor_timeout`, `extractor_crash`, `extractor_broken`, `sandbox_unavailable`, `no_extractor`, `request_timeout` |
| `tabagent_errors_total` | counter | `class`: an extractor failure class, `request_timeout`, `bad_request` (400) or `internal` (500) |
| `tabagent_decision_seconds` | histogram | `stage` = `total`, `prefix`, `extraction`, `scoring` |
| `tabagent_http_requests_total` | counter | `endpoint`, `code` |
| `tabagent_inflight_decisions` (computed by a thread now), `tabagent_queued_decisions` (waiting for a thread), `tabagent_ready`, `tabagent_extractor_broken`, `tabagent_extractor_degraded` | gauge | |
| `tabagent_extractor_workers` | gauge | `state` = `size`, `min_ready`, `live`, `idle`, `busy`, `starting` |
| `tabagent_extractor_restarts_total`, `_idle_deaths_total`, `_recycled_total`, `_start_failures_total`, `_timeouts_total`, `_requests_total` | counter | |
| `tabagent_head_info` | gauge (1) | `id`, `card`, `target`, `extractor_sha256`, `tabagent_version` |
| `tabagent_feature_psi`, `tabagent_feature_null_rate_shift`, `tabagent_feature_unseen_share`, `tabagent_feature_drifting` | gauge | `feature` |
| `tabagent_drift_window_decisions`, `tabagent_drift_features_drifting` | gauge | |

Suggested alerts (tune the thresholds to your traffic):

```yaml
- alert: TabAgentNotReady
  expr: tabagent_ready == 0
  for: 5m
- alert: TabAgentExtractorBroken           # extract.py does not compile or breaks the rules: nothing runs
  expr: tabagent_extractor_broken == 1
- alert: TabAgentExtractorFailing          # extraction failures, not confidence deferrals
  expr: sum(rate(tabagent_deferrals_total{reason=~"extractor_.*|sandbox_unavailable"}[10m]))
        / sum(rate(tabagent_decisions_total[10m])) > 0.05
  for: 10m
- alert: TabAgentWorkersCannotStart
  expr: increase(tabagent_extractor_start_failures_total[15m]) > 3
- alert: TabAgentPoolDegraded              # fewer workers than configured for a while
  expr: tabagent_extractor_degraded == 1
  for: 10m
- alert: TabAgentDeferralRateHigh          # the gate defers more than calibrated (onboard --gate-coverage)
  expr: sum(rate(tabagent_deferrals_total{reason="low_confidence"}[1h]))
        / sum(rate(tabagent_decisions_total[1h])) > 0.2
  for: 1h
- alert: TabAgentWorkersRestarting
  expr: increase(tabagent_extractor_restarts_total[15m]) > 5
- alert: TabAgentSlow
  expr: histogram_quantile(0.95, sum by (le) (rate(tabagent_decision_seconds_bucket{stage="total"}[5m]))) > 0.1
  for: 10m
- alert: TabAgentFeatureDrift
  expr: tabagent_drift_features_drifting > 0
  for: 1h
```

**Logs.** The service logs one JSON object per line on stderr (`--log-format text` for plain text). Each
decision is one record with `event: "decision"`, `request_id`, `head_id`, `endpoint`, `status`,
`latency_ms` (`total`, `prefix`, `extraction`, `scoring`), `deferred`, `defer_reason`, `error_class`,
`error_type` (the exception name only), `n_candidates`, `k`, `confidence` and `input` (`prefix` or `run`). A
decision is logged at INFO, a deferral caused by a failure or a timeout and a 400 at WARNING, and a 500 at
ERROR with its stack frames. The request's content (prefix, run, candidates), the answer and the full error
message are logged only with `--log-content`, since they are the agent's data and an error message can
quote it.

## Drift, and when to re-onboard

When a head is fitted, `TabAgent.fit` records the training distribution of every input feature of the head
(`model/feature_stats.json`, covered by the manifest hash; see `tabagent.drift`): the null rate, the
quantiles of numbers, the top categories and the categories seen, and the length of texts. A feature that is
constant within a decision, such as an extractor column, is counted once per decision; a candidate feature is
counted once per candidate row. While serving, the service compares the last `--drift-window` decisions
(default 500) with that distribution, per feature:

- the population stability index (PSI) over the training deciles (numbers and text lengths) or over the top
  categories plus one bin for every other value (bins with fewer than 5 training values are merged);
- the null-rate shift: the live null rate minus the training one;
- the unseen share: the share of live categorical values never seen in training, compared with the share the
  training data itself predicts (the Good-Turing estimate: the values whose category occurs in one training
  task only). A long-tail category (cities, account names) always shows new values.

These statistics move on their own when samples are small, and live decisions are not independent: a
decision's candidates share its context, and one task's runs and steps share theirs. At fit time,
`feature_stats` therefore measures each feature's design effect over the training table's task groups: how
much clustering inflates the variance of its bin shares. A feature drifts only when a statistic passes both its
practical threshold (PSI 0.2, a null-rate change of 0.1, or 20% more unseen values than expected) and the
limit a window without drift crosses with probability `alpha` (0.01) split over every test of every feature.
For the PSI the limit is about `deff × (1/n_train + 1/n_live) × χ²(B − 1)` at that level. In simulations with
40 features, heads fitted on 60 decisions, candidate features sharing 90% of their variance within a decision,
and a long-tail category, at most 3% of no-drift windows reported any drifting feature. A 1 standard deviation
shift of one feature was still found 80% of the time with 60 training decisions and every time with candidate
features. On real τ²-bench and SkillsBench logs, 36 replays of held-out tasks (no drift) raised no alarm.
Replaying another τ²-bench domain than the head was fitted on raised one in 9 of 9 replays. Nothing drifts
before the window holds 100 decisions, nor for a head fitted on fewer than 30 decisions (status
`insufficient_training`).

**What it can detect.** The other side of so few false alarms is that small shifts go unnoticed, the more so
for small or clustered training sets. Shifting the mean of one of 16 numeric features over a 500-decision
window, in 20 replays each, gave:

| head fitted on | shift of 0.25 SD | 0.5 SD | 1 SD | null rate 0 → 15% |
|---|---|---|---|---|
| 100 independent decisions | never | never | always | 85% |
| 300 or 1,000 independent decisions | never | 95% | always | always |
| 244 decisions in task groups of 4 (τ²-bench-like: 4 runs per task) | never | 5% | 70% | always |
| 1,000 decisions in task groups of 4 | 5% | 20% | always | always |

For each feature, `GET /drift` reports `psi` next to `psi_limit`, the value a window without drift exceeds
with probability `alpha` (and `null_rate_limit` and `unseen_limit` for the other tests). A feature whose
`psi` stays close below its limit is worth a look even when it is not drifting. Drift monitoring catches
large shifts, such as another task domain or an extractor that starts failing. Subtle changes need labeled
evaluation.

A feature that starts drifting is logged once at WARNING (`event: "feature_drift"`), and again at INFO when it
stops; `GET /drift` and the `tabagent_feature_*` gauges give the details. The monitor works in a background
thread: a decision only queues its features. Heads fitted with an earlier version of tabagent have no
statistics. `/drift` then answers `"status": "unavailable"` and the service runs without drift monitoring;
refit and save such a head to enable it.

How to read it:

| what drifts | what it usually means | what to do |
|---|---|---|
| the null rate of extractor columns, with `extractor_error` deferrals | live prefixes no longer have the fields the extractor reads: the agent's log format, its prompts or the run-level task fields changed | compare a live prefix with a training one (`head.live_prefix`); fix the integration, or re-onboard on the new logs |
| unseen categories in candidate columns | new tools or new candidate names that the head never saw | re-onboard on logs that include them; until then the head has no training signal for them |
| PSI of context features (text lengths, counts) | the mix of tasks changed | check the head's quality on recent labeled runs; re-onboard if it dropped |
| the deferral rate rises with no drift | the gate's threshold no longer fits | recalibrate (`onboard --gate-coverage`) on recent logs |

Re-onboard when quality on recent labeled runs drops, and plan it whenever the agent, its prompts, its tools or
its log format change. Drift tells you when to check; it does not measure quality, which needs labels. Serve the
new head next to the old one, compare their deferral rates and `/drift` reports, then switch over.

## Upgrading

What changes for a head served by an earlier version of tabagent:

- **Extractors are checked again before they run.** A sandbox chosen by name (`warm`, the default,
  `subprocess` or `docker`) re-applies the security-relevant static checks of the compile step to
  `extract.py`. An extractor that fails them is no longer run. That happens when it imports a module outside
  the allowed standard-library list (`time`, `os`, `signal`, ...), calls a forbidden builtin, or reaches for
  interpreter internals or frames. `/ready` then answers 503 with the reason, `tabagent_extractor_broken` is 1,
  and every decision is deferred to the LLM (`extractor_broken`). Extractors that the current TabSchema
  writes pass these checks. For an older head:
  - onboard it again, which is the clean fix;
  - or review `extract.py` and serve it with `tabagent serve --allow-unchecked-extractor`
    (`TabAgentHead.load(..., allow_unchecked_extractor=True)`). The extractor still runs in the sandbox, and a
    warning names the rules it breaks. A sandbox instance (`sandbox=WarmSandbox(...)`) runs what it is given,
    as before.
- **Warm workers are the default.** `--sandbox subprocess` keeps one fresh interpreter per decision.
- **Drift monitoring needs the training statistics** that `TabAgent.fit` now records. A head fitted earlier
  reports `/drift` as unavailable until it is fitted and saved again.
- **Status codes.** A request the head cannot answer as given (`RequestError`) is a 400, as before. A body
  outside the schema is a 422. Any other failure of the head is now a 500 (a plain `ValueError` from a custom
  head used to be a 400); `RemoteHead` and `ToolFilter("http://...")` defer on it. A `k` below 1 is a 400.
- **A decision past its deadline** keeps the response's shape: every candidate id in input order, empty
  `scores`, null `confidence`.

## Security model

The service has no authentication and no TLS: bind it to localhost or a private network, or put it behind an
authenticated proxy. Never expose it to the internet. An untrusted extractor runs in warm workers with an empty
environment, isolated mode, no site packages, and file-size, core-dump and (Linux) memory and CPU-time
limits. That isolates accidents, not adversaries. A worker serves many requests, so a malicious extractor
could observe later prefixes and forge later answers, and the static checks do not prevent that. Review
`extract.py`, and serve a head whose extractor you have not reviewed with `--sandbox subprocess` (a fresh
interpreter per decision) or `--sandbox docker`. `SECURITY.md` explains the model and what stays out of the
logs.

## Benchmark

`experiments/efficiency/serving.py` measures the extraction stage and the HTTP service on a representative
synthetic head: a 30-tool shortlisting decision over chat prefixes of about 2 KB, a stdlib extractor of about 60
lines, and a CatBoost head with the paper's preset over 16 features. These runs come from a 10-core Apple M1
Max that other jobs shared (load average 10 to 17), with 200 decisions per mode for the extraction and 1500
requests per run over HTTP (400 for the cold sandbox):

| per decision, ms (p50 / p95 / p99) | extraction alone | whole `decide` |
|---|---|---|
| `subprocess`: a fresh interpreter per decision (the previous default) | 44.9 / 54.9 / 65.2 | 55.0 / 64.5 / 67.5 |
| `warm`: warm workers (the default) | 0.10 / 0.19 / 0.36 | 6.40 / 7.16 / 7.60 |
| `--trust-extractor`: in-process | 0.05 / 0.08 / 0.27 | 6.12 / 6.73 / 6.96 |

Over HTTP, with `POST /decide` from 1, 8 or 32 concurrent clients:

| extractor | clients | decisions/s | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| warm | 1 | 118 | 8.2 | 9.2 | 11.1 |
| warm | 8 | 113 | 60.8 | 83.6 | 118.2 |
| warm | 32 | 91 | 246.5 | 310.6 | 342.5 |
| subprocess | 1 | 15 | 61.2 | 82.9 | 117.3 |
| subprocess | 8 | 36 | 145.6 | 190.3 | 242.5 |
| subprocess | 32 | 19 | 647.7 | 784.5 | 831.1 |

The warm pool removes the interpreter start-up from every decision. A decision then costs what the trusted
in-process mode costs, and one process serves about eight times as many decisions per second. One process tops
out near 115 decisions per second for this head. Its scoring is mostly Python (the card's features, TF-IDF,
decoding; CatBoost's own pass takes about 1 ms) that holds the interpreter lock, so further clients only wait
longer. Add replicas for more throughput. The full tables, with the in-process mode and the server's time per
stage, are in [`experiments/efficiency/README.md`](https://github.com/dolev31/TabAgent/blob/main/experiments/efficiency/README.md#serving-servingpy).
