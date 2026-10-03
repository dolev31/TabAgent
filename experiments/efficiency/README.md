# Efficiency experiments

Per-decision latency and cost of TabAgent (paper Sec. 5 "The swap is far cheaper", Fig. 4, App. F with
Tables 15 and 16). Run from the repository root. Outputs go to `results/efficiency/` (git-ignored).

## Cost formulas (`cost.py`)

The formulas are those of the paper (App. C, "Cost estimation" and "CPU cost"; App. F):

- CPU heads: `C_cpu(t) = threads * t / 3600 * p_vCPU-h`, with 4 threads and p_vCPU-h = $0.063, the
  on-demand AWS EC2 r6i price (r6i.xlarge, 4 vCPU at about $0.252 per hour). For 2.682 ms this is
  1.9e-7 dollars per decision; Table 15 rounds it up to 2.0e-7.
- GPU-served small LLMs: `C_local(t) = (P/1000)(t/3600) PUE p_kWh + (t/3600) p_GPU-h`, with P = 250 W,
  PUE = 1.4, $0.20/kWh and $0.20/GPU-h.

```bash
python -m experiments.efficiency.cost                          # recompute the paper's Table 15
python -m experiments.efficiency.cost --cpu-ms 2.682 --gpu-s 0.658
```

## Latency (`latency.py`)

TabAgent is fitted on four pinned folds and timed on decisions of the fifth. Each decision scores its
full candidate set; CatBoost, BLAS/OpenMP and torch are capped at `--threads` (default 4). The script
reports the mean, median and p90 over N decisions at three nested levels:

| level | what one call does |
|---|---|
| `model` | CatBoost `predict_proba` on the decision's featurized matrix (the classifier pass) |
| `head` | `TabHead.score` on the card's feature frame (TF-IDF, categorical and numeric encoding, then the model) |
| `agent` / `agent_cold` | `TabAgent.score` on the raw rows (every card feature, then the head). `agent_cold` is the first call per decision, which also embeds the task text with E5 |

Each level is priced with `C_cpu`. The script also records held-out-fold P@R, so the run yields a
(latency, cost, quality) point.

```bash
python -m experiments.efficiency.latency --download             # tau2/all-domains, presets paper + catboost-default
python -m experiments.efficiency.latency --datasets skillsbench/skill-select mind2web/element-150 \
    tau2/gpt-4.1-2025-04-14 --n 200 --repeats 5
```

The paper's headline is 2.682 ms per decision at about 48 candidates. It was measured on the CUGA agent's
AppWorld shortlisting trajectories, which are not released, so it cannot be recomputed from public data.
The closest public analog is tool shortlisting on τ²-bench over the merged catalog (`tau2/all-domains`, 38
candidates per decision). Report such runs as a public analog, not as the paper's number. Latency grows with
the number of trees: `paper` (the paper's configuration) uses 300 and `catboost-default` uses 1000.

## Pareto plot (`pareto.py`)

The plot shows latency (log ms) against P@R, with marker area growing as cost falls; it also computes
the Pareto front over the three axes.

```bash
python -m experiments.efficiency.pareto --paper                 # the paper's Fig. 4 from its reported points
python -m experiments.efficiency.pareto --latency results/efficiency/latency.json --level model \
    --points my_slm_points.json
```

`--points` takes a JSON list of `{"name", "latency_ms", "cost_usd", "quality", "kind", "note"}`, where
`kind` is one of tabagent, finetuned, fewshot, api or other. Points measured on different decisions
cannot be compared on quality, and each point's `note` records its source.

## Reference run

A 10-core laptop CPU with 4 threads and the `paper` preset (`catboost-default` is within 0.2 ms). Other jobs
were running at the same time (load average about 10), so the means are upper bounds; the JSON also records
medians and minima. The cost column uses $0.063 per vCPU-hour.

| card | candidates | model | head | agent (cold) | C_cpu(head) | held-out P@R |
|---|---|---|---|---|---|---|
| tau2/all-domains | 39¹ | 0.73 ms | 3.19 ms | 5.55 ms (19.2 ms) | $2.2e-7 | 0.760 |
| mind2web/element-150 | 50 | 0.93 ms | 4.28 ms | 4.86 ms (5.1 ms) | $3.0e-7 | 0.331² |
| tau2/gpt-4.1-2025-04-14 | 67 | 0.56 ms | 10.2 ms | 13.6 ms (34.3 ms) | $7.1e-7 | 0.821 |
| skillsbench/skill-select | 195 | 1.34 ms | 47.0 ms | 64.0 ms (106 ms) | $3.3e-6 | 0.792 |

¹ This run ranked a 39-tool union; the loader's union now has the paper's 38 tools (App. B.2).
² Recomputed on the current table, whose rows follow the Mind2Web task split (CatBoost depends on row order).

The head level of `skillsbench/skill-select` is dominated by TF-IDF encoding. The long task text is
re-vectorized for each of the 195 candidate rows.

## Serving (`serving.py`)

The latency of an onboarded head behind `tabagent serve`, which is not a paper number. The head decides in three
stages: build the prefix, run the TabSchema extractor on it (sandboxed unless trusted), then score the
candidates. `extract` times the extraction alone and whole `TabAgentHead.decide` calls with the extractor in a
fresh sandboxed interpreter per decision (`subprocess`, the serving default before the warm pool), in the warm
worker pool (`warm`, the default now), and in-process (`--trust-extractor`). `load` starts `tabagent serve` in
a subprocess, waits for `/ready`, and sends `POST /decide` from N client threads with one keep-alive connection
each. It reports the throughput and client-side p50/p95/p99, and the server's mean time per stage from
`/metrics`.

Unless `--head` names a saved head (then also pass `--prefixes`, a JSONL file of live prefixes, and
`--candidates`, a JSON list), both commands fit a synthetic but representative head. It shortlists among 30
tools over chat prefixes of 2 to 16 steps (2.2 KB of JSON on average). Its extractor uses only the standard
library, as TabSchema's do: step counts, the last user message, tool calls, parsed tool results and keywords,
about 60 lines with the appended context text. The head is CatBoost with the `paper` preset over 16 card
features, among them a lexical overlap and a fitted candidate prior.

```bash
python -m experiments.efficiency.serving extract --n 300
python -m experiments.efficiency.serving load --modes warm in-process subprocess --concurrency 1 8 32
```

A reference run on a 10-core Apple M1 Max shared with other jobs (load average 10 to 17). Extraction used
200 decisions per mode, about 2.2 KB of JSON each:

| stage, mode | p50 ms | p95 ms | p99 ms | mean ms |
|---|---|---|---|---|
| extraction, subprocess | 44.93 | 54.88 | 65.19 | 45.76 |
| extraction, warm | 0.10 | 0.19 | 0.36 | 0.11 |
| extraction, in-process | 0.05 | 0.08 | 0.27 | 0.05 |
| decide, subprocess | 54.97 | 64.47 | 67.48 | 55.17 |
| decide, warm | 6.40 | 7.16 | 7.60 | 6.43 |
| decide, in-process | 6.12 | 6.73 | 6.96 | 6.09 |

The warm pool started its worker in 49 ms. Over HTTP, each run timed 1500 requests (400 for `subprocess`) after
20 warm-up requests per client, with the default 4 workers and 8 decision threads:

| mode | clients | decisions/s | p50 ms | p95 ms | p99 ms | server: extraction / scoring / total ms |
|---|---|---|---|---|---|---|
| warm | 1 | 118 | 8.2 | 9.2 | 11.1 | 0.2 / 6.6 / 7.0 |
| warm | 8 | 113 | 60.8 | 83.6 | 118.2 | 9.1 / 19.7 / 37.5 |
| warm | 32 | 91 | 246.5 | 310.6 | 342.5 | 18.9 / 22.4 / 91.9 |
| in-process | 1 | 117 | 8.2 | 9.4 | 12.9 | 0.1 / 6.8 / 7.1 |
| in-process | 8 | 109 | 62.1 | 82.8 | 137.1 | 0.1 / 27.6 / 39.7 |
| in-process | 32 | 84 | 261.4 | 383.8 | 506.4 | 0.1 / 43.3 / 128.8 |
| subprocess | 1 | 15 | 61.2 | 82.9 | 117.3 | 50.1 / 10.2 / 60.8 |
| subprocess | 8 | 36 | 145.6 | 190.3 | 242.5 | 80.6 / 22.7 / 114.5 |
| subprocess | 32 | 19 | 647.7 | 784.5 | 831.1 | 118.5 / 40.4 / 462.1 |

No request was deferred or failed. The warm pool brings a decision to the cost of the trusted in-process mode,
with an eighth of the cold sandbox's p50 and about eight times its throughput. One process tops out near 115
decisions per second. Scoring is mostly Python that holds the interpreter lock: in a profile of one decision,
featurization and TF-IDF take about 40%, decoding and the confidence about 25%, and the card's features 15%,
while CatBoost's own pass takes about 1 ms. More clients therefore only queue, and the server's stage times grow
with lock contention. Scale out with more processes or replicas. The server's `total` is shorter than the
client's latency, which also includes HTTP parsing and the wait in the server's event loop.
