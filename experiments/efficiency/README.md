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
