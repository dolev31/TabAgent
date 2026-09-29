# Public datasets

TabAgent ships no benchmark data. `tabagent.datasets.load(name, download=True)` rebuilds each table from
the upstream release at a pinned revision, verifies every file by SHA-256, and caches the result under
`$TABAGENT_CACHE` (default `~/.cache/tabagent`). Each dataset comes with the paper's feature card
(`datasets.card(name)`) and the paper's task-level folds (`datasets.folds(name)`).

| dataset | decision | source (pinned revision) | license | paper |
|---|---|---|---|---|
| `tau2/claude-3-7-sonnet-20250219` | tool shortlisting (ranking, 67-tool dual-control catalog, one decision per task) | https://github.com/sierra-research/tau2-bench @ `cf71a80702` | MIT | Table 2 (tab:transfer_new); App. B.2; trajectory efficiency (tab:traj_per_tool) |
| `tau2/gpt-4.1-2025-04-14` | tool shortlisting (ranking, 67-tool dual-control catalog, one decision per task) | https://github.com/sierra-research/tau2-bench @ `cf71a80702` | MIT | Table 2 (tab:transfer_new); App. B.2; trajectory efficiency (tab:traj_per_tool) |
| `tau2/gpt-4.1-mini-2025-04-14` | tool shortlisting (ranking, 67-tool dual-control catalog, one decision per task) | https://github.com/sierra-research/tau2-bench @ `cf71a80702` | MIT | Table 2 (tab:transfer_new); App. B.2; trajectory efficiency (tab:traj_per_tool) |
| `tau2/o4-mini-2025-04-16` | tool shortlisting (ranking, 67-tool dual-control catalog, one decision per task) | https://github.com/sierra-research/tau2-bench @ `cf71a80702` | MIT | Table 2 (tab:transfer_new); App. B.2; trajectory efficiency (tab:traj_per_tool) |
| `tau2/airline` | tool shortlisting (ranking over the airline agent toolkit, one decision per trajectory) | https://huggingface.co/datasets/Jarrodbarnes/tau2-sft-final @ `b6a74d4c62` | Apache-2.0 | Table 1 (τ²-bench rows); App. B.2 (tab:taubench_new) |
| `tau2/retail` | tool shortlisting (ranking over the retail agent toolkit, one decision per trajectory) | https://huggingface.co/datasets/Jarrodbarnes/tau2-sft-final @ `b6a74d4c62` | Apache-2.0 | Table 1 (τ²-bench rows); App. B.2 (tab:taubench_new) |
| `tau2/telecom` | tool shortlisting (ranking over the telecom agent toolkit, one decision per trajectory) | https://huggingface.co/datasets/Jarrodbarnes/tau2-sft-final @ `b6a74d4c62` | Apache-2.0 | Table 1 (τ²-bench rows); App. B.2 (tab:taubench_new) |
| `tau2/all-domains` | tool shortlisting (ranking over the merged agent toolkits of airline, retail and telecom, one decision per trajectory) | https://huggingface.co/datasets/Jarrodbarnes/tau2-sft-final @ `b6a74d4c62` | Apache-2.0 | Table 1 (τ²-bench rows); App. B.2 (tab:taubench_new) |
| `skillsbench/skill-select` | skill selection (set retrieval) | https://huggingface.co/datasets/benchflow/skillsbench-leaderboard @ `f104580363` | Apache-2.0 | Table 1 skill selection (P@R 74, R@9 87); Fig. 2a; Fig. 3 |
| `skillsbench/skill-gating` | skill gating (policy) | https://huggingface.co/datasets/benchflow/skillsbench-leaderboard @ `f104580363` | Apache-2.0 | Table 1 skill gating (gated reward 49, always-activate 50, chance 46, oracle 56) |
| `skillsbench/skill-gating-enriched` | skill gating (policy) | https://huggingface.co/datasets/benchflow/skillsbench-leaderboard @ `f104580363` | Apache-2.0 | not in the paper (pre-release three-backbone variant) |
| `mind2web/operation` | single | https://huggingface.co/datasets/osunlp/Mind2Web @ `17ece8eb89` | CC-BY-4.0 | Table 1, Operation selection (Mind2Web, 3-way, macro-F1) |
| `mind2web/element` | ranking | https://huggingface.co/datasets/osunlp/Mind2Web @ `17ece8eb89` | CC-BY-4.0 | Table 1, Mind2Web GUI element selection (R@3 per web domain; R@5 in the R@k column) |
| `mind2web/element-150` | ranking | https://huggingface.co/datasets/osunlp/Mind2Web @ `17ece8eb89` | CC-BY-4.0 | Table 1, Mind2Web GUI element selection (R@3 per web domain; R@5 in the R@k column) |
| `llmrouterbench/tau2` | model routing (policy): route each tau2-bench task to one of 12 LLMs | https://huggingface.co/datasets/NPULH/LLMRouterBench @ `0e5af1b84b` | license unresolved upstream (the README states MIT, but no license text is published) (opt-in) | Table 1, model routing: route@1 68 (best fixed model 70, chance 43, oracle 94) |
| `appworld/subagent-routing` | sub-agent routing (policy): route each AppWorld test task to one of 14 agent systems | https://github.com/StonyBrookNLP/appworld @ `a072b7a86e` | Apache-2.0, with the added condition that data and outputs are redistributed only encrypted | Table 1, sub-agent routing: route@1 38 (best fixed system 36, chance 16, oracle 58) |

**Access notes.**

* **LLMRouterBench**: the dataset card states MIT but the pinned repository has no license file; loading requires
  `accept_license=True` to confirm your use complies with the upstream terms.
* **AppWorld**: redistribution is restricted, so TabAgent never downloads it. Install it with AppWorld's
  official CLI (`pip install appworld && appworld install && appworld download data`, plus
  `appworld download experiment-outputs` for the baseline systems) and point `APPWORLD_ROOT` at it.
* **Mind2Web** (CC-BY-4.0): attribution to the Mind2Web authors is required for derived work.
* **τ²-bench per-agent tables**: as in the paper's Table 2, the context text is the task's user scenario, which
  the evaluated agent itself never sees; the per-domain tables use the first user message instead.
