# Public datasets

TabAgent ships no benchmark data. `tabagent.datasets.load(name, download=True)` rebuilds each table from
the upstream release at a pinned revision, verifies every file by SHA-256, and caches the result under
`$TABAGENT_CACHE` (default `~/.cache/tabagent`). Each dataset comes with the paper's feature card
(`datasets.card(name)`) and the paper's task-level folds (`datasets.folds(name)`). `make data` (or
`tabagent datasets --all --download`) fetches every public dataset at once.

| dataset | decision | source (pinned revision) | download | license | paper |
|---|---|---|---|---|---|
| `tau2/claude-3-7-sonnet-20250219` | tool shortlisting (ranking, 67-tool dual-control catalog, one decision per task) | https://github.com/sierra-research/tau2-bench @ `cf71a80702` | 604 MB (shared by the four agents) | MIT | Table 2; App. B.2; App. F.1, Table 17; App. N.4, Table 21 |
| `tau2/gpt-4.1-2025-04-14` | same | same | same | MIT | same |
| `tau2/gpt-4.1-mini-2025-04-14` | same | same | same | MIT | same |
| `tau2/o4-mini-2025-04-16` | same | same | same | MIT | same |
| `tau2/airline` | tool shortlisting (ranking over the airline agent toolkit, 14 tools, one decision per trajectory) | https://huggingface.co/datasets/Jarrodbarnes/tau2-sft-final @ `b6a74d4c62` | 8.2 MB (shared by the four tables) | Apache-2.0 | Table 1 (τ²-bench rows); App. B.2, Table 11 |
| `tau2/retail` | tool shortlisting (retail agent toolkit, 15 tools) | same | same | Apache-2.0 | same |
| `tau2/telecom` | tool shortlisting (telecom agent toolkit, 13 tools) | same | same | Apache-2.0 | same |
| `tau2/all-domains` | tool shortlisting over the merged toolkits (38 distinct tools) | same | same | Apache-2.0 | same |
| `skillsbench/skill-select` | skill selection (set retrieval over 195 skills) | https://huggingface.co/datasets/benchflow/skillsbench-leaderboard @ `f104580363` | 3.5 MB | Apache-2.0 | Table 1 skill selection (P@R 80, R@9 91); Fig. 2a; Fig. 3 |
| `skillsbench/skill-gating` | skill gating (policy) | same | same | Apache-2.0 | Table 1 skill gating (gated reward 48, always-activate 50, chance 46, oracle 56) |
| `skillsbench/skill-gating-enriched` | skill gating (policy), three backbones | same | same | Apache-2.0 | not in the paper (an extension of skill gating to three backbones) |
| `mind2web/operation` | operation selection (single-label, CLICK / TYPE / SELECT) | https://huggingface.co/datasets/osunlp/Mind2Web @ `17ece8eb89` | 5.9 GB (shared by the three tables) | CC-BY-4.0 | Table 1, operation selection (macro-F1 92) |
| `mind2web/element` | GUI element selection (ranking, 50 candidates per step, all 1,009 tasks) | same | same | CC-BY-4.0 | Fig. 2b; the +synth training pool of the element row |
| `mind2web/element-150` | GUI element selection on the 150 evaluated tasks (1,019 steps) | same | same | CC-BY-4.0 | Table 1, GUI element selection (R@3 per web domain; R@5 in the R@k column) |
| `llmrouterbench/tau2` | model routing (policy): route each τ²-bench task to one of 12 LLMs | https://huggingface.co/datasets/NPULH/LLMRouterBench @ `0e5af1b84b` | 1.35 GB | MIT, as declared in the upstream README (the repository ships no LICENSE file); opt-in download | Table 1, model routing: route@1 71 (best fixed model 70, chance 43, oracle 94) |
| `appworld/subagent-routing` | sub-agent routing (policy): route each AppWorld test task to one of 14 agent systems | https://github.com/StonyBrookNLP/appworld @ `a072b7a86e` | through AppWorld's CLI | Apache-2.0, with the added condition that data and outputs are redistributed only encrypted | Table 1, sub-agent routing: route@1 38 (best fixed system 36, chance 16, oracle 58) |

The card's embedding-similarity features also download the `intfloat/e5-small-v2` encoder (about 128 MB)
from Hugging Face once (the `embed` extra).

**Access notes.**

* **LLMRouterBench** is MIT-licensed as declared in its README; the pinned repository ships no LICENSE file, so
  the download is opt-in: pass `accept_license=True` (`--accept-license` / `--accept-licenses` on the command
  line) to confirm your use complies with the upstream terms.
* **AppWorld** is public, but its terms allow redistributing the data and experiment outputs only in encrypted
  form, so TabAgent never downloads them. Install them with AppWorld's official CLI (`pip install appworld &&
  appworld install && appworld download data`, plus `appworld download experiment-outputs` for the baseline
  systems) and point `APPWORLD_ROOT` at them.
* **Mind2Web** (CC-BY-4.0): attribution to the Mind2Web authors is required for derived work. The first build
  downloads the original training files (11 files, about 5.9 GB; parsing one needs about 2 GB of RAM) and keeps a
  reduced, HTML-free copy in the cache.
* **τ²-bench per-agent tables**: the context text is what the agent sees before it acts: the customer's first
  message in the task's first successful run, or the task's ticket in the release's solo modes, where no
  customer speaks (paper App. B.2). `load(..., context="user_scenario")` uses the simulator's task
  description instead.
