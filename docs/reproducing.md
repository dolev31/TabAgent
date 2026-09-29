# Reproducing the paper

`experiments/paper_cells.yaml` lists every result of the paper that public data can reproduce, with the value
the paper reports. `python experiments/reproduce.py` recomputes each one with the released code under the
paper's protocol (the bundled task-level folds, the three seeds, every data-dependent statistic fitted inside the
training folds) and writes `results/paper_cells.md` with the paper value, the released value, the difference and
a 95% bootstrap interval. Nothing is tuned to match.

```bash
make reproduce                                          # all public cells, CPU
python experiments/reproduce.py --accept-licenses       # include LLMRouterBench (license unresolved upstream)
APPWORLD_ROOT=/path/to/appworld python experiments/reproduce.py --cells subagent-routing
```

Baselines that need a GPU (DSR fine-tuning, few-shot and fine-tuned small LLMs), the candidate-overload and
head-ablation figures, and the efficiency benchmark live under `experiments/` with their own READMEs.

Results that depend on private agent trajectories (AppWorld tool shortlisting with CUGA, application selection,
OAK/BPO, the in-loop hot swap) are reported in the paper; the adapters and integration code to rerun them on
your own CUGA traces are included.
