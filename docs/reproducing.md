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

The AppWorld results that use the CUGA agent's own trajectories (tool shortlisting, application selection, the
in-loop hot swap) are reported in the paper; those trajectories are not released, and the adapters and
integration code to rerun them on your own CUGA traces are included. The OAK and BPO benchmarks are public;
loaders for them are not included yet.

## Results (released code, the paper's protocol)

CatBoost TabHead (`tmlr-runs` preset: 300 iterations, learning rate 0.08, depth 6), the paper's three seeds,
the bundled folds, every statistic fitted in-fold. Percentages; 95% bootstrap intervals over decisions.

| decision | paper | released |
|---|---|---|
| τ²-bench airline / retail / telecom / combined, P@R | 71 / 76 / 84 / 75 | 71.9 / 75.4 / 84.7 / 76.8 |
| … +synth (TabSynth chosen by nested inner CV) | 75 / 75 / 85 / 78 | 72.6 / 75.2 / 84.9 / 77.0 |
| … Recall@9 | 96 / 96 / 100 / 94 | 98.1 / 95.7 / 100 / 97.0 |
| SkillsBench skill selection, P@R (no-prior card) | 74 | 79.6 [74.0, 85.4] |
| … Recall@9 | 87 | 90.5 |
| Mind2Web operation selection, macro-F1 | 92 | 91.9 |
| Mind2Web element selection, R@3 (Entertainment / Shopping / Travel)¹ | 58 / 65 / 64 | 57.9 / 65.7 / 63.6 |
| … +synth (domain-matched, the paper's recipe)¹ | 68 / 74 / 72 | 68.4 / 74.6 / 70.9 |
| … R@5 | 73 / 78 / 75 | 73.7 / 77.7 / 75.5 |
| AppWorld sub-agent routing, route@1 (best fixed 35.6) | 38 | 37.9 [34.0, 41.7] |
| LLMRouterBench model routing, route@1 (best fixed 70.1) | 68 | 70.5 [65.9, 75.3] |
| SkillsBench skill gating, reward (always-activate 49.6) | 49 | 48.3 [40.0, 56.7] |

¹ Mind2Web element follows the paper's own protocol by default (`element_synth: paper` in
`experiments/paper_cells.yaml`): CatBoost seed 0, the evaluator's row order, and for +synth one model per web
domain, trained on the fold's real rows plus paraphrases of that domain's rows from the evaluator's whole
706-task train split (tasks of the test fold removed). The +synth model therefore also learns from paraphrases
of train-split tasks outside the 150 evaluated tasks, which the real-only model does not see;
`--element-synth in-fold` restricts TabSynth to the training folds of the 150 tasks (the paper's three seeds:
57.0 / 68.5 / 64.0 real, 58.5 / 67.1 / 65.0 +synth). The paper's scripts give exactly the released values on
the same machine. The paper's run, on an x86-64 CPU with the same library versions, gave 57.9 / 65.4 / 63.6
and 68.4 / 74.3 / 71.7: floating-point differences between CPU architectures move these fits by up to 0.8
points.

**BM25** is computed as in the paper: each row's own query and document text (τ²: the first 400 characters
of the customer's message against the tool name; SkillsBench: the task against the skill name and the first
200 characters of its description; Mind2Web element: the instruction against the first 200 characters of the
attribute text), scored in expectation over a uniformly random order of tied scores
(`metrics.p_at_r(..., tie_break="expected")`). BM25 gives most candidates the same score of 0 (the
demonstrated element in 78% of Mind2Web steps), so a fixed tie order would decide much of its result.
Released: τ² airline / retail / telecom / combined P@R 28.0 / 33.0 / 40.8 / 28.3 (paper 28 / 33 / 41 / 28),
skill selection 61.0 (paper 60; the released skill descriptions come from each `SKILL.md`), Mind2Web element
R@3 13.2 / 17.5 / 21.8 (paper 13 / 18 / 22). Under the pre-release evaluator's sort, which kept tied
candidates in table order with the demonstrated element first, element BM25 was 60.9 / 52.7 / 59.0.

Policy rows against their strong default (paired task-level bootstrap, 2,000 resamples): sub-agent routing
+2.3 pp [−0.5, +5.1], model routing +0.4 pp [−3.5, +4.6], skill gating −1.3 pp [−3.8, +0.7]: parity in every
case, as the paper states.

**τ²-bench cross-agent transfer** (`experiments/tau2_studies.py --study transfer`, the context each agent
actually sees): own-agent mean P@R 76.6, cross-agent mean 73.5. **Trajectories to 90% of full-pool quality**
(`--study efficiency`, ten seeds): Claude-3.7 37 [32, 43], GPT-4.1 45 [40, 51], GPT-4.1-mini 45 [38, 53],
o4-mini 51 [45, 58].

Notes on differences from the pre-release code, all deliberate: candidate priors and embedding centering are
fitted inside the training folds (the paper's protocol; the pre-release skill-selection prior was computed on a
fixed split that included test tasks, which is why the released no-prior card is the default); the τ² per-agent
context is the customer's first message rather than the simulator's hidden task description; SkillsBench skill
descriptions are read from each skill's `SKILL.md`.
