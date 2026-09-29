# Figure experiments

Reproduces the paper's candidate-overload figure (Fig. 2, `fig:overload_new`) and head ablation
(Fig. 3, `fig:card_new`) from the public benchmark tables. Run every command from the repository root.
Outputs (JSON, PDF, PNG) go to `results/figures/` (git-ignored). Change the location with `--out` or
`TABAGENT_RESULTS`.

The tables are rebuilt from the upstream releases (see `tabagent.datasets`). Pass `download=True` once, or
point `TABAGENT_SOURCE_MIRROR` at a local mirror. The card's embedding feature uses `intfloat/e5-small-v2`,
which needs the `embed` extra (`torch`, `transformers`). Set `HF_HUB_OFFLINE=1` once the model is cached.

## Fig. 3: head ablation (`head_ablation.py`)

Each head is cross-validated on SkillsBench skill selection (`skillsbench/skill-select`) on the paper's
pinned folds with the three paper seeds, in two conditions:

- **With the card:** `TabAgent(card, TabHead(estimator=head))`.
- **Without the card** (as in the pre-release E28/E28b runs):
  - Featurizing heads get only E5(task text) concatenated with E5(skill name — description). Both blocks
    are centered on the training fold.
  - TabSTAR gets only the raw text columns.

```bash
python -m experiments.figures.head_ablation                                  # CPU heads, registered card
python -m experiments.figures.head_ablation --cards paper skillsbench_skill_select_noprior
python -m experiments.figures.head_ablation --heads tabstar xgboost lightgbm # optional heads
python -m experiments.figures.head_ablation --plot-only
```

- Heads live in `experiments/heads/`: CatBoost (TabHead), HistGBM, random forest, logistic regression,
  MLP, and XGBoost/LightGBM/TabSTAR when those are installed.
- TabSTAR uses the paper's LoRA settings (lr 0.001, rank 16, 50 epochs) and the pre-release
  collapse-retry guard. It takes hours per condition on CPU, so run it on a GPU.
- The CatBoost preset defaults to `tmlr-runs` (300 trees, depth 6, learning rate 0.08), the configuration
  that produced the figure.
- The run resumes from `head_ablation.json`: finished cells are skipped unless `--force` is given, or the
  seeds, preset or encoder change.

**Card choice matters.** The paper's figure used the pre-release card. That card fitted the skill prior on
a fixed split that overlaps the test folds. The registered card `skillsbench_skill_select` fits the prior
inside each fold, which drops CatBoost from 0.796 to 0.727. The recommended card
`skillsbench_skill_select_noprior` omits the prior and recovers 0.796. Run both with `--cards`.

## Fig. 2: candidate overload (`overload.py`, `overload_llm.py`)

This is the controlled candidate-set scaling of the pre-release E12v2 run:

1. TabAgent (CatBoost, `tmlr-runs`, seed 0) is cross-validated once on the pinned folds.
2. For each task and each K, the slate is all positives plus K negatives sampled without replacement,
   drawn 30 times.
3. P@R and acc@1 are averaged over draws, then over tasks.
4. A power law `P@R = a K^b` is fitted in log-log space and projected to K = 400 and 800.

```bash
python -m experiments.figures.overload                                        # SkillsBench (CPU, minutes)
python -m experiments.figures.overload --datasets skillsbench/skill-select mind2web/element
python -m experiments.figures.overload --card skillsbench_skill_select_noprior --out results/figures/noprior
python -m experiments.figures.overload --llm sft_gen fewshot cross_enc --llm-model llama8b --dry-run
python -m experiments.figures.overload --skip-tabagent --llm cross_enc          # GPU, merges into overload.json
```

**Mind2Web** (368K rows, about 3 GB of dense features while fitting). The pre-release curve pooled every
step of a task into one unit, which is how K reaches 200 even though a step has 50 candidates. That pooling
is the default (`--unit task_uid`). Use `--unit decision` to rank within a step (K ≤ 49).

**LLM curves** (GPU; needs `torch`, `transformers` and `peft`):

- Generative SFT: LoRA, 6 draws per K.
- Cross-encoder: LoRA sequence classifier.
- Few-shot: pointwise p(yes).

All are Llama-3.1-8B-Instruct in the paper, and all run on the same folds and slates. In the pre-release
run the few-shot prompt had no demonstrations (it was zero-shot yes/no scoring), and the port keeps that.
`--dry-run` prints the number of generations or scored pairs for each regime without importing torch.

## Paper values and a reference run

Both JSON files carry the values plotted in the paper (`paper` key). The scripts print each run next to
those values.

Reference run on a 10-core laptop CPU (pinned folds; head ablation with the three paper seeds; P@R).

Fig. 3 (mean over seeds):

| head | without | with (registered card) | with (no-prior card) | paper without / with |
|---|---|---|---|---|
| CatBoost | 0.184 | 0.727 | 0.796 | 0.193 / 0.796 |
| HistGBM | 0.123 | 0.581 | 0.766 | 0.143 / 0.740 |
| Random forest | 0.201 | 0.707 | 0.740 | 0.207 / 0.722 |
| Logistic regression | 0.110 | 0.701 | 0.706 | 0.099 / 0.690 |
| MLP | 0.214 | 0.656 | 0.633 | 0.213 / 0.582 |

Fig. 2 (TabAgent, seed 0):

| panel | card | K=3 | K=10 | K=50 | K=100 | largest K | R² |
|---|---|---|---|---|---|---|---|
| SkillsBench | registered | 0.952 | 0.913 | 0.826 | 0.781 | 0.756 (K=190) | 0.978 |
| SkillsBench | no-prior | 0.970 | 0.944 | 0.883 | 0.845 | 0.822 (K=190) | 0.970 |
| Mind2Web | registered | 0.923 | 0.835 | 0.653 | 0.571 | 0.515 (K=200) | 0.976 |

The paper's values for the same points: SkillsBench 0.965, 0.930, 0.857, 0.817 and 0.785; Mind2Web 0.923,
0.835, 0.652, 0.571 and 0.510.
