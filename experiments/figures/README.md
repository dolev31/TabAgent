# Figure experiments

Reproduces the paper's candidate-overload figure (Fig. 2) and head ablation (Fig. 3) from the public
benchmark tables. Run every command from the repository root. Outputs (JSON, PDF, PNG) go to
`results/figures/` (git-ignored). Change the location with `--out` or `TABAGENT_RESULTS`.

The tables are rebuilt from the upstream releases (see `tabagent.datasets`): pass `--download` once (or run
`make data`), or point `TABAGENT_SOURCE_MIRROR` at a local mirror. The card's embedding feature uses
`intfloat/e5-small-v2`, which needs the `embed` extra (`torch`, `transformers`) and is downloaded from Hugging
Face once (about 128 MB). Set `HF_HUB_OFFLINE=1` once the model is cached.

## Fig. 3: head ablation (`head_ablation.py`)

Each head is cross-validated on SkillsBench skill selection (`skillsbench/skill-select`) on the paper's
pinned folds with the three paper seeds, in two conditions:

- **With the card:** `TabAgent(card, TabHead(estimator=head))`.
- **Without the card:**
  - Featurizing heads get only E5(task text) concatenated with E5(skill name — description). Both blocks
    are centered on the training fold.
  - TabSTAR gets only the raw text columns.

```bash
python -m experiments.figures.head_ablation --download                        # CPU heads, registered card
python -m experiments.figures.head_ablation --cards paper skillsbench_skill_select
python -m experiments.figures.head_ablation --heads tabstar xgboost lightgbm  # optional heads
python -m experiments.figures.head_ablation --plot-only
```

- Heads live in `experiments/heads/`: CatBoost (TabHead), HistGBM, random forest, logistic regression,
  MLP, and XGBoost/LightGBM/TabSTAR when those are installed.
- TabSTAR uses the paper's LoRA settings (lr 0.001, rank 16, 50 epochs, App. K.3) and retries a fold once
  when its outputs collapse to a constant. It takes hours per condition on CPU, so run it on a GPU.
- The CatBoost preset defaults to `paper` (300 trees, depth 6, learning rate 0.08), the configuration the
  paper states (Sec. 4.2).
- The run resumes from `head_ablation.json`: finished cells are skipped unless `--force` is given, or the
  seeds, preset or encoder change.

`--cards paper` is the dataset's registered card, which has no skill-prior feature. `skillsbench_skill_select`
adds a skill prior fitted inside each training fold; 180 of the 195 skills occur in a single task's bundle,
so such a prior is 0 for a held-out task's own skills, and it lowers CatBoost from 0.796 to 0.727.

## Fig. 2: candidate overload (`overload.py`, `overload_llm.py`)

The controlled candidate-set scaling:

1. TabAgent (CatBoost, `paper` preset, seed 0) is cross-validated once on the pinned folds.
2. For each task and each K, the slate is all positives plus K negatives sampled without replacement,
   drawn 30 times.
3. P@R and acc@1 are averaged over draws, then over tasks. Tied scores are ordered uniformly at random and
   each draw contributes its exact expected value (paper App. L).
4. A power law `P@R = a K^b` is fitted in log-log space and projected to K = 400 and 800.

```bash
python -m experiments.figures.overload --download                             # SkillsBench (CPU, minutes)
python -m experiments.figures.overload --datasets skillsbench/skill-select mind2web/element
python -m experiments.figures.overload --card skillsbench_skill_select --out results/figures/prior
python -m experiments.figures.overload --llm sft_gen fewshot cross_enc --llm-model llama8b --dry-run
python -m experiments.figures.overload --skip-tabagent --llm cross_enc          # GPU, merges into overload.json
```

**Mind2Web** (368K rows, about 3 GB of dense features while fitting). Every step of a task is pooled into one
unit, which is how K reaches 200 even though a step has 50 candidates. That pooling is the default
(`--unit task_uid`). Use `--unit decision` to rank within a step (K ≤ 49).

**LLM curves** (GPU; needs `torch`, `transformers` and `peft`):

- Generative SFT: LoRA, 6 draws per K.
- Cross-encoder: LoRA sequence classifier.
- Few-shot: pointwise p(yes); the prompt carries no demonstrations.

All run Llama-3.1-8B-Instruct by default (`--llm-model llama8b`) on the same folds and slates. `--dry-run`
prints the number of generations or scored pairs for each regime without importing torch.

## Paper values and a reference run

Both JSON files carry the values plotted in the paper (`paper` key). The scripts print each run next to
those values.

Reference run on a 10-core laptop CPU (pinned folds; head ablation with the three paper seeds; P@R).

Fig. 3 (mean over seeds):

| head | without | with (registered card) | with (`skillsbench_skill_select`, in-fold prior) | paper without / with |
|---|---|---|---|---|
| CatBoost | 0.184 | 0.796 | 0.727 | 0.193 / 0.796 |
| HistGBM | 0.126 | 0.766 | 0.581 | 0.143 / 0.740 |
| Random forest | 0.202 | 0.737 | 0.706 | 0.207 / 0.722 |
| Logistic regression | 0.110 | 0.706 | 0.701 | 0.099 / 0.690 |
| MLP | 0.214 | 0.633 | 0.656 | 0.213 / 0.582 |

Fig. 2 (TabAgent, seed 0):

| panel | card | K=3 | K=10 | K=50 | K=100 | largest K | R² |
|---|---|---|---|---|---|---|---|
| SkillsBench | registered | 0.970 | 0.944 | 0.883 | 0.845 | 0.822 (K=190) | 0.970 |
| SkillsBench | `skillsbench_skill_select` | 0.952 | 0.913 | 0.826 | 0.781 | 0.756 (K=190) | 0.978 |
| Mind2Web | registered | 0.922 | 0.835 | 0.650 | 0.568 | 0.511 (K=200) | 0.976 |

The paper's values for the same points: SkillsBench 0.965, 0.930, 0.857, 0.817 and 0.785; Mind2Web 0.923,
0.835, 0.652, 0.571 and 0.510. [docs/reproducing.md](../../docs/reproducing.md) discusses the differences.
