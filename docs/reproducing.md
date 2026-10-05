# Reproducing the paper

`experiments/paper_cells.yaml` lists every result of the paper that public data can reproduce, with the value
the paper reports. `python -m experiments.reproduce` recomputes each one with the released code under the
paper's protocol (the bundled task-level folds, the three seeds, every data-dependent statistic fitted inside
the training folds, tied scores ordered uniformly at random with the exact expected metric reported) and
writes `results/paper_cells.md` with the paper value, the released value, the difference and a 95% bootstrap
interval. Nothing is tuned to match.

Every cell runs on the feature card bundled with its dataset, so reproducing calls no LLM. A card is *frozen*
when its feature list was taken from the paper's experiment code for that benchmark and fixed, every feature
recomputed by the release's code (a few frozen cards are variants; each card's `provenance` note says how), and
*hand-authored* when it was written for the release from the paper's description of the benchmark's features
(App. B); `provenance.kind` in each card says which. No cell runs TabSchema, and the released TabSchema
implementation has not yet been validated against a paper number (see "Differences from the paper").

```bash
uv sync --extra embed      # the exact environment of uv.lock, with the e5 encoder's dependencies
make data                  # every public dataset, pinned and SHA-256 verified (about 7.9 GB)
make reproduce             # every public Table 1 and Table 2 cell, on CPU
```

With pip instead (`pip install -e ".[embed]"`), run `python -m experiments.reproduce --download`; pip resolves
the newest dependency releases rather than the locked ones. `--cells` selects cells (for example
`--cells skill-selection transfer`), `--accept-licenses` adds LLMRouterBench (an opt-in download; MIT as
declared upstream), and the AppWorld cell needs `APPWORLD_ROOT`. The values below were computed on an Apple M1
Max (10 cores) with Python 3.13, CatBoost 1.2.10, pandas 3.0 and scikit-learn 1.9.

## What each paper result needs

Times are for a 10-core laptop CPU, after the data is cached. Downloads are counted once: the three
Mind2Web tables share one release, as do the four per-agent and the four per-domain τ² tables. Every
card with an embedding-similarity feature also downloads the e5-small-v2 encoder once (about 128 MB).

| paper result | command (`python -m ...`) | hardware | time | download | access |
|---|---|---|---|---|---|
| all public data | `make data` | CPU | download-bound | 7.9 GB | public; LLMRouterBench and AppWorld as below |
| Table 1, τ²-bench rows (TabAgent, +synth, R@k, chance, BM25) | `experiments.reproduce --cells tau2-airline tau2-retail tau2-telecom tau2-combined` | CPU | 25 min | 8.2 MB | public |
| Table 1, skill selection; Table 20 rows 1–2 | `experiments.reproduce --cells skill-selection` | CPU | 2 min | 3.5 MB | public |
| Table 1, operation selection | `experiments.reproduce --cells operation-selection` | CPU | 2 min, after the download-bound first build of the Mind2Web tables | 5.9 GB | public |
| Table 1, element selection | `experiments.reproduce --cells element-selection` | CPU | 15 min | 5.9 GB (shared) | public |
| Table 1, sub-agent routing; Table 20 | `experiments.reproduce --cells subagent-routing` | CPU | 1 min | AppWorld data and experiment outputs | AppWorld's CLI (`APPWORLD_ROOT`) |
| Table 1, model routing; Table 20 | `experiments.reproduce --cells model-routing --accept-licenses` | CPU | 1 min | 1.35 GB | public (MIT as declared upstream); opt-in download |
| Table 1, skill gating; Table 20 | `experiments.reproduce --cells skill-gating` | CPU | 1 min | 3.5 MB (shared) | public |
| Table 1, DSR-ZS and DSR-FT columns | `experiments.baselines.run --dataset <name> --method dsr-zs` (or `dsr-ft`) | CPU or Apple MPS | seconds to 12 min per row | as the row | as the row |
| Table 1, Few-shot and FT-SLM columns; App. F, Fig. 8b | `experiments.baselines.run --dataset <name> --method fewshot` (or `ft-slm`) | CUDA GPU (40–80 GB) | hours per row | as the row, plus model weights | Few-shot on skill gating: an endpoint serving GPT-5.4 |
| Table 2 | `experiments.reproduce --cells transfer` | CPU | 5 min | 604 MB | public |
| App. F.1, Table 17 (τ² rows) | `experiments.tau2_studies --study efficiency` | CPU | hours (ten seeds) | 604 MB (shared) | public |
| App. N.4, Table 21 (τ² row) | `experiments.tau2_studies --study rename` | CPU | 5 min | 604 MB (shared) | public |
| Fig. 2, TabAgent curves | `experiments.figures.overload --datasets skillsbench/skill-select mind2web/element` | CPU (about 3 GB RAM for Mind2Web) | 2 min + 10 min | 3.5 MB + 5.9 GB | public |
| Fig. 2, LLM curves | `experiments.figures.overload --skip-tabagent --llm sft_gen fewshot cross_enc` | CUDA GPU | hours | Llama-3.1-8B weights | public |
| Fig. 3, CPU heads | `experiments.figures.head_ablation` | CPU | 1 hour | 3.5 MB | public |
| Fig. 3, TabSTAR | `experiments.figures.head_ablation --heads tabstar` | GPU | hours | TabSTAR weights | public |
| Table 15 and Fig. 4 (cost formulas and the reported points) | `experiments.efficiency.cost`, `experiments.efficiency.pareto --paper` | CPU | seconds | none | public |
| per-decision latency on public data | `experiments.efficiency.latency` | CPU | 5 min | 8.2 MB | public |
| every result on the CUGA agent's AppWorld trajectories: AppWorld tool shortlisting and application selection (Table 1, Tables 5–10, 12–14, 19; Figs. 5–7), the in-loop hot swap (Tables 3, 23), Table 16, Figs. 8a, 9–14, Tables 17, 21 and 22 (AppWorld rows), Tables 18, 24 and 25 | none | | | | not released |
| OAK and BPO rows of Table 1 | none (the benchmarks are public; loaders are not included yet) | | | | public |

The AppWorld benchmark itself is public; the results above use the CUGA agent's own trajectories on it, which
are not released. The CUGA trajectory adapter (both decisions, `ShortlisterAgent` and `TaskAnalyzerAgent`) and
the hot-swap helpers (`tabagent.integrations.cuga`, see [Deployment](deployment.md#cuga)) are included, so the
pipeline runs on your own CUGA traces and the heads plug back into CUGA. Table 15's and Fig. 4's TabAgent latency was measured on those trajectories;
`experiments.efficiency.latency` measures the same quantity on public data (see
`experiments/efficiency/README.md`). The candidate-overload and head-ablation figures and the efficiency
benchmark have their own READMEs under `experiments/`.

## Results (released code, the paper's protocol)

CatBoost TabHead (the paper's configuration: 300 iterations, learning rate 0.08, depth 6), the paper's three
seeds, the bundled folds, every statistic fitted in-fold. Percentages; 95% bootstrap intervals over decisions.
"R@k" is Table 1's R@k column: the recall of the +synth head where the row reports a +synth value, of the
real-only head otherwise.

| decision | paper | released |
|---|---|---|
| τ²-bench airline / retail / telecom / combined, P@R | 72 / 75 / 85 / 77 | 71.9 / 75.4 / 84.7 / 76.8 |
| … +synth (the extended TabSynth operators of App. H, nested choice) | 71 / 75 / 85 / 78 | 71.3 / 75.2 / 85.2 / 77.8 |
| … R@k: Recall@9 of the +synth head | 97 / 96 / 100 / 97 | 96.7 / 95.7 / 99.7 / 97.3 |
| … Recall@9 of the real-only head (Table 11) | 98 / 96 / 100 / 97 | 98.1 / 95.7 / 100 / 97.0 |
| … chance | 21 / 28 / 26 / 10 | 21.1 / 27.8 / 25.5 / 9.5 |
| … BM25 | 28 / 33 / 41 / 28 | 28.0 / 33.0 / 40.8 / 28.3 |
| SkillsBench skill selection, P@R | 80 | 79.6 [74.0, 85.4] |
| … R@k: Recall@9 | 91 | 90.5 |
| … chance / BM25 | 1 / 65 | 1.4 / 64.8 |
| Mind2Web operation selection, macro-F1 | 92 | 91.9 |
| Mind2Web element selection, R@3 (Entertainment / Shopping / Travel)¹ | 58 / 65 / 64 | 57.7 / 65.7 / 63.6 |
| … +synth (domain-matched)¹ | 68 / 74 / 72 | 68.1 / 74.6 / 70.8 |
| … R@k: Recall@5 of the +synth head¹ | 82 / 84 / 83 | 82.2 / 84.1 / 82.8 |
| … chance / BM25 | 6 / 13, 18, 22 | 6.0 / 13.2, 17.5, 21.8 |
| AppWorld sub-agent routing, route@1 | 38 | 37.9 [34.0, 41.7] |
| LLMRouterBench model routing, route@1 | 71 | 70.5 [65.9, 75.3] |
| SkillsBench skill gating, reward | 48 | 48.3 [40.0, 56.7] |
| Table 2, all 16 cells; own-agent / cross-agent mean P@R | 76.6 / 73.5 | every cell as in the paper; 76.6 / 73.5 |

¹ Under the paper's element protocol; see "Differences from the paper".

Every released value in this table is within about one point of the paper's; the element +synth cells
follow the protocol described under "Differences from the paper".

**BM25** scores each row's own query and document text: τ², the first 400 characters of the customer's
message against the tool name; skill selection, the task against the skill name and description (the card's
text); Mind2Web element, the instruction against the first 200 characters of the element's attribute text.
BM25 gives most candidates the same score of 0 (the demonstrated element in 78% of Mind2Web steps), so its
value depends on how ties are ordered; like every metric here, it is the expectation over a uniformly random
order.

**Table 20** (paired task-level bootstrap, 2,000 resamples), released: skill selection 79.6 [74.0, 85.4] and
+14.8 pp over BM25 (paper +14.7); sub-agent routing +2.3 pp [−0.5, +5.1] over the best fixed system (35.6),
model routing +0.4 pp [−3.5, +4.6] over the best fixed model (70.1), skill gating −1.3 pp [−3.8, +0.7]
against always-activate (49.6), as in the paper. Skill selection over fine-tuned dense retrieval
(DSR-FT 74.5): +5.1 pp, as in the paper.

**τ²-bench four-agent studies** (`experiments.tau2_studies`, with the context each agent sees before it
acts): Table 2 as above. Trajectories to 90% of full-pool quality (Table 17, ten seeds): Claude-3.7 37
[32, 43], GPT-4.1 45 [40, 51], GPT-4.1-mini 45 [38, 53], o4-mini 51 [45, 58], as in the paper. Hashed tool
names at test time with the prior carried (Table 21, τ² row): P@R 76.6 → 54.3, −22.3 pp, as in the paper.

## Differences from the paper

Every remaining difference between the released code and the paper, with its effect.

- **Mind2Web element selection, +synth pool and seed.** The element row follows the protocol behind the
  paper's values (`element_synth: paper` in `paper_cells.yaml`). CatBoost runs with seed 0 only, where App. M
  states three seeds for every decision-quality score. For +synth, one model per web domain is trained on the
  fold's real rows plus TabSynth paraphrases (k=2, the instruction) of that domain's rows of the 706-task
  Mind2Web train split, minus the test fold's tasks; the +synth model therefore also sees paraphrases of
  train-split tasks outside the 150 evaluated tasks, which the real-only model does not. App. B.6 describes
  domain-matched TabSynth without this pool. `--element-synth in-fold` restricts TabSynth to the training
  folds of the 150 evaluated tasks. CatBoost fits on this row also move by up to about 0.8 points between CPU
  architectures (x86-64 against arm64).
- **τ²-bench +synth.** TabSynth uses the extended operators of App. H (text features recomputed per
  paraphrase, entity swap, 1/√f coverage allocation). On each outer training fold, a task-grouped 4-fold inner
  cross-validation chooses among no synthesis and four configurations; over the 15 outer folds (5 folds × 3
  seeds) it chose synthesis in 10 (airline), 8 (retail), 14 (telecom) and 13 (combined).
- **Feature cards.** The paper runs the TabSchema loop inside each training fold (Sec. 4.3). The bundled
  cards are frozen or hand-authored (defined above; `provenance.kind` in each card), so no cell runs TabSchema;
  only their data-dependent statistics (candidate priors, embedding centering, k-NN outcomes) are fitted inside
  each training fold. On your own logs, `tabagent.cross_validate_discovery` runs discovery inside each training
  fold as the paper describes; the released TabSchema has not been compared with a paper number yet. The routing
  cards (sub-agent and model routing) omit a `task_family_solve_rate` feature (the mean success of all arms on
  similar tasks) that the paper's cards had. The skill-gating card encodes the skill bundle as a passage and
  computes its similarity on both arms.
- **Fig. 3 (head ablation, skill selection).** The paper's "with card" bars are CatBoost 0.796, HistGBM 0.740,
  random forest 0.722, logistic regression 0.690 and MLP 0.582. The released code with the registered card
  gives CatBoost 0.796, HistGBM 0.766, random forest 0.737, logistic regression 0.706 and MLP 0.633
  (`experiments.figures.head_ablation`). A variant of the card with a skill prior fitted inside each fold
  (`skillsbench_skill_select`) gives 72.7 with CatBoost.
- **Skill descriptions.** A skill's text is the first 800 characters of its `SKILL.md` in the pinned
  SkillsBench release. In the paper's runs, 6 of the 195 skill texts came from another Markdown file of the
  skill's folder (a README or reference page).
- **Baselines.** Skill-selection DSR reads the full task and skill texts (no character cut), as in the
  paper's runs: DSR-ZS 47.8 against 51, and DSR-FT (in-batch negatives) 74.5 against 74. Mind2Web element
  FT-SLM (31 / 48 / 41) is a pointwise LoRA cross-encoder on Llama-3.1-8B, not the generative fine-tuning
  Sec. 4.2 describes, and the per-domain τ² FT-SLM values were computed with three task-grouped folds instead
  of the paper's five. The Mind2Web element Few-shot values (15 / 14 / 20) and the few-shot curves of Fig. 2
  score each candidate with a yes/no prompt that carries no demonstrations. The release computes no BM25 or
  chance for operation selection and no BM25 for the routing rows, and has no few-shot prompt for model or
sub-agent routing. `experiments/baselines/README.md` lists every released baseline value.
