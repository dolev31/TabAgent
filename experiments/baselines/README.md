# Learned baselines

The paper compares TabAgent against a shared ladder of baselines at every decision type (Sec. 4.2; Table 1).
The training-free rungs (chance, BM25, e5 zero-shot) live in the package (`tabagent.baselines`). This
directory holds the learned rungs, which need PyTorch and, for the LLM rungs, a GPU:

| Table 1 column | Method here | What it is |
|---|---|---|
| DSR-ZS | `dsr-zs` | e5-small-v2 bi-encoder as released: inner product of the L2-normalized, mean-pooled embeddings of the context (`query: `) and the candidate text (`passage: `). No training. |
| DSR-FT | `dsr-ft` | The same encoder, contrastively fine-tuned with LoRA on the training folds of each split with in-batch negatives (Sec. 4.2), then scoring the held-out fold. `--synth K` adds TabSynth paraphrases of the training contexts (DSR-FT Real+synth). |
| Few-shot | `fewshot` | Llama-3.1-8B-Instruct with two demonstrations on the retrieval and classification rows; on the policy/reward rows the paper uses GPT-5.4 (Sec. 4.2), which runs here through any OpenAI-compatible endpoint (the skill-gating prompt is included). |
| FT-SLM | `ft-slm` | Llama-3.1-8B-Instruct fine-tuned with LoRA per fold to generate the relevant candidate numbers (set retrieval) or the class word (operation). On Mind2Web element selection (about 50 candidates per step) it is a pointwise LoRA cross-encoder on the same model. |
| SLM family (App. F, Fig. 8) | `fewshot`, `ft-slm`, `ft-ce` with `--model llama-3.2-1b`, `llama-3.2-3b`, `llama-3.1-8b`, `qwen2.5-7b` | The same prompts and recipes for each model; `ft-ce` is the LoRA cross-encoder. |

Every trainable baseline is trained only on the training folds of the paper's task folds
(`tabagent.datasets.folds(name)`); rows of a task never straddle folds, and the few-shot demonstrations'
tasks are excluded from evaluation. Metrics order tied scores uniformly at random and report the exact
expected value (paper App. L).

## Commands

```bash
# DSR on a CPU or Apple-silicon laptop (skill selection: seconds; Mind2Web element-150: minutes)
python -m experiments.baselines.run --dataset skillsbench/skill-select --method dsr-zs --download
python -m experiments.baselines.run --dataset skillsbench/skill-select --method dsr-ft \
    --seeds 20260720 20260726 20260727
python -m experiments.baselines.run --dataset mind2web/element-150 --method dsr-zs

# LLM baselines (CUDA GPU; bfloat16 Llama-3.1-8B fits a 40 GB card, prompts of up to 8K tokens)
python -m experiments.baselines.run --dataset skillsbench/skill-select --method fewshot
python -m experiments.baselines.run --dataset skillsbench/skill-select --method ft-slm --model llama-3.1-8b
python -m experiments.baselines.run --dataset mind2web/element-150 --method ft-slm
OPENAI_API_KEY=... OPENAI_BASE_URL=... \
python -m experiments.baselines.run --dataset skillsbench/skill-gating --method fewshot --model gpt-5.4

# Build and inspect every prompt without loading a model (no GPU)
python -m experiments.baselines.run --dataset tau2/airline --method ft-slm --dry-run \
    --prompts-out results/baselines/tau2-airline.prompts.jsonl
```

`--dataset` accepts every table with a preset (`presets.py`): `skillsbench/skill-select`,
`skillsbench/skill-gating`, `mind2web/operation`, `mind2web/element-150`, `mind2web/element`, the four
per-domain τ² tables (`tau2/airline`, `tau2/retail`, `tau2/telecom`, `tau2/all-domains`), the four
per-agent τ² tables, and `llmrouterbench/tau2` (add `--accept-license`). Other flags: `--recipe
in-batch|hard-negatives` (DSR-FT negatives, below), `--device`, `--max-groups N` (smoke tests), `--download`.

The output JSON (default `results/baselines/<dataset>__<method>[__<model>].json`, never committed) holds:

- `per_decision[metric]`: the value of every evaluated decision, averaged over seeds;
- `decision_group`: decision id to task id, for task-level resampling;
- `summary[metric]`: mean over decisions (the number the paper reports) with a 95% percentile bootstrap
  CI over decisions, and the same over tasks (`group_*`); macro-F1 for single-class rows; chance and oracle
  where defined;
- `per_seed`, `slices` (per web domain or τ² domain), `config`, `source` (the paper result the preset
  reproduces) and `paper` (its Table 1 value).

`python -m experiments.reproduce` reads these files into the baseline columns of its comparison. A paired CI
against TabAgent on the decisions both cover (the comparisons of App. M, Table 20):

```python
import json, pandas as pd
from tabagent.evaluation import stats

b = pd.Series(
    json.load(open("results/baselines/skillsbench__skill-select__dsr-ft.json"))["per_decision"]["p@r"]
)
a = tabagent_cv_result.per_decision("p@r")  # tabagent.evaluation.cv.CVResult
a, b = a.align(b, join="inner")
print(stats.paired_bootstrap(a.to_numpy(), b.to_numpy()))
```

## Recipes

**DSR.** Texts are cut to 256 characters (skill selection: `skill_name` plus the first 200 characters of the
SKILL.md; Mind2Web: the element's attribute text capped at 200), or 200 characters and 96 tokens on the
per-domain τ² tables. Fine-tuning: LoRA r=16, alpha=32, dropout 0.05 on the query/value projections, AdamW
2e-4, temperature 0.05, batch 32, 3 epochs (2 on Mind2Web and per-domain τ²).

- `--recipe in-batch` (default) is the recipe of Sec. 4.2: each positive pair is contrasted against the other
  passages of its batch.
- `--recipe hard-negatives` adds one mined hard negative per positive, drawn from the 10 non-relevant
  candidates of the same decision that the zero-shot encoder ranks highest (mined on the training folds
  only), and masks in-batch passages that are relevant to the query instead of counting them as negatives.
  Its output file is `<dataset>__dsr-ft__hard-negatives.json`.

**Few-shot and FT-SLM.** Listwise prompts (`TASK: ...`, a numbered candidate list, `Relevant numbers:`) for
skill selection and τ² tool shortlisting; a yes/no scorer and a LoRA cross-encoder for Mind2Web elements; a
class-word classifier for the Mind2Web operation; a yes/no gate for skill gating. LoRA r=16, alpha=32 on
q_proj/v_proj, AdamW 2e-4, batch 1, 3 epochs (4 per-domain τ², 2 operation); cross-encoders train 2 epochs
at batch 16 with class-balanced BCE. Models load from ungated mirrors of the paper's checkpoints (`MODELS` in
`llm.py`); any Hugging Face id works with `--model`. Every prompt's `source` names the Table 1 cell it
reproduces; `reconstructed` marks a configuration written for this release without the settings of the
paper's run (DSR on Mind2Web operation, where the classes are ranked as passages; model routing; the
skill-gating FT-SLM), so its value is not expected to match Table 1 exactly. No few-shot prompt is included
for model routing, and sub-agent routing has no learned-baseline preset.

## Hardware and dependencies

- DSR: `torch` and `transformers` (the package's `embed` extra). A CPU or Apple MPS laptop is enough for
  skill selection, τ² and Mind2Web element-150 (on an M-series laptop: DSR-ZS seconds; DSR-FT 0.5 min on
  skill selection, under 2 min on element-150 and τ², 12 min on operation). The full Mind2Web element
  table (368K rows) wants a CUDA GPU.
- LLM baselines: a CUDA GPU with bfloat16 (8B-model LoRA SFT on prompts of up to 8K tokens, batch 1, no
  gradient checkpointing, fits one 80 GB A100), `torch`, `transformers`, `tokenizers`; `openai` (the
  `tabschema` extra) for the gate.
- LoRA is implemented in `lora.py` (peft's initialization and scaling), so `peft` is not required.

## Released DSR values (Apple MPS, paper folds)

Seed 0, and in parentheses the mean over the three paper seeds where it was run; P@R for set retrieval,
per-step R@3 per web domain for elements, macro-F1 for the operation. "Paper" is Table 1.

| Decision | Method | This code | Paper |
|---|---|---|---|
| Skill selection | DSR-ZS | 0.478 | 51 |
| Skill selection | DSR-FT, `in-batch` | 0.745 | 74 |
| τ² airline / retail / telecom / combined | DSR-ZS | .299 / .417 / .462 / .322 | 30 / 48 / 46 / 34 |
| τ² airline / retail / telecom / combined | DSR-FT, `in-batch` | .456 / .406 / .291 / .392 | 44 / 42 / 28 / 42 |
| τ² airline / retail / telecom / combined | DSR-FT, `hard-negatives` | .637 / .673 / .642 / .623 | |
| Element-150, Ent. / Shop. / Travel | DSR-ZS | .113 / .132 / .117 | 11 / 13 / 12 |
| Element-150, Ent. / Shop. / Travel | DSR-FT, `in-batch` | .140 / .174 / .214 | 15 / 18 / 23 |
| Element-150, Ent. / Shop. / Travel | DSR-FT, `hard-negatives` | .205 / .243 / .299 | |
| Operation | DSR-ZS / DSR-FT `in-batch` / DSR-FT `hard-negatives` | 0.276 / 0.838 / 0.925 | 28 / 84 |

Where the released value differs from Table 1:

- **Skill selection.** The DSR text views are the full task text and the skill name with its full text (the
  first 800 characters of its SKILL.md), with no character cut; a 256-character cut keeps little more than the
  YAML front matter of a SkillsBench task file. Released DSR-ZS 47.8 and DSR-FT 74.5 (in-batch, the paper's
  three seeds) against Table 1's 51 and 74; TabAgent's margin over DSR-FT is +5.1 pp, as in Table 20.
- **Mind2Web operation.** Released DSR-ZS 27.6 and DSR-FT 83.8 (macro-F1, in-batch, the paper's three seeds)
  against the paper's 28 and 84; 92.5 with hard negatives. The zero-shot value depends strongly on the wording of the query and class
  passages (macro-F1 from 0.08 to 0.80 across reasonable wordings).
- **Recipes.** On the small τ² catalogs the `hard-negatives` recipe is far stronger than the paper's in-batch
  recipe (.62–.67 against .29–.46 P@R): there, most in-batch "negatives" are other queries' positives for the
  same few tools, which that recipe masks.
- **Mind2Web element.** On element-150, 47% of rows tie with another candidate of their step (identical
  attribute text), so the tie rule moves R@3 by several points; the values above are the expectation over a
  uniformly random order of tied candidates, as for every method (Table 1: DSR-ZS 11 / 13 / 12).

## Protocol notes

- Fine-tuned baselines use the paper's five task folds everywhere. The Table 1 per-domain τ² FT-SLM values
  were computed with three task-grouped folds.
- DSR-FT Real+synth generates the paraphrases inside each training fold.
- Skill gating: the gate sees the skill bundle stored on the `with` arm's row.
- Seeds are explicit (`--seeds`); the LoRA adapters' initialization is seeded too.
