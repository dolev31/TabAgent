# Learned baselines

The paper compares TabAgent against a shared ladder of baselines at every decision type (Sec. 4,
Implementation Details; Table 1). The training-free rungs (chance, BM25, e5 zero-shot) live in the package
(`tabagent.baselines`). This directory holds the learned rungs, which need PyTorch and, for the LLM rungs,
a GPU:

| Table 1 column | Method here | What it is |
|---|---|---|
| DSR-ZS | `dsr-zs` | e5-small-v2 bi-encoder as released: inner product of the L2-normalized, mean-pooled embeddings of the context (`query: `) and the candidate text (`passage: `). No training. |
| DSR-FT | `dsr-ft` | The same encoder, contrastively fine-tuned with LoRA on the training folds of each split, then scoring the held-out fold. `--synth K` adds TabSynth paraphrases of the training contexts (DSR-FT Real+synth, AppWorld appendix tables). |
| Few-shot | `fewshot` | Llama-3.1-8B-Instruct with two demonstrations (retrieval and classification rows); GPT-5.4 through any OpenAI-compatible endpoint on skill gating. |
| FT-SLM | `ft-slm` | Llama-3.1-8B-Instruct fine-tuned with LoRA per fold: generating the relevant candidate numbers (set retrieval) or the class word (operation), or as a pointwise cross-encoder where the paper used one (Mind2Web element selection). |
| SLM family (appendix) | `fewshot`, `ft-slm`, `ft-ce` with `--model llama-3.2-1b`, `llama-3.2-3b`, `llama-3.1-8b`, `qwen2.5-7b` | The same prompts and recipes for each model; `ft-ce` is the LoRA cross-encoder. |

Every trainable baseline is trained only on the training folds of the paper's task folds
(`tabagent.datasets.folds(name)`); rows of a task never straddle folds, and the few-shot demonstrations'
tasks are excluded from evaluation.

## Commands

```bash
# DSR on a CPU or Apple-silicon laptop (skill selection: seconds; Mind2Web element-150: minutes)
python experiments/baselines/run.py --dataset skillsbench/skill-select --method dsr-zs
python experiments/baselines/run.py --dataset skillsbench/skill-select --method dsr-ft \
    --seeds 20260720 20260726 20260727
python experiments/baselines/run.py --dataset mind2web/element-150 --method dsr-zs

# LLM baselines (CUDA GPU; bfloat16 Llama-3.1-8B fits a 40 GB card, prompts of up to 8K tokens)
python experiments/baselines/run.py --dataset skillsbench/skill-select --method fewshot
python experiments/baselines/run.py --dataset skillsbench/skill-select --method ft-slm --model llama-3.1-8b
python experiments/baselines/run.py --dataset mind2web/element-150 --method ft-slm
OPENAI_API_KEY=... OPENAI_BASE_URL=... \
python experiments/baselines/run.py --dataset skillsbench/skill-gating --method fewshot --model gpt-5.4

# Build and inspect every prompt without loading a model (no GPU)
python experiments/baselines/run.py --dataset tau2/airline --method ft-slm --dry-run \
    --prompts-out results/baselines/tau2-airline.prompts.jsonl
```

`--dataset` accepts every table with a preset (`presets.py`): `skillsbench/skill-select`,
`skillsbench/skill-gating`, `mind2web/operation`, `mind2web/element-150`, `mind2web/element`, the four
per-domain τ² tables (`tau2/airline`, `tau2/retail`, `tau2/telecom`, `tau2/all-domains`), the four
per-agent τ² tables, and `llmrouterbench/tau2` (add `--accept-license`). Other flags: `--recipe
paper|legacy` (DSR-FT negatives, below), `--device`, `--max-groups N` (smoke tests), `--download`.

The output JSON (default `results/baselines/<dataset>__<method>[__<model>].json`, never committed) holds:

- `per_decision[metric]`: the value of every evaluated decision, averaged over seeds;
- `decision_group`: decision id to task id, for task-level resampling;
- `summary[metric]`: mean over decisions (the number the paper reports) with a 95% percentile bootstrap
  CI over decisions, and the same over tasks (`group_*`); macro-F1 for single-class rows; chance and oracle
  where defined;
- `per_seed`, `slices` (per web domain or τ² domain), `config`, `paper` and `legacy` reference values.

A paired CI against TabAgent on the decisions both cover:

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

## Recipes and their provenance

Each preset ports the pre-release script that produced the published cell (`source` in the output).

**DSR** (`run_e32_bm25_dsr.py`, `run_e40_element_perstep.py`, `run_e45b_tau_domain_fixed.py`). Texts
are cut to 256 characters (skill selection: `skill_name` plus the first 200 characters of the SKILL.md;
Mind2Web: the element's attribute text capped at 200), or 200 characters and 96 tokens on the per-domain τ²
tables. Fine-tuning: LoRA r=16, alpha=32, dropout 0.05 on the query/value projections, AdamW 2e-4,
temperature 0.05, batch 32, 3 epochs (2 on Mind2Web and per-domain τ²).

- `--recipe paper` (default) is what the paper describes: in-batch negatives plus one mined hard negative
  per positive, drawn from the 10 non-relevant candidates of the same decision that the zero-shot encoder
  ranks highest (mined on the training folds only), and in-batch passages that are relevant to the query
  are masked instead of counted as negatives.
- `--recipe legacy` is what produced the published numbers: in-batch negatives only, no masking.

**Few-shot and FT-SLM.** Listwise prompts (`TASK: ...`, a numbered candidate list, `Relevant numbers:`)
from `run_e23_llama_family.py` (skill selection), `run_e36_tau_llm.py` (per-agent τ²) and
`run_e45b_tau_domain_fixed.py` (per-domain τ²); the element yes/no scorer and cross-encoder from
`run_e49_element_slm.py`; the operation classifier from `run_e48_operation_slm.py`; the gate from
`run_e51_gate_gpt.py`. LoRA r=16, alpha=32 on q_proj/v_proj, AdamW 2e-4, batch 1, 3 epochs (4 per-domain
τ², 2 operation); cross-encoders train 2 epochs at batch 16 with class-balanced BCE. Models load from the
ungated mirrors the pre-release runs used (`MODELS` in `llm.py`); any Hugging Face id works with `--model`.

## Hardware and dependencies

- DSR: `torch` and `transformers` (the package's `embed` extra). A CPU or Apple MPS laptop is enough for
  skill selection, τ² and Mind2Web element-150 (on an M-series laptop: DSR-ZS seconds; DSR-FT 0.5 min on
  skill selection, under 2 min on element-150 and τ², 12 min on operation). The full Mind2Web element
  table (368K rows) wants a CUDA GPU.
- LLM baselines: a CUDA GPU with bfloat16 (the pre-release runs used one 80 GB A100: 8B-model LoRA SFT on
  prompts of up to 8K tokens, batch 1, no gradient checkpointing), `torch`, `transformers`, `tokenizers`;
  `openai` (the `tabschema` extra) for the gate.
- LoRA is implemented in `lora.py` (peft's initialization and scaling), so `peft` is not required.

## Reproduced DSR numbers (Apple MPS, paper folds)

Seed 0 unless noted; P@R for set retrieval, per-step R@3 per web domain for elements, macro-F1 for the
operation. "Paper" is Table 1, "pre-release" the stored result of the script the preset ports.

| Decision | Method | This code | Paper | Pre-release |
|---|---|---|---|---|
| Skill selection | DSR-ZS | 0.171 | 17 | 0.171 |
| Skill selection | DSR-FT, `legacy` (3 paper seeds) | 0.388 (0.389) | 38 | 0.384 |
| Skill selection | DSR-FT, `paper` (3 paper seeds) | 0.391 (0.404) | 38 | |
| Skill selection | DSR-FT Real+synth k=2, `paper` | 0.408 | | 0.437 (synthetic rows leaked held-out tasks) |
| τ² airline / retail / telecom / all | DSR-ZS | .299 / .417 / .462 / .322 | 30 / 48 / 46 / 34 | |
| τ² airline / retail / telecom / all | DSR-FT, `legacy` | .456 / .406 / .291 / .392 | 45 / 42 / 27 / 42 | |
| τ² airline / retail / telecom / all | DSR-FT, `paper` | .637 / .673 / .642 / .623 | | |
| τ² per agent (gpt-4.1) | DSR-ZS | 0.420 | | 0.420 |
| Element-150, Ent. / Shop. / Travel | DSR-ZS | .158 / .136 / .134 | 15 / 16 / 15 | .147 / .157 / .148 |
| Element-150, Ent. / Shop. / Travel | DSR-FT, `legacy` | .131 / .167 / .215 | 15 / 18 / 23 | .147 / .178 / .226 |
| Element-150, Ent. / Shop. / Travel | DSR-FT, `paper` | .214 / .245 / .309 | | |
| Operation (reconstructed) | DSR-ZS / DSR-FT `paper` | 0.276 / 0.925 | 48 / 54 | 0.483 / 0.539 (script not found) |

Notes. On element-150, 47% of rows tie with another candidate of their step (identical attribute text), so
the tie rule alone moves R@3 by several points (candidate-id ties: 0.139 overall; pre-release sort order:
0.143; random: 0.120). The paper recipe is far stronger than the legacy one on the small τ² catalogs, where
in-batch "negatives" are mostly other queries' positives for the same few tools. Operation DSR depends
strongly on the unported design (macro-F1 from 0.08 to 0.80 zero-shot across reasonable query and passage
wordings); the preset uses the step context the LLM baselines see and one-line class descriptions.

## Differences from the pre-release scripts

- Fine-tuned baselines always use the paper's five task folds. The per-domain τ² SFT script used its own
  three task-grouped folds.
- DSR-FT Real+synth generates the paraphrases inside each training fold. The pre-release run appended a
  synthetic file built once from a fixed split, which contained paraphrases of held-out tasks.
- Skill gating: the gate sees the skill bundle stored on the `with` arm's row.
- Cross-encoders shuffle every epoch (`run_e23`'s cross-encoder drew a fresh permutation per batch in its
  first epoch).
- Seeds are explicit: the DSR-FT default seed reproduces each script's permutation stream; the adapters'
  initialization is seeded too (it was not before).
- Model routing (LLMRouterBench) and the gating FT-SLM have no pre-release script; their presets are
  marked `reconstructed`, as is DSR on Mind2Web operation (classes ranked as passages).
