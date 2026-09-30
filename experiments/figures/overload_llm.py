"""The LLM curves of Fig. 2 (GPU): generative SFT, cross-encoder and few-shot, on the controlled K sweep.

The regimes run Llama-3.1-8B-Instruct by default (``--llm-model llama8b``, the model of the LLM curves in
``overload.PAPER_FIG2``). All three use the task-grouped pinned folds and the units of ``overload``.

sft_gen
    LoRA supervised fine-tuning (r 16, alpha 32, dropout 0.05 on q/v projections, AdamW lr 2e-4, 3 epochs,
    one example per training unit) to read the whole slate in the prompt and emit the relevant candidate
    numbers, most relevant first. Evaluation builds the controlled slates itself (all positives plus K sampled
    negatives, shuffled, ``reps`` draws, 6 by default because generation is costly), decodes greedily (80 new
    tokens) and ranks the parsed numbers first and the rest in a seeded random order. Mind2Web training slates
    are capped at 180 negatives to fit the context.
cross_enc
    LoRA sequence classifier (same adapter, 2 epochs, batch 16, positive-class weighting) over (task,
    candidate) pairs, trained per fold; its out-of-fold scores go through ``overload.simulate_k``.
fewshot
    Pointwise p(yes) of the instruction-tuned model for "is this candidate relevant to the task?". The prompt
    carries no demonstrations (zero-shot yes/no scoring), whatever the name.

Needs ``torch``, ``transformers`` and ``peft`` and, in practice, a CUDA GPU. ``plan`` (the ``--dry-run`` of
``overload``) needs none of them.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

from tabagent.table import DecisionTable

REGIMES = ("sft_gen", "fewshot", "cross_enc")
MODELS = {
    "llama1b": "unsloth/Llama-3.2-1B-Instruct",
    "llama3b": "unsloth/Llama-3.2-3B-Instruct",
    "llama8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}
SEED = 20260726
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"]}
LR = 2e-4
SFT = {"epochs": 3, "max_len": 8192, "new_tokens": 80, "reps": 6, "max_listed": 15}
CROSS = {"epochs": 2, "batch": 16, "max_len": 192, "eval_batch": 64}
FEWSHOT = {"batch": 16, "max_len": 1024}
SPEC = {
    "skillsbench/skill-select": {
        "cand": "skill_name",
        "text": "task_text",
        "noun": "skills",
        "cand_chars": 80,
        "train_cap": None,
    },
    "mind2web/element": {
        "cand": "elem_attr_text",
        "text": "instruction",
        "noun": "elements",
        "cand_chars": 50,
        "train_cap": 180,
    },
}


def spec_for(dataset: str) -> dict:
    if dataset not in SPEC:
        raise KeyError(f"no LLM prompt spec for {dataset!r}; known: {list(SPEC)}")
    return SPEC[dataset]


# --------------------------------------------------------------------------- prompts (pure)
def slate_prompt(task: str, candidates: Sequence[str], noun: str, cand_chars: int) -> str:
    lines = "\n".join(f"{i + 1}. {str(c)[:cand_chars]}" for i, c in enumerate(candidates))
    return (
        f"Select which {noun} are relevant to the task. "
        "Output ONLY the relevant candidate numbers, most relevant first, comma-separated.\n\n"
        f"TASK: {str(task)[:500]}\nCANDIDATES:\n{lines}\n\nRelevant numbers:"
    )


def fewshot_prompt(task: str, candidate: str) -> str:
    return (
        f"Task: {str(task)[:400]}\nCandidate: {str(candidate)[:120]}\n"
        "Is this candidate relevant to the task? Answer yes or no."
    )


def pair_text(task: str, candidate: str) -> str:
    return f"Task: {str(task)[:400]}\nCandidate: {str(candidate)[:100]}\nRelevant?"


def parse_rank(text: str, n: int, seed: int = 0) -> np.ndarray:
    """Slate positions in generated order (1-based, deduplicated, in range), then the rest shuffled."""
    order: list[int] = []
    for k in (int(x) for x in re.findall(r"\d+", text)):
        if 1 <= k <= n and (k - 1) not in order:
            order.append(k - 1)
    rest = [i for i in range(n) if i not in order]
    np.random.default_rng(seed).shuffle(rest)
    return np.asarray(order + rest, dtype=int)


def _units(table: DecisionTable, unit: str | None) -> dict:
    key = table.df[unit].astype(str) if unit else table.decision_ids
    return {
        u: np.asarray(ix) for u, ix in pd.Series(np.arange(len(table))).groupby(key.to_numpy()).groups.items()
    }


def _fold_of(table: DecisionTable, folds: dict) -> np.ndarray:
    g = table.df[table.spec.group]
    return g.map(lambda v: folds.get(v, folds.get(str(v)))).astype(int).to_numpy()


def plan(
    table: DecisionTable,
    dataset: str,
    regime: str,
    model: str,
    ks: Sequence[int],
    unit: str | None,
    reps: int | None = None,
) -> dict:
    """What a regime would run: units per K and the number of generations or scored pairs (no GPU)."""
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}")
    y = table.labels.astype(int)
    per_unit = [(int(y[ix].sum()), int((y[ix] == 0).sum())) for ix in _units(table, unit).values()]
    per_unit = [(p, n) for p, n in per_unit if p and n]
    eligible = {int(k): sum(n >= k for _, n in per_unit) for k in ks}
    out = {
        "dataset": dataset,
        "regime": regime,
        "model": MODELS.get(model, model),
        "rows": len(table),
        "units": len(per_unit),
        "units_per_k": eligible,
    }
    if regime == "sft_gen":
        r = reps or SFT["reps"]
        out.update(reps=r, train_examples_per_fold_x_epochs=f"~{len(per_unit) * 4 // 5} x {SFT['epochs']}")
        out["generations"] = r * sum(eligible.values())
    elif regime == "cross_enc":
        out.update(
            reps=reps or 30, train_pairs_per_fold_x_epochs=f"~{len(table) * 4 // 5} x {CROSS['epochs']}"
        )
        out["scored_pairs"] = len(table)
    else:
        out.update(reps=reps or 30, scored_pairs=len(table))
    return out


# --------------------------------------------------------------------------- GPU runners
def _torch_setup():
    import torch
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    return torch, AutoTokenizer, device, dtype


def _chat(tok, content: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
    )


def fewshot_scores(table: DecisionTable, sp: dict, model_id: str) -> np.ndarray:
    torch, AutoTokenizer, device, dtype = _torch_setup()
    from transformers import AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"  # last-token logits
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    yes = tok(" yes", add_special_tokens=False).input_ids[-1]
    no = tok(" no", add_special_tokens=False).input_ids[-1]
    df = table.df
    texts = [_chat(tok, fewshot_prompt(t, c)) for t, c in zip(df[sp["text"]], df[sp["cand"]], strict=True)]
    out = np.zeros(len(texts))
    for i in range(0, len(texts), FEWSHOT["batch"]):
        enc = tok(
            texts[i : i + FEWSHOT["batch"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=FEWSHOT["max_len"],
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        out[i : i + len(logits)] = torch.softmax(logits[:, [yes, no]].float(), -1)[:, 0].cpu().numpy()
    del model
    return out


def crossenc_scores(
    table: DecisionTable, sp: dict, folds: dict, model_id: str, seed: int = SEED
) -> np.ndarray:
    torch, AutoTokenizer, device, dtype = _torch_setup()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    df, y = table.df, table.labels.astype(np.float32)
    texts = [pair_text(t, c) for t, c in zip(df[sp["text"]], df[sp["cand"]], strict=True)]
    fold = _fold_of(table, folds)
    oof = np.full(len(df), np.nan)
    torch.manual_seed(seed)
    for f in sorted(set(fold)):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        base = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1, torch_dtype=dtype)
        base.config.pad_token_id = tok.pad_token_id
        model = get_peft_model(base, LoraConfig(task_type="SEQ_CLS", **LORA)).to(device)
        pos_weight = torch.tensor([(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())], device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        rng = np.random.default_rng(seed)
        model.train()
        for _ in range(CROSS["epochs"]):
            idx = tr[rng.permutation(len(tr))]
            for i in range(0, len(idx), CROSS["batch"]):
                b = idx[i : i + CROSS["batch"]]
                enc = tok(
                    [texts[j] for j in b],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=CROSS["max_len"],
                ).to(device)
                loss_fn(model(**enc).logits.squeeze(-1).float(), torch.tensor(y[b], device=device)).backward()
                opt.step()
                opt.zero_grad()
        model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(te), CROSS["eval_batch"]):
                b = te[i : i + CROSS["eval_batch"]]
                enc = tok(
                    [texts[j] for j in b],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=CROSS["max_len"],
                ).to(device)
                scores.append(torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy())
        oof[te] = np.concatenate(scores)
        del model, base
    return oof


def sft_gen_curve(
    table: DecisionTable,
    sp: dict,
    folds: dict,
    model_id: str,
    ks: Sequence[int],
    unit: str | None,
    reps: int = SFT["reps"],
    seed: int = SEED,
) -> dict:
    torch, AutoTokenizer, device, dtype = _torch_setup()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    df, y = table.df, table.labels.astype(int)
    cands, texts = df[sp["cand"]].astype(str).to_numpy(), df[sp["text"]].astype(str).to_numpy()
    fold = _fold_of(table, folds)
    units = {u: ix for u, ix in _units(table, unit).items() if y[ix].any()}
    par: dict[int, list] = {int(k): [] for k in ks}
    acc: dict[int, list] = {int(k): [] for k in ks}

    def ids_of(prompt: str):
        return tok(
            _chat(tok, prompt), return_tensors="pt", truncation=True, max_length=SFT["max_len"]
        ).input_ids

    for f in sorted(set(fold)):
        train_units = [ix for ix in units.values() if fold[ix[0]] != f]
        test_units = [ix for ix in units.values() if fold[ix[0]] == f]
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **LORA))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        rng = np.random.default_rng(seed)
        examples = []
        for ix in train_units:
            pos, neg = ix[y[ix] == 1], ix[y[ix] == 0]
            if sp["train_cap"] and len(neg) > sp["train_cap"]:
                neg = neg[rng.choice(len(neg), sp["train_cap"], replace=False)]
            slate = np.concatenate([pos, neg])[rng.permutation(len(pos) + len(neg))]
            rel = [str(i + 1) for i, r in enumerate(y[slate]) if r == 1][: SFT["max_listed"]]
            prompt = slate_prompt(texts[ix[0]], cands[slate].tolist(), sp["noun"], sp["cand_chars"])
            examples.append((prompt, (", ".join(rel) or "none") + tok.eos_token))
        model.train()
        for _ in range(SFT["epochs"]):
            for i in rng.permutation(len(examples)):
                p_ids = ids_of(examples[i][0])
                c_ids = tok(examples[i][1], return_tensors="pt", add_special_tokens=False).input_ids
                ids = torch.cat([p_ids, c_ids], 1).to(device)
                labels = ids.clone()
                labels[0, : p_ids.shape[1]] = -100
                model(ids, labels=labels).loss.backward()
                opt.step()
                opt.zero_grad()
        model.eval()
        for ix in test_units:
            pos, neg = ix[y[ix] == 1], ix[y[ix] == 0]
            for k in ks:
                if len(neg) < k:
                    continue
                srng = np.random.default_rng(seed + int(k))
                p_r, a_r = [], []
                for _ in range(reps):
                    slate = np.concatenate([pos, neg[srng.choice(len(neg), k, replace=False)]])
                    slate = slate[srng.permutation(len(slate))]
                    ids = ids_of(
                        slate_prompt(texts[ix[0]], cands[slate].tolist(), sp["noun"], sp["cand_chars"])
                    )
                    ids = ids.to(device)
                    with torch.no_grad():
                        gen = model.generate(
                            ids,
                            max_new_tokens=SFT["new_tokens"],
                            do_sample=False,
                            pad_token_id=tok.eos_token_id,
                        )
                    text = tok.decode(gen[0, ids.shape[1] :], skip_special_tokens=True)
                    ranked = y[slate][parse_rank(text, len(slate))]
                    p_r.append(ranked[: len(pos)].sum() / len(pos))
                    a_r.append(float(ranked[0]))
                par[int(k)].append(float(np.mean(p_r)))
                acc[int(k)].append(float(np.mean(a_r)))
        del model, base
    return {
        int(k): {
            "p@r": round(float(np.mean(par[k])), 4) if par[k] else None,
            "acc@1": round(float(np.mean(acc[k])), 4) if acc[k] else None,
            "chance_p@r": None,
            "n_units": len(par[k]),
        }
        for k in par
    }


def run(
    table: DecisionTable,
    dataset: str,
    regime: str,
    folds: dict,
    *,
    model: str = "llama8b",
    ks: Sequence[int] = (3, 5, 10, 20, 50, 100),
    unit: str | None = None,
    reps: int | None = None,
) -> dict:
    """One LLM curve (GPU), in the ``overload.json`` curve format."""
    from experiments.figures.overload import REPS, scored_frame, simulate_k

    sp, model_id = spec_for(dataset), MODELS.get(model, model)
    t0 = time.time()
    if regime == "sft_gen":
        reps = reps or SFT["reps"]
        curve = sft_gen_curve(table, sp, folds, model_id, ks, unit, reps)
    elif regime in ("fewshot", "cross_enc"):
        reps = reps or REPS
        scores = (
            fewshot_scores(table, sp, model_id)
            if regime == "fewshot"
            else crossenc_scores(table, sp, folds, model_id)
        )
        curve = simulate_k(scored_frame(table, scores, unit), ks, reps)
    else:
        raise ValueError(f"regime must be one of {REGIMES}")
    return {
        "curve": curve,
        "config": {"model": model_id, "reps": reps, "unit": unit or "decision", "ks": list(ks)},
        "secs": round(time.time() - t0, 1),
    }
