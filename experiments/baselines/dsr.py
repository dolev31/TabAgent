"""Dense semantic retrieval (DSR) baselines (paper Sec. 4.2; Table 1 columns DSR-ZS and DSR-FT).

A bi-encoder scores every (decision context, candidate) row by the inner product of L2-normalized e5
embeddings of the context ("query: " prefix) and the candidate text ("passage: " prefix), mean-pooled over
tokens. The encoder is intfloat/e5-small-v2, the one TabHead's similarity features use.

* DSR-ZS: the encoder as released. No training, no folds.
* DSR-FT (Real): per fold, contrastive fine-tuning with LoRA on the training folds only, then the held-out
  fold is scored by the adapted encoder. Each positive (context, relevant candidate) pair is contrasted
  against the other passages of its batch (in-batch negatives, as the paper describes).
* DSR-FT (Real+synth): the same, with TabSynth paraphrases of the training contexts added as extra positive
  pairs. The paraphrases are generated inside each training fold (paper App. H, Algorithm 1).

``RECIPES`` names the negatives: ``in-batch`` (the default, the paper's recipe) and ``hard-negatives``, which
adds one mined hard negative per positive (a non-relevant candidate of the same decision, drawn from the
``hard_pool`` candidates the zero-shot encoder ranks highest) and masks in-batch passages that are relevant
to the query (another relevant candidate of the same decision, or the same candidate text) instead of
counting them as negatives.

Single-class decisions (Mind2Web operation) are scored as a ranking over the class names; see
:func:`expand_classes`.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from tabagent.table import DecisionSpec, DecisionTable

from .common import TextCols, fold_array, free_memory, pick_device, seed_everything, text_view
from .lora import apply_lora

E5 = "intfloat/e5-small-v2"
QUERY, PASSAGE = "query: ", "passage: "


@dataclass(frozen=True)
class DSRConfig:
    model_id: str = E5
    max_chars: int | None = 256  # each text is cut to this many characters before tokenization
    max_length: int = 512  # tokens
    encode_batch_size: int = 128
    # fine-tuning: LoRA r=16, alpha=32 on query/value, AdamW 2e-4, 3 epochs, batch 32, temperature 0.05
    epochs: int = 3
    batch_size: int = 32
    lr: float = 2e-4
    temperature: float = 0.05
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ("query", "value")
    hard_negatives: int = 0  # mined hard negatives per positive pair (0: in-batch negatives only)
    hard_pool: int = 10  # mine among the top-k non-relevant candidates under the zero-shot encoder
    mask_false_negatives: bool = False  # mask in-batch passages that are relevant to the query


RECIPES: dict[str, DSRConfig] = {
    "in-batch": DSRConfig(),
    "hard-negatives": DSRConfig(hard_negatives=1, mask_false_negatives=True),
}

Loader = Callable[[], tuple]  # () -> (tokenizer, model)


def e5_loader(model_id: str = E5) -> Loader:
    def load():
        from transformers import AutoModel, AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id), AutoModel.from_pretrained(model_id)

    return load


# --------------------------------------------------------------------------- encoding
def _embed(model, tok, texts: list[str], prefix: str, cfg: DSRConfig, device: str):
    import torch

    enc = tok(
        [prefix + t for t in texts],
        padding=True,
        truncation=True,
        max_length=cfg.max_length,
        return_tensors="pt",
    ).to(device)
    hidden = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(emb, p=2, dim=1)


def encode(model, tok, texts, prefix: str, cfg: DSRConfig, device: str) -> dict[str, np.ndarray]:
    """Embeddings of the distinct ``texts`` (sorted by length so batches pad little)."""
    import torch

    uniq = sorted(set(texts), key=len)
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(uniq), cfg.encode_batch_size):
            chunk = uniq[i : i + cfg.encode_batch_size]
            vecs = _embed(model, tok, chunk, prefix, cfg, device).float().cpu().numpy()
            out.update(zip(chunk, vecs, strict=True))
    return out


def pair_scores(model, tok, queries, docs, cfg: DSRConfig, device: str) -> np.ndarray:
    """Inner product of every row's query and passage embeddings."""
    if len(queries) == 0:
        return np.zeros(0)
    q = encode(model, tok, queries, QUERY, cfg, device)
    d = encode(model, tok, docs, PASSAGE, cfg, device)
    return np.array([float(q[a] @ d[b]) for a, b in zip(queries, docs, strict=True)])


# --------------------------------------------------------------------------- training data
@dataclass(frozen=True)
class Pair:
    query: str
    positive: str
    decision: str
    hard: tuple[str, ...] = ()


def training_pairs(
    queries: np.ndarray,
    docs: np.ndarray,
    labels: np.ndarray,
    decisions: np.ndarray,
    train: np.ndarray,
    zs: np.ndarray | None,
    cfg: DSRConfig,
) -> tuple[list[Pair], dict[str, set[str]]]:
    """Positive pairs of the training rows, each with the hard-negative pool of its decision, and the set
    of relevant passage texts per decision (for false-negative masking)."""
    f = pd.DataFrame(
        {
            "q": queries,
            "d": docs,
            "y": np.asarray(labels).astype(int),
            "dec": decisions,
            "zs": np.zeros(len(queries)) if zs is None else zs,
        }
    )[train]
    positives = f[f["y"] == 1]
    rel: dict[str, set[str]] = defaultdict(set)
    for dec, d in zip(positives["dec"], positives["d"], strict=True):
        rel[dec].add(d)
    pools: dict[str, tuple[str, ...]] = {}
    if cfg.hard_negatives and zs is not None:
        neg = f[f["y"] == 0]
        neg = neg[[d not in rel.get(dec, ()) for dec, d in zip(neg["dec"], neg["d"], strict=True)]]
        neg = neg.drop_duplicates(["dec", "d"]).sort_values(
            ["dec", "zs"], ascending=[True, False], kind="stable"
        )
        for dec, g in neg.groupby("dec", sort=False):
            pools[dec] = tuple(g["d"].head(cfg.hard_pool))
    pairs = [
        Pair(q, d, dec, pools.get(dec, ()))
        for q, d, dec in zip(positives["q"], positives["d"], positives["dec"], strict=True)
    ]
    return pairs, dict(rel)


def synth_pairs(
    table: DecisionTable,
    query_cols: TextCols | str,
    pairs: list[Pair],
    train: np.ndarray,
    synth,
    cfg: DSRConfig,
) -> list[Pair]:
    """TabSynth paraphrases of the training contexts as extra positive pairs (training rows only).

    ``synth`` is the number of variants k, or a :class:`tabagent.tabsynth.TabSynth`; by default the
    paraphrased columns are the query columns.
    """
    from tabagent.tabsynth import TabSynth

    cols = list(dict.fromkeys(c if isinstance(c, str) else c[0] for c in _as_list(query_cols)))
    gen = synth if isinstance(synth, TabSynth) else TabSynth(k=int(synth), text_cols=cols)
    if gen.text_cols is None:
        gen = copy.copy(gen)
        gen.text_cols = cols
    df = table.df[train]
    X = df[cols].assign(_decision=table.decision_ids[train].to_numpy())
    X_syn, _ = gen.augment(X, df[table.spec.label].to_numpy())
    syn = pd.DataFrame({"dec": X_syn["_decision"], "q": text_view(X_syn, query_cols, cfg.max_chars)})
    by_dec: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        by_dec[p.decision].append(p)
    return [
        replace(p, query=q)
        for dec, q in syn.drop_duplicates().itertuples(index=False)
        for p in by_dec.get(dec, ())
        if q != p.query
    ]


def _as_list(cols) -> list:
    return [cols] if isinstance(cols, str) else list(cols)


# --------------------------------------------------------------------------- fine-tuning
def false_negatives(
    batch: list[Pair], passages: list[str], relevant: dict[str, set[str]]
) -> list[list[bool]]:
    """``mask[i][j]``: passage j is relevant to query i but is not its own positive, so it must not count
    as a negative (another relevant candidate of the decision, or the same candidate text)."""
    return [
        [j != i and passages[j] in relevant.get(p.decision, ()) for j in range(len(passages))]
        for i, p in enumerate(batch)
    ]


def contrastive_finetune(
    model, tok, pairs: list[Pair], relevant: dict[str, set[str]], cfg: DSRConfig, seed: int, device: str
) -> list[float]:
    """Contrastive LoRA fine-tuning (InfoNCE over the in-batch passages, plus mined hard negatives when the
    config asks for them)."""
    import torch

    params = apply_lora(model, cfg.lora_targets, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)
    model.to(device).train()
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(cfg.epochs):
        order = rng.permutation(len(pairs))
        for i in range(0, len(order), cfg.batch_size):
            batch = [pairs[j] for j in order[i : i + cfg.batch_size]]
            if len(batch) < 2:  # a single pair has no in-batch negative
                continue
            passages = [p.positive for p in batch]
            if cfg.hard_negatives:
                for p in batch:
                    if p.hard:
                        k = min(cfg.hard_negatives, len(p.hard))
                        passages += [p.hard[j] for j in rng.choice(len(p.hard), size=k, replace=False)]
            q = _embed(model, tok, [p.query for p in batch], QUERY, cfg, device)
            d = _embed(model, tok, passages, PASSAGE, cfg, device)
            logits = (q @ d.T).float() / cfg.temperature
            if cfg.mask_false_negatives:
                mask = torch.tensor(false_negatives(batch, passages, relevant), device=logits.device)
                logits = logits.masked_fill(mask, float("-inf"))
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(batch), device=logits.device))
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
    model.eval()
    return losses


# --------------------------------------------------------------------------- entry point
def dsr_scores(
    table: DecisionTable,
    query_cols: TextCols | str,
    doc_cols: TextCols | str,
    folds: Mapping | None = None,
    finetune: bool = False,
    synth=None,
    *,
    config: DSRConfig | None = None,
    seed: int = 0,
    device: str | None = None,
    loader: Loader | None = None,
    log: Callable[[str], None] | None = print,
) -> np.ndarray:
    """One DSR score per row of ``table``.

    Zero-shot (``finetune=False``) needs no folds. Fine-tuned scores are out-of-fold: each row is scored by
    an encoder adapted on the other folds only. ``synth`` (an int k or a ``TabSynth``) adds in-fold TabSynth
    paraphrases of the training contexts (DSR-FT Real+synth).
    """
    cfg = config or DSRConfig()
    loader = loader or e5_loader(cfg.model_id)
    device = pick_device(device)
    queries = text_view(table.df, query_cols, cfg.max_chars)
    docs = text_view(table.df, doc_cols, cfg.max_chars)

    tok, model = loader()
    model.to(device).eval()
    if not finetune:
        return pair_scores(model, tok, queries, docs, cfg, device)
    if folds is None:
        raise ValueError("fine-tuned DSR needs folds (it is trained on the training folds only)")
    if not table.has_labels:
        raise ValueError("fine-tuning needs a labeled table")

    fold = fold_array(table, folds)
    labels, decisions = table.labels, table.decision_ids.to_numpy()
    # The zero-shot encoder is label-free, so mining with it on every row leaks nothing: labels are read
    # only from the training folds when the pools are built.
    zs = pair_scores(model, tok, queries, docs, cfg, device) if cfg.hard_negatives else None
    del model
    out = np.full(len(table), np.nan)
    for f in sorted(set(fold)):
        test = fold == f
        pairs, relevant = training_pairs(queries, docs, labels, decisions, ~test, zs, cfg)
        n_real = len(pairs)
        if synth is not None:
            pairs = pairs + synth_pairs(table, query_cols, pairs, ~test, synth, cfg)
        seed_everything(seed)
        tok, model = loader()
        losses = contrastive_finetune(model, tok, pairs, relevant, cfg, seed, device)
        out[test] = pair_scores(model, tok, queries[test], docs[test], cfg, device)
        if log:
            first, last = (f"{losses[0]:.3f}", f"{losses[-1]:.3f}") if losses else ("-", "-")
            n_syn = len(pairs) - n_real
            log(f"  dsr-ft fold {f}: {n_real} real + {n_syn} synth pairs, loss {first} -> {last}")
        del model
        free_memory()
    return out


# --------------------------------------------------------------------------- single-class decisions
CLASS_COL, CLASS_TEXT = "_class", "_class_text"


def expand_classes(table: DecisionTable, classes=None, class_text: Mapping | None = None) -> DecisionTable:
    """A single-class table as a ranking over its classes: one row per (decision, class), labeled 1 for
    the decision's class. The class passage is ``class_text[c]`` (default: the lower-cased class name)."""
    s = table.spec
    if s.kind != "single":
        raise ValueError("expand_classes is for single-class decisions")
    classes = sorted(pd.unique(table.df[s.label].astype(str))) if classes is None else list(classes)
    text = {c: (class_text or {}).get(c, str(c).lower()) for c in classes}
    parts = []
    for c in classes:
        part = table.df.assign(**{CLASS_COL: c, CLASS_TEXT: text[c]})
        part[s.label] = (table.df[s.label].astype(str) == str(c)).astype(int)
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    spec = DecisionSpec(
        kind="ranking",
        group=s.group,
        label=s.label,
        candidate=CLASS_COL,
        decision=s.decision_cols,
        eval=s.eval,
    )
    return DecisionTable(df, spec)


def argmax_class(expanded: DecisionTable, scores: np.ndarray, original: DecisionTable) -> np.ndarray:
    """Predicted class per row of ``original`` from scores over its expanded table (ties: class order)."""
    f = pd.DataFrame(
        {"dec": expanded.decision_ids.to_numpy(), "cls": expanded.df[CLASS_COL].to_numpy(), "s": scores}
    )
    best = f.loc[f.groupby("dec", sort=False)["s"].idxmax()].set_index("dec")["cls"]
    return original.decision_ids.map(best).to_numpy(dtype=object)
