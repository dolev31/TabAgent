"""Small-LLM baselines (paper Sec. 4.2; Table 1 columns Few-shot and FT-SLM; the SLM family, App. F, Fig. 8).

Each task's ``source`` names the paper result its prompt reproduces. Four task shapes cover the paper's rows:

``ListwiseTask`` (set retrieval: skill selection, tau2 tool shortlisting)
    The model reads the task and the numbered candidate catalog and generates the relevant candidate
    numbers, most relevant first. Few-shot: two demonstrations, chosen once, whose groups are then excluded
    from evaluation. FT-SLM: per fold, LoRA supervised fine-tuning on the training folds to generate the
    relevant numbers. The generated order ranks the candidates; unlisted candidates follow in a seeded
    random order, so P@R and Recall@k are computed exactly as for TabAgent.
``PointwiseTask`` (Mind2Web element selection, about 50 candidates per step)
    Generative ranking is impractical, so every (instruction, element) pair is scored. Few-shot: the
    next-token probability of "yes" against "no" (zero-shot: the prompt carries no demonstrations).
    FT-SLM: a LoRA cross-encoder (sequence classification, class-balanced BCE) fitted per fold.
``ClassifyTask`` (Mind2Web operation, CLICK / TYPE / SELECT)
    Few-shot: two demonstrations per class drawn from the training folds of each test fold. FT-SLM: per
    fold, LoRA fine-tuning to generate the class word.
``GateTask`` (SkillsBench skill gating, policy)
    A frontier model (GPT-5.4 in the paper) decides per task whether to activate the skill bundle, through
    any OpenAI-compatible endpoint. Undecidable replies are excluded, not counted as "no".

Models run with Hugging Face transformers (bfloat16 on CUDA). LoRA is ``lora.apply_lora`` (r=16, alpha=32,
dropout 0.05 on q_proj/v_proj, AdamW 2e-4, batch 1 for generation), so ``peft`` is not needed.
``DryRunLM`` builds and records every prompt without loading weights.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from tabagent.table import DecisionTable

from .common import TextCols, fold_array, free_memory, pick_device, seed_everything, text_view
from .lora import apply_lora

# Ungated mirrors of the paper's checkpoints; any Hugging Face id works too.
MODELS = {
    "llama-3.2-1b": "unsloth/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "unsloth/Llama-3.2-3B-Instruct",
    "llama-3.1-8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}
DEFAULT_MODEL = "llama-3.1-8b"
SLM_FAMILY = tuple(MODELS)


def resolve_model(name: str) -> str:
    return MODELS.get(name, name)


@dataclass(frozen=True)
class LoraSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    targets: tuple[str, ...] = ("q_proj", "v_proj")
    lr: float = 2e-4


# --------------------------------------------------------------------------- task shapes
@dataclass(frozen=True)
class ListwiseTask:
    task_cols: TextCols
    cand_cols: TextCols
    instruction: str  # prepended to the first user turn (no system role: some chat templates reject it)
    header: str = "CANDIDATES"
    task_chars: int = 500
    cand_chars: int | None = 80
    max_relevant: int = 15  # relevant numbers written in a demonstration or training target
    shots: str = "first"  # "first": the first two decisions by id; "random": two seeded distinct groups
    shot_seed: int = 20260727
    max_input: int = 8192
    max_new_tokens: int = 60
    sft_epochs: int = 3
    source: str = ""


@dataclass(frozen=True)
class CrossEncoderTask:
    task_cols: TextCols
    cand_cols: TextCols
    template: str  # with {task} and {cand}
    task_chars: int = 400
    cand_chars: int = 100
    max_length: int = 224
    epochs: int = 2
    batch_size: int = 16
    source: str = ""


@dataclass(frozen=True)
class PointwiseTask:
    task_cols: TextCols
    cand_cols: TextCols
    yes_template: str  # user turn for the next-token yes/no score, with {task} and {cand}
    task_chars: int = 200
    cand_chars: int = 100
    max_length: int = 512
    batch_size: int = 32
    cross_encoder: CrossEncoderTask | None = None  # the FT-SLM of this row
    source: str = ""


@dataclass(frozen=True)
class ClassifyTask:
    context: Callable[[Mapping], str]
    classes: tuple[str, ...]
    instruction: str
    parse_order: tuple[str, ...]  # substrings checked in this order; the last class is the fallback
    shots_per_class: int = 2
    max_input: int = 1024
    max_new_tokens: int = 4
    batch_size: int = 16
    sft_epochs: int = 2
    source: str = ""


@dataclass(frozen=True)
class GateTask:
    task_col: str
    bundle_col: str
    arm_col: str
    on_arm: str
    template: str  # with {task} and {bundle}
    task_chars: int = 900
    bundle_chars: int = 900
    max_tokens: int = 512
    retries: int = 4
    source: str = ""


LLMTask = ListwiseTask | PointwiseTask | ClassifyTask | GateTask


# --------------------------------------------------------------------------- prompts (pure functions)
def listwise_target(labels: Sequence[int], max_relevant: int = 15) -> str:
    rel = [i + 1 for i, y in enumerate(labels) if y == 1][:max_relevant]
    return ", ".join(map(str, rel)) or "none"


def listwise_block(task: ListwiseTask, text: str, cands: Sequence[str]) -> str:
    cap = task.cand_chars
    lines = "\n".join(f"{i + 1}. {c if cap is None else c[:cap]}" for i, c in enumerate(cands))
    return f"TASK: {str(text)[: task.task_chars]}\n{task.header}:\n{lines}\n\nRelevant numbers:"


def listwise_messages(task: ListwiseTask, text: str, cands: Sequence[str], shots=()) -> list[dict]:
    """Chat turns: each demonstration as a user/assistant pair, then the query; the instruction is
    prepended to the first user turn."""
    msgs: list[dict] = []
    for s_text, s_cands, s_labels in shots:
        msgs.append({"role": "user", "content": listwise_block(task, s_text, s_cands)})
        msgs.append({"role": "assistant", "content": listwise_target(s_labels, task.max_relevant)})
    msgs.append({"role": "user", "content": listwise_block(task, text, cands)})
    msgs[0] = {**msgs[0], "content": task.instruction + msgs[0]["content"]}
    return msgs


def parse_ranking(text: str, n: int, seed: int = 0) -> np.ndarray:
    """Candidate order from a generated list of 1-based numbers: listed candidates first (duplicates and
    out-of-range numbers ignored), then the rest in a seeded random order."""
    seen: set[int] = set()
    order: list[int] = []
    for k in (int(x) for x in re.findall(r"\d+", text or "")):
        if 1 <= k <= n and k - 1 not in seen:
            seen.add(k - 1)
            order.append(k - 1)
    rest = [i for i in range(n) if i not in seen]
    np.random.default_rng(seed).shuffle(rest)
    return np.array(order + rest, dtype=int)


def order_scores(order: np.ndarray) -> np.ndarray:
    """Scores that sort candidates in ``order`` (first = highest)."""
    s = np.empty(len(order))
    s[order] = np.arange(len(order), 0, -1, dtype=float)
    return s


def classify_messages(task: ClassifyTask, context: str, shots=()) -> list[dict]:
    msgs: list[dict] = []
    for s_ctx, s_label in shots:
        msgs.append({"role": "user", "content": s_ctx})
        msgs.append({"role": "assistant", "content": str(s_label).lower()})
    msgs.append({"role": "user", "content": context})
    msgs[0] = {**msgs[0], "content": task.instruction + msgs[0]["content"]}
    return msgs


def parse_class(task: ClassifyTask, text: str) -> str:
    t = (text or "").lower()
    for c in task.parse_order:
        if c.lower() in t:
            return c
    return task.parse_order[-1]


def parse_yes_no(text: str) -> bool | None:
    """The last explicit yes/no word of a reply (after any reasoning); None if there is none."""
    t = (text or "").lower()
    ys = [m.start() for m in re.finditer(r"\byes\b", t)]
    ns = [m.start() for m in re.finditer(r"\bno\b", t)]
    if ys and (not ns or ys[-1] > ns[-1]):
        return True
    if ns and (not ys or ns[-1] > ys[-1]):
        return False
    return None


# --------------------------------------------------------------------------- decisions
@dataclass
class Decision:
    id: str
    group: object
    rows: np.ndarray
    text: str
    cands: list[str]
    labels: np.ndarray


def decisions(table: DecisionTable, task_cols: TextCols, cand_cols: TextCols) -> list[Decision]:
    """Decisions in id order; candidates in table row order (the catalog order the prompt lists)."""
    ids = table.decision_ids.to_numpy()
    ctx, cand = text_view(table.df, task_cols), text_view(table.df, cand_cols)
    groups = table.df[table.spec.group].to_numpy()
    labels = table.labels if table.has_labels else np.zeros(len(table), dtype=int)
    out = []
    order = np.argsort(ids, kind="stable")
    bounds = np.flatnonzero(np.r_[True, ids[order][1:] != ids[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        rows = np.sort(order[a:b])
        out.append(
            Decision(ids[rows[0]], groups[rows[0]], rows, ctx[rows[0]], list(cand[rows]), labels[rows])
        )
    return out


def select_shots(items: list[Decision], how: str, n: int = 2, seed: int = 20260727) -> list[Decision]:
    """``n`` demonstrations from distinct groups: the first by decision id, or a seeded random draw."""
    order = range(len(items)) if how == "first" else np.random.default_rng(seed).permutation(len(items))
    picked, groups = [], set()
    for i in order:
        if items[i].group not in groups:
            picked.append(items[i])
            groups.add(items[i].group)
        if len(picked) == n:
            break
    return picked


# --------------------------------------------------------------------------- language-model backends
def dtype_kwarg(dtype: str) -> dict:
    """``from_pretrained`` takes ``dtype`` from transformers 4.56 on and ``torch_dtype`` before."""
    import torch
    import transformers
    from packaging.version import Version

    key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    return {key: getattr(torch, dtype)}


class HFCausalLM:
    """A Hugging Face causal LM: greedy generation, next-token yes/no scores and LoRA SFT."""

    def __init__(
        self, model_id: str, device: str | None = None, dtype: str | None = None, lora: LoraSpec | None = None
    ):
        from transformers import AutoTokenizer

        self.model_id, self.device, self.lora = model_id, pick_device(device), lora or LoraSpec()
        self.dtype = dtype or ("bfloat16" if self.device == "cuda" else "float32")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = None

    @property
    def eos(self) -> str:
        return self.tok.eos_token

    def _load(self):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(self.model_id, **dtype_kwarg(self.dtype))
        return model.to(self.device).eval()

    def release(self) -> None:
        self.model = None
        free_memory()

    def render(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _ids(self, prompt: str, max_input: int):
        # The rendered template already carries the BOS token and the tokenizer adds another; the paper's
        # runs tokenized this way, so it is kept.
        return self.tok(prompt, return_tensors="pt", truncation=True, max_length=max_input).input_ids

    def generate(
        self, prompts: list[str], max_new_tokens: int, max_input: int, batch_size: int = 1, fold=None
    ) -> list[str]:
        import torch

        if self.model is None:
            self.model = self._load()
        out = []
        if batch_size == 1:
            for p in prompts:
                ids = self._ids(p, max_input).to(self.device)
                with torch.no_grad():
                    gen = self.model.generate(
                        ids,
                        attention_mask=torch.ones_like(ids),
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tok.eos_token_id,
                    )
                out.append(self.tok.decode(gen[0, ids.shape[1] :], skip_special_tokens=True))
            return out
        self.tok.padding_side = "left"
        for i in range(0, len(prompts), batch_size):
            enc = self.tok(
                prompts[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input,
            ).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id
                )
            n = enc.input_ids.shape[1]
            out += [self.tok.decode(g[n:], skip_special_tokens=True) for g in gen]
        return out

    def yes_prob(self, prompts: list[str], max_input: int, batch_size: int = 32) -> np.ndarray:
        """P(" yes") / (P(" yes") + P(" no")) for the next token of each prompt (distinct prompts once)."""
        import torch

        if self.model is None:
            self.model = self._load()
        self.tok.padding_side = "left"
        yes = self.tok(" yes", add_special_tokens=False).input_ids[-1]
        no = self.tok(" no", add_special_tokens=False).input_ids[-1]
        uniq = list(dict.fromkeys(prompts))
        val: dict[str, float] = {}
        for i in range(0, len(uniq), batch_size):
            chunk = uniq[i : i + batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_input)
            with torch.no_grad():
                logits = self.model(**enc.to(self.device)).logits[:, -1, :]
            p = torch.softmax(logits[:, [yes, no]].float(), -1)[:, 0].cpu().numpy()
            val.update(zip(chunk, p.tolist(), strict=True))
        return np.array([val[p] for p in prompts])

    def sft(
        self, examples: list[tuple[str, str]], epochs: int, seed: int, max_input: int, fold=None
    ) -> list[float]:
        """LoRA SFT from fresh base weights: loss on the completion tokens only, batch 1."""
        import torch

        seed_everything(seed)
        self.model = self._load()
        params = apply_lora(self.model, self.lora.targets, self.lora.r, self.lora.alpha, self.lora.dropout)
        self.model.to(self.device).train()
        opt = torch.optim.AdamW(params, lr=self.lora.lr)
        rng, losses = np.random.default_rng(seed), []
        for _ in range(epochs):
            for i in rng.permutation(len(examples)):
                prompt, completion = examples[i]
                pid = self._ids(prompt, max_input)
                cid = self.tok(completion, return_tensors="pt", add_special_tokens=False).input_ids
                ids = torch.cat([pid, cid], 1).to(self.device)
                labels = ids.clone()
                labels[0, : pid.shape[1]] = -100
                loss = self.model(ids, labels=labels).loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                losses.append(loss.item())
        self.model.eval()
        return losses


class HFCrossEncoder:
    """A LoRA sequence-classification head on a decoder LM (one logit), trained with class-balanced BCE."""

    def __init__(
        self, model_id: str, device: str | None = None, dtype: str | None = None, lora: LoraSpec | None = None
    ):
        from transformers import AutoTokenizer

        self.model_id, self.device, self.lora = model_id, pick_device(device), lora or LoraSpec()
        self.dtype = dtype or ("bfloat16" if self.device == "cuda" else "float32")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = None

    def release(self) -> None:
        self.model = None
        free_memory()

    def _logits(self, enc):
        return self.model(**enc).logits.squeeze(-1).float()

    def fit(
        self, texts: list[str], labels: np.ndarray, task: CrossEncoderTask, seed: int, fold=None
    ) -> list[float]:
        import torch
        from transformers import AutoModelForSequenceClassification

        seed_everything(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, num_labels=1, **dtype_kwarg(self.dtype)
        )
        model.config.pad_token_id = self.tok.pad_token_id
        heads = ("score", "classifier")
        params = apply_lora(model, self.lora.targets, self.lora.r, self.lora.alpha, self.lora.dropout, heads)
        for name, module in model.named_modules():  # train the new head in float32
            if name.rsplit(".", 1)[-1] in heads:
                module.float()
                module.register_forward_pre_hook(lambda m, args: tuple(a.float() for a in args))
        self.model = model.to(self.device).train()
        y = np.asarray(labels, dtype=np.float32)
        pos_weight = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())], device=self.device)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(params, lr=self.lora.lr)
        rng, losses = np.random.default_rng(seed), []
        for _ in range(task.epochs):
            idx = rng.permutation(len(texts))
            for i in range(0, len(idx), task.batch_size):
                b = idx[i : i + task.batch_size]
                enc = self.tok(
                    [texts[j] for j in b],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=task.max_length,
                ).to(self.device)
                loss = lossf(self._logits(enc), torch.tensor(y[b], device=self.device))
                loss.backward()
                opt.step()
                opt.zero_grad()
                losses.append(loss.item())
        self.model.eval()
        return losses

    def score(self, texts: list[str], task: CrossEncoderTask, batch_size: int = 64, fold=None) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(
                    texts[i : i + batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=task.max_length,
                ).to(self.device)
                out.append(torch.sigmoid(self._logits(enc)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


@dataclass
class DryRunLM:
    """Builds and records prompts without loading weights. Renders with the model's chat template when its
    tokenizer is available locally, else with a plain role-tagged layout; token counts are then estimated
    as characters / 4."""

    model_id: str
    tok: object | None = None
    records: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.tok is None:
            try:
                from transformers import AutoTokenizer

                self.tok = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
            except Exception:  # not cached (or transformers missing): fall back to the plain layout
                self.tok = None

    @property
    def templated(self) -> bool:
        return self.tok is not None and getattr(self.tok, "chat_template", None) is not None

    @property
    def eos(self) -> str:
        return getattr(self.tok, "eos_token", None) or "</s>"

    def render(self, messages: list[dict]) -> str:
        if self.templated:
            return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages) + "<|assistant|>\n"

    def n_tokens(self, text: str) -> int:
        if self.tok is not None:
            return len(self.tok(text).input_ids)
        return max(1, len(text) // 4)

    def _record(self, kind: str, prompt: str, target: str | None = None, fold: int | None = None) -> None:
        self.records.append(
            {
                "kind": kind,
                "fold": fold,
                "prompt": prompt,
                "target": target,
                "n_tokens": self.n_tokens(prompt),
            }
        )

    def release(self) -> None:
        pass

    # causal LM interface
    def generate(self, prompts, max_new_tokens, max_input, batch_size=1, fold=None) -> list[str]:
        for p in prompts:
            self._record("generate", p, fold=fold)
        return [""] * len(prompts)

    def yes_prob(self, prompts, max_input, batch_size=32) -> np.ndarray:
        for p in dict.fromkeys(prompts):
            self._record("score", p)
        return np.zeros(len(prompts))

    def sft(self, examples, epochs, seed, max_input, fold=None) -> list[float]:
        for p, c in examples:
            self._record("train", p, c, fold)
        return []

    # cross-encoder interface
    def fit(self, texts, labels, task, seed, fold=None) -> list[float]:
        for t, y in zip(texts, labels, strict=True):
            self._record("train", t, str(int(y)), fold)
        return []

    def score(self, texts, task, batch_size=64, fold=None) -> np.ndarray:
        for t in texts:
            self._record("score", t, fold=fold)
        return np.zeros(len(texts))

    # API client interface (tabagent.tabschema.llm.LLM)
    def complete(self, system: str, *, role: str = ""):
        from tabagent.tabschema.llm import LLMResponse

        self._record("complete", system)
        return LLMResponse("", self.model_id)


def _dry(lm) -> bool:
    return isinstance(lm, DryRunLM)


# --------------------------------------------------------------------------- listwise (set retrieval)
def fewshot_listwise(table: DecisionTable, task: ListwiseTask, lm, n_shots: int = 2, log=print) -> np.ndarray:
    """Scores for the decisions outside the demonstrations' groups (NaN elsewhere)."""
    items = [d for d in decisions(table, task.task_cols, task.cand_cols) if d.labels.sum() > 0]
    shots = select_shots(items, task.shots, n_shots, task.shot_seed)
    demo = [(s.text, s.cands, s.labels) for s in shots]
    held = {s.group for s in shots}
    ev = [d for d in items if d.group not in held]
    prompts = [lm.render(listwise_messages(task, d.text, d.cands, demo)) for d in ev]
    t0 = time.time()
    outs = lm.generate(prompts, task.max_new_tokens, task.max_input)
    scores = np.full(len(table), np.nan)
    for d, text in zip(ev, outs, strict=True):
        scores[d.rows] = order_scores(parse_ranking(text, len(d.cands)))
    if log:
        log(
            f"  few-shot: {len(ev)} decisions ({len(shots)} demonstrations held out), {time.time() - t0:.0f}s"
        )
    return scores


def sft_listwise(table: DecisionTable, task: ListwiseTask, lm, folds: Mapping, seed: int = 0, log=print):
    """Out-of-fold scores of a LoRA-SFT generator trained per fold on the training folds."""
    items = [d for d in decisions(table, task.task_cols, task.cand_cols) if d.labels.sum() > 0]
    fold = fold_array(table, folds)
    item_fold = np.array([fold[d.rows[0]] for d in items])
    scores = np.full(len(table), np.nan)
    for f in sorted(set(item_fold)):
        train = [d for d, g in zip(items, item_fold, strict=True) if g != f]
        test = [d for d, g in zip(items, item_fold, strict=True) if g == f]
        examples = [
            (
                lm.render(listwise_messages(task, d.text, d.cands)),
                listwise_target(d.labels, task.max_relevant) + lm.eos,
            )
            for d in train
        ]
        kw = {"fold": int(f)}
        losses = lm.sft(examples, task.sft_epochs, seed, task.max_input, **kw)
        prompts = [lm.render(listwise_messages(task, d.text, d.cands)) for d in test]
        for d, text in zip(
            test, lm.generate(prompts, task.max_new_tokens, task.max_input, **kw), strict=True
        ):
            scores[d.rows] = order_scores(parse_ranking(text, len(d.cands)))
        lm.release()
        if log:
            log(f"  ft-slm fold {f}: {len(train)} train / {len(test)} test decisions, {_loss(losses)}")
    return scores


def _loss(losses) -> str:
    return f"loss {np.mean(losses[:20]):.3f} -> {np.mean(losses[-20:]):.3f}" if losses else "no loss"


# --------------------------------------------------------------------------- pointwise (element selection)
def pair_texts(table: DecisionTable, task, template: str) -> list[str]:
    t = text_view(table.df, task.task_cols, task.task_chars)
    c = text_view(table.df, task.cand_cols, task.cand_chars)
    return [template.format(task=a, cand=b) for a, b in zip(t, c, strict=True)]


def fewshot_pointwise(table: DecisionTable, task: PointwiseTask, lm, log=print) -> np.ndarray:
    prompts = [
        lm.render([{"role": "user", "content": u}]) for u in pair_texts(table, task, task.yes_template)
    ]
    t0 = time.time()
    scores = lm.yes_prob(prompts, task.max_length, task.batch_size)
    if log:
        log(f"  few-shot pointwise: {len(prompts)} pairs, {time.time() - t0:.0f}s")
    return scores


def cross_encoder_scores(
    table: DecisionTable, task: CrossEncoderTask, ce, folds: Mapping, seed: int = 0, log=print
) -> np.ndarray:
    """Out-of-fold probabilities of a LoRA cross-encoder fitted per fold on the training folds."""
    texts = np.array(pair_texts(table, task, task.template), dtype=object)
    y = table.labels.astype(float)
    fold = fold_array(table, folds)
    out = np.full(len(table), np.nan)
    for f in sorted(set(fold)):
        test = fold == f
        kw = {"fold": int(f)}
        losses = ce.fit(list(texts[~test]), y[~test], task, seed, **kw)
        out[test] = ce.score(list(texts[test]), task, **kw)
        ce.release()
        if log:
            log(f"  cross-encoder fold {f}: {int((~test).sum())} train pairs, {_loss(losses)}")
    return out


# --------------------------------------------------------------------------- single-class (operation)
def _rows(table: DecisionTable) -> list[dict]:
    return table.df.to_dict("records")


def fewshot_classify(table: DecisionTable, task: ClassifyTask, lm, folds: Mapping, log=print) -> np.ndarray:
    """Predicted class per row; the demonstrations of each test fold come from its training folds."""
    rows, fold = _rows(table), fold_array(table, folds)
    y = table.df[table.spec.label].astype(str).to_numpy()
    preds = np.empty(len(table), dtype=object)
    for f in sorted(set(fold)):
        train = np.flatnonzero(fold != f)
        shots = [
            (task.context(rows[i]), c)
            for c in task.classes
            for i in train[y[train] == c][: task.shots_per_class]
        ]
        test = np.flatnonzero(fold == f)
        prompts = [lm.render(classify_messages(task, task.context(rows[i]), shots)) for i in test]
        kw = {"fold": int(f)}
        outs = lm.generate(prompts, task.max_new_tokens, task.max_input, task.batch_size, **kw)
        preds[test] = [parse_class(task, o) for o in outs]
        if log:
            log(f"  few-shot fold {f}: {len(test)} decisions")
    return preds


def sft_classify(table: DecisionTable, task: ClassifyTask, lm, folds: Mapping, seed: int = 0, log=print):
    rows, fold = _rows(table), fold_array(table, folds)
    y = table.df[table.spec.label].astype(str).to_numpy()
    preds = np.empty(len(table), dtype=object)
    for f in sorted(set(fold)):
        train, test = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        examples = [
            (lm.render(classify_messages(task, task.context(rows[i]))), y[i].lower() + lm.eos) for i in train
        ]
        kw = {"fold": int(f)}
        losses = lm.sft(examples, task.sft_epochs, seed, task.max_input, **kw)
        prompts = [lm.render(classify_messages(task, task.context(rows[i]))) for i in test]
        outs = lm.generate(prompts, task.max_new_tokens, task.max_input, task.batch_size, **kw)
        preds[test] = [parse_class(task, o) for o in outs]
        lm.release()
        if log:
            log(f"  ft-slm fold {f}: {len(train)} train / {len(test)} test, {_loss(losses)}")
    return preds


# --------------------------------------------------------------------------- policy (gating)
def gate_prompt(task: GateTask, text: str, bundle: str) -> str:
    return task.template.format(task=str(text)[: task.task_chars], bundle=str(bundle)[: task.bundle_chars])


def fewshot_gate(table: DecisionTable, task: GateTask, client, log=print) -> np.ndarray:
    """Score 1 for the arm the model picks and 0 for the other; NaN when the reply stays undecidable."""
    df = table.df
    ids = table.decision_ids.to_numpy()
    on = (df[task.arm_col].astype(str) == task.on_arm).to_numpy()
    scores = np.full(len(table), np.nan)
    n_undecided = 0
    for d in dict.fromkeys(ids):
        rows = np.flatnonzero(ids == d)
        on_row = rows[on[rows]][0] if on[rows].any() else rows[0]  # the bundle is stored on the "with" row
        prompt = gate_prompt(task, df[task.task_col].iloc[on_row], df[task.bundle_col].iloc[on_row])
        choice = None
        for attempt in range(1 if _dry(client) else task.retries):
            try:
                choice = parse_yes_no(client.complete(prompt).text)
            except Exception as e:  # transient endpoint errors: retry
                if log:
                    log(f"  retry {attempt + 1}: {str(e)[:120]}")
                time.sleep(4)
            if choice is not None:
                break
        if choice is None:
            n_undecided += 1
            continue
        scores[rows] = (on[rows] == choice).astype(float)
    if log:
        log(f"  gate: {len(set(ids)) - n_undecided} decided, {n_undecided} undecidable (excluded)")
    return scores
