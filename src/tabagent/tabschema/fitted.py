"""Fitted (dependency) features: statistics learned from training rows only (paper Sec. 4.3).

The paper requires every data-dependent statistic, such as candidate priors, co-usage counts and
embedding vocabularies, to be fitted inside the training portion of each fold. Each class here is fitted
on the *real* training rows of a fold (labels and outcome columns available) and then applied to any rows,
training or held-out, reading only non-outcome columns at transform time. ``seen_groups_`` records the
groups a feature was fitted on, so tests can prove that no held-out task was ever seen.

A card references one as ``fitted:<name>`` (the keys of ``FITTED``): ``candidate_prior``, ``co_usage``,
``embedding_cosine`` and ``knn_outcome``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

from tabagent.table import DecisionSpec
from tabagent.tabschema.features import join_text

logger = logging.getLogger(__name__)


class FittedFeature:
    """Base class. Subclasses implement ``_fit`` and ``_transform``."""

    def __init__(self, **args: Any):
        self.args = args
        self.seen_groups_: set | None = None

    def fit(self, df: pd.DataFrame, spec: DecisionSpec) -> FittedFeature:
        """Fit on labeled training rows; records the groups seen in ``seen_groups_``."""
        self.seen_groups_ = set(pd.unique(df[spec.group]))
        self._fit(df, spec)
        return self

    def transform(self, df: pd.DataFrame, spec: DecisionSpec) -> pd.Series:
        """The feature for any rows (training or held-out), reading only non-outcome columns."""
        if self.seen_groups_ is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        return self._transform(df, spec)

    # Pickle-free persistence: JSON-able scalars/maps plus named numpy arrays (see tabagent.artifact).
    def state(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        return {"seen_groups": sorted(map(str, self.seen_groups_ or ()))}, {}

    def restore(self, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> FittedFeature:
        self.seen_groups_ = set(meta.get("seen_groups", []))
        return self

    def _fit(self, df: pd.DataFrame, spec: DecisionSpec) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _transform(self, df: pd.DataFrame, spec: DecisionSpec) -> pd.Series:  # pragma: no cover
        raise NotImplementedError


class CandidatePrior(FittedFeature):
    """Per-candidate statistic of a training column: the usage prior, a success rate, a mean cost.

    args: ``key`` (candidate column, default the spec's candidate), ``value`` (default the label),
    ``stat`` (``mean`` or ``rank``: rank 1 = highest mean), ``fill`` (``zero`` or ``global``: the value
    for candidates unseen in training).
    """

    def _fit(self, df, spec):
        key = self.args.get("key", spec.candidate)
        value = self.args.get("value", spec.label)
        self.key_ = key
        rate = df.groupby(key)[value].mean()
        self.global_ = float(df[value].mean())
        if self.args.get("stat", "mean") == "rank":
            self.map_ = rate.rank(ascending=False, method="min")
            self.fill_ = float(self.map_.max() + 1) if len(self.map_) else 1.0
        else:
            self.map_ = rate
            self.fill_ = self.global_ if self.args.get("fill", "zero") == "global" else 0.0

    def _transform(self, df, spec):
        return df[self.key_].map(self.map_).fillna(self.fill_).astype(float)

    def state(self):
        meta, arrays = super().state()
        meta.update(
            key=self.key_,
            fill=self.fill_,
            global_mean=self.global_,
            keys=[str(k) for k in self.map_.index],
            values=[float(v) for v in self.map_.to_numpy()],
        )
        return meta, arrays

    def restore(self, meta, arrays):
        super().restore(meta, arrays)
        self.key_, self.fill_, self.global_ = meta["key"], meta["fill"], meta["global_mean"]
        self.map_ = pd.Series(meta["values"], index=meta["keys"], dtype=float)
        return self


_ITEM_SEP = re.compile(r"[,;|\s]+")


def context_items(value: Any) -> list[str]:
    """The candidate ids in one cell of a co-usage context column, in order and without repeats.

    A cell is a list (or tuple, set, array) of ids, a JSON list, or a string of ids separated by commas,
    semicolons, pipes or whitespace; a missing value or an empty string or list is an empty context.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                return context_items(json.loads(text))
            except ValueError:
                pass
        items = [t.strip("[]'\"") for t in _ITEM_SEP.split(text)]
    elif isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        items = [str(v) for v in value if v is not None and not (isinstance(v, float) and np.isnan(v))]
    elif isinstance(value, float) and np.isnan(value):
        return []
    else:
        items = [str(value)]
    return list(dict.fromkeys(t for t in items if t))


class CoUsage(FittedFeature):
    """Co-usage: how often the candidate was chosen when the candidates of a context column had already been
    used (the co-usage counts of paper Sec. 4.3; "whether the candidate has co-occurred with other tools",
    Sec. 1).

    args: ``context`` (a column listing the candidates used before the decision, see :func:`context_items`;
    for example an extractor feature with the tools the agent called earlier in the run), ``key`` (the
    candidate column, default the spec's candidate), ``value`` (default the label), ``alpha`` (smoothing
    strength, default 1) and ``agg`` (``mean``, the default, or ``max`` over the context's candidates).

    Fitted on the training rows: for a context candidate ``u`` and a candidate ``c``, ``n(u, c)`` counts the
    rows of ``c`` whose context contains ``u`` and ``s(u, c)`` the chosen ones among them. The smoothed rate
    ``(s(u, c) + alpha * prior(c)) / (n(u, c) + alpha)`` shrinks toward ``prior(c)``, the candidate's training
    mean label (the global mean for a candidate unseen in training). A row's value aggregates the rates of its
    context's candidates; a pair never seen in training has rate ``prior(c)``, so an empty context, or one
    with no candidate seen in training, gives the prior: it carries no co-usage evidence.
    """

    def _fit(self, df, spec):
        self.key_ = self.args.get("key", spec.candidate)
        self.value_ = self.args.get("value", spec.label)
        self.context_ = self.args["context"]
        self.alpha_ = float(self.args.get("alpha", 1.0))
        self.agg_ = self.args.get("agg", "mean")
        if self.agg_ not in ("mean", "max"):
            raise ValueError(f"co_usage: agg must be 'mean' or 'max', not {self.agg_!r}")
        y = pd.to_numeric(df[self.value_], errors="coerce").astype(float)
        cand = df[self.key_].astype(str)
        self.global_ = float(y.mean()) if len(y) else 0.0
        self.prior_ = {str(k): float(v) for k, v in y.groupby(cand.to_numpy()).mean().items()}
        pairs = pd.DataFrame({"u": df[self.context_].map(context_items).to_numpy(), "c": cand, "y": y})
        pairs = pairs.explode("u").dropna(subset=["u"])
        stats = pairs.groupby(["u", "c"])["y"].agg(["sum", "count"])
        self.pairs_ = {
            (str(u), str(c)): (float(s), float(n))
            for (u, c), s, n in zip(stats.index, stats["sum"], stats["count"], strict=True)
        }

    def _rate(self, u: str, c: str) -> float:
        prior = self.prior_.get(c, self.global_)
        s, n = self.pairs_.get((u, c), (0.0, 0.0))
        return (s + self.alpha_ * prior) / (n + self.alpha_) if n + self.alpha_ > 0 else prior

    def _transform(self, df, spec):
        agg = max if self.agg_ == "max" else (lambda r: sum(r) / len(r))
        out = []
        for ctx, c in zip(df[self.context_].map(context_items), df[self.key_].astype(str), strict=True):
            rates = [self._rate(u, c) for u in ctx]
            out.append(agg(rates) if rates else self.prior_.get(c, self.global_))
        return pd.Series(out, index=df.index, dtype=float)

    def state(self):
        meta, arrays = super().state()
        pairs = sorted(self.pairs_.items())
        meta.update(
            key=self.key_,
            value=self.value_,
            context=self.context_,
            alpha=self.alpha_,
            agg=self.agg_,
            global_mean=self.global_,
            prior=self.prior_,
            pairs=[[u, c, s, n] for (u, c), (s, n) in pairs],
        )
        return meta, arrays

    def restore(self, meta, arrays):
        super().restore(meta, arrays)
        self.key_, self.value_, self.context_ = meta["key"], meta["value"], meta["context"]
        self.alpha_, self.agg_, self.global_ = meta["alpha"], meta["agg"], meta["global_mean"]
        self.prior_ = {str(k): float(v) for k, v in meta["prior"].items()}
        self.pairs_ = {(str(u), str(c)): (float(s), float(n)) for u, c, s, n in meta["pairs"]}
        return self


# --------------------------------------------------------------------------- text encoders
class HashingEncoder:
    """Deterministic, dependency-free text encoder (character trigram hashing, L2-normalized).

    Used in tests and as a fallback when the ``embed`` extra is not installed. Not a substitute for e5 when
    reproducing published numbers.
    """

    model_id = "hashing-trigram-256"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str], prefix: str = "") -> np.ndarray:
        """L2-normalized trigram-count vectors (``prefix`` is ignored)."""
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            s = f"  {(t or '').lower()}  "
            for j in range(len(s) - 2):
                h = int.from_bytes(hashlib.md5(s[j : j + 3].encode()).digest()[:4], "little")
                out[i, h % self.dim] += 1.0
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norm, 1e-9, None)


@contextmanager
def _quiet_hub() -> Iterator[None]:
    """Silence Hugging Face progress bars, info logs and advisory warnings while the encoder loads.

    Settings are restored afterwards; exceptions propagate unchanged.
    """
    from huggingface_hub import utils as hub_utils
    from transformers.utils import logging as hf_logging

    verbosity, bars = hf_logging.get_verbosity(), hf_logging.is_progress_bar_enabled()
    hub_bars_off = hub_utils.are_progress_bars_disabled()
    hub_logger = logging.getLogger("huggingface_hub")
    hub_level = hub_logger.level
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    hub_utils.disable_progress_bars()
    hub_logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module=r"(huggingface_hub|transformers)(\..*)?$")
            yield
    finally:
        hf_logging.set_verbosity(verbosity)
        if bars:
            hf_logging.enable_progress_bar()
        if not hub_bars_off:
            hub_utils.enable_progress_bars()
        hub_logger.setLevel(hub_level)


class E5Encoder:
    """intfloat/e5-small-v2 with mean pooling and L2 normalization (the encoder shared with the DSR
    baselines, paper Sec. 4.2). Requires ``pip install "tabagent[embed]"``; the weights (about 130 MB) are
    downloaded from the Hugging Face Hub on first use and cached there."""

    model_id = "intfloat/e5-small-v2"

    def __init__(self, batch_size: int = 32, max_length: int = 256):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                f"embedding features ({self.model_id}) need the embed extra: pip install 'tabagent[embed]'. "
                "Alternatively use a card without embedding features (pair_embeddings=False)."
            ) from e

        self.torch = torch
        self.batch_size, self.max_length = batch_size, max_length
        logger.info("loading the %s encoder", self.model_id)
        try:
            with _quiet_hub():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id).eval()
        except OSError as e:
            cause = str(e).splitlines()[0] if str(e) else type(e).__name__
            raise OSError(
                f"cannot load the {self.model_id} weights (about 130 MB) from the Hugging Face Hub or its "
                "local cache. Run once with network access to cache them, or use a card without embedding "
                f"features (pair_embeddings=False). Cause: {cause}"
            ) from e

    def encode(self, texts: list[str], prefix: str = "query: ") -> np.ndarray:
        """L2-normalized embeddings of ``prefix + text`` for each text (``query: `` or ``passage: ``)."""
        torch, out = self.torch, []
        for i in range(0, len(texts), self.batch_size):
            chunk = [prefix + (t or " ") for t in texts[i : i + self.batch_size]]
            enc = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(emb, dim=1).cpu().numpy())
        return np.vstack(out).astype("float32")


_ENCODERS: dict[str, Callable[[], Any]] = {"e5-small-v2": E5Encoder, "hashing": HashingEncoder}
_ENCODER_CACHE: dict[str, Any] = {}


class CachedEncoder:
    """Memoizes embeddings per (encoder, prefix, text); encoders are deterministic, so caching across
    folds changes nothing but speed."""

    def __init__(self, name: str):
        if name not in _ENCODER_CACHE:
            _ENCODER_CACHE[name] = _ENCODERS[name]()
        self.name, self.encoder = name, _ENCODER_CACHE[name]
        self.memo: dict[tuple[str, str], np.ndarray] = {}

    def encode(self, texts: list[str], prefix: str) -> np.ndarray:
        """Embeddings of ``texts``, encoding only those not seen before with this prefix."""
        todo = list(dict.fromkeys(t for t in texts if (prefix, t) not in self.memo))
        if todo:
            for t, v in zip(todo, self.encoder.encode(todo, prefix=prefix), strict=True):
                self.memo[(prefix, t)] = v
        return np.stack([self.memo[(prefix, t)] for t in texts]) if texts else np.zeros((0, 1), "float32")


_SHARED: dict[str, CachedEncoder] = {}


def encoder(name: str) -> CachedEncoder:
    """The process-wide cached encoder ``name`` (``e5-small-v2`` or ``hashing``)."""
    if name not in _SHARED:
        _SHARED[name] = CachedEncoder(name)
    return _SHARED[name]


class EmbeddingCosine(FittedFeature):
    """Centered cosine between a context text (query) and a candidate text (passage).

    E5 vectors occupy a narrow cone, so raw cosines barely vary. The common direction (the mean of the
    unique query and passage vectors) is therefore removed and the vectors are renormalized. That mean is
    fitted on the **training** rows only, so held-out tasks never influence the representation.

    args: ``query`` and ``passage`` (a column or list of columns), ``encoder`` (``e5-small-v2``).
    """

    def _fit(self, df, spec):
        enc = encoder(self.args.get("encoder", "e5-small-v2"))
        q = list(dict.fromkeys(join_text(df, self.args["query"])))
        p = list(dict.fromkeys(join_text(df, self.args["passage"])))
        stacked = np.vstack([enc.encode(q, "query: "), enc.encode(p, "passage: ")])
        self.mu_ = stacked.mean(axis=0, keepdims=True)

    def _center(self, X: np.ndarray) -> np.ndarray:
        Y = X - self.mu_
        return Y / np.clip(np.linalg.norm(Y, axis=1, keepdims=True), 1e-9, None)

    def state(self):
        meta, arrays = super().state()
        return meta, {**arrays, "mu": self.mu_}

    def restore(self, meta, arrays):
        super().restore(meta, arrays)
        self.mu_ = arrays["mu"]
        return self

    def _transform(self, df, spec):
        enc = encoder(self.args.get("encoder", "e5-small-v2"))
        Q = self._center(enc.encode(join_text(df, self.args["query"]).tolist(), "query: "))
        P = self._center(enc.encode(join_text(df, self.args["passage"]).tolist(), "passage: "))
        return pd.Series((Q * P).sum(axis=1).astype(float), index=df.index)


class KnnOutcome(FittedFeature):
    """Mean outcome of each candidate on the k most similar *training* tasks (e.g. a model's success on
    similar tasks, for routing). args: ``query`` (context text), ``value`` (default the label), ``k`` (15),
    ``encoder``."""

    def _fit(self, df, spec):
        enc = encoder(self.args.get("encoder", "e5-small-v2"))
        value = self.args.get("value", spec.label)
        per_task = df.groupby(spec.group).agg(text=(self.args["query"], "first"))
        self.tasks_ = per_task.index.to_numpy()
        self.E_ = enc.encode(per_task["text"].fillna("").astype(str).tolist(), "query: ")
        self.V_ = df.pivot_table(index=spec.group, columns=spec.candidate, values=value, aggfunc="mean")
        self.V_ = self.V_.reindex(self.tasks_)
        self.colmean_ = self.V_.mean(axis=0)

    def state(self):
        meta, arrays = super().state()
        meta.update(tasks=[str(t) for t in self.tasks_], candidates=[str(c) for c in self.V_.columns])
        arrays = {
            **arrays,
            "E": self.E_,
            "V": self.V_.to_numpy(dtype=float),
            "colmean": self.colmean_.to_numpy(),
        }
        return meta, arrays

    def restore(self, meta, arrays):
        super().restore(meta, arrays)
        self.tasks_ = np.asarray(meta["tasks"], dtype=object)
        self.E_ = arrays["E"]
        self.V_ = pd.DataFrame(arrays["V"], index=self.tasks_, columns=meta["candidates"])
        self.colmean_ = pd.Series(arrays["colmean"], index=meta["candidates"])
        return self

    def _transform(self, df, spec):
        enc = encoder(self.args.get("encoder", "e5-small-v2"))
        k = int(self.args.get("k", 15))
        per_task = df.groupby(spec.group, sort=False)[self.args["query"]].first()
        Eq = enc.encode(per_task.fillna("").astype(str).tolist(), "query: ")
        sim = Eq @ self.E_.T
        same = per_task.index.astype(str).to_numpy()[:, None] == np.asarray(self.tasks_, dtype=str)[None, :]
        sim[same] = -np.inf  # a training task never counts as its own neighbour
        kk = max(1, min(k, len(self.tasks_) - 1))
        # Deterministic neighbours: highest similarity first, ties broken by task id (duplicate texts occur).
        tie = np.argsort(np.argsort(np.asarray(self.tasks_, dtype=str)))
        idx = np.lexsort((np.broadcast_to(tie, sim.shape), -sim), axis=1)[:, :kk]
        vals = self.V_.to_numpy()
        means = pd.DataFrame(np.nanmean(vals[idx], axis=1), index=per_task.index, columns=self.V_.columns)
        means = means.fillna(self.colmean_)
        look = means.stack()
        keys = list(zip(df[spec.group], df[spec.candidate], strict=True))
        return pd.Series([look.get(key, np.nan) for key in keys], index=df.index, dtype=float)


FITTED: dict[str, type[FittedFeature]] = {
    "candidate_prior": CandidatePrior,
    "co_usage": CoUsage,
    "embedding_cosine": EmbeddingCosine,
    "knn_outcome": KnnOutcome,
}
