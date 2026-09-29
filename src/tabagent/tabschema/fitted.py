"""Fitted (dependency) features: statistics learned from training rows only (paper L443).

The paper requires every data-dependent statistic, such as candidate priors, co-usage counts and
embedding vocabularies, to be fitted inside the training portion of each fold. Each class here is fitted
on the *real* training rows of a fold (labels and outcome columns available) and then applied to any rows,
training or held-out, reading only non-outcome columns at transform time. ``seen_groups_`` records the
groups a feature was fitted on, so tests can prove that no held-out task was ever seen.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from tabagent.table import DecisionSpec


class FittedFeature:
    """Base class. Subclasses implement ``_fit`` and ``_transform``."""

    def __init__(self, **args: Any):
        self.args = args
        self.seen_groups_: set | None = None

    def fit(self, df: pd.DataFrame, spec: DecisionSpec) -> FittedFeature:
        self.seen_groups_ = set(pd.unique(df[spec.group]))
        self._fit(df, spec)
        return self

    def transform(self, df: pd.DataFrame, spec: DecisionSpec) -> pd.Series:
        if self.seen_groups_ is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        return self._transform(df, spec)

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
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            s = f"  {(t or '').lower()}  "
            for j in range(len(s) - 2):
                h = int.from_bytes(hashlib.md5(s[j : j + 3].encode()).digest()[:4], "little")
                out[i, h % self.dim] += 1.0
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norm, 1e-9, None)


class E5Encoder:
    """intfloat/e5-small-v2 with mean pooling and L2 normalization (the encoder shared with the DSR
    baselines, paper L389). Requires ``pip install "tabagent[embed]"``."""

    model_id = "intfloat/e5-small-v2"

    def __init__(self, batch_size: int = 32, max_length: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.batch_size, self.max_length = batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).eval()

    def encode(self, texts: list[str], prefix: str = "query: ") -> np.ndarray:
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
        todo = list(dict.fromkeys(t for t in texts if (prefix, t) not in self.memo))
        if todo:
            for t, v in zip(todo, self.encoder.encode(todo, prefix=prefix), strict=True):
                self.memo[(prefix, t)] = v
        return np.stack([self.memo[(prefix, t)] for t in texts]) if texts else np.zeros((0, 1), "float32")


_SHARED: dict[str, CachedEncoder] = {}


def encoder(name: str) -> CachedEncoder:
    if name not in _SHARED:
        _SHARED[name] = CachedEncoder(name)
    return _SHARED[name]


def _text(df: pd.DataFrame, cols) -> pd.Series:
    cols = [cols] if isinstance(cols, str) else list(cols)
    out = df[cols[0]].fillna("").astype(str)
    for c in cols[1:]:
        out = out + " " + df[c].fillna("").astype(str)
    return out


class EmbeddingCosine(FittedFeature):
    """Centered cosine between a context text (query) and a candidate text (passage).

    E5 vectors occupy a narrow cone, so raw cosines barely vary. As in the pre-release code, the common
    direction (the mean of the unique query and passage vectors) is removed and the vectors are
    renormalized. Here that mean is fitted on the **training** rows only, so held-out tasks never
    influence the representation (the pre-release code centered on all tasks).

    args: ``query`` and ``passage`` (a column or list of columns), ``encoder`` (``e5-small-v2``).
    """

    def _fit(self, df, spec):
        enc = encoder(self.args.get("encoder", "e5-small-v2"))
        q = list(dict.fromkeys(_text(df, self.args["query"])))
        p = list(dict.fromkeys(_text(df, self.args["passage"])))
        stacked = np.vstack([enc.encode(q, "query: "), enc.encode(p, "passage: ")])
        self.mu_ = stacked.mean(axis=0, keepdims=True)

    def _center(self, X: np.ndarray) -> np.ndarray:
        Y = X - self.mu_
        return Y / np.clip(np.linalg.norm(Y, axis=1, keepdims=True), 1e-9, None)

    def _transform(self, df, spec):
        enc = encoder(self.args.get("encoder", "e5-small-v2"))
        Q = self._center(enc.encode(_text(df, self.args["query"]).tolist(), "query: "))
        P = self._center(enc.encode(_text(df, self.args["passage"]).tolist(), "passage: "))
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

    def _transform(self, df, spec):
        enc = encoder(self.args.get("encoder", "e5-small-v2"))
        k = int(self.args.get("k", 15))
        per_task = df.groupby(spec.group, sort=False)[self.args["query"]].first()
        Eq = enc.encode(per_task.fillna("").astype(str).tolist(), "query: ")
        sim = Eq @ self.E_.T
        same = per_task.index.to_numpy()[:, None] == self.tasks_[None, :]
        sim[same] = -np.inf  # a training task never counts as its own neighbour
        kk = max(1, min(k, len(self.tasks_) - 1))
        idx = np.argpartition(sim, -kk, axis=1)[:, -kk:]
        vals = self.V_.to_numpy()
        means = pd.DataFrame(np.nanmean(vals[idx], axis=1), index=per_task.index, columns=self.V_.columns)
        means = means.fillna(self.colmean_)
        look = means.stack()
        keys = list(zip(df[spec.group], df[spec.candidate], strict=True))
        return pd.Series([look.get(key, np.nan) for key in keys], index=df.index, dtype=float)


FITTED: dict[str, type[FittedFeature]] = {
    "candidate_prior": CandidatePrior,
    "embedding_cosine": EmbeddingCosine,
    "knn_outcome": KnnOutcome,
}
