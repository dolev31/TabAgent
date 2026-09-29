"""Training-free reference baselines used by the evaluation report (paper L385-389).

* chance: a uniformly random ranking (expected P@R = R/n) or policy (mean arm reward);
* BM25: Okapi BM25 between the decision context and each candidate's text (k1=1.5, b=0.75);
* e5 zero-shot: cosine between e5 embeddings of the context (query) and the candidate text (passage).

The fine-tuned baselines (DSR-FT, few-shot and fine-tuned small LLMs) are paper-reproduction code and live
in ``experiments/``.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from tabagent.table import DecisionTable


def _tok(text: str) -> list[str]:
    return [w for w in re.split(r"\W+", str(text).lower()) if len(w) > 1]


def _text(df: pd.DataFrame, cols) -> pd.Series:
    cols = [cols] if isinstance(cols, str) else list(cols)
    out = df[cols[0]].fillna("").astype(str)
    for c in cols[1:]:
        out = out + " " + df[c].fillna("").astype(str)
    return out


def bm25(table: DecisionTable, query, doc, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Okapi BM25 score of every row's candidate document for its decision's query text."""
    docs = [_tok(d) for d in _text(table.df, doc)]
    n = len(docs)
    lengths = np.array([len(d) for d in docs], dtype=float)
    avg = lengths.mean() if n and lengths.mean() > 0 else 1.0
    df_count = Counter()
    for d in docs:
        df_count.update(set(d))
    idf = {t: math.log((n - m + 0.5) / (m + 0.5) + 1.0) for t, m in df_count.items()}
    queries = _text(table.df, query).tolist()
    out = np.zeros(n)
    for i, (d, q) in enumerate(zip(docs, queries, strict=True)):
        tf = Counter(d)
        denom = k1 * (1 - b + b * lengths[i] / avg)
        out[i] = sum(
            idf[t] * tf[t] * (k1 + 1) / (tf[t] + denom) for t in set(_tok(q)) if t in tf and t in idf
        )
    return out


def embedding_zero_shot(table: DecisionTable, query, doc, encoder: str = "e5-small-v2") -> np.ndarray:
    from tabagent.tabschema.fitted import encoder as get_encoder

    enc = get_encoder(encoder)
    Q = enc.encode(_text(table.df, query).tolist(), "query: ")
    P = enc.encode(_text(table.df, doc).tolist(), "passage: ")
    return (Q * P).sum(axis=1)


def random_scores(table: DecisionTable, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(len(table))
