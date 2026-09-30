"""Helpers shared by the DSR and LLM baselines: text views, folds, devices, seeding."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from tabagent.table import DecisionTable

# A text view is a sequence of columns, each optionally capped to its first N characters, joined by a space:
# ("skill_name", ("skill_desc", 200)) renders "<skill_name> <first 200 chars of skill_desc>".
TextCols = Sequence["str | tuple[str, int]"]


def _cell(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def text_view(df: pd.DataFrame, cols: TextCols | str, max_chars: int | None = None) -> np.ndarray:
    """One string per row: the (capped) columns joined by a space, then capped to ``max_chars``."""
    cols = [cols] if isinstance(cols, str) else list(cols)
    parts = []
    for c in cols:
        name, cap = (c, None) if isinstance(c, str) else c
        s = df[name].map(_cell)
        parts.append(s if cap is None else s.str.slice(0, cap))
    out = parts[0]
    for p in parts[1:]:
        out = out + " " + p
    if max_chars is not None:
        out = out.str.slice(0, max_chars)
    return out.to_numpy(dtype=object)


def fold_array(table: DecisionTable, folds: Mapping) -> np.ndarray:
    """Fold id of every row (rows of a group share its fold). Every group must have a fold."""
    col = table.df[table.spec.group]
    fold = col.map(lambda g: folds.get(g, folds.get(str(g))))
    if fold.isna().any():
        raise ValueError(f"{int(fold.isna().sum())} rows belong to groups without a fold")
    return fold.astype(int).to_numpy()


def pick_device(device: str | None = None) -> str:
    """``device`` if given, else CUDA, then Apple MPS, then CPU."""
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def free_memory() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()
