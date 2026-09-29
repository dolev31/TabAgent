"""TabSynth: schema-aligned, label-preserving paraphrase augmentation (paper Sec. 3.2; App. G, Algorithm 1).

For each real training row, TabSynth produces ``k`` variants that paraphrase only the row's free-text
state columns (the agent's thought, status and progress summaries, and the task text) and copies the
candidate, the label and every schema, state and dependency feature verbatim. A variant whose feature key
duplicates a real training row is dropped. The paraphraser is deterministic and rule-based, so the
augmentation is exactly reproducible. TabSynth never creates a new (context, candidate) relationship and
never changes a label: a paraphrase of a positive row stays positive, and no negatives are synthesized.

TabSynth runs only inside ``TabAgent.fit`` on the training rows of a fold, after the feature card has been
applied; scoring never synthesizes.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

GENERATOR_VERSION = "det-paraphrase-v1"


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _sentences(text: str) -> list[str]:
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]


def det_paraphrase_v1(text, j: int) -> str:
    """Meaning-preserving variant ``j`` of ``text``; a pure function, so results are reproducible and a
    decision's candidate rows stay coherent (every row of a decision gets the same rewrite)."""
    text = text if isinstance(text, str) else ("" if text is None or pd.isna(text) else str(text))
    t = _collapse_ws(text)
    if j % 4 == 0:
        return "Task: " + t
    if j % 4 == 1:
        s = _sentences(text)
        return _collapse_ws(" ".join(s[1:] + s[:1])) if len(s) >= 2 else "Please handle the following. " + t
    if j % 4 == 2:
        return "User request — " + t
    return t + "\n(Please proceed accordingly.)"


class TabSynth:
    """Algorithm 1 of the paper.

    Parameters
    ----------
    k: variants per real row (the paper uses k in {2, 4}).
    text_cols: free-text columns to paraphrase; defaults to the card's ``synth_text``.
    paraphrase: ``(text, j) -> text``; the default is the paper's deterministic rule-based paraphraser.
    max_source_rows: optionally paraphrase a seeded random subset of the real rows (for very large cards).
    seed: seeds only the ``max_source_rows`` subsample; the paraphrases themselves are deterministic.
    """

    def __init__(
        self,
        k: int = 4,
        text_cols: Sequence[str] | None = None,
        paraphrase: Callable[[str, int], str] = det_paraphrase_v1,
        max_source_rows: int | None = None,
        seed: int = 20260720,
    ):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.text_cols = list(text_cols) if text_cols is not None else None
        self.paraphrase = paraphrase
        self.max_source_rows = max_source_rows
        self.seed = seed
        self.stats_: dict = {}

    def augment(
        self, X: pd.DataFrame, y: np.ndarray, table=None, card=None
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Return the synthetic rows (feature frame, labels) for the real training rows ``X, y``.

        ``X`` is the card's feature frame of the real training rows; ``table`` and ``card`` supply the
        default text columns and the source group of each row (recorded in ``stats_`` for provenance).
        """
        text_cols = self.text_cols if self.text_cols is not None else list(getattr(card, "synth_text", []))
        missing = [c for c in text_cols if c not in X.columns]
        if missing:
            raise KeyError(f"TabSynth text columns {missing} are not card features")
        y = np.asarray(y)
        if not text_cols:
            self.stats_ = {"n_source": 0, "n_synthetic": 0, "dropped_duplicates": 0}
            return X.iloc[0:0].copy(), y[:0]

        src_idx = np.arange(len(X))
        if self.max_source_rows and len(X) > self.max_source_rows:
            src_idx = np.sort(
                np.random.default_rng(self.seed).choice(len(X), self.max_source_rows, replace=False)
            )
        src = X.iloc[src_idx]

        real_keys = set(map(tuple, X.astype(str).itertuples(index=False, name=None)))
        parts, labels, dropped = [], [], 0
        for j in range(self.k):
            var = src.copy()
            changed = np.zeros(len(var), dtype=bool)
            for c in text_cols:
                new = var[c].map(lambda s, j=j: self.paraphrase(s, j))
                changed |= (new.astype(str) != var[c].astype(str)).to_numpy()
                var[c] = new
            keys = var.astype(str).itertuples(index=False, name=None)
            keep = changed & np.array([key not in real_keys for key in keys], dtype=bool)
            dropped += int((changed & ~keep).sum())
            parts.append(var[keep].assign(_src=np.flatnonzero(keep), _j=j))
            labels.append(pd.Series(y[src_idx][keep], index=parts[-1].index))
        # Row-major order (each source row followed by its variants), as in the original generator.
        X_syn = pd.concat(parts)
        y_syn = pd.concat(labels)
        order = np.lexsort((X_syn["_j"].to_numpy(), X_syn["_src"].to_numpy()))
        X_syn = X_syn.iloc[order].drop(columns=["_src", "_j"]).reset_index(drop=True)
        y_syn = y_syn.iloc[order].to_numpy()
        groups = None
        if table is not None:
            groups = table.df[table.spec.group].to_numpy()[src_idx]
        self.stats_ = {
            "generator": GENERATOR_VERSION,
            "k": self.k,
            "text_cols": text_cols,
            "n_source": int(len(src)),
            "n_synthetic": int(len(X_syn)),
            "dropped_duplicates": dropped,
            "source_groups": None if groups is None else sorted(map(str, set(groups))),
        }
        return X_syn, y_syn
