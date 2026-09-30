"""TabSynth: schema-aligned, label-preserving paraphrase augmentation (paper Sec. 3.2; App. H, Algorithm 1).

For each real training row, TabSynth produces ``k`` variants that paraphrase only the row's free-text
state columns (the agent's thought, status and progress summaries, and the task text) and copies the
candidate, the label and every schema, state and dependency feature verbatim. A variant whose feature key
duplicates a real training row is dropped. The paraphraser is deterministic and rule-based, so the
augmentation is exactly reproducible. TabSynth never creates a new (context, candidate) relationship and
never changes a label: a paraphrase of a positive row stays positive, and no negatives are synthesized.

TabSynth runs only inside ``TabAgent.fit`` on the training rows of a fold, after the feature card has been
applied; scoring never synthesizes.

The extended operators of App. H ("Extended operators", used on the τ²-bench per-domain cards) are options:
``recompute`` recomputes the text-derived features of each paraphrase instead of copying them, ``entity_swap``
replaces names, identifiers, codes, contact details and amounts with other well-formed values before
paraphrasing, and ``allocation="inverse_sqrt_frequency"`` gives each decision variants in proportion to
``1/sqrt(f)``, where ``f`` is the training frequency of its rarest relevant candidate.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from tabagent.defaults import DEFAULT_SEED

GENERATOR_VERSION = "det-paraphrase-v1"
DEFAULT_K = 4


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


_FIRST = [
    "Olivia",
    "Liam",
    "Emma",
    "Noah",
    "Ava",
    "Ethan",
    "Mia",
    "Lucas",
    "Amelia",
    "Mason",
    "Harper",
    "Aiden",
    "Ella",
    "Omar",
    "Priya",
    "Yuki",
    "Mateo",
    "Chloe",
    "Daniel",
    "Sara",
]
_LAST = [
    "Garcia",
    "Nguyen",
    "Patel",
    "Kim",
    "Brown",
    "Rossi",
    "Cohen",
    "Silva",
    "Novak",
    "Lopez",
    "Muller",
    "Khan",
    "Ito",
    "Dubois",
    "Moreno",
    "Walker",
    "Lee",
    "Fischer",
    "Costa",
    "Reyes",
]


def swap_entities(text, seed: int) -> str:
    """Replace the entities of a customer request with other well-formed values (App. H, extended operators):
    e-mail addresses, user identifiers (``name_surname_1234``), order identifiers (``#W1234567``), phone
    numbers, six-character reservation codes, amounts, five-digit zip codes, and self-introduced names. A pure
    function of ``(text, seed)``; the rest of the text is unchanged."""
    import random

    rng = random.Random(seed)
    t = text if isinstance(text, str) else ("" if text is None or pd.isna(text) else str(text))
    first, last = rng.choice(_FIRST), rng.choice(_LAST)

    def digits(n: int) -> str:
        return "".join(rng.choice("0123456789") for _ in range(n))

    t = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", f"{first.lower()}.{last.lower()}{digits(3)}@example.com", t)
    t = re.sub(r"\b[a-z]+_[a-z]+_\d{3,5}\b", f"{first.lower()}_{last.lower()}_{digits(4)}", t)
    t = re.sub(r"#?W\d{7}\b", lambda m: ("#" if m.group(0).startswith("#") else "") + "W" + digits(7), t)
    t = re.sub(r"\b\d{3}-\d{3}-\d{4}\b", f"555-{digits(3)}-{digits(4)}", t)
    t = re.sub(
        r"\b(?=[A-Z0-9]{6}\b)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b",
        lambda m: "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6)),
        t,
    )
    t = re.sub(r"\$\s?\d+(?:\.\d+)?", lambda m: f"${rng.randint(20, 900)}", t)
    t = re.sub(r"\b\d{5}\b", lambda m: digits(5), t)
    t = re.sub(
        r"(?i)\b(my name is|i am|i'm|this is|you are)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?",
        lambda m: f"{m.group(1)} {first} {last}",
        t,
    )
    return t


def inverse_sqrt_frequency_allocation(table, k: int, max_variants: int = 12) -> dict:
    """Variants per decision in proportion to ``1/sqrt(f)``, where ``f`` is the share of the table's decisions
    whose relevant set contains the decision's rarest relevant candidate; the mean is ``k`` and every decision
    gets between 1 and ``max_variants`` (App. H, extended operators). Decisions without a relevant candidate
    get ``k``."""
    spec = table.spec
    ids = table.decision_ids.to_numpy()
    df = pd.DataFrame({"d": ids, "c": table.df[spec.candidate].to_numpy(), "y": table.labels})
    pos = df[df["y"] == 1]
    freq = pos.groupby("c")["d"].nunique() / df["d"].nunique()
    ratio = {d: max(1.0 / np.sqrt(freq[c]) for c in g["c"]) for d, g in pos.groupby("d")}
    if not ratio:
        return {d: k for d in pd.unique(ids)}
    mean = float(np.mean(list(ratio.values())))
    return {
        d: int(np.clip(round(k * ratio[d] / mean), 1, max_variants)) if d in ratio else k
        for d in pd.unique(ids)
    }


class TabSynth:
    """Algorithm 1 of the paper (App. H).

    Parameters
    ----------
    k: variants per real row (the paper uses two to four, Sec. 3.2). None uses the card's ``synth_k``
        (``DEFAULT_K`` when the card sets none).
    text_cols: free-text columns to paraphrase; defaults to the card's ``synth_text``.
    paraphrase: ``(text, j) -> text``; the default is the paper's deterministic rule-based paraphraser.
    max_source_rows: optionally paraphrase a seeded random subset of the real rows (for very large cards).
    seed: seeds only the ``max_source_rows`` subsample; the paraphrases themselves are deterministic.
    where: paraphrase only the source rows matching ``{column: value}`` or a ``frame -> bool mask`` callable.
        This is domain-matched synthesis (App. B.6): e.g. ``where={"domain": "Shopping"}`` adds
        paraphrases of Shopping rows only, for the model evaluated on Shopping.
    recompute: build each variant from the table's raw row with the paraphrased text, then apply the card,
        so text-derived features (length, lexical and embedding similarity) match the paraphrase instead of
        being copied. Needs ``table`` and ``card`` in ``augment`` and text columns that are raw table columns.
    entity_swap: swap entities (``swap_entities``) before paraphrasing variant ``j``, seeded by ``j``.
    allocation: ``"uniform"`` (``k`` variants per row) or ``"inverse_sqrt_frequency"``
        (``inverse_sqrt_frequency_allocation``: per-decision budgets averaging ``k``, between 1 and
        ``max_variants``). Needs ``table`` in ``augment``.

    After ``augment``: ``stats_`` (counts and the source groups), ``k_`` (the ``k`` used), ``source_rows_``
    and ``source_groups_`` (see ``augment``).
    """

    def __init__(
        self,
        k: int | None = None,
        text_cols: Sequence[str] | None = None,
        paraphrase: Callable[[str, int], str] = det_paraphrase_v1,
        max_source_rows: int | None = None,
        seed: int = DEFAULT_SEED,
        where=None,
        *,
        recompute: bool = False,
        entity_swap: bool = False,
        allocation: str = "uniform",
        max_variants: int = 12,
    ):
        if k is not None and k < 1:
            raise ValueError("k must be at least 1")
        if allocation not in ("uniform", "inverse_sqrt_frequency"):
            raise ValueError('allocation must be "uniform" or "inverse_sqrt_frequency"')
        self.k = k
        self.text_cols = list(text_cols) if text_cols is not None else None
        self.paraphrase = paraphrase
        self.max_source_rows = max_source_rows
        self.seed = seed
        self.where = where
        self.recompute = recompute
        self.entity_swap = entity_swap
        self.allocation = allocation
        self.max_variants = max_variants
        self.stats_: dict = {}
        self.k_: int | None = None
        self.source_rows_: np.ndarray | None = None
        self.source_groups_: np.ndarray | None = None

    def augment(
        self, X: pd.DataFrame, y: np.ndarray, table=None, card=None
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Return the synthetic rows (feature frame, labels) for the real training rows ``X, y``.

        ``X`` is the card's feature frame of the real training rows; ``table`` and ``card`` supply the
        default text columns and ``k`` and the source group of each row (recorded in ``stats_`` for
        provenance). After the call, ``source_rows_`` holds the row position in ``X`` that each synthetic
        row paraphrases, and ``source_groups_`` its group (when ``table`` is given).
        """
        if (self.recompute or self.allocation != "uniform") and (table is None or card is None):
            raise ValueError("recompute and allocation need the training table and the fitted card")
        text_cols = self.text_cols if self.text_cols is not None else list(getattr(card, "synth_text", []))
        k = self.k if self.k is not None else (getattr(card, "synth_k", 0) or DEFAULT_K)
        self.k_ = k
        missing = [c for c in text_cols if c not in X.columns]
        if missing:
            raise KeyError(f"TabSynth text columns {missing} are not card features")
        y = np.asarray(y)
        if not text_cols:
            self.stats_ = {"n_source": 0, "n_synthetic": 0, "dropped_duplicates": 0}
            self.source_rows_ = np.zeros(0, dtype=int)
            self.source_groups_ = None if table is None else np.zeros(0, dtype=object)
            return X.iloc[0:0].copy(), y[:0]

        src_idx = np.arange(len(X))
        if self.where is not None:
            if callable(self.where):
                mask = np.asarray(self.where(X), dtype=bool)
            else:
                mask = np.ones(len(X), dtype=bool)
                for col, value in self.where.items():
                    mask &= (X[col] == value).to_numpy()
            src_idx = src_idx[mask]
        if self.max_source_rows and len(src_idx) > self.max_source_rows:
            pick = np.random.default_rng(self.seed).choice(len(src_idx), self.max_source_rows, replace=False)
            src_idx = np.sort(src_idx[pick])
        src = X.iloc[src_idx]

        if self.recompute:
            raw_missing = [c for c in text_cols if c not in table.df.columns]
            if raw_missing:
                raise KeyError(f"recompute needs raw table columns for {raw_missing}")
        n_variants = np.full(len(src_idx), k)
        if self.allocation == "inverse_sqrt_frequency":
            budget = inverse_sqrt_frequency_allocation(table, k, self.max_variants)
            n_variants = np.array([budget[d] for d in table.decision_ids.to_numpy()[src_idx]])

        real_keys = set(map(tuple, X.astype(str).itertuples(index=False, name=None)))
        raw_src = table.df.iloc[src_idx] if self.recompute else None
        parts, labels, dropped = [], [], 0
        for j in range(int(n_variants.max(initial=0))):
            active = n_variants > j

            def rewrite(s, j=j):
                return self.paraphrase(swap_entities(s, 7919 * j + 17) if self.entity_swap else s, j)

            base = raw_src if self.recompute else src
            new_text = {c: base[c].map(rewrite) for c in text_cols}
            changed = np.zeros(len(src), dtype=bool)
            for c in text_cols:
                changed |= (new_text[c].astype(str) != base[c].astype(str)).to_numpy()
            if self.recompute:
                raw = raw_src.assign(**new_text)
                var = card.transform(table.with_df(raw.reset_index(drop=True)))
                var.index = src.index
            else:
                var = src.assign(**new_text)
            keys = var.astype(str).itertuples(index=False, name=None)
            keep = active & changed & np.array([key not in real_keys for key in keys], dtype=bool)
            dropped += int((active & changed & ~keep).sum())
            parts.append(var[keep].assign(_src=np.flatnonzero(keep), _j=j))
            labels.append(pd.Series(y[src_idx][keep], index=parts[-1].index))
        if not parts:
            parts, labels = [src.iloc[0:0].assign(_src=0, _j=0)], [pd.Series([], dtype=y.dtype)]
        # Row-major order: the variants of each source row are consecutive, in source-row order.
        X_syn = pd.concat(parts)
        y_syn = pd.concat(labels)
        order = np.lexsort((X_syn["_j"].to_numpy(), X_syn["_src"].to_numpy()))
        self.source_rows_ = src_idx[X_syn["_src"].to_numpy()[order]]
        X_syn = X_syn.iloc[order].drop(columns=["_src", "_j"]).reset_index(drop=True)
        y_syn = y_syn.iloc[order].to_numpy()
        groups = None
        self.source_groups_ = None
        if table is not None:
            groups = table.df[table.spec.group].to_numpy()[src_idx]
            self.source_groups_ = table.df[table.spec.group].to_numpy()[self.source_rows_]
        self.stats_ = {
            "generator": GENERATOR_VERSION,
            "k": k,
            "operators": {
                "recompute": self.recompute,
                "entity_swap": self.entity_swap,
                "allocation": self.allocation,
            },
            "text_cols": text_cols,
            "n_source": int(len(src)),
            "n_synthetic": int(len(X_syn)),
            "dropped_duplicates": dropped,
            "source_groups": None if groups is None else sorted(map(str, set(groups))),
        }
        return X_syn, y_syn
