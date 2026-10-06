"""Automatic configuration: build a feature card from any DataFrame and select the training recipe.

``infer_card`` turns a pointwise frame into a ``FeatureCard`` with declared types:

* numeric and boolean columns -> ``num``;
* string columns -> ``text`` when long (mean length > 40 characters) or multi-word (3+ words), else ``cat``;
* the candidate column -> ``cat`` (family ``schema``);
* text columns that are constant within each decision group are the decision's *context* (state); they are
  the columns TabSynth may paraphrase. Candidate-side text (descriptions) is never paraphrased.

``select`` chooses among a small grid of recipes (TabSynth off / k=2 / k=4, candidate prior on / off, head
preset) by nested, grouped inner cross-validation on the **training data only**, and returns the winner and
the full table of inner scores. This operationalizes the paper's rule of adopting +synth only where it beats
the real-only head (App. A, "Where synthesis backfires") without ever looking at held-out data. For a
TabSchema card, the winner is then tried once more without the generic context text and the context-candidate
similarities (``CONTEXT``), which rich LLM features can make redundant; they are dropped only when that scores
strictly better.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tabagent.agent import TabAgent
from tabagent.evaluation.cv import cross_validate
from tabagent.tabhead import TabHead, resolve_text_repr
from tabagent.table import DecisionSpec, DecisionTable
from tabagent.tabschema.card import (
    Feature,
    FeatureCard,
    has_relative,
    relative_bases,
    with_relative,
    without_features,
    without_relative,
)
from tabagent.tabsynth import TabSynth

PRIMARY_METRIC = {"ranking": "p@r", "multilabel": "ml_f1", "policy": "route@1", "single": "accuracy"}
PRIOR = "candidate_prior"
# The context text TabSchema appends to every extractor, and the context-candidate similarities (computed over
# the extractor's text columns).
CONTEXT = ("context_text", "context_candidate_lex", "context_candidate_sim")


def infer_type(s: pd.Series) -> str:
    """``num`` for numeric or boolean columns; for strings ``text`` when long or multi-word, else ``cat``."""
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return "num"
    text = s.dropna().astype(str)
    if text.empty:
        return "cat"
    long = text.str.len().mean() > 40
    wordy = text.str.split().str.len().mean() >= 3
    return "text" if long or wordy else "cat"


def embeddings_available() -> bool:
    """True when the embed extra (torch and transformers) is installed."""
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


def infer_card(
    df: pd.DataFrame,
    spec: DecisionSpec,
    columns: list[str],
    *,
    add_prior: bool = True,
    text_cols: list[str] | None = None,
    pair_embeddings: bool | None = None,
    name: str = "auto",
) -> FeatureCard:
    """A feature card for the pointwise frame ``df`` over ``columns`` (see the module docstring).

    ``add_prior`` adds the in-fold candidate prior; ``text_cols`` forces columns to ``text``;
    ``pair_embeddings`` adds the e5 context-candidate similarity (None: when the embed extra is installed).
    """
    features, synth_text = [], []
    for c in columns:
        if c in spec.structural_cols and c != spec.candidate:
            continue
        kind = (
            "cat" if c == spec.candidate else ("text" if text_cols and c in text_cols else infer_type(df[c]))
        )
        is_context = df.groupby(spec.group, sort=False)[c].nunique(dropna=False).max() <= 1
        family = "schema" if (c == spec.candidate or not is_context) else "state"
        features.append(Feature(c, family, kind))
        if kind == "text" and (is_context or spec.kind == "single"):
            synth_text.append(c)  # decision-context text only; candidate descriptions are never paraphrased
    base = FeatureCard(name, spec, features)
    context, candidate_text = base.context_text_columns(), base.candidate_text_columns()
    if spec.candidate and context and candidate_text:
        # Context-candidate overlap, the signal every paper card carries (Sec. 3.3; App. N.4, Table 21).
        pair = {"a": context, "b": candidate_text}
        features.append(Feature("context_candidate_lex", "schema", "num", "jaccard", pair))
        if pair_embeddings is None:
            pair_embeddings = embeddings_available()
        if pair_embeddings:
            args = {"query": context, "passage": candidate_text, "encoder": "e5-small-v2"}
            features.append(
                Feature("context_candidate_sim", "dependency", "num", "fitted:embedding_cosine", args)
            )
    if add_prior and spec.candidate:
        features.append(
            Feature(PRIOR, "dependency", "num", "fitted:candidate_prior", {"key": spec.candidate})
        )
    return FeatureCard(
        name,
        spec,
        features,
        synth_text=synth_text,
        synth_k=4,
        provenance={"kind": "auto-inferred", "columns": list(columns)},
    )


@dataclass
class Recipe:
    """One training recipe: TabSynth variants per row (0 = off), the candidate prior on or off, the CatBoost
    preset, the context text (``CONTEXT``) on or off, and the within-decision relative features (each
    feature's rank and gap within its decision, ``card.with_relative``) on (added to the card when it has
    none) or off (removed when it has them); None keeps the card as it is (relative features not searched).
    ``scores`` holds its inner-validation scores."""

    synth_k: int = 0
    prior: bool = True
    preset: str = "paper"
    scores: dict = field(default_factory=dict)
    context: bool = True
    text_repr: str = "auto"
    relative: bool | None = None

    def label(self) -> str:
        """A short description, for example ``synth=k2 prior=on paper`` (``relative=on`` or ``relative=off``
        when the relative features were searched, ``context=off`` when off, ``text=lsa`` or ``text=tfidf``
        when the text representation was chosen)."""
        synth = f"k{self.synth_k}" if self.synth_k else "off"
        relative = "" if self.relative is None else f" relative={'on' if self.relative else 'off'}"
        context = "" if self.context else " context=off"
        text = "" if self.text_repr == "auto" else f" text={self.text_repr}"
        return f"synth={synth} prior={'on' if self.prior else 'off'} {self.preset}{relative}{context}{text}"


def can_drop_context(card: FeatureCard) -> bool:
    """Whether ``card`` has context-text features (``CONTEXT``) and other state features to rank without
    them."""
    names = {f.name for f in card.features}
    state = [f for f in card.features if f.family == "state" and f.name not in CONTEXT]
    return bool(names & set(CONTEXT)) and bool(state)


def build_agent(card: FeatureCard, recipe: Recipe, seed: int = 0, **head_overrides) -> TabAgent:
    """The ``TabAgent`` of ``recipe`` on ``card`` (``head_overrides`` go to the CatBoost head)."""
    has_prior = any(f.name == PRIOR for f in card.features)
    c = card if (recipe.prior or not has_prior) else without_features(card, {PRIOR})
    if recipe.relative is True and not has_relative(c):
        c = with_relative(c)
    elif recipe.relative is False:
        c = without_relative(c)
    if not recipe.context and can_drop_context(c):
        c = without_features(c, set(CONTEXT))
    synth = TabSynth(k=recipe.synth_k) if recipe.synth_k and c.synth_text else None
    head_overrides = {"text_repr": recipe.text_repr, **head_overrides}
    head = TabHead(c.spec.kind, preset=recipe.preset, seed=seed, **head_overrides)
    return TabAgent(c, head=head, synth=synth, seed=seed)


def searches_relative(card: FeatureCard, relative: bool = False) -> bool:
    """Whether the recipe search tries the within-decision relative features on ``card``: it has them, or
    ``relative`` asks for them and the card is a ranking or multi-label card with a feature to rank
    (``card.relative_bases``)."""
    return has_relative(card) or (relative and bool(relative_bases(card)))


def recipes(card: FeatureCard, search: str = "fast", relative: bool = False) -> list[Recipe]:
    """The recipe grid: TabSynth off / k=2 / k=4 (when the card has text to paraphrase), prior on / off (when
    it has one), and the ``paper`` preset (``fast``) or both presets (``full``). The relative features are
    off in every recipe when they are searched (:func:`searches_relative`); :func:`select` tries them once on
    the winner."""
    if search not in ("fast", "full"):
        raise ValueError(f"search must be 'fast' or 'full', got {search!r}")
    synth = [0, 2, 4] if card.synth_text else [0]
    prior = [True, False] if any(f.name == PRIOR for f in card.features) else [True]
    presets = ["paper"] if search == "fast" else ["paper", "catboost-default"]
    rel = False if searches_relative(card, relative) else None  # off; "on" is tried once on the winner
    return [Recipe(k, p, s, relative=rel) for s, p, k in itertools.product(presets, prior, synth)]


def select(
    table: DecisionTable,
    card: FeatureCard,
    *,
    search: str = "fast",
    inner_folds: int = 3,
    seed: int = 0,
    time_budget: float | None = None,
    relative: bool = False,
    **head_overrides,
) -> tuple[Recipe, pd.DataFrame]:
    """Nested selection of the training recipe on ``table`` (training data only).

    Every recipe of :func:`recipes` is scored by ``inner_folds``-fold grouped cross-validation with one
    seed on the card's primary metric; the best wins, ties going to the simpler (earlier) recipe. When the
    card has context-text features and other state features (:func:`can_drop_context`), the winner is scored
    once more without the context text, which replaces it only when strictly better. With ``relative`` (and
    for a card that already has them), the within-decision relative features (``card.with_relative``) are a
    challenger: off in the grid, the winner is scored once with them, which they join only when strictly
    better (ties, and too few groups to search, keep them off). With ``time_budget``
    (seconds), recipes left when it runs out are skipped. Returns the winner and a table of every recipe's
    inner score.
    """
    metric = PRIMARY_METRIC[card.spec.kind]
    grid = recipes(card, search, relative)
    n_groups = len(table.groups)
    if len(grid) == 1 or n_groups < 2 * inner_folds:
        return grid[0], pd.DataFrame([{"recipe": grid[0].label(), metric: np.nan, "note": "no search"}])
    rows, start = [], time.monotonic()

    def score(r: Recipe) -> bool:
        if time_budget is not None and time.monotonic() - start > time_budget and rows:
            rows.append({"recipe": r.label(), metric: np.nan, "note": "skipped (time budget)"})
            return False
        res = cross_validate(
            build_agent(card, r, seed, **head_overrides),
            table,
            n_splits=inner_folds,
            fold_seed=seed,
            seeds=(seed,),
        )
        r.scores[metric] = float(res.per_decision(metric).mean())
        rows.append({"recipe": r.label(), metric: r.scores[metric], "note": ""})
        return True

    for r in grid:
        score(r)
    scored = [r for r in grid if metric in r.scores]
    best = max(scored, key=lambda r: r.scores[metric])  # ties keep the simpler (earlier) recipe
    if can_drop_context(card):
        lean = dataclasses.replace(best, scores={}, context=False)
        if score(lean) and lean.scores[metric] > best.scores[metric]:
            best = lean
    if best.relative is False:  # the relative features on, once: kept only when strictly better
        rich = dataclasses.replace(best, scores={}, relative=True)
        if score(rich) and rich.scores[metric] > best.scores[metric]:
            best = rich
    n_text = sum(f.type == "text" for f in build_agent(card, best, seed).card.features)
    if n_text and "text_repr" not in head_overrides:  # the other text representation, once
        default = resolve_text_repr("auto", n_text)
        other = dataclasses.replace(best, scores={}, text_repr="lsa" if default == "tfidf" else "tfidf")
        if score(other) and other.scores[metric] > best.scores[metric]:
            best = other
    return best, pd.DataFrame(rows)
