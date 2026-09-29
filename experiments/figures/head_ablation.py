"""Fig. 3 (``fig:card_new``): the head ablation. Every head WITH the TabSchema card vs WITHOUT it.

On SkillsBench skill selection (87 tasks x 195 skills), each head is cross-validated on the paper's pinned
task-level folds over three seeds, in two conditions:

WITH the card
    ``TabAgent(card, TabHead(estimator=head))``: the card's features (task text, skill name and
    description, and the engineered lexical, embedding-similarity and prior features), fitted in-fold.
    Featurizing heads see the TabHead matrix (TF-IDF text, numeric, categorical codes); TabSTAR sees the
    card's feature frame directly.
WITHOUT the card (as in the pre-release E28/E28b runs)
    featurizing heads get only raw embeddings: E5(task text) concatenated with E5(skill name + " — " +
    skill description), each block centered and renormalized (``RawEmbeddingModel``); TabSTAR gets only the
    raw text columns (task text, skill name, skill description), no engineered feature.

The metric is P@R, averaged over decisions for each seed; the figure plots the mean and the population s.d.
over seeds. One difference from the pre-release runs: the embedding centering is fitted on each training
fold (the pre-release code centered on all tasks, a label-free transductive step).

    python -m experiments.figures.head_ablation                       # CPU heads, paper card, 3 seeds
    python -m experiments.figures.head_ablation --cards paper skillsbench_skill_select_noprior
    python -m experiments.figures.head_ablation --heads tabstar       # GPU recommended
    python -m experiments.figures.head_ablation --plot-only
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments import heads as H
from experiments.figures import common as C
from tabagent import TabAgent, cross_validate
from tabagent.evaluation.cv import PAPER_SEEDS
from tabagent.table import DecisionTable
from tabagent.tabschema.card import Feature, FeatureCard

DATASET = "skillsbench/skill-select"
RAW_QUERY = ("task_text",)
RAW_PASSAGE = ("skill_name", "skill_desc")
RAW_SEP = " — "

#: The values plotted in the paper's Fig. 3 (pre-release E28/E28b/E28c/E28d runs; P@R, mean and s.d. of
#: seeds 0-2). Their card fitted the skill prior on a fixed split that overlaps the test folds.
PAPER_FIG3 = {
    "catboost": {"card": (0.7963, 0.0041), "nocard": (0.1925, 0.0051)},
    "histgbm": {"card": (0.7401, 0.0094), "nocard": (0.1429, 0.0012)},
    "random_forest": {"card": (0.7221, 0.0055), "nocard": (0.2069, 0.0006)},
    "logreg": {"card": (0.6898, 0.0), "nocard": (0.0989, 0.0)},
    "mlp": {"card": (0.5815, 0.0059), "nocard": (0.2130, 0.0053)},
    "tabstar": {"card": (0.7373, 0.0160), "nocard": (0.1411, 0.0099)},
    "tabpfn": {"card": (0.7299, 0.0100), "nocard": (0.1202, 0.0169)},
    "ft_transformer": {"card": (0.6613, 0.0231), "nocard": (0.1046, 0.0261)},
    "carte": {"card": (0.7020, 0.0171), "nocard": (0.1179, 0.0034)},
    "tpberta": {"card": (0.6681, 0.0164), "nocard": (0.0981, 0.0132)},
}


def _join(df: pd.DataFrame, cols: Sequence[str], sep: str = " ") -> list[str]:
    out = df[cols[0]].fillna("").astype(str)
    for c in cols[1:]:
        out = out + sep + df[c].fillna("").astype(str)
    return out.tolist()


class RawEmbeddingModel:
    """The WITHOUT-card condition for featurizing heads: [E5(query) ; E5(passage)] -> head.

    Follows the ``cross_validate`` model protocol (``clone``/``fit``/``score``). The common direction of
    the query and passage vectors (their mean over the training fold's unique texts) is removed and each
    block renormalized, as ``fitted.EmbeddingCosine`` does for the card's similarity feature.
    """

    def __init__(
        self,
        estimator: Callable[[int], Any],
        query: Sequence[str] = RAW_QUERY,
        passage: Sequence[str] = RAW_PASSAGE,
        encoder: str = "e5-small-v2",
        sep: str = RAW_SEP,
        seed: int = 0,
    ):
        self.estimator, self.query, self.passage = estimator, tuple(query), tuple(passage)
        self.encoder, self.sep, self.seed = encoder, sep, seed
        self.mu_: np.ndarray | None = None
        self.model_: Any = None

    def clone(self, seed: int | None = None) -> RawEmbeddingModel:
        s = self.seed if seed is None else seed
        return RawEmbeddingModel(self.estimator, self.query, self.passage, self.encoder, self.sep, s)

    def _vectors(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        from tabagent.tabschema.fitted import encoder

        enc = encoder(self.encoder)
        q = enc.encode(_join(df, self.query), "query: ")
        p = enc.encode(_join(df, self.passage, self.sep), "passage: ")
        return q, p

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        q, p = self._vectors(df)

        def center(v: np.ndarray) -> np.ndarray:
            y = v - self.mu_
            return y / np.clip(np.linalg.norm(y, axis=1, keepdims=True), 1e-9, None)

        return np.hstack([center(q), center(p)]).astype("float32")

    def fit(self, table: DecisionTable) -> RawEmbeddingModel:
        from tabagent.tabschema.fitted import encoder

        enc = encoder(self.encoder)
        df = table.df
        uq = list(dict.fromkeys(_join(df, self.query)))
        up = list(dict.fromkeys(_join(df, self.passage, self.sep)))
        self.mu_ = np.vstack([enc.encode(uq, "query: "), enc.encode(up, "passage: ")]).mean(0, keepdims=True)
        self.model_ = self.estimator(self.seed).fit(self._matrix(df), table.labels.astype(int))
        return self

    def score(self, table: DecisionTable) -> np.ndarray:
        proba = np.asarray(self.model_.predict_proba(self._matrix(table.df)))
        return proba[:, list(np.asarray(self.model_.classes_).ravel()).index(1)]


def raw_text_card(card: FeatureCard, columns: Sequence[str]) -> FeatureCard:
    """The WITHOUT-card condition for heads that read raw columns (TabSTAR): raw text columns only."""
    feats = [Feature(c, "state" if c in RAW_QUERY else "schema", "text") for c in columns]
    return FeatureCard(
        name=f"{card.name}__raw_text",
        spec=card.spec,
        features=feats,
        provenance={"kind": "ablation", "note": "raw text columns only, no engineered feature"},
    )


def model_for(
    head: str,
    card: FeatureCard | None,
    base_card: FeatureCard,
    *,
    preset: str = "tmlr-runs",
    encoder: str = "e5-small-v2",
    query: Sequence[str] = RAW_QUERY,
    passage: Sequence[str] = RAW_PASSAGE,
    head_kwargs: dict | None = None,
):
    """The model of one ablation cell. ``card=None`` is the WITHOUT condition."""
    kind = base_card.spec.kind
    kw = head_kwargs or {}
    if card is not None:
        return TabAgent(card, H.make_head(head, kind, preset=preset, **kw))
    if H.featurizes(head):
        return RawEmbeddingModel(H.factory(head, preset=preset, kind=kind, **kw), query, passage, encoder)
    return TabAgent(
        raw_text_card(base_card, [*query, *passage]), H.make_head(head, kind, preset=preset, **kw)
    )


def run(
    heads: Sequence[str],
    cards: Sequence[str] = ("paper",),
    *,
    dataset: str = DATASET,
    table: DecisionTable | None = None,
    folds: dict | None = None,
    seeds: Sequence[int] = PAPER_SEEDS,
    preset: str = "tmlr-runs",
    encoder: str = "e5-small-v2",
    query: Sequence[str] = RAW_QUERY,
    passage: Sequence[str] = RAW_PASSAGE,
    metric: str = "p@r",
    out: Path | None = None,
    force: bool = False,
    card_objects: Sequence[FeatureCard] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run (or resume) the ablation grid; results are written after every cell."""
    from tabagent import datasets

    out = C.out_dir("figures", out)
    path = out / "head_ablation.json"
    table = table if table is not None else datasets.load(dataset)
    folds = folds if folds is not None else datasets.folds(dataset)
    loaded = list(card_objects) if card_objects is not None else [C.load_card(c, dataset) for c in cards]
    base = loaded[0]
    config = {
        "seeds": list(seeds),
        "preset": preset,
        "encoder": encoder,
        "metric": metric,
        "dataset": dataset,
    }
    state = C.read_json(path)
    if state.get("config") != config:
        state = {}
    state.update(
        config=config,
        meta=state.get("meta") or C.run_meta(folds="pinned (datasets.folds)"),
        cards=[c.name for c in loaded],
        raw_inputs={"query": list(query), "passage": list(passage), "sep": RAW_SEP},
        paper=PAPER_FIG3,
    )
    results = state.setdefault("results", {})
    conditions = [("nocard", None)] + [(f"card:{c.name}", c) for c in loaded]
    for head in heads:
        if not H.is_available(head):
            log(f"skip {head}: package not installed")
            continue
        for cond, card in conditions:
            key = f"{head}|{cond}"
            if key in results and not force:
                log(f"{key} cached: {results[key]['mean']:.4f}")
                continue
            t0 = time.time()
            model = model_for(head, card, base, preset=preset, encoder=encoder, query=query, passage=passage)
            res = cross_validate(model, table, folds=folds, seeds=tuple(seeds))
            results[key] = {**C.summarize(C.per_seed(res, metric)), "secs": round(time.time() - t0, 1)}
            log(f"{key}: {results[key]['mean']:.4f} +- {results[key]['sd']:.4f} ({results[key]['secs']}s)")
            C.write_json(path, state)
    C.write_json(path, state)
    return state


def comparison(state: dict) -> pd.DataFrame:
    """One row per head: this run's WITHOUT / WITH values next to the paper's Fig. 3 values."""
    rows = []
    res = state.get("results", {})
    for head in dict.fromkeys(k.split("|")[0] for k in res):
        row = {"head": head, "without": res.get(f"{head}|nocard", {}).get("mean")}
        for name in state.get("cards", []):
            row[f"with:{name}"] = res.get(f"{head}|card:{name}", {}).get("mean")
        paper = PAPER_FIG3.get(head, {})
        row["paper_without"] = paper.get("nocard", (None,))[0]
        row["paper_with"] = paper.get("card", (None,))[0]
        rows.append(row)
    return pd.DataFrame(rows).set_index("head") if rows else pd.DataFrame()


def plot(state: dict, stem: Path) -> list[Path]:
    """Dumbbell per head: WITHOUT (orange) to WITH the first card (blue); further cards as open markers."""
    plt = C.mpl()
    from matplotlib.lines import Line2D

    res, cards = state.get("results", {}), state.get("cards", [])
    if not cards:
        raise ValueError("no card results to plot")
    primary = f"card:{cards[0]}"
    heads = [h for h in dict.fromkeys(k.split("|")[0] for k in res) if f"{h}|{primary}" in res]
    heads.sort(key=lambda h: res[f"{h}|{primary}"]["mean"], reverse=True)
    n = len(heads)
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * n + 1.0))
    for i, h in enumerate(heads):
        y = n - 1 - i
        cm, cs = res[f"{h}|{primary}"]["mean"], res[f"{h}|{primary}"]["sd"]
        if f"{h}|nocard" in res:
            nm, ns = res[f"{h}|nocard"]["mean"], res[f"{h}|nocard"]["sd"]
            ax.plot([nm, cm], [y, y], color="#c9ccd1", lw=2, zorder=1, solid_capstyle="round")
            ax.errorbar(nm, y, xerr=ns, fmt="o", color=C.ORANGE, ms=6, elinewidth=1, mec="white", zorder=3)
        for extra in cards[1:]:
            e = res.get(f"{h}|card:{extra}")
            if e:
                ax.errorbar(
                    e["mean"],
                    y,
                    xerr=e["sd"],
                    fmt="o",
                    mfc="white",
                    mec=C.BLUE,
                    ecolor=C.BLUE,
                    ms=6,
                    mew=1.6,
                    elinewidth=1,
                    zorder=4,
                )
        ax.errorbar(cm, y, xerr=cs, fmt="o", color=C.BLUE, ms=6, elinewidth=1, mec="white", zorder=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([H.LABELS.get(h, h) for h in reversed(heads)], color=C.INK)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P@R (skill selection, 5-fold, mean ± 1 s.d. over seeds)", color=C.INK2)
    ax.grid(axis="x", color=C.GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D([], [], marker="o", ls="", color=C.BLUE, mec="white", ms=7, label=f"with card ({cards[0]})"),
        Line2D([], [], marker="o", ls="", color=C.ORANGE, mec="white", ms=7, label="without card (raw only)"),
    ]
    handles += [
        Line2D([], [], marker="o", ls="", mfc="white", mec=C.BLUE, mew=1.6, ms=7, label=f"with card ({c})")
        for c in cards[1:]
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    paths = C.save_figure(fig, stem)
    plt.close(fig)
    return paths


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--heads", nargs="+", default=None, help=f"default: installed CPU heads {H.CPU_HEADS}")
    ap.add_argument(
        "--cards", nargs="+", default=["paper"], help="'paper' (registered card), names, or paths"
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    ap.add_argument("--preset", default="tmlr-runs", help="CatBoost preset (tmlr-runs produced the figure)")
    ap.add_argument("--encoder", default="e5-small-v2")
    ap.add_argument("--raw-query", nargs="+", default=list(RAW_QUERY))
    ap.add_argument("--raw-passage", nargs="+", default=list(RAW_PASSAGE))
    ap.add_argument("--out", type=Path, default=None, help="output directory (default results/figures)")
    ap.add_argument("--force", action="store_true", help="recompute cached cells")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    out = C.out_dir("figures", a.out)
    if a.plot_only:
        state = C.read_json(out / "head_ablation.json")
    else:
        heads = a.heads or H.available(H.CPU_HEADS)
        state = run(
            heads,
            a.cards,
            dataset=a.dataset,
            seeds=a.seeds,
            preset=a.preset,
            encoder=a.encoder,
            query=a.raw_query,
            passage=a.raw_passage,
            out=out,
            force=a.force,
        )
    print(comparison(state).round(4).to_string())
    for p in plot(state, out / "head_ablation"):
        print("wrote", p)
    return state


if __name__ == "__main__":
    main()
