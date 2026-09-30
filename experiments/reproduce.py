"""Recompute the paper's public-data results with the released code and compare them with the paper.

    python -m experiments.reproduce --download                # every cell, the paper's three seeds
    python -m experiments.reproduce --cells skill-selection --seeds 20260720
    python -m experiments.reproduce --accept-licenses         # + LLMRouterBench (opt-in download)
    APPWORLD_ROOT=... python -m experiments.reproduce --cells subagent-routing
    python -m experiments.reproduce --cells element-selection --element-synth in-fold

Writes results/paper_cells.json and results/paper_cells.md: for every cell, the paper's value, the value the
released code produces under the paper's protocol (pinned task folds, the given seeds, every data-dependent
statistic fitted in-fold, tied scores ordered uniformly at random with the exact expected metric reported),
the difference, and a 95% bootstrap interval. The protocol is fixed in paper_cells.yaml; nothing is tuned to
match.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from tabagent import TabAgent, TabHead, TabSynth, cross_validate, datasets
from tabagent.baselines import bm25
from tabagent.evaluation import metrics as M
from tabagent.evaluation import stats as S
from tabagent.tabschema import FeatureCard

HERE = Path(__file__).resolve().parent


def pct(x: float | None) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(100 * float(x), 1)


def text_columns(card: FeatureCard) -> tuple[list[str], list[str]]:
    query = [f.name for f in card.features if f.type == "text" and f.family == "state"]
    docs = [f.name for f in card.features if f.family == "schema" and f.type in ("text", "cat")]
    return query, docs


def fit_cv(table, card, folds, seeds, preset, synth_k: int = 0):
    head = TabHead(card.spec.kind, preset=preset)
    synth = TabSynth(k=synth_k) if synth_k and card.synth_text else None
    return cross_validate(TabAgent(card, head=head, synth=synth), table, folds=folds, seeds=seeds)


def summarize(values: pd.Series) -> dict[str, Any]:
    iv = S.bootstrap_ci(values.to_numpy())
    return {"value": pct(iv.estimate), "ci": [pct(iv.low), pct(iv.high)], "n": iv.n}


def by_domain(result, metric: str) -> dict[str, float | None]:
    """Mean per-step ``metric`` of a ``CVResult`` per web domain, plus ``overall`` (Mind2Web element)."""
    from tabagent.datasets.mind2web import per_domain

    return {k: pct(v) for k, v in per_domain(result, metric).items()}


def chance(table, metric: str) -> float | None:
    """The expected ``metric`` of a uniformly random ranking: a constant score, every candidate tied."""
    return pct(M.per_decision(table, np.zeros(len(table)), metric, tie_break="expected").mean())


class NestedChoice:
    """Choose among candidate agents by grouped inner cross-validation on the training data only, then fit
    the winner (the paper's "applied where it helps" rule for TabSynth), so the choice never sees the outer
    held-out fold."""

    def __init__(self, options: dict[str, TabAgent], metric: str, inner_folds: int = 3, seed: int = 0):
        self.options, self.metric, self.inner_folds, self.seed = options, metric, inner_folds, seed
        self.choice_: str | None = None
        self.fitted_ = None

    def clone(self, seed: int) -> NestedChoice:
        return NestedChoice(
            {k: a.clone(seed) for k, a in self.options.items()}, self.metric, self.inner_folds, seed
        )

    def fit(self, table) -> NestedChoice:
        scores = {}
        for name, agent in self.options.items():
            res = cross_validate(
                agent.clone(self.seed),
                table,
                n_splits=self.inner_folds,
                fold_seed=self.seed,
                seeds=(self.seed,),
            )
            scores[name] = float(res.per_decision(self.metric).mean())
        self.choice_ = max(scores, key=scores.get)  # ties keep the first (simplest) option
        self.scores_ = scores
        self.fitted_ = self.options[self.choice_].clone(self.seed).fit(table)
        return self

    def score(self, table):
        return self.fitted_.score(table)


# --------------------------------------------------------------------------- Mind2Web element, paper protocol
#: Missing-value strings of pandas' CSV reader. The paper's element runs read the table from CSV files, so an
#: attribute text that is literally one of these (``None``, on nine Shopping rows) reached the model empty.
CSV_NA_STRINGS = frozenset(
    ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A"]
    + ["NA", "NULL", "NaN", "None", "n/a", "nan", "null"]
)


def as_read_from_csv(table):
    """The table's text as a CSV reader returns it: missing-value literals become empty strings."""
    df = table.df.copy()
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].mask(df[c].isin(CSV_NA_STRINGS), "")
    return table.with_df(df)


class PaperElementSynth:
    """Domain-matched TabSynth for the Mind2Web element row, as the paper's +synth column was computed.

    The deterministic paraphraser (k=2, the instruction) runs on one web domain's rows of the 706-task train
    split of Mind2Web (``mind2web.evaluator_split``), not only on the 150 evaluated tasks, and a variant whose
    feature key matches a real row of that split is dropped. In each outer fold the variants of the test
    fold's tasks are removed and the rest are added to the fold's real training rows. The +synth model
    therefore also learns from paraphrases of train-split tasks outside the 150 evaluated tasks, which the
    real-only model does not see (``--element-synth in-fold`` paraphrases the training folds only).
    """

    def __init__(self, pool, domain: str, evaluated_groups, k: int = 2):
        self.pool, self.domain, self.k = pool, domain, k
        self.evaluated = frozenset(map(str, evaluated_groups))
        self.rows_ = None

    def __deepcopy__(self, memo):
        return self  # fold clones share the paraphrased pool, which depends only on the stateless card

    def augment(self, X, y, table, card):
        if self.rows_ is None:
            if any(f.fitted for f in card.features):
                raise ValueError("this recipe paraphrases a fixed pool; use a stateless card")
            synth = TabSynth(k=self.k, where={"domain": self.domain})
            X_syn, y_syn = synth.augment(card.transform(self.pool), self.pool.labels, self.pool, card)
            self.rows_ = (X_syn, y_syn, synth.source_groups_.astype(str))
        X_syn, y_syn, groups = self.rows_
        test_fold = self.evaluated - set(map(str, table.groups))
        keep = ~np.isin(groups, list(test_fold))
        return X_syn[keep].reset_index(drop=True), y_syn[keep]


def bm25_scores(cell, table, card) -> np.ndarray | None:
    """BM25 over the cell's own query and document columns when it names them (the row's baseline: e.g.
    Mind2Web element scores the instruction against the first 200 characters of the attribute text; tau2 the
    first 400 characters of the customer's message against the tool name), else the card's state text
    against its candidate text."""
    spec = cell.get("bm25")
    if spec:
        query, docs = spec["query"], spec["doc"]
        df = table.df.copy()
        for cols, chars in ((query, spec.get("query_chars")), (docs, spec.get("doc_chars"))):
            for c in cols if chars else ():
                df[c] = df[c].fillna("").astype(str).str[:chars]
        table = table.with_df(df)
    else:
        query, docs = text_columns(card)
        if not (query and docs):
            return None
    return bm25(table, query, docs)


def mean_by_domain(table, per_step: pd.Series) -> dict[str, float | None]:
    dom = pd.Series(table.df["domain"].to_numpy(), index=table.decision_ids.to_numpy())
    dom = dom[~dom.index.duplicated()].reindex(per_step.index)
    return {k: pct(v) for k, v in per_step.groupby(dom.to_numpy()).mean().items()}


def element_cell_paper(cell, table, card, folds, preset, download: bool = False) -> dict[str, Any]:
    """The Mind2Web element row under the paper's protocol (``element_synth: paper``): CatBoost seeds
    ``paper_seeds``, CSV-read text, and domain-matched TabSynth over the train split
    (:class:`PaperElementSynth`), one +synth model per web domain."""
    from tabagent.datasets.mind2web import evaluator_split

    seeds, metric, k = tuple(cell["paper_seeds"]), cell["metric"], cell.get("recall_at", 9)
    table = as_read_from_csv(table)
    full = as_read_from_csv(datasets.load(cell["synth_pool"], download=download))
    split = evaluator_split()
    pool = full.subset([g for g in full.groups if split.get(str(g)) == "train"])

    res = fit_cv(table, card, folds, seeds, preset)
    out: dict[str, Any] = {"protocol": "paper", "seeds": list(seeds)}
    out["tabagent"], out[f"recall@{k}"] = by_domain(res, metric), by_domain(res, f"recall@{k}")
    out["chance"] = chance(table, metric)
    scores = bm25_scores(cell, table, card)
    if scores is not None:
        out["bm25"] = mean_by_domain(table, M.per_decision(table, scores, metric))
    synth, synth_recall = {}, {}
    for dom in sorted(table.df["domain"].unique()):
        pool_synth = PaperElementSynth(pool, dom, table.groups, cell.get("synth_k", 2))
        agent = TabAgent(card, head=TabHead(card.spec.kind, preset=preset), synth=pool_synth)
        res_d = cross_validate(agent, table, folds=folds, seeds=seeds)
        synth[dom], synth_recall[dom] = by_domain(res_d, metric)[dom], by_domain(res_d, f"recall@{k}")[dom]
    out["synth"], out[f"synth_recall@{k}"] = synth, synth_recall
    return out


# --------------------------------------------------------------------------- cells
def extended_synth_options(card: FeatureCard, head: TabHead, k: int = 4) -> dict[str, TabAgent]:
    """The five candidates of the paper's inner choice on the τ²-bench per-domain cards (App. H, extended
    operators): no synthesis, template paraphrases, template paraphrases with recomputed text features, entity
    swap with recomputed features, and entity swap with coverage allocation. Ties keep the earlier option."""
    return {
        "none": TabAgent(card, head=head),
        "template": TabAgent(card, head=head, synth=TabSynth(k=k)),
        "template+recompute": TabAgent(card, head=head, synth=TabSynth(k=k, recompute=True)),
        "swap+recompute": TabAgent(card, head=head, synth=TabSynth(k=k, recompute=True, entity_swap=True)),
        "swap+recompute+allocation": TabAgent(
            card,
            head=head,
            synth=TabSynth(k=k, recompute=True, entity_swap=True, allocation="inverse_sqrt_frequency"),
        ),
    }


def ranking_cell(cell, table, card, folds, seeds, preset, download: bool = False) -> dict[str, Any]:
    if cell.get("element_synth") == "paper":
        return element_cell_paper(cell, table, card, folds, preset, download)
    metric, k, out = cell["metric"], cell.get("recall_at", 9), {}
    res = fit_cv(table, card, folds, seeds, preset)
    if cell.get("per_domain"):
        out["tabagent"], out[f"recall@{k}"] = by_domain(res, metric), by_domain(res, f"recall@{k}")
    else:
        out["tabagent"] = summarize(res.per_decision(metric))
        out[f"recall@{k}"] = pct(res.per_decision(f"recall@{k}").mean())
    out["chance"] = chance(table, metric)
    scores = bm25_scores(cell, table, card)
    if scores is not None:
        per_step = M.per_decision(table, scores, metric)
        out["bm25"] = mean_by_domain(table, per_step) if cell.get("per_domain") else pct(per_step.mean())
    if cell.get("synth") == "domain_matched" and card.synth_text:
        # element_synth: in-fold. Paraphrase only the evaluated domain's rows of the training folds (k=2), one
        # model per web domain, with no rows from outside the evaluated tasks.
        synth, synth_recall = {}, {}
        for dom in sorted(table.df["domain"].unique()):
            agent = TabAgent(
                card,
                head=TabHead(card.spec.kind, preset=preset),
                synth=TabSynth(k=cell.get("synth_k", 2), where={"domain": dom}),
            )
            res_d = cross_validate(agent, table, folds=folds, seeds=seeds)
            synth[dom], synth_recall[dom] = (
                by_domain(res_d, metric)[dom],
                by_domain(res_d, f"recall@{k}")[dom],
            )
        out["synth"], out[f"synth_recall@{k}"] = synth, synth_recall
    elif cell.get("synth") == "extended" and card.synth_text:
        # App. H: on each outer training fold, a task-grouped 4-fold inner cross-validation chooses among no
        # synthesis and four TabSynth configurations (template paraphrases, recomputed text features, entity
        # swap, coverage allocation); the outer test fold never informs the choice.
        head = TabHead(card.spec.kind, preset=preset)
        nested = NestedChoice(extended_synth_options(card, head), metric, inner_folds=4)
        res_s = cross_validate(nested, table, folds=folds, seeds=seeds)
        out["synth"] = summarize(res_s.per_decision(metric))
        out[f"synth_recall@{k}"] = pct(res_s.per_decision(f"recall@{k}").mean())
    elif cell.get("synth") and card.synth_text:
        # TabSynth off / k=2 / k=4, chosen by inner cross-validation on each outer training fold.
        head = TabHead(card.spec.kind, preset=preset)
        options = {"real": TabAgent(card, head=head)}
        options.update({f"k{k_}": TabAgent(card, head=head, synth=TabSynth(k=k_)) for k_ in (2, 4)})
        res_s = cross_validate(NestedChoice(options, metric), table, folds=folds, seeds=seeds)
        out["synth"] = summarize(res_s.per_decision(metric))
        out[f"synth_recall@{k}"] = pct(res_s.per_decision(f"recall@{k}").mean())
    return out


def single_cell(cell, table, card, folds, seeds, preset, download: bool = False) -> dict[str, Any]:
    res = fit_cv(table, card, folds, seeds, preset)
    f1 = [M.macro_f1(table, res.oof[s].to_numpy()) for s in res.seeds]
    return {
        "tabagent": {"value": pct(np.mean(f1)), "per_seed": [pct(x) for x in f1]},
        "accuracy": pct(res.per_decision("accuracy").mean()),
    }


def policy_cell(cell, table, card, folds, seeds, preset, download: bool = False) -> dict[str, Any]:
    res = fit_cv(table, card, folds, seeds, preset)
    chosen = res.per_decision("route@1")
    spec, df = table.spec, table.df
    arm_mean = df.groupby(spec.candidate)[spec.reward].mean()
    best_arm = arm_mean.idxmax()
    best = df[df[spec.candidate] == best_arm].assign(
        _d=table.decision_ids[df[spec.candidate] == best_arm].to_numpy()
    )
    best_per_decision = best.set_index("_d")[spec.reward].reindex(chosen.index)
    diff = S.paired_bootstrap(chosen.to_numpy(), best_per_decision.to_numpy())
    return {
        "tabagent": summarize(chosen),
        "best_fixed": pct(arm_mean.max()),
        "best_fixed_arm": str(best_arm),
        "vs_best_fixed_pp": {"value": pct(diff.estimate), "ci": [pct(diff.low), pct(diff.high)]},
        "chance": pct(M.chance_route(table).mean()),
        "oracle": pct(M.oracle_route(table).mean()),
    }


def transfer_table(spec, seeds, preset, download: bool = False) -> dict[str, Any]:
    names = spec["agents"]
    tables = {a: datasets.load(n, download=download) for a, n in names.items()}
    card = datasets.card(next(iter(names.values())))
    folds = datasets.folds(next(iter(names.values())))
    per: dict[tuple[str, str], list[pd.Series]] = {}
    for a, train_table in tables.items():
        for seed in seeds:
            for f in sorted(set(folds.values())):
                train = train_table.subset([g for g in train_table.groups if folds[str(g)] != f])
                agent = TabAgent(card, head=TabHead(card.spec.kind, preset=preset), seed=seed).fit(train)
                for b, test_table in tables.items():
                    test = test_table.subset([g for g in test_table.groups if folds[str(g)] == f])
                    if len(test):
                        per.setdefault((a, b), []).append(M.p_at_r(test, agent.score(test.drop_labels())))
    matrix = {a: {b: None for b in names} for a in names}
    for (a, b), parts in per.items():
        by_seed = pd.concat(parts)  # decisions repeat once per seed
        matrix[a][b] = pct(by_seed.groupby(level=0).mean().mean())
    diag = np.mean([matrix[a][a] for a in names])
    off = np.mean([matrix[a][b] for a in names for b in names if a != b])
    return {"matrix": matrix, "own_mean": round(diag, 1), "cross_mean": round(off, 1)}


BASELINE_METHODS = {"dsr_zs": "dsr-zs", "dsr_ft": "dsr-ft", "fewshot": "fewshot", "ft_slm": "ft-slm"}


def attach_baselines(cell, cell_result: dict, out_dir: Path) -> None:
    """Fill the baseline columns from ``experiments.baselines.run`` outputs (results/baselines/*.json):
    DSR files are ``<dataset>__<method>.json``, LLM files ``<dataset>__<method>__<model>.json``."""
    stem, metric = cell["dataset"].replace("/", "__"), cell["metric"]
    for key, method in BASELINE_METHODS.items():
        folder = out_dir / "baselines"
        files = [folder / f"{stem}__{method}.json"] + sorted(folder.glob(f"{stem}__{method}__*.json"))
        files = [f for f in files if f.exists() and not f.name.endswith(".dryrun.json")]
        if not files:
            continue
        res = json.loads(files[0].read_text())
        if cell.get("per_domain"):
            value = {d: pct(v.get(metric)) for d, v in res.get("slices", {}).items()} or None
        else:
            value = pct((res.get("summary", {}).get(metric) or {}).get("value"))
        for vals in cell_result.values():
            if isinstance(vals, dict) and value is not None:
                vals[key] = value


RUNNERS = {"ranking": ranking_cell, "multilabel": ranking_cell, "single": single_cell, "policy": policy_cell}


def run_cell(cell, seeds, preset, accept_licenses: bool, download: bool = False) -> dict[str, Any]:
    if cell.get("requires") and not os.environ.get(cell["requires"]):
        return {"skipped": f"set {cell['requires']} (see the dataset's documentation)"}
    if cell.get("accept_license") and not accept_licenses:
        return {"skipped": "opt-in download (MIT as declared upstream): rerun with --accept-licenses"}
    table = datasets.load(cell["dataset"], download=download, accept_license=bool(cell.get("accept_license")))
    folds = datasets.folds(cell["dataset"])
    out = {}
    for name in cell.get("variants") or [datasets.info(cell["dataset"]).card]:
        card = FeatureCard.load(name)
        t0 = time.time()
        out[name] = RUNNERS[card.spec.kind](cell, table, card, folds, seeds, preset, download)
        out[name]["seconds"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- report
#: Result keys that describe the run rather than a quantity to compare.
RUN_KEYS = {"protocol", "seeds", "seconds", "best_fixed_arm"}


def _fmt(v) -> str:
    if isinstance(v, dict) and "value" in v:
        ci = v.get("ci")
        return f"{v['value']}" + (f" [{ci[0]}, {ci[1]}]" if ci else "")
    return "" if v is None else str(v)


def _rows(cell_id: str, card: str, q: str, paper, got) -> list[str]:
    card = card.replace("|", "\\|")
    if isinstance(paper, dict) or (paper is None and isinstance(got, dict) and "value" not in got):
        rows = []
        for dom in paper or got:
            pv, gv = (paper or {}).get(dom), got.get(dom) if isinstance(got, dict) else None
            delta = "" if gv is None or pv is None else f"{gv - pv:+.1f}"
            rows.append(f"| {cell_id} | {card} | {q} ({dom}) | {_fmt(pv)} | {_fmt(gv)} | {delta} |")
        return rows
    gv = got["value"] if isinstance(got, dict) and "value" in got else got
    delta = "" if not isinstance(gv, int | float) or paper is None else f"{gv - paper:+.1f}"
    return [f"| {cell_id} | {card} | {q} | {_fmt(paper)} | {_fmt(got)} | {delta} |"]


def markdown(cells, results, transfer=None) -> str:
    lines = [
        "# Paper results reproduced from public data",
        "",
        "Paper values are percentages from Table 1 / Table 2 (Table 11 for the real-only τ² Recall@9); "
        "released values use the paper's folds and seeds with every statistic fitted in-fold and tied scores "
        "ordered uniformly at random (exact expected metric). CI: 95% bootstrap over decisions. "
        "`synth_recall@k` is the +synth head's recall, the quantity of Table 1's R@k column where a +synth "
        "value is reported.",
        "",
        "| cell | card | quantity | paper | released | Δ (pp) |",
        "|---|---|---|---|---|---|",
    ]
    for cell in cells:
        res = results.get(cell["id"], {})
        if "skipped" in res:
            lines.append(f"| {cell['id']} | | | | skipped: {res['skipped']} | |")
            continue
        for card, vals in res.items():
            for q, paper in cell["paper"].items():
                lines += _rows(cell["id"], card, q, paper, vals.get(q))
            for q, got in vals.items():  # released quantities the paper does not report
                if q not in cell["paper"] and q not in RUN_KEYS:
                    lines += _rows(cell["id"], card, q, None, got)
    for cell in cells:
        for vals in results.get(cell["id"], {}).values():
            if isinstance(vals, dict) and vals.get("protocol") == "paper":
                lines.append(
                    f"\n{cell['id']}: the paper's protocol (CatBoost seeds {vals['seeds']}; +synth "
                    "paraphrases each web domain's rows of the 706-task train split, minus the test fold's "
                    "tasks)."
                )
    if transfer:
        lines += [
            "",
            "## Table 2: cross-agent transfer (P@R)",
            "",
            "| train \\ test | " + " | ".join(transfer["matrix"]) + " |",
            "|---|" + "---|" * len(transfer["matrix"]),
        ]
        for a, row in transfer["matrix"].items():
            lines.append(
                f"| {a} | " + " | ".join(f"{row[b]} (paper {transfer['paper'][a][b]})" for b in row) + " |"
            )
        lines.append(f"\nown-agent mean {transfer['own_mean']} · cross-agent mean {transfer['cross_mean']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    spec = yaml.safe_load((HERE / "paper_cells.yaml").read_text())
    ids = [c["id"] for c in spec["cells"]] + ["transfer"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--cells",
        nargs="*",
        choices=ids,
        metavar="ID",
        help=f"cells to recompute (default: all): {', '.join(ids)}",
    )
    ap.add_argument("--seeds", nargs="*", type=int, help="CatBoost seeds (default: the paper's three seeds)")
    ap.add_argument(
        "--preset",
        default="paper",
        choices=["paper", "catboost-default"],
        help="TabHead configuration: paper (300 iterations, learning rate 0.08, depth 6; default) or "
        "catboost-default (1000 iterations)",
    )
    ap.add_argument(
        "--accept-licenses",
        action="store_true",
        help="include LLMRouterBench, whose download is opt-in (MIT as declared in its README)",
    )
    ap.add_argument("--download", action="store_true", help="fetch upstream releases that are not cached")
    ap.add_argument(
        "--element-synth",
        choices=["paper", "in-fold"],
        help="Mind2Web element +synth: the paper's protocol (default) or TabSynth on the training folds only",
    )
    ap.add_argument("--out", default="results", help="output directory (default: results)")
    args = ap.parse_args(argv)

    seeds = tuple(args.seeds or spec["seeds"])
    wanted = set(args.cells or ids)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "paper_cells.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    cells = [c for c in spec["cells"] if c["id"] in wanted]
    for c in cells:
        if args.element_synth and "element_synth" in c:
            c["element_synth"] = args.element_synth
    for cell in cells:
        print(f"[{cell['id']}] ...", flush=True)
        results[cell["id"]] = run_cell(cell, seeds, args.preset, args.accept_licenses, args.download)
        attach_baselines(cell, results[cell["id"]], out_dir)
        results["_meta"] = {"seeds": list(seeds), "preset": args.preset}
        path.write_text(json.dumps(results, indent=1, default=str))
    transfer = None
    if "transfer" in wanted:
        print("[transfer] ...", flush=True)
        transfer = {
            **transfer_table(spec["transfer"], seeds, args.preset, args.download),
            "paper": spec["transfer"]["paper"],
        }
        results["transfer"] = transfer
        path.write_text(json.dumps(results, indent=1, default=str))
    elif "transfer" in results:
        transfer = results["transfer"]
    all_cells = [c for c in spec["cells"] if c["id"] in results]
    (out_dir / "paper_cells.md").write_text(markdown(all_cells, results, transfer))
    print(f"wrote {path} and {out_dir / 'paper_cells.md'}")


if __name__ == "__main__":
    main()
