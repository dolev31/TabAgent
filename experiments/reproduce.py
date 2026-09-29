"""Recompute the paper's public-data results with the released code and compare them with the paper.

    python experiments/reproduce.py                          # every cell, the paper's three seeds
    python experiments/reproduce.py --cells skill-selection --seeds 20260720
    python experiments/reproduce.py --accept-licenses        # + LLMRouterBench (license unresolved upstream)
    APPWORLD_ROOT=... python experiments/reproduce.py --cells subagent-routing

Writes results/paper_cells.json and results/paper_cells.md: for every cell, the paper's value, the value the
released code produces under the paper's protocol (pinned task folds, the given seeds, every data-dependent
statistic fitted in-fold), the difference, and a 95% bootstrap interval. Nothing is tuned to match.
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


class NestedChoice:
    """Choose among candidate agents by grouped inner cross-validation on the training data only, then fit
    the winner (the paper's "applied where it helps" rule for TabSynth, and the prior / no-prior choice for
    skill selection), so the choice never sees the outer held-out fold."""

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


# --------------------------------------------------------------------------- cells
def ranking_cell(cell, table, card, folds, seeds, preset) -> dict[str, Any]:
    metric, out = cell["metric"], {}
    res = fit_cv(table, card, folds, seeds, preset)
    if cell.get("per_domain"):
        from tabagent.datasets.mind2web import per_domain

        out["tabagent"] = {k: pct(v) for k, v in per_domain(res, metric).items()}
        out["recall@5"] = {k: pct(v) for k, v in per_domain(res, "recall@5").items()}
    else:
        out["tabagent"] = summarize(res.per_decision(metric))
        out["recall@9"] = pct(res.per_decision("recall@9").mean())
    out["chance"] = pct(M.chance_p_at_r(table).mean()) if metric == "p@r" else None
    query, docs = text_columns(card)
    if query and docs:
        scores = bm25(table, query, docs)
        if cell.get("per_domain"):
            per_step = M.per_decision(table, scores, metric)
            dom = pd.Series(table.df["domain"].to_numpy(), index=table.decision_ids.to_numpy())
            dom = dom[~dom.index.duplicated()].reindex(per_step.index)
            out["bm25"] = {k: pct(v) for k, v in per_step.groupby(dom.to_numpy()).mean().items()}
        else:
            out["bm25"] = pct(M.per_decision(table, scores, metric).mean())
    if cell.get("synth") == "domain_matched" and card.synth_text:
        # Paper L1177: paraphrase only the evaluated domain's rows (k=2), one model per web domain.
        from tabagent.datasets.mind2web import per_domain

        synth_scores = {}
        for dom in sorted(table.df["domain"].unique()):
            agent = TabAgent(
                card,
                head=TabHead(card.spec.kind, preset=preset),
                synth=TabSynth(k=cell.get("synth_k", 2), where={"domain": dom}),
            )
            res_d = cross_validate(agent, table, folds=folds, seeds=seeds)
            synth_scores[dom] = pct(per_domain(res_d, metric).get(dom))
        out["synth"] = synth_scores
    elif cell.get("synth") and card.synth_text:
        head = TabHead(card.spec.kind, preset=preset)
        options = {"real": TabAgent(card, head=head)}
        options.update({f"k{k}": TabAgent(card, head=head, synth=TabSynth(k=k)) for k in (2, 4)})
        res_s = cross_validate(NestedChoice(options, metric), table, folds=folds, seeds=seeds)
        if cell.get("per_domain"):
            from tabagent.datasets.mind2web import per_domain

            out["synth"] = {k: pct(v) for k, v in per_domain(res_s, metric).items()}
        else:
            out["synth"] = summarize(res_s.per_decision(metric))
    return out


def single_cell(cell, table, card, folds, seeds, preset) -> dict[str, Any]:
    res = fit_cv(table, card, folds, seeds, preset)
    f1 = [M.macro_f1(table, res.oof[s].to_numpy()) for s in res.seeds]
    return {
        "tabagent": {"value": pct(np.mean(f1)), "per_seed": [pct(x) for x in f1]},
        "accuracy": pct(res.per_decision("accuracy").mean()),
    }


def policy_cell(cell, table, card, folds, seeds, preset) -> dict[str, Any]:
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


def transfer_table(spec, seeds, preset) -> dict[str, Any]:
    names = spec["agents"]
    tables = {a: datasets.load(n) for a, n in names.items()}
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
    """Fill the baseline columns from experiments/baselines/run.py outputs (results/baselines/*.json)."""
    stem = cell["dataset"].replace("/", "__")
    for key, method in BASELINE_METHODS.items():
        files = sorted((out_dir / "baselines").glob(f"{stem}__{method}*.json"))
        files = [f for f in files if ".dryrun" not in f.name and "+synth" not in f.name]
        if not files:
            continue
        summary = json.loads(files[0].read_text()).get("summary", {})
        metric = "macro_f1" if cell["metric"] == "macro_f1" else cell["metric"]
        value = (summary.get(metric) or {}).get("value")
        for vals in cell_result.values():
            if isinstance(vals, dict) and value is not None:
                vals[key] = pct(value)


RUNNERS = {"ranking": ranking_cell, "multilabel": ranking_cell, "single": single_cell, "policy": policy_cell}


def run_cell(cell, seeds, preset, accept_licenses: bool) -> dict[str, Any]:
    if cell.get("requires") and not os.environ.get(cell["requires"]):
        return {"skipped": f"set {cell['requires']} (see the dataset's documentation)"}
    if cell.get("accept_license") and not accept_licenses:
        return {"skipped": "upstream license unresolved: rerun with --accept-licenses to include it"}
    table = datasets.load(cell["dataset"], accept_license=bool(cell.get("accept_license")))
    folds = datasets.folds(cell["dataset"])
    variants = cell.get("variants") or [datasets.info(cell["dataset"]).card]
    out = {}
    if cell.get("nested_variants"):
        cards = {v: FeatureCard.load(v) for v in variants}
        kind = next(iter(cards.values())).spec.kind
        options = {v: TabAgent(c, head=TabHead(kind, preset=preset)) for v, c in cards.items()}
        res = cross_validate(NestedChoice(options, cell["metric"]), table, folds=folds, seeds=seeds)
        out["nested:" + "|".join(variants)] = {
            "tabagent": summarize(res.per_decision(cell["metric"])),
            "recall@9": pct(res.per_decision("recall@9").mean()),
        }
    for name in variants:
        card = FeatureCard.load(name)
        t0 = time.time()
        out[name] = RUNNERS[card.spec.kind](cell, table, card, folds, seeds, preset)
        out[name]["seconds"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- report
def _fmt(v) -> str:
    if isinstance(v, dict) and "value" in v:
        ci = v.get("ci")
        return f"{v['value']}" + (f" [{ci[0]}, {ci[1]}]" if ci else "")
    return "" if v is None else str(v)


def markdown(cells, results, transfer=None) -> str:
    lines = [
        "# Paper results reproduced from public data",
        "",
        "Paper values are percentages from Table 1 / Table 2; released values use the paper's folds and "
        "seeds with every statistic fitted in-fold. CI: 95% bootstrap over decisions.",
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
                got = vals.get(q)
                if isinstance(paper, dict):
                    for dom, pv in paper.items():
                        gv = got.get(dom) if isinstance(got, dict) else None
                        delta = "" if gv is None else f"{gv - pv:+.1f}"
                        lines.append(f"| {cell['id']} | {card} | {q} ({dom}) | {pv} | {_fmt(gv)} | {delta} |")
                    continue
                gv = got["value"] if isinstance(got, dict) and "value" in got else got
                delta = "" if not isinstance(gv, (int, float)) else f"{gv - paper:+.1f}"
                lines.append(f"| {cell['id']} | {card} | {q} | {paper} | {_fmt(got)} | {delta} |")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", nargs="*", help="cell ids (default: all), or 'transfer'")
    ap.add_argument("--seeds", nargs="*", type=int, help="default: the paper's three seeds")
    ap.add_argument("--preset", default="tmlr-runs", choices=["paper", "tmlr-runs"])
    ap.add_argument("--accept-licenses", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    spec = yaml.safe_load((HERE / "paper_cells.yaml").read_text())
    seeds = tuple(args.seeds or spec["seeds"])
    wanted = set(args.cells or [c["id"] for c in spec["cells"]] + ["transfer"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "paper_cells.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    cells = [c for c in spec["cells"] if c["id"] in wanted]
    for cell in cells:
        print(f"[{cell['id']}] ...", flush=True)
        results[cell["id"]] = run_cell(cell, seeds, args.preset, args.accept_licenses)
        attach_baselines(cell, results[cell["id"]], out_dir)
        results["_meta"] = {"seeds": list(seeds), "preset": args.preset}
        path.write_text(json.dumps(results, indent=1, default=str))
    transfer = None
    if "transfer" in wanted:
        print("[transfer] ...", flush=True)
        transfer = {
            **transfer_table(spec["transfer"], seeds, args.preset),
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
