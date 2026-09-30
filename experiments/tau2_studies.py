"""τ²-bench four-agent studies: cross-agent transfer (paper Table 2), trajectory efficiency (App. F.1,
Table 17) and the rename row of App. N.4 (Table 21, τ² row).

    python -m experiments.tau2_studies --study transfer --download  # Table 2: cross-agent transfer
    python -m experiments.tau2_studies --study efficiency           # N* to 90% of full-pool P@R (Table 17)
    python -m experiments.tau2_studies --study rename               # names hashed at test time, prior kept

Protocol: the pinned 594-task folds shared by the four agents, CatBoost TabHead, every fold-level statistic
fitted inside the training subsample. Results go to results/tau2_studies.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tabagent import TabAgent, TabHead, datasets
from tabagent.evaluation import metrics as M

AGENTS = {
    "claude-3.7-sonnet": "tau2/claude-3-7-sonnet-20250219",
    "gpt-4.1": "tau2/gpt-4.1-2025-04-14",
    "gpt-4.1-mini": "tau2/gpt-4.1-mini-2025-04-14",
    "o4-mini": "tau2/o4-mini-2025-04-16",
}
PAPER_SEEDS = (20260720, 20260726, 20260727)
GRID = (4, 8, 16, 32, 64, 128, 256)


def load(download: bool = False):
    tables = {a: datasets.load(n, download=download) for a, n in AGENTS.items()}
    first = next(iter(AGENTS.values()))
    return tables, datasets.card(first), datasets.folds(first)


def split(table, folds, f):
    fold_of = {g: folds[str(g)] for g in table.groups}
    train = [g for g in table.groups if fold_of[g] != f]
    test = [g for g in table.groups if fold_of[g] == f]
    return train, test


def fit(card, table, groups, seed, preset):
    return TabAgent(card, head=TabHead(card.spec.kind, preset=preset), seed=seed).fit(table.subset(groups))


# --------------------------------------------------------------------------- Table 2
def transfer(seeds, preset) -> dict:
    tables, card, folds = load()
    par: dict[tuple[str, str], list[pd.Series]] = {}
    r9: dict[str, list[pd.Series]] = {}
    for a, ta in tables.items():
        for seed in seeds:
            for f in range(5):
                train, _ = split(ta, folds, f)
                agent = fit(card, ta, train, seed, preset)
                for b, tb in tables.items():
                    _, test = split(tb, folds, f)
                    t = tb.subset(test)
                    s = agent.score(t.drop_labels())
                    par.setdefault((a, b), []).append(M.p_at_r(t, s))
                    if a == b:
                        r9.setdefault(a, []).append(M.recall_at_k(t, s, 9))

    def mean_over_seeds(parts):  # each decision appears once per seed
        return pd.concat(parts).groupby(level=0).mean()

    matrix = {a: {b: round(100 * mean_over_seeds(par[(a, b)]).mean(), 1) for b in tables} for a in tables}
    own = np.mean([matrix[a][a] for a in tables])
    cross = np.mean([matrix[a][b] for a in tables for b in tables if a != b])
    diag_r9 = {a: round(100 * mean_over_seeds(r9[a]).mean(), 1) for a in tables}
    return {
        "p_at_r": matrix,
        "own_mean": round(own, 1),
        "cross_mean": round(cross, 1),
        "diag_recall@9": diag_r9,
    }


# --------------------------------------------------------------------------- trajectory efficiency
def _power(n, y_inf, a, b):
    return y_inf - a * np.power(n, -b)


def _n_star(ns, ys, y_full, frac=0.9):
    from scipy.optimize import curve_fit

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (y_inf, a, b), _ = curve_fit(_power, ns, ys, p0=(ys[-1], 1.0, 0.5), maxfev=50_000)
    target = frac * y_full
    if y_inf <= target or a <= 0 or b <= 0:
        return float("inf")
    return float((a / (y_inf - target)) ** (1 / b))


def efficiency(name, seeds, preset, grid=GRID, n_boot=200) -> dict:
    tables, card, folds = load()
    table = tables[name]
    rows = []  # (N label, seed, decision id, P@R)
    full_n = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for f in range(5):
            train, test = split(table, folds, f)
            t = table.subset(test)
            full_n.append(len(train))
            for n in [*grid, None]:
                if n is not None and n >= len(train):
                    continue
                groups = (
                    train if n is None else list(rng.choice(np.array(train, dtype=object), n, replace=False))
                )
                s = fit(card, table, groups, seed, preset).score(t.drop_labels())
                v = M.p_at_r(t, s)
                rows += [("all" if n is None else n, seed, d, x) for d, x in v.items()]
    df = pd.DataFrame(rows, columns=["n", "seed", "decision", "par"])
    n_all = float(np.mean(full_n))
    labels = [n for n in [*grid, "all"] if n in set(df["n"])]
    ns = np.array([n_all if n == "all" else n for n in labels], dtype=float)

    def curve(sub):
        per_seed = sub.groupby(["n", "seed"])["par"].mean().unstack("seed")
        return per_seed.mean(axis=1).reindex(labels).to_numpy()

    ys = curve(df)
    n_star = _n_star(ns, ys, ys[-1])
    decisions = df["decision"].unique()
    rng = np.random.default_rng(20260720)
    boot = []
    for _ in range(n_boot):
        d = rng.choice(decisions, len(decisions))
        s = rng.choice(np.array(seeds), len(seeds))
        weights = pd.Series(1, index=d).groupby(level=0).sum()
        sub = df[df["decision"].isin(weights.index) & df["seed"].isin(s)].copy()
        sub["w"] = sub["decision"].map(weights) * sub["seed"].map(pd.Series(s).value_counts())
        per = sub.groupby(["n", "seed"]).apply(
            lambda g: np.average(g["par"], weights=g["w"]), include_groups=False
        )
        yb = per.unstack("seed").mean(axis=1).reindex(labels).to_numpy()
        boot.append(_n_star(ns, yb, yb[-1]))
    finite = np.array([b for b in boot if np.isfinite(b)])
    ci = (
        [round(float(np.percentile(finite, 2.5)), 1), round(float(np.percentile(finite, 97.5)), 1)]
        if len(finite)
        else None
    )
    return {
        "curve": {str(n): round(100 * y, 1) for n, y in zip(labels, ys, strict=True)},
        "full_pool_train_tasks": round(n_all, 1),
        "n_star_90": round(n_star, 1),
        "ci95": ci,
        "bootstrap_finite": int(len(finite)),
    }


# --------------------------------------------------------------------------- rename (prior carried)
def opaque(name: str) -> str:
    return "t" + hashlib.sha1(name.encode()).hexdigest()[:10]


def rename(seeds, preset) -> dict:
    tables, card, folds = load()
    out = {}
    for a, table in tables.items():
        ctrl, hashed = [], []
        for seed in seeds:
            for f in range(5):
                train, test = split(table, folds, f)
                agent = fit(card, table, train, seed, preset)
                t = table.subset(test)
                ctrl.append(M.p_at_r(t, agent.score(t.drop_labels())))
                h = t.with_df(t.df.assign(tool_name=t.df["tool_name"].map(opaque)))  # `tool` keeps the prior
                hashed.append(M.p_at_r(h, agent.score(h.drop_labels())))
        c = pd.concat(ctrl).groupby(level=0).mean().mean()
        x = pd.concat(hashed).groupby(level=0).mean().mean()
        out[a] = {
            "control": round(100 * c, 1),
            "hashed": round(100 * x, 1),
            "delta_pp": round(100 * (x - c), 1),
        }
    out["mean"] = {
        k: round(float(np.mean([v[k] for v in out.values()])), 1) for k in ("control", "hashed", "delta_pp")
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", choices=["transfer", "efficiency", "rename", "all"], default="all")
    ap.add_argument(
        "--seeds", nargs="*", type=int, help="default: the paper's seeds (ten seeds for efficiency)"
    )
    ap.add_argument("--preset", default="paper", choices=["paper", "catboost-default"])
    ap.add_argument("--download", action="store_true", help="fetch the τ²-bench release if not cached")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.download:
        load(download=True)  # every study reads the cached tables afterwards
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "tau2_studies.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["_meta"] = {"preset": args.preset}
    if args.study in ("transfer", "all"):
        res["transfer"] = transfer(tuple(args.seeds or PAPER_SEEDS), args.preset)
        path.write_text(json.dumps(res, indent=1))
    if args.study in ("rename", "all"):
        res["rename"] = rename(tuple(args.seeds or PAPER_SEEDS), args.preset)
        path.write_text(json.dumps(res, indent=1))
    if args.study in ("efficiency", "all"):
        seeds = tuple(args.seeds or range(10))
        res["efficiency"] = {a: efficiency(a, seeds, args.preset) for a in AGENTS}
        path.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "_meta"}, indent=1))


if __name__ == "__main__":
    main()
