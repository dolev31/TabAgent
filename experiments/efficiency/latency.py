"""Per-decision latency and cost of a fitted TabAgent on a public card (paper Sec. 5, "The swap is far
cheaper"; App. F, Tables 15-16).

Protocol (paper): one decision scores the full candidate set of one decision in a single pass; the head runs
on CPU with four worker threads; latency is the mean over N decisions; cost is ``cost.c_cpu`` of that time.
Here TabAgent is fitted on four of the five pinned folds and timed on decisions of the held-out fold, at three
nested levels:

model       CatBoost ``predict_proba`` on the decision's already featurized matrix: the classifier pass
            alone
head        ``TabHead.score`` on the card's feature frame: TF-IDF, categorical and numeric encoding + model
agent       ``TabAgent.score`` on the decision's raw, label-free rows: every card feature (lexical overlap,
            fitted priors, embedding similarity) + head. ``agent_cold`` is each decision's first call, which
            also embeds its task text; ``agent`` repeats are warm (the text embedding is cached).

After ``warmup`` untimed decisions, each of ``n`` decisions is timed ``repeats`` times per level with
``time.perf_counter``; the machine's load average is recorded with each run, since other processes inflate
the mean (the median and minimum are more robust). Threads are capped at ``--threads`` for CatBoost,
BLAS/OpenMP (threadpoolctl) and torch, so the cost formula's thread count is honest. The held-out P@R of
the same fitted model is recorded so that a (latency, cost, quality) point can be placed on the Pareto plot.

The paper's headline (2.682 ms per decision, about 48 candidates) was measured on the CUGA agent's AppWorld
shortlisting trajectories, which are not released; the public analog closest in shape is tau2-bench tool
shortlisting over the merged catalog (``tau2/all-domains``, 38 candidates per decision).

    python -m experiments.efficiency.latency --download                         # tau2/all-domains
    python -m experiments.efficiency.latency --datasets skillsbench/skill-select mind2web/element-150
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from experiments.efficiency import cost
from experiments.figures import common as C
from tabagent import TabAgent, TabHead
from tabagent.evaluation import metrics as M
from tabagent.table import DecisionTable
from tabagent.tabschema.card import FeatureCard

DEFAULT_DATASETS = ("tau2/all-domains",)
LEVELS = ("model", "head", "agent_cold", "agent")


@contextlib.contextmanager
def thread_limit(threads: int):
    """Cap BLAS/OpenMP pools (threadpoolctl) and torch intra-op threads for the duration of the block."""
    stack = contextlib.ExitStack()
    try:
        from threadpoolctl import threadpool_limits

        stack.enter_context(threadpool_limits(threads))
    except ImportError:  # pragma: no cover - threadpoolctl ships with scikit-learn
        pass
    prev = None
    try:
        import torch

        prev = torch.get_num_threads()
        torch.set_num_threads(threads)
    except ImportError:  # pragma: no cover - torch is optional
        torch = None
    try:
        yield
    finally:
        stack.close()
        if prev is not None:
            torch.set_num_threads(prev)


def fit_holdout(
    table: DecisionTable,
    card: FeatureCard,
    folds: dict,
    *,
    preset: str = "paper",
    threads: int = 4,
    holdout_fold: int = 0,
    seed: int = 0,
) -> tuple[TabAgent, DecisionTable, float]:
    """Fit on every fold but ``holdout_fold``; return the agent, the held-out table and the fit time (s)."""
    group = table.df[table.spec.group]
    fold = group.map(lambda g: folds.get(g, folds.get(str(g))))
    test_groups = set(group[fold == holdout_fold])
    train = table.subset(set(group) - test_groups)
    test = table.subset(test_groups)
    head = TabHead(card.spec.kind, preset=preset, thread_count=threads)
    t0 = time.perf_counter()
    agent = TabAgent(card, head, seed=seed).fit(train)
    return agent, test, time.perf_counter() - t0


def _stats(ms: list[float]) -> dict:
    a = np.asarray(ms, dtype=float)
    return {
        "mean_ms": round(float(a.mean()), 4),
        "median_ms": round(float(np.median(a)), 4),
        "p90_ms": round(float(np.percentile(a, 90)), 4),
        "min_ms": round(float(a.min()), 4),
        "calls": len(a),
    }


def evict_query_embeddings() -> int:
    """Drop cached query-side ("query: ") text embeddings, so each decision's first call embeds its task
    text as a deployed agent would. Passage (catalog) embeddings stay cached, as they would in deployment."""
    from tabagent.tabschema.fitted import _SHARED

    n = 0
    for enc in _SHARED.values():
        for key in [k for k in enc.memo if k[0] == "query: "]:
            del enc.memo[key]
            n += 1
    return n


def time_decisions(
    agent: TabAgent,
    test: DecisionTable,
    *,
    n: int = 200,
    repeats: int = 5,
    warmup: int = 5,
    seed: int = 0,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Per-decision wall-clock at the model, head and agent levels (see the module docstring)."""
    evict_query_embeddings()
    free = test.drop_labels()
    ids = free.decision_ids.to_numpy()
    uniq = np.unique(ids)
    order = np.random.default_rng(seed).permutation(len(uniq))
    chosen = uniq[order[: warmup + n]]
    slates = [DecisionTable(free.df[ids == d], free.spec, has_labels=False) for d in chosen]
    head = agent.head_
    times: dict[str, list[float]] = {lvl: [] for lvl in LEVELS}
    sizes = []
    for i, slate in enumerate(slates):
        timed = i >= warmup
        t0 = clock()
        agent.score(slate)  # first call: embeds the unseen task text
        cold = clock() - t0
        X = agent.features(slate)
        Z = head._matrix(X)
        runs = {"agent": lambda s=slate: agent.score(s), "head": lambda x=X: head.score(x)}
        runs["model"] = lambda z=Z: head.model_.predict_proba(z)
        if not timed:
            for fn in runs.values():
                fn()
            continue
        sizes.append(len(slate))
        times["agent_cold"].append(cold * 1000)
        for lvl, fn in runs.items():
            for _ in range(repeats):
                t0 = clock()
                fn()
                times[lvl].append((clock() - t0) * 1000)
    return {
        "decisions": len(sizes),
        "repeats": repeats,
        "warmup": warmup,
        "candidates_per_decision": {
            "mean": round(float(np.mean(sizes)), 2),
            "min": int(np.min(sizes)),
            "max": int(np.max(sizes)),
        },
        "levels": {lvl: _stats(v) for lvl, v in times.items()},
    }


def holdout_quality(
    agent: TabAgent, test: DecisionTable, metrics: Sequence[str] = ("p@r", "recall@9")
) -> dict:
    scores = agent.score(test.drop_labels())
    return {m: round(float(M.per_decision(test, scores, m).mean()), 4) for m in metrics}


def run(
    dataset: str,
    *,
    table: DecisionTable | None = None,
    card: FeatureCard | None = None,
    folds: dict | None = None,
    preset: str = "paper",
    threads: int = 4,
    n: int = 200,
    repeats: int = 5,
    warmup: int = 5,
    holdout_fold: int = 0,
    download: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Fit, time and price one dataset; returns the JSON record."""
    from tabagent import datasets

    table = table if table is not None else datasets.load(dataset, download=download)
    card = card if card is not None else datasets.card(dataset)
    folds = folds if folds is not None else datasets.folds(dataset)
    with thread_limit(threads):
        agent, test, fit_s = fit_holdout(
            table, card, folds, preset=preset, threads=threads, holdout_fold=holdout_fold
        )
        timing = time_decisions(agent, test, n=n, repeats=repeats, warmup=warmup)
        quality = holdout_quality(agent, test)
    for lvl in timing["levels"].values():
        lvl["usd_per_decision"] = cost.c_cpu(lvl["mean_ms"] / 1000, threads)
    rec = {
        "dataset": dataset,
        "card": card.name,
        "preset": preset,
        "threads": threads,
        "holdout_fold": holdout_fold,
        "train_rows": len(table) - len(test),
        "fit_seconds": round(fit_s, 2),
        **timing,
        "quality_holdout_fold": quality,
        "load_average_1_5_15min": [round(x, 2) for x in os.getloadavg()],
        "pricing": {"formula": "C_cpu = threads * t / 3600 * p_vCPU-h", **cost.CPU, "threads": threads},
    }
    lv = rec["levels"]
    log(
        f"{dataset} [{preset}, {threads} thr, {rec['candidates_per_decision']['mean']:.0f} cand/decision]: "
        f"model {lv['model']['mean_ms']:.3f} ms, head {lv['head']['mean_ms']:.3f} ms, "
        f"agent {lv['agent']['mean_ms']:.3f} ms (cold {lv['agent_cold']['mean_ms']:.3f} ms); "
        f"C_cpu(model) ${lv['model']['usd_per_decision']:.2g}; holdout P@R {quality['p@r']:.3f}"
    )
    return rec


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    ap.add_argument(
        "--presets", nargs="+", default=["paper", "catboost-default"], help="CatBoost presets to time"
    )
    ap.add_argument("--threads", type=int, default=cost.CPU["threads"])
    ap.add_argument("--n", type=int, default=200, help="timed decisions (capped by the held-out fold)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--holdout-fold", type=int, default=0)
    ap.add_argument("--download", action="store_true", help="fetch upstream releases that are not cached")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default results/efficiency)")
    a = ap.parse_args(argv)
    out = C.out_dir("efficiency", a.out)
    path = out / "latency.json"
    state = C.read_json(path)
    state["meta"] = C.run_meta()
    runs = state.setdefault("runs", {})
    for name in a.datasets:
        for preset in a.presets:
            runs[f"{name}|{preset}"] = run(
                name,
                preset=preset,
                threads=a.threads,
                n=a.n,
                repeats=a.repeats,
                warmup=a.warmup,
                holdout_fold=a.holdout_fold,
                download=a.download,
            )
            C.write_json(path, state)
    print("wrote", path)
    return state


if __name__ == "__main__":
    main()
