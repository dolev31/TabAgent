"""Fig. 2: resilience to candidate overload, by controlled candidate-set scaling.

Protocol: TabAgent is cross-validated once on the paper's pinned folds, and its out-of-fold scores are reused
for every K, with no retraining. For each unit (a task) and each candidate-set size K, the slate is the unit's
positives plus K negatives drawn without replacement from its own negatives, 30 times; P@R (precision of the
top R, R = number of positives) and acc@1 are averaged over draws, then over units. The answer set is held
fixed and only the number of distractors grows, so the curve measures overload rather than a change in task
mix. Units with fewer than K negatives are skipped at that K (``n_units`` records how many remain). Tied
scores are ordered uniformly at random and each draw contributes its exact expected P@R and acc@1 (paper
Sec. 4.3 and App. L).

Units: SkillsBench skill selection uses its decisions (one per task, 195 candidates). For Mind2Web element
selection every step of a task is pooled into one unit (``unit="task_uid"``; the positives are the task's
target elements across steps), which is what allows K up to 200 although a single step has 50 candidates;
``--unit decision`` scores per step instead (K <= 49).

The trend is a power law ``P@R = a K^b`` fitted by least squares in log-log space; R^2 is computed on the
linear scale and the fit is projected to K = 400 and 800, as in the paper figure.

LLM curves (generative SFT, cross-encoder, few-shot; GPU) are produced by ``overload_llm`` with the same
sampler and merged into the same JSON; this script runs them with ``--llm``.

    python -m experiments.figures.overload --download                        # SkillsBench, CPU
    python -m experiments.figures.overload --datasets skillsbench/skill-select mind2web/element
    python -m experiments.figures.overload --llm sft_gen fewshot cross_enc --dry-run
    python -m experiments.figures.overload --plot-only
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.figures import common as C
from tabagent import TabAgent, TabHead, cross_validate
from tabagent.table import DecisionTable
from tabagent.tabschema.card import FeatureCard

REPS = 30
SAMPLER_SEED = 0
PROJECT = (400, 800)
SETTINGS = {
    "skillsbench/skill-select": {
        "ks": (3, 5, 10, 20, 50, 100, 150, 190),
        "unit": None,
        "title": "Skill selection (SkillsBench)",
    },
    "mind2web/element": {
        "ks": (3, 5, 10, 20, 50, 100, 200),
        "unit": "task_uid",
        "title": "Element selection (Mind2Web)",
    },
}

#: The curves plotted in the paper's Fig. 2 (P@R by K). LLM curves: Llama-3.1-8B-Instruct, 6 draws per K for
#: generative SFT and 30 for the cross-encoder and few-shot regimes.
PAPER_FIG2 = {
    "skillsbench/skill-select": {
        "tabagent": {
            3: 0.9654,
            5: 0.9511,
            10: 0.9303,
            20: 0.9014,
            50: 0.8566,
            100: 0.8167,
            150: 0.786,
            190: 0.785,
        },
        "sft_gen": {3: 0.8056, 5: 0.7426, 10: 0.7306, 20: 0.6844, 50: 0.6372, 100: 0.6036, 190: 0.4851},
        "cross_enc": {3: 0.8171, 5: 0.7766, 10: 0.6906, 20: 0.5767, 50: 0.4668, 100: 0.3648, 190: 0.2964},
        "fewshot": {3: 0.8412, 5: 0.8097, 10: 0.7238, 20: 0.609, 50: 0.4639, 100: 0.3454, 190: 0.257},
    },
    "mind2web/element": {
        "tabagent": {3: 0.9226, 5: 0.8919, 10: 0.8353, 20: 0.7651, 50: 0.6518, 100: 0.5708, 200: 0.5097},
        "sft_gen": {3: 0.7378, 5: 0.6403, 10: 0.5618, 20: 0.4624, 50: 0.3608, 100: 0.2967, 200: 0.2515},
        "cross_enc": {3: 0.8144, 5: 0.7554, 10: 0.6504, 20: 0.5286, 50: 0.3854, 100: 0.3016, 200: 0.2549},
        "fewshot": {3: 0.7208, 5: 0.6182, 10: 0.4763, 20: 0.3516, 50: 0.2278, 100: 0.1637, 200: 0.1094},
    },
}
CURVES = {  # regime -> (label, color, marker)
    "tabagent": ("TabAgent (CatBoost TabHead)", C.BLUE, "o"),
    "sft_gen": ("Generative SFT", C.ORANGE, "D"),
    "cross_enc": ("Cross-encoder", C.AQUA, "s"),
    "fewshot": ("Few-shot LLM", C.YELLOW, "^"),
}


def default_ks(dataset: str) -> tuple[int, ...]:
    return tuple(SETTINGS.get(dataset, {}).get("ks", (3, 5, 10, 20, 50, 100)))


def resolve_unit(dataset: str, unit: str | None = "default") -> str | None:
    """``default`` = the dataset's setting; ``decision`` (or None) = the table's decisions; else a column."""
    if unit == "default":
        unit = SETTINGS.get(dataset, {}).get("unit")
    return None if unit in (None, "decision") else unit


# --------------------------------------------------------------------------- scores
def scored_frame(table: DecisionTable, scores: np.ndarray, unit: str | None = None) -> pd.DataFrame:
    """(unit, candidate, y, score) per row; ``unit=None`` uses the table's decisions."""
    s = table.spec
    return pd.DataFrame(
        {
            "unit": table.df[unit].astype(str).to_numpy() if unit else table.decision_ids.to_numpy(),
            "candidate": table.df[s.candidate].astype(str).to_numpy(),
            "y": table.labels.astype(int),
            "score": np.asarray(scores, dtype=float),
        }
    )


def tabagent_scores(
    table: DecisionTable,
    card: FeatureCard,
    folds: dict,
    *,
    preset: str = "paper",
    seeds: Sequence[int] = (0,),
) -> pd.DataFrame:
    """Out-of-fold TabAgent scores, one column per seed (``cross_validate`` on the pinned folds)."""
    model = TabAgent(card, TabHead(card.spec.kind, preset=preset))
    return cross_validate(model, table, folds=folds, seeds=tuple(seeds)).oof


# --------------------------------------------------------------------------- controlled K sweep
def expected_hits(scores: np.ndarray, labels: np.ndarray, top: int) -> float:
    """Expected number of positives among the ``top`` highest scores when tied scores are ordered uniformly
    at random: every positive above the cutoff score, plus the free slots times the positive share of the
    candidates tied at the cutoff."""
    cut = np.sort(scores)[::-1][top - 1]
    above, tied = scores > cut, scores == cut
    return float(labels[above].sum() + (top - above.sum()) * labels[tied].sum() / tied.sum())


def simulate_k(
    frame: pd.DataFrame, ks: Sequence[int], reps: int = REPS, seed: int = SAMPLER_SEED
) -> dict[int, dict]:
    """Sweep the number of distractors K with the answer set fixed (see the module docstring)."""
    units = []
    for _, idx in frame.groupby("unit", sort=True).indices.items():
        y = frame["y"].to_numpy()[idx]
        s = frame["score"].to_numpy(dtype=float)[idx]
        pos, neg = y == 1, y == 0
        if pos.any() and neg.any():
            units.append((s[pos], s[neg]))
    out: dict[int, dict] = {}
    for k in ks:
        rng = np.random.default_rng(seed)
        par, acc, chance = [], [], []
        for ps, ns in units:
            if len(ns) < k:
                continue
            r = len(ps)
            lab = np.r_[np.ones(r), np.zeros(k)]
            p_r, a_r = [], []
            for _ in range(reps):
                pick = rng.choice(len(ns), k, replace=False)
                slate = np.r_[ps, ns[pick]]
                p_r.append(expected_hits(slate, lab, r) / r)
                a_r.append(expected_hits(slate, lab, 1))
            par.append(np.mean(p_r))
            acc.append(np.mean(a_r))
            chance.append(r / (r + k))
        out[int(k)] = {
            "p@r": round(float(np.mean(par)), 4) if par else None,
            "acc@1": round(float(np.mean(acc)), 4) if acc else None,
            "chance_p@r": round(float(np.mean(chance)), 4) if chance else None,
            "n_units": len(par),
        }
    return out


def power_law(ks: Sequence[float], ys: Sequence[float]) -> dict:
    """``y = a K^b`` fitted in log-log space; R^2 on the linear scale; projections at ``PROJECT``."""
    k, y = np.asarray(ks, float), np.asarray(ys, float)
    b, log_a = np.polyfit(np.log(k), np.log(y), 1)
    a = float(np.exp(log_a))
    pred = a * k**b
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 0 else 1.0
    return {"a": a, "b": float(b), "r2": r2, "projection": {int(p): float(a * p**b) for p in PROJECT}}


def curve_points(curve: dict) -> tuple[list[int], list[float]]:
    pts = sorted((int(k), v["p@r"] if isinstance(v, dict) else v) for k, v in curve.items())
    pts = [(k, v) for k, v in pts if v is not None]
    return [k for k, _ in pts], [v for _, v in pts]


def aggregate(curves: list[dict]) -> dict:
    """Mean (and s.d.) over seeds of per-seed curves."""
    out = {}
    for k in curves[0]:
        vals = [c[k] for c in curves]
        entry = {}
        for m in ("p@r", "acc@1", "chance_p@r"):
            xs = [v[m] for v in vals if v[m] is not None]
            entry[m] = round(float(np.mean(xs)), 4) if xs else None
            if m == "p@r" and len(xs) > 1:
                entry["p@r_sd"] = round(float(np.std(xs)), 4)
        entry["n_units"] = vals[0]["n_units"]
        out[k] = entry
    return out


# --------------------------------------------------------------------------- driver
def run_tabagent(
    dataset: str,
    *,
    table: DecisionTable | None = None,
    card: FeatureCard | None = None,
    folds: dict | None = None,
    ks: Sequence[int] | None = None,
    unit: str | None = "default",
    preset: str = "paper",
    seeds: Sequence[int] = (0,),
    reps: int = REPS,
    download: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    from tabagent import datasets

    ks = tuple(ks or default_ks(dataset))
    unit = resolve_unit(dataset, unit)
    table = table if table is not None else datasets.load(dataset, download=download)
    card = card if card is not None else datasets.card(dataset)
    folds = folds if folds is not None else datasets.folds(dataset)
    t0 = time.time()
    oof = tabagent_scores(table, card, folds, preset=preset, seeds=seeds)
    log(f"{dataset}: out-of-fold scores in {time.time() - t0:.0f}s")
    per_seed = [simulate_k(scored_frame(table, oof[s].to_numpy(float), unit), ks, reps) for s in oof.columns]
    curve = aggregate(per_seed)
    kk, yy = curve_points(curve)
    return {
        "curve": curve,
        "fit": power_law(kk, yy) if len(kk) >= 2 else None,
        "config": {
            "card": card.name,
            "preset": preset,
            "seeds": list(seeds),
            "reps": reps,
            "unit": unit or "decision",
            "ks": list(ks),
            "sampler_seed": SAMPLER_SEED,
        },
        "secs": round(time.time() - t0, 1),
    }


def plot(state: dict, stem: Path, paper_reference: bool = True) -> list[Path]:
    """One panel per dataset: TabAgent (solid, power law, open projections) and LLM curves (dashed)."""
    plt = C.mpl()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter

    names = [d for d in state.get("curves", {}) if "tabagent" in state["curves"][d]]
    if not names:
        raise ValueError("no TabAgent curve to plot")
    fig, axes = plt.subplots(1, len(names), figsize=(5.6 * len(names), 4.3), sharey=True, squeeze=False)
    seen = set()
    for ax, name in zip(axes[0], names, strict=True):
        cur = state["curves"][name]
        for regime in ("sft_gen", "cross_enc", "fewshot"):
            if regime in cur:
                label, color, marker = CURVES[regime]
                kk, yy = curve_points(cur[regime]["curve"])
                ax.plot(kk, yy, color=color, lw=1.6, ls=(0, (4, 2)), zorder=3)
                ax.scatter(kk, yy, facecolor="white", edgecolor=color, s=26, lw=1.3, marker=marker, zorder=4)
                seen.add(regime)
        if paper_reference and name in PAPER_FIG2:
            kk, yy = curve_points(PAPER_FIG2[name]["tabagent"])
            ax.plot(kk, yy, color=C.MUTED, lw=1.2, ls=(0, (1, 1.5)), zorder=2)
            seen.add("paper")
        kk, yy = curve_points(cur["tabagent"]["curve"])
        fit = cur["tabagent"]["fit"]
        color = CURVES["tabagent"][1]
        if fit:
            kmax = max(kk)
            ax.axvspan(kmax, 900, color="#f4f4f4", zorder=0)
            xs = np.logspace(np.log10(min(kk)), np.log10(kmax), 100)
            ax.plot(xs, fit["a"] * xs ** fit["b"], color=color, lw=2.2, zorder=6)
            xp = np.logspace(np.log10(kmax), np.log10(max(PROJECT)), 60)
            ax.plot(xp, fit["a"] * xp ** fit["b"], color=color, lw=1.8, ls=(0, (5, 3)), zorder=6)
            for p, v in fit["projection"].items():
                ax.scatter([int(p)], [v], facecolor="white", edgecolor=color, s=44, lw=1.8, zorder=7)
            ax.text(
                np.sqrt(kmax * 900),
                0.97,
                f"R² = {fit['r2']:.3f}",
                ha="center",
                va="top",
                fontsize=9,
                color=C.INK2,
            )
        ax.scatter(kk, yy, color=color, s=34, edgecolor="white", lw=1, zorder=7)
        ax.set_xscale("log", base=2)
        ax.set_xticks([3, 5, 10, 20, 50, 100, 200, 400, 800])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xlim(2.6, 900)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(SETTINGS.get(name, {}).get("title", name), loc="left", fontsize=10.5, color=C.INK)
        ax.grid(axis="y", color=C.GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("candidate-set size K", color=C.INK2)
    axes[0][0].set_ylabel("P@R", color=C.INK2)
    handles = [Line2D([], [], color=C.BLUE, lw=2.2, marker="o", mec="white", label=CURVES["tabagent"][0])]
    for regime in ("sft_gen", "cross_enc", "fewshot"):
        if regime in seen:
            label, color, marker = CURVES[regime]
            handles.append(
                Line2D(
                    [],
                    [],
                    color=color,
                    lw=1.6,
                    ls=(0, (4, 2)),
                    marker=marker,
                    mfc="white",
                    mec=color,
                    label=label,
                )
            )
    if "paper" in seen:
        handles.append(
            Line2D([], [], color=C.MUTED, lw=1.2, ls=(0, (1, 1.5)), label="TabAgent, paper Fig. 2")
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    paths = C.save_figure(fig, stem)
    plt.close(fig)
    return paths


def main(argv: Sequence[str] | None = None) -> dict:
    from experiments.figures import overload_llm as L

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["skillsbench/skill-select"])
    ap.add_argument("--card", default=None, help="card name or path (default: the dataset's registered card)")
    ap.add_argument("--unit", choices=["default", "decision", "task_uid"], default="default")
    ap.add_argument("--ks", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0], help="model seeds (default: 0)")
    ap.add_argument("--preset", default="paper")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--llm", nargs="*", default=[], choices=list(L.REGIMES), help="also run LLM curves (GPU)")
    ap.add_argument("--llm-model", default="llama8b", choices=list(L.MODELS))
    ap.add_argument(
        "--llm-reps", type=int, default=None, help="draws per K (default: 6 for sft_gen, else 30)"
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan (sizes, prompt counts) and exit")
    ap.add_argument("--skip-tabagent", action="store_true", help="keep the cached TabAgent curve")
    ap.add_argument("--no-paper-reference", action="store_true")
    ap.add_argument("--download", action="store_true", help="fetch upstream releases that are not cached")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    out = C.out_dir("figures", a.out)
    path = out / "overload.json"
    state = C.read_json(path)
    state.setdefault("curves", {})
    state["meta"] = C.run_meta(folds="pinned (datasets.folds)")
    state["paper"] = PAPER_FIG2
    if a.dry_run:
        from tabagent import datasets

        for name in a.datasets:
            unit = resolve_unit(name, a.unit)
            table = datasets.load(name, download=a.download)
            ks = tuple(a.ks or default_ks(name))
            print(f"[dry-run] {name}: {len(table)} rows; TabAgent 5-fold x {len(a.seeds)} seed(s); K {ks}")
            for regime in a.llm:
                print("[dry-run]", L.plan(table, name, regime, a.llm_model, ks, unit, reps=a.llm_reps))
        return state
    if not a.plot_only:
        from tabagent import datasets

        for name in a.datasets:
            unit = resolve_unit(name, a.unit)
            cur = state["curves"].setdefault(name, {})
            if not a.skip_tabagent:
                card = C.load_card(a.card, name)
                cur["tabagent"] = run_tabagent(
                    name,
                    card=card,
                    ks=a.ks,
                    unit=unit,
                    preset=a.preset,
                    seeds=a.seeds,
                    reps=a.reps,
                    download=a.download,
                )
                C.write_json(path, state)
            for regime in a.llm:
                table = datasets.load(name, download=a.download)
                ks = tuple(a.ks or default_ks(name))
                cur[regime] = L.run(
                    table,
                    name,
                    regime,
                    datasets.folds(name),
                    model=a.llm_model,
                    ks=ks,
                    unit=unit,
                    reps=a.llm_reps,
                )
                C.write_json(path, state)
        C.write_json(path, state)
    for name, cur in state["curves"].items():
        if "tabagent" in cur:
            kk, yy = curve_points(cur["tabagent"]["curve"])
            fit = cur["tabagent"]["fit"] or {}
            paper = PAPER_FIG2.get(name, {}).get("tabagent", {})
            print(f"{name} (card {cur['tabagent']['config']['card']}):")
            for k, v in zip(kk, yy, strict=True):
                print(f"  K={k:4d}  P@R {v:.4f}  paper {paper.get(k, float('nan')):.4f}")
            if fit:
                proj = ", ".join(f"K={k}: {v:.3f}" for k, v in fit["projection"].items())
                print(f"  power law a={fit['a']:.3f} b={fit['b']:.4f} R2={fit['r2']:.3f}; {proj}")
    for p in plot(state, out / "overload", paper_reference=not a.no_paper_reference):
        print("wrote", p)
    return state


if __name__ == "__main__":
    main()
