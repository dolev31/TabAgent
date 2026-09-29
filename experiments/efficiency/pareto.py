"""Efficiency Pareto plot (Fig. 4, ``fig:pareto_new``): latency x quality, marker area = cost-efficiency.

Takes measured (latency, cost, quality) points: x is per-decision latency (log ms), y is decision quality
(P@R), and marker area grows as cost falls (area = 24 + 9 * (-log10 $)^2, TabAgent's star scaled by 2.9, as in
the paper figure). ``pareto_front`` returns the points no other point beats on all three axes.

Points come from a JSON list (``--points``; keys ``name``, ``latency_ms``, ``cost_usd``, ``quality``, optional
``kind`` in tabagent / finetuned / fewshot / api / other and ``note``), from ``latency.json`` records
(``--latency``, one TabAgent point per dataset and preset at the chosen ``--level``), and optionally from
the paper's reported AppWorld points (``--paper``). Points measured on different decisions are not
comparable on the quality axis; the plot labels every point with its source.

    python -m experiments.efficiency.pareto --paper                      # the paper's Fig. 4 from its table
    python -m experiments.efficiency.pareto --latency results/efficiency/latency.json --points slm.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from experiments.figures import common as C

KINDS = ("tabagent", "finetuned", "fewshot", "api", "other")


@dataclass(frozen=True)
class Point:
    name: str
    latency_ms: float
    cost_usd: float
    quality: float
    kind: str = "other"
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.latency_ms <= 0 or self.cost_usd <= 0:
            raise ValueError(f"{self.name}: latency and cost must be positive")


#: The points of the paper's Fig. 4 (AppWorld shortlisting, private data; GPU-served SLMs on a V100, costs by
#: ``cost.c_local``; TabAgent CPU, ``cost.c_cpu``; GPT-5.4 at its API latency and metered charge).
PAPER_POINTS = [
    Point("TabAgent (paper, AppWorld)", 2.682, 2.0e-7, 0.794, "tabagent", "paper, AppWorld"),
    Point("Llama-1B FT", 277.6, 2.08e-5, 0.669, "finetuned", "paper, AppWorld"),
    Point("Llama-3B FT", 531.0, 3.98e-5, 0.803, "finetuned", "paper, AppWorld"),
    Point("Qwen-7B FT", 474.0, 3.56e-5, 0.832, "finetuned", "paper, AppWorld"),
    Point("Llama-8B FT", 658.0, 4.94e-5, 0.848, "finetuned", "paper, AppWorld"),
    Point("Llama-1B few-shot", 277.6, 2.08e-5, 0.321, "fewshot", "paper, AppWorld"),
    Point("Llama-3B few-shot", 531.0, 3.98e-5, 0.497, "fewshot", "paper, AppWorld"),
    Point("Qwen-7B few-shot", 474.0, 3.56e-5, 0.600, "fewshot", "paper, AppWorld"),
    Point("Llama-8B few-shot", 658.0, 4.94e-5, 0.567, "fewshot", "paper, AppWorld"),
    Point("GPT-5.4", 7500.0, 0.052, 1.0, "api", "paper, AppWorld (quality = ceiling)"),
]


def dominates(a: Point, b: Point) -> bool:
    """``a`` is at least as good as ``b`` on latency, cost and quality, and strictly better on one."""
    ge = a.latency_ms <= b.latency_ms and a.cost_usd <= b.cost_usd and a.quality >= b.quality
    gt = a.latency_ms < b.latency_ms or a.cost_usd < b.cost_usd or a.quality > b.quality
    return ge and gt


def pareto_front(points: Sequence[Point]) -> list[Point]:
    return [p for p in points if not any(dominates(q, p) for q in points if q is not p)]


def marker_area(cost_usd: float, hero: bool = False) -> float:
    return (24 + 9 * (-math.log10(cost_usd)) ** 2) * (2.9 if hero else 1.0)


def points_from_latency(path: str | Path, level: str = "model", preset: str | None = None) -> list[Point]:
    """One TabAgent point per record of ``latency.json`` (held-out-fold P@R, mean latency at ``level``),
    optionally only the records of one CatBoost ``preset``."""
    runs = json.loads(Path(path).read_text()).get("runs", {})
    out = []
    for rec in runs.values():
        if preset and rec["preset"] != preset:
            continue
        lv = rec["levels"][level]
        cands = rec["candidates_per_decision"]["mean"]
        note = f"measured, {rec['dataset']}, {cands:.0f} cand., {rec['preset']}, {level}"
        out.append(
            Point(
                f"TabAgent: {rec['dataset']}",
                lv["mean_ms"],
                lv["usd_per_decision"],
                rec["quality_holdout_fold"]["p@r"],
                "tabagent",
                note,
            )
        )
    return out


def load_points(path: str | Path) -> list[Point]:
    return [Point(**d) for d in json.loads(Path(path).read_text())]


def _label(ax, points: Sequence[Point], front: set) -> None:
    """Greedy label placement in points space: the first candidate box that overlaps no marker disc and no
    label already placed (falls back to the first candidate)."""
    scale = 72.0 / ax.figure.dpi
    discs = []
    for q in points:
        x, y = ax.transData.transform((q.latency_ms, q.quality))
        discs.append((x * scale, y * scale, np.sqrt(marker_area(q.cost_usd, q.kind == "tabagent")) / 2))
    bb = ax.get_window_extent()
    frame = (bb.x0 * scale, bb.y0 * scale, bb.x1 * scale, bb.y1 * scale)
    boxes: list[tuple[float, float, float, float]] = []
    for p, (x, y, r) in sorted(
        zip(points, discs, strict=True), key=lambda t: (-t[0].quality, t[0].latency_ms)
    ):
        text = p.name + (" *" if p in front else "")
        w, h = 4.1 * len(text), 9.0
        pad = r + 2
        ring = [
            (1, 0, "left"),
            (-1, 0, "right"),
            (0, 1, "center"),
            (0, -1, "center"),
            (0.7, 0.9, "left"),
            (0.7, -0.9, "left"),
            (-0.7, 0.9, "right"),
            (-0.7, -0.9, "right"),
        ]
        cands = [(dx * pad * m, dy * (pad + 5) * m, ha) for m in (1, 1.8, 2.6) for dx, dy, ha in ring]

        def box(c, x=x, y=y, w=w, h=h):
            lx = x + c[0] - {"left": 0, "right": w, "center": w / 2}[c[2]]
            return lx, y + c[1] - h / 2, lx + w, y + c[1] + h / 2

        def free(b, me=(x, y)) -> bool:
            if b[0] < frame[0] or b[2] > frame[2] or b[1] < frame[1] or b[3] > frame[3]:
                return False
            for dx, dy, dr in discs:
                if (dx, dy) == me:
                    continue
                nx, ny = min(max(dx, b[0]), b[2]), min(max(dy, b[1]), b[3])
                if (nx - dx) ** 2 + (ny - dy) ** 2 < dr**2:
                    return False
            return all(b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3] for o in boxes)

        k = next((k for k, c in enumerate(cands) if free(box(c))), 0)
        c = cands[k]
        boxes.append(box(c))
        leader = {"arrowstyle": "-", "color": C.MUTED, "lw": 0.6, "shrinkB": r} if k >= len(ring) else None
        ax.annotate(
            text,
            (p.latency_ms, p.quality),
            xytext=c[:2],
            textcoords="offset points",
            ha=c[2],
            va="center",
            fontsize=7.5,
            color=C.INK2,
            arrowprops=leader,
        )


def plot(points: Sequence[Point], stem: Path, title: str | None = None) -> list[Path]:
    plt = C.mpl()
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    ax.set_xscale("log")
    style = {  # kind -> scatter kwargs (hue: TabAgent orange, SLMs blue filled/open, API neutral ink)
        "tabagent": {"marker": "*", "color": C.ORANGE, "edgecolor": "white", "linewidth": 1.4},
        "finetuned": {"marker": "o", "color": C.BLUE, "edgecolor": "white", "linewidth": 1.1},
        "fewshot": {"marker": "o", "facecolor": "white", "edgecolor": C.BLUE, "linewidth": 1.6},
        "api": {"marker": "s", "color": C.INK, "edgecolor": "white", "linewidth": 1.0},
        "other": {"marker": "D", "color": C.MUTED, "edgecolor": "white", "linewidth": 1.0},
    }
    front = set(pareto_front(points))
    for p in points:
        ax.scatter(
            p.latency_ms,
            p.quality,
            s=marker_area(p.cost_usd, p.kind == "tabagent"),
            zorder=4,
            **style[p.kind],
        )
    lats = [p.latency_ms for p in points]
    ax.set_xlim(min(lats) / 2, max(lats) * 3)
    ax.set_ylim(min(0.0, min(p.quality for p in points) - 0.05), 1.06)
    _label(ax, points, front)
    ax.set_xlabel("latency per decision (ms, log scale)", color=C.INK2)
    ax.set_ylabel("P@R", color=C.INK2)
    ax.grid(axis="x", color=C.GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    kinds = [k for k in KINDS if any(p.kind == k for p in points)]
    names = {
        "tabagent": "TabAgent",
        "finetuned": "SLM, fine-tuned",
        "fewshot": "SLM, few-shot",
        "api": "API LLM",
        "other": "other",
    }
    handles = []
    for k in kinds:
        st = style[k]
        face = st.get("facecolor", st.get("color"))
        edge = st["edgecolor"] if k == "fewshot" else "white"
        handles.append(Line2D([], [], marker=st["marker"], ls="", mfc=face, mec=edge, ms=9, label=names[k]))
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=8.5,
        title="marker area = cheaper; * = Pareto front",
        title_fontsize=8,
    )
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=C.INK)
    paths = C.save_figure(fig, stem)
    plt.close(fig)
    return paths


def main(argv: Sequence[str] | None = None) -> list[Point]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=Path, nargs="*", default=[])
    ap.add_argument("--latency", type=Path, nargs="*", default=[])
    ap.add_argument("--level", default="model", choices=["model", "head", "agent", "agent_cold"])
    ap.add_argument("--preset", default=None, help="only latency records of this CatBoost preset")
    ap.add_argument("--paper", action="store_true", help="include the paper's reported AppWorld points")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default results/efficiency)")
    a = ap.parse_args(argv)
    pts: list[Point] = list(PAPER_POINTS) if a.paper else []
    for p in a.points:
        pts += load_points(p)
    for p in a.latency:
        pts += points_from_latency(p, a.level, a.preset)
    if not pts:
        ap.error("no points: pass --paper, --points or --latency")
    out = C.out_dir("efficiency", a.out)
    front = pareto_front(pts)
    C.write_json(
        out / "pareto_points.json", {"points": [asdict(p) for p in pts], "front": [p.name for p in front]}
    )
    for p in plot(pts, out / "pareto", a.title):
        print("wrote", p)
    print("Pareto front:", ", ".join(p.name for p in front))
    return pts


if __name__ == "__main__":
    main()
