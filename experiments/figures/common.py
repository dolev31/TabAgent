"""Helpers shared by the figure and efficiency experiments: output paths, run metadata, plot style.

Outputs go under ``results/`` (git-ignored, see ``.gitignore``); nothing here writes data into the repo.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

RESULTS = Path(os.environ.get("TABAGENT_RESULTS", "results"))

# Categorical slots of the validated default palette (fixed order; light surface).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def out_dir(sub: str, root: str | Path | None = None) -> Path:
    d = Path(root) if root is not None else RESULTS / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_meta(**extra: Any) -> dict[str, Any]:
    import tabagent

    return {
        "tabagent": tabagent.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra,
    }


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2))
    tmp.replace(p)
    return p


def read_json(path: str | Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def load_card(name_or_path: str | None, dataset: str):
    """The dataset's registered (paper) card when ``name_or_path`` is None, else a bundled name or a path."""
    from tabagent import datasets
    from tabagent.tabschema.card import FeatureCard

    return datasets.card(dataset) if name_or_path in (None, "", "paper") else FeatureCard.load(name_or_path)


def per_seed(result, metric: str = "p@r", **kw) -> list[float]:
    """Mean per-decision ``metric`` for each seed of a ``CVResult`` (the figure's mean and s.d. inputs)."""
    from tabagent.evaluation import metrics as M

    return [
        float(M.per_decision(result.table, result.oof[s].to_numpy(), metric, **kw).mean())
        for s in result.seeds
    ]


def summarize(values: list[float]) -> dict[str, Any]:
    """Mean and population s.d. (ddof=0, as the pre-release figure scripts) of per-seed scores."""
    v = np.asarray(values, dtype=float)
    return {
        "seeds": [round(float(x), 4) for x in v],
        "mean": round(float(v.mean()), 4),
        "sd": round(float(v.std()), 4),
    }


def mpl():
    """matplotlib with a non-interactive backend and the paper-figure defaults."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def save_figure(fig, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", dpi=200)
    return paths
