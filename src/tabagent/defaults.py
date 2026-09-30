"""Library-wide defaults: the paper's seeds and the local cache location.

Every decision-quality score in the paper is repeated over ``PAPER_SEEDS`` (App. M). ``DEFAULT_SEED`` is the
first of them and seeds everything that needs one fixed seed: fold assignment, bootstrap resampling,
TabSchema's decision-point sample and TabSynth's optional row subsample.
"""

from __future__ import annotations

import os
from pathlib import Path

PAPER_SEEDS: tuple[int, ...] = (20260720, 20260726, 20260727)
DEFAULT_SEED: int = PAPER_SEEDS[0]


def cache_root() -> Path:
    """Root of the local cache: ``$TABAGENT_CACHE``, else ``~/.cache/tabagent``.

    Dataset sources and built tables live under ``sources/`` and ``tables/``, TabSchema run directories under
    ``runs/``.
    """
    return Path(os.environ.get("TABAGENT_CACHE", Path.home() / ".cache" / "tabagent"))
