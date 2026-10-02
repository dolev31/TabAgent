"""Which decision points the FeatureAnalyzer reads.

The analyzer reads only ``n_analyze`` decision points, so the sample decides which features can be found
at all. Two strategies:

* ``coverage`` (the default, the release's addition): every pick goes to a task not sampled yet, as long
  as one is left. Among those, the pick covers the most of the onboarding data the sample has not covered
  yet: the source (the log file, often a domain), the position in the run (its first, a middle or its last
  decision), the prefix-length quartile and the decision's labels (up to ``LABEL_GAIN`` of them, so long
  runs with many labels do not dominate). When the sample covers everything, coverage starts over. Ties
  break in a seeded order. Without the position rule, the analyzer of a decision that recurs in a run (the
  next tool call) reads mostly openings and finds few features of the run's later state.
* ``tasks``: one decision point per task, the first of each run, in a seeded task order; then the second
  of each, and so on (the sample of earlier releases).

Labels come from the adapter's ``labels(dp)``. Reading them uses only the onboarding data, which is the
training fold under in-fold discovery.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from tabagent.tabschema.trajectory import DecisionPoint

STRATEGIES = ("coverage", "tasks")
LABEL_GAIN = 3  # most label tokens one pick is credited for
MAX_POOL = 20_000  # decision points considered; a seeded subsample beyond this

LabelFn = Callable[[DecisionPoint], Iterable[Any]]


def sample_decision_points(
    dps: list[DecisionPoint],
    n: int,
    seed: int,
    label_of: LabelFn | None = None,
    strategy: str = "coverage",
) -> list[DecisionPoint]:
    """A seeded sample of ``n`` decision points (all of them when there are fewer); see the module text."""
    if strategy not in STRATEGIES:
        raise ValueError(f"sample_strategy must be one of {STRATEGIES}, not {strategy!r}")
    if strategy == "tasks":
        return _by_task(dps, n, seed)
    rng = np.random.default_rng(seed)
    pool = list(range(len(dps)))
    if len(pool) > MAX_POOL:
        pool = sorted(rng.choice(len(dps), MAX_POOL, replace=False).tolist())
    cands = [dps[i] for i in pool]
    desc = descriptors(cands, label_of)
    tiebreak = rng.permutation(len(cands))
    groups = [str(dp.group) for dp in cands]
    covered: set[str] = set()
    seen: set[str] = set()
    chosen: list[int] = []
    free = set(range(len(cands)))
    while free and len(chosen) < n:
        best, key = _best(free, desc, groups, seen, covered, tiebreak)
        if key[1] == 0 and covered:  # everything is covered: start a new round
            covered = set()
            best, key = _best(free, desc, groups, seen, covered, tiebreak)
        chosen.append(best)
        free.discard(best)
        seen.add(groups[best])
        covered |= desc[best]
    return [cands[j] for j in chosen]


def _best(free, desc, groups, seen, covered, tiebreak) -> tuple[int, tuple]:
    best, best_key = -1, None
    for j in free:
        key = (groups[j] not in seen, _gain(desc[j] - covered), -int(tiebreak[j]))
        if best_key is None or key > best_key:
            best, best_key = j, key
    return best, best_key


def _gain(fresh: frozenset[str] | set[str]) -> int:
    labels = sum(t.startswith("label:") for t in fresh)
    return len(fresh) - labels + min(labels, LABEL_GAIN)


def descriptors(dps: list[DecisionPoint], label_of: LabelFn | None = None) -> list[frozenset[str]]:
    """The coverage tokens of each decision point: ``source:``, ``position:``, ``length:`` and ``label:``."""
    runs: dict[str, list[int]] = {}
    for i, dp in enumerate(dps):
        runs.setdefault(dp.trajectory_id, []).append(i)
    position = [""] * len(dps)
    for idx in runs.values():
        idx.sort(key=lambda i: dps[i].step_index)
        last = len(idx) - 1
        for k, i in enumerate(idx):
            position[i] = "only" if last == 0 else "first" if k == 0 else "last" if k == last else "middle"
    cuts = np.quantile([dp.step_index for dp in dps], [0.25, 0.5, 0.75]) if dps else np.array([])
    out = []
    for i, dp in enumerate(dps):
        quartile = int(np.searchsorted(cuts, dp.step_index, side="right"))
        tokens = {f"source:{dp.source}", f"position:{position[i]}", f"length:{quartile}"}
        if label_of is not None:
            try:
                tokens.update(f"label:{y}" for y in label_of(dp))
            except NotImplementedError:  # the adapter has no labels: cover the rest
                label_of = None
            except Exception:  # noqa: BLE001 - one unreadable label does not stop the sample
                pass
        out.append(frozenset(tokens))
    return out


def coverage(
    dps: list[DecisionPoint], sample: list[DecisionPoint], label_of: LabelFn | None = None
) -> dict[str, list[int]]:
    """``{kind: [values the sample covers, values in the data]}`` for tasks, sources, positions, prefix-length
    quartiles and labels (labels only when the adapter has them)."""
    every = descriptors(dps, label_of)
    ids = {dp.id for dp in sample}
    picked = [d for dp, d in zip(dps, every, strict=True) if dp.id in ids]
    out = {"tasks": [len({str(dp.group) for dp in sample}), len({str(dp.group) for dp in dps})]}
    for kind in ("source", "position", "length", "label"):
        total = {t for d in every for t in d if t.startswith(kind + ":")}
        if total:
            out[kind + "s"] = [len({t for d in picked for t in d if t.startswith(kind + ":")}), len(total)]
    return out


def _by_task(dps: list[DecisionPoint], n: int, seed: int) -> list[DecisionPoint]:
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[DecisionPoint]] = {}
    for dp in dps:
        by_group.setdefault(str(dp.group), []).append(dp)
    groups = sorted(by_group)
    rng.shuffle(groups)
    out: list[DecisionPoint] = []
    rnd = 0
    while len(out) < n and any(len(v) > rnd for v in by_group.values()):
        for g in groups:
            if rnd < len(by_group[g]) and len(out) < n:
                out.append(by_group[g][rnd])
        rnd += 1
    return out
