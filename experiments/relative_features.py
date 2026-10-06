"""Within-decision relative features on public data: each decision's rank and gap to the best of a card's
numeric candidate features (``tabagent.tabschema.card.with_relative``), with and without, on the paper's
folds.

    python -m experiments.relative_features --download

Protocol: the bundled public tables, cards and pinned folds (``tabagent.datasets``), the paper preset of the
CatBoost TabHead, seeds 0, 1 and 2, no TabSynth. The relative features are added for the card features listed
in ``JOBS`` (its numeric context-candidate overlaps and priors). The report gives each metric's mean over
decisions (averaged over seeds) with and without them, and the paired bootstrap 95% interval of the difference
over tasks (``CVResult.compare``). Results go to results/relative_features.json.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

from tabagent import TabAgent, TabHead, cross_validate, datasets
from tabagent.tabschema.card import with_relative

#: Public decisions and the card features whose rank and gap within the decision are added.
JOBS = {
    "tau2/gpt-4.1-2025-04-14": ["tool_task_lex", "tool_prior_usage", "tool_task_overlap"],
    "skillsbench/skill-select": ["skill_task_lex_overlap", "skill_task_overlap"],
}
METRICS = ("p@r", "recall@3", "recall@5")
SEEDS = (0, 1, 2)


def run(name: str, download: bool = False) -> dict:
    """Cross-validate ``name``'s card with and without the relative features; the means and the paired
    intervals of the difference."""
    table = datasets.load(name, download=download)
    card, folds = datasets.card(name), datasets.folds(name)
    results = {}
    for label, c in (("base", card), ("relative", with_relative(card, of=JOBS[name]))):
        agent = TabAgent(c, head=TabHead(c.spec.kind, preset="paper"))
        results[label] = cross_validate(agent, table, folds=folds, seeds=SEEDS)
    out: dict = {"decision": name, "relative_of": JOBS[name], "seeds": list(SEEDS)}
    for m in METRICS:
        diff = results["relative"].compare(results["base"], m)
        out[m] = {
            "base": round(float(results["base"].per_decision(m).mean()), 4),
            "relative": round(float(results["relative"].per_decision(m).mean()), 4),
            "difference": round(diff.estimate, 4),
            "ci95": [round(diff.low, 4), round(diff.high, 4)],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", default=list(JOBS), help="decisions (default: all of JOBS)")
    parser.add_argument("--download", action="store_true", help="download missing public data first")
    parser.add_argument("--out", default="results/relative_features.json")
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    rows = []
    for name in args.names:
        t0 = time.time()
        rows.append(run(name, args.download))
        print(json.dumps(rows[-1]), f"({time.time() - t0:.0f} s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
