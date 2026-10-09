"""TabSchema onboarded with an LLM on public τ²-bench logs: the GPT-4.1 agent's next tool call
(``NextToolCall``, a ranking decision), three pipeline versions, one onboarding each, compared on identical
rows and folds.

    python -m experiments.tabschema_tau2 download    # the three public result files, checksummed, cached
    python -m experiments.tabschema_tau2 commands    # the onboarding command of each pipeline version
    python -m experiments.tabschema_tau2 evaluate --previous RUN_A --run1 RUN_B --run2 RUN_C

Data: ``data/tau2/results/final/gpt-4.1-2025-04-14_{airline,retail,telecom}_default_gpt-4.1-2025-04-14_4trials
.json`` of github.com/sierra-research/tau2-bench at ``cf71a8070269883e38a365ffa85f78f46844c1f4`` (MIT), each
checked against the SHA-256 the repository records (``tabagent/datasets/manifests/tau2_official_final.json``).

The pipeline versions, each onboarded once with the same settings (``commands`` prints them):

* ``previous``: the pipeline before the per-candidate function (PR #43, the parent of PR #47);
* ``run1``: PR #47, whose FeatureAnalyzer is not told that the decision chooses among candidates;
* ``run2``: PR #48, the FeatureAnalyzer told so (``tabagent_analyzer_candidates``). Its run was onboarded from
  the pull request's branch, whose prompts and stages are those of the merged commit (which also holds PR
  #49's handling of Ctrl+C).

Each version is found on the main branch by its pull request number, which ends the commit's subject, so the
commands work in any clone of the repository.

``evaluate`` reads each run directory's ``card.yaml`` and ``extract.py``, extracts every card's features on
the same decision points, and cross-validates TabAgent on each card and on three variants: run 1 without
``last_step_text``, run 2 without its per-candidate features, and run 2 with the within-decision relative
features (``card.with_relative``). Protocol: 5-fold task-grouped cross-validation, head seeds 0, 1 and 2, the
CatBoost paper preset, no TabSynth; intervals over decisions (95% bootstrap) and paired differences over tasks
(``CVResult.compare``, paired bootstrap 95%). Results go to results/tabschema_tau2.json.

An onboarding asks an LLM, so a card, and the numbers it gives, vary with the model and from run to run; the
evaluation of given run directories is deterministic.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import warnings
from pathlib import Path

from tabagent import TabAgent, TabHead, cross_validate
from tabagent.datasets.tau2 import OFFICIAL, RESULTS_DIR
from tabagent.defaults import cache_root

FILES = [
    f"gpt-4.1-2025-04-14_{d}_default_gpt-4.1-2025-04-14_4trials.json"
    for d in ("airline", "retail", "telecom")
]
DECISION = "NextToolCall"
METRICS = ("p@r", "recall@3", "recall@5")
SEEDS = (0, 1, 2)
#: The pipeline versions, by the pull request that merged each into main (github.com/dolev31/TabAgent), and
#: the onboarding settings.
VERSIONS = {
    "previous": "#43",
    "run1": "#47",
    "run2": "#48",
}
SETTINGS = (
    "--decision NextToolCall --kind ranking --model aws/claude-opus-5-5 "
    "--stage-model extractor=azure/gpt-6-sol --n-analyze 20 --accept accept"
)
#: The paired comparisons over tasks: (a, b) reports a - b.
PAIRS = (
    ("run 2", "previous pipeline"),
    ("run 2", "run 1"),
    ("run 2", "run 2 without per-candidate features"),
    ("run 2 + relative features", "run 2"),
    ("run 1", "run 1 without last_step_text"),
)


def logs_dir() -> Path:
    """The folder that holds the three result files (links into the dataset cache)."""
    return cache_root() / "experiments" / "tabschema_tau2" / "logs"


def download() -> Path:
    """Fetch the three files at the pinned revision (verified against the recorded SHA-256) and gather them
    in :func:`logs_dir`."""
    out = logs_dir()
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src, dest = OFFICIAL.path(RESULTS_DIR + name, download=True), out / name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            os.symlink(src, dest)
        except OSError:  # no symlinks on this file system
            shutil.copyfile(src, dest)
    return out


def revision(pr: str) -> str:
    """A shell expression for the main-branch commit of pull request ``pr`` (``"#47"``): the one whose subject
    ends in ``(#47)``. Commit hashes are not used, since they differ between clones of a rewritten history."""
    return f"\"$(git log -1 --format=%H --grep='({pr})$' main)\""


def commands(root: Path) -> list[str]:
    """The onboarding command of each pipeline version (run directories and heads under ``root``)."""
    logs = logs_dir()
    out = []
    for name, rev in VERSIONS.items():
        out += [
            f"# {name}: TabAgent at PR {rev}",
            f"git checkout {revision(rev)} && uv sync --extra tabschema --extra llm",
            f"tabagent onboard {logs} {SETTINGS} --run-dir {root / name / 'run'} "
            f"--out {root / name / 'head'}",
        ]
    return out


def evaluate(runs: dict[str, Path], logs: Path, trust_extractor: bool = False) -> dict:
    """Cross-validate the three cards and their variants on the same decision points (see the module text)."""
    from tabagent.adapters import detect_adapter
    from tabagent.tabschema.card import FeatureCard, with_relative, without_features
    from tabagent.tabschema.compile import ExtractorRunner
    from tabagent.tabschema.orchestrator import extract_rows, rows_to_table

    adapter = detect_adapter(str(logs))
    trajectories = adapter.load(str(logs))
    good = [t for t in trajectories if t.success]
    dps = [dp for t in good for dp in adapter.decision_points(t, DECISION)]

    def table_for(run: Path):
        card = FeatureCard.load(run)
        runner = ExtractorRunner((run / "extract.py").read_text(), trust_extractor)
        feats, per = extract_rows(runner, card, dps, adapter)
        return card, rows_to_table(card, dps, feats, adapter, per)

    cards = {
        "previous pipeline": table_for(runs["previous"]),
        "run 1": table_for(runs["run1"]),
        "run 2": table_for(runs["run2"]),
    }
    if len({len(t.df) for _, t in cards.values()}) != 1:
        raise ValueError("the cards do not score the same rows")
    (c1, t1), (c2, t2) = cards["run 1"], cards["run 2"]
    per_candidate = [f.name for f in c2.features if f.source == "extractor_candidate"]
    variants = {
        **cards,
        "run 1 without last_step_text": (without_features(c1, {"last_step_text"}), t1),
        "run 2 without per-candidate features": (without_features(c2, set(per_candidate)), t2),
        "run 2 + relative features": (with_relative(c2), t2),
    }
    report: dict = {
        "data": {
            "source": f"{OFFICIAL.url} @ {OFFICIAL.revision}",
            "files": [RESULTS_DIR + f for f in FILES],
            "runs": len(trajectories),
            "successful_runs": len(good),
            "decisions": len(dps),
            "rows": len(t2.df),
        },
        "protocol": {
            "folds": 5,
            "grouped_by": "task",
            "seeds": list(SEEDS),
            "preset": "paper",
            "synth": None,
        },
        "per_candidate_features": per_candidate,
        "variants": {},
        "paired_over_tasks": {},
    }
    results = {}
    for name, (card, table) in variants.items():
        t0 = time.time()
        agent = TabAgent(card, head=TabHead("ranking", preset="paper"))
        results[name] = cross_validate(agent, table, n_splits=5, seeds=SEEDS)
        cis = {m: results[name].ci(m, level="decision") for m in METRICS}
        report["variants"][name] = {
            "features": len(card.features),
            **{m: [round(v.estimate, 4), round(v.low, 4), round(v.high, 4)] for m, v in cis.items()},
        }
        print(
            f"{name:<38} {len(card.features):3d} features  "
            + "  ".join(f"{m} {v.estimate:.3f} [{v.low:.3f}, {v.high:.3f}]" for m, v in cis.items())
            + f"  ({time.time() - t0:.0f} s)",
            flush=True,
        )
    for a, b in PAIRS:
        diffs = {m: results[a].compare(results[b], m) for m in METRICS}
        report["paired_over_tasks"][f"{a} - {b}"] = {
            m: [round(d.estimate, 4), round(d.low, 4), round(d.high, 4)] for m, d in diffs.items()
        }
        print(
            f"{a} - {b}: "
            + "  ".join(f"{m} {d.estimate:+.4f} [{d.low:+.4f}, {d.high:+.4f}]" for m, d in diffs.items()),
            flush=True,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="fetch the three public result files (checksummed) into the cache")
    cmd = sub.add_parser("commands", help="print the onboarding command of each pipeline version")
    cmd.add_argument("--root", default="tabschema_tau2", help="where the run directories and heads go")
    ev = sub.add_parser("evaluate", help="cross-validate the cards of three run directories")
    for name in VERSIONS:
        ev.add_argument(f"--{name}", required=True, type=Path, help=f"the {name} run directory")
    ev.add_argument("--logs", type=Path, default=None, help="the result files (default: the downloaded ones)")
    ev.add_argument(
        "--trust-extractor",
        action="store_true",
        help="run the cards' extract.py in-process (faster; review the code first) instead of in the sandbox",
    )
    ev.add_argument("--out", default="results/tabschema_tau2.json")
    args = parser.parse_args()
    if args.command == "download":
        print(download())
    elif args.command == "commands":
        print("\n".join(commands(Path(args.root))))
    else:
        warnings.simplefilter("ignore")
        logs = args.logs or download()
        runs = {name: getattr(args, name) for name in VERSIONS}
        report = evaluate(runs, logs, args.trust_extractor)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
