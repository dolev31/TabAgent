"""Refit a discovered head on a split's train tasks, reusing its feature card and extractor (no LLM call).

TabSchema discovery writes a run directory with ``card.yaml`` and ``extract.py``. For a clean held-out
evaluation of the live path, this refits the head on the split's train tasks only: the extractor runs on the
train decision points as ``build_table`` runs it, the training recipe is selected by nested validation on the
train tasks (``--select``; else ``--recipe``), the report is grouped cross-validation on the train tasks, and
the head is saved with the adapter, so ``live_prefix`` builds live prefixes as in training. Calibrated gate
thresholds (keeping 80/90/95% of the cross-validated decisions) go to ``gates.json``; the saved head has no
gate. The card itself was discovered on whatever runs discovery saw: say so when it saw held-out tasks.

    python -m experiments.live.refit --logs DIR --run-dir RUN --target NextToolCall --split split.json \
        --out HEAD --select

When ``--split`` does not exist yet, a seeded 70/30 task split (stratified by domain) is written there.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

from tabagent import auto
from tabagent.adapters import NEXT_TOOL_CALL, Tau2Adapter
from tabagent.api import TabAgentHead
from tabagent.report import evaluate
from tabagent.tabschema.card import FeatureCard
from tabagent.tabschema.compile import ExtractorRunner
from tabagent.tabschema.orchestrator import extract_rows, rows_to_table

from .common import split_tasks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs", required=True, help="τ²-bench results files (a file or a directory)")
    ap.add_argument("--run-dir", required=True, help="the TabSchema run directory (card.yaml, extract.py)")
    ap.add_argument("--target", default=NEXT_TOOL_CALL, help="the decision (NextToolCall or ToolShortlist)")
    ap.add_argument("--split", required=True, help="split JSON (written when missing)")
    ap.add_argument("--seed", type=int, default=0, help="seed of a new split")
    ap.add_argument("--held-out", type=float, default=0.3, help="held-out share of a new split")
    ap.add_argument("--select", action="store_true", help="select the recipe by nested validation")
    ap.add_argument(
        "--recipe", default="2,off,paper", help="synth k, prior on/off, preset (without --select)"
    )
    ap.add_argument("--out", required=True, help="directory for the refitted head")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    adapter = Tau2Adapter(granularity="step" if args.target == NEXT_TOOL_CALL else "task")
    trajs = adapter.load(args.logs)
    split_path = Path(args.split)
    if not split_path.exists():
        split_path.write_text(json.dumps(split_tasks(trajs, args.held_out, args.seed), indent=0))
    train = set(json.loads(split_path.read_text())["train"])
    good = [t for t in trajs if t.success and t.group in train]
    dps = [dp for t in good for dp in adapter.decision_points(t, args.target)]
    print(f"train: {len(train)} tasks, {len(good)} successful runs, {len(dps)} decision points", flush=True)

    run = Path(args.run_dir)
    card = FeatureCard.load(run / "card.yaml")
    code = (run / "extract.py").read_text()
    t0 = time.time()
    feats, per = extract_rows(
        ExtractorRunner(code, trust_extractor=False), card, dps, adapter
    )  # as build_table
    print(f"extracted in {time.time() - t0:.1f}s; failures {sum(f is None for f in feats)}", flush=True)
    table = rows_to_table(card, dps, feats, adapter, per)
    if args.select:
        recipe, grid = auto.select(table, card, search="fast")
        print(grid.to_string(), flush=True)
    else:
        k, prior, preset = args.recipe.split(",")
        recipe = auto.Recipe(synth_k=int(k), prior=prior == "on", preset=preset)
    agent = auto.build_agent(card, recipe)
    rep = evaluate(agent, table, n_splits=5)
    rep.notes.append(f"recipe: {recipe.label()}; refitted on the split's {len(train)} train tasks only")
    print(rep, flush=True)
    head = TabAgentHead(agent.clone().fit(table), args.target, code, rep, adapter=adapter)
    gates = {str(c): head.calibrate_gate(rep.cv, c) for c in (0.8, 0.9, 0.95)}
    head.gate_threshold = None
    head.save(args.out)
    Path(args.out, "gates.json").write_text(json.dumps(gates))
    print("gate thresholds by coverage:", gates, "| saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
