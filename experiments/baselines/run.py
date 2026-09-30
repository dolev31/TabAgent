"""Run one learned baseline on one public decision table and write per-decision metric vectors.

    python -m experiments.baselines.run --dataset skillsbench/skill-select --method dsr-ft --download

Methods: ``dsr-zs``, ``dsr-ft`` (add ``--synth K`` for Real+synth), ``fewshot``, ``ft-slm``, ``ft-ce`` (the
LoRA cross-encoder). Fine-tuned methods are trained on the training folds of the paper's task folds only
(``tabagent.datasets.folds``) and score the held-out fold. ``--dry-run`` builds every LLM prompt without
loading model weights.

The output holds, per metric, the value of every decision (averaged over seeds), the decision -> group map,
and a summary with bootstrap CIs (decision level and task level), so that paired CIs against TabAgent can
be computed with ``tabagent.evaluation.stats.paired_bootstrap`` on the shared decisions.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.baselines import dsr as D
from experiments.baselines import llm as L
from experiments.baselines.presets import PRESETS, Preset
from tabagent import datasets
from tabagent.evaluation import metrics as M
from tabagent.evaluation import stats as S
from tabagent.table import DecisionTable

ROOT = Path(__file__).resolve().parents[2]

METHODS = ("dsr-zs", "dsr-ft", "fewshot", "ft-slm", "ft-ce")
DETERMINISTIC = {"dsr-zs", "fewshot"}
DEFAULT_RECIPE = "in-batch"


# --------------------------------------------------------------------------- scoring
def run_dsr(table, preset: Preset, folds, method, seed, args):
    if preset.query is None:
        raise SystemExit(f"DSR is not defined for {args.dataset} (the paper reports '--')")
    cfg = replace(D.RECIPES[args.recipe], **_dsr_overrides(preset.dsr))
    if args.epochs is not None:
        cfg = replace(cfg, epochs=args.epochs)
    single = table.spec.kind == "single"
    target = D.expand_classes(table, preset.classes, preset.class_text) if single else table
    scores = D.dsr_scores(
        target,
        preset.query,
        preset.doc,
        folds,
        finetune=method == "dsr-ft",
        synth=args.synth,
        config=cfg,
        seed=seed,
        device=args.device,
    )
    if single:
        return D.argmax_class(target, scores, table), asdict(cfg)
    return scores, asdict(cfg)


def _dsr_overrides(preset_cfg: D.DSRConfig) -> dict:
    """The preset's text caps and schedule; the recipe decides negatives and masking."""
    keys = ("model_id", "max_chars", "max_length", "epochs")
    return {k: getattr(preset_cfg, k) for k in keys}


class Backends:
    """One backend per kind (causal LM, cross-encoder, API client), created on first use; in a dry run a
    single prompt recorder stands in for all of them."""

    def __init__(self, args):
        self.args, self.cache = args, {}
        self.model = L.resolve_model(args.model)

    def __call__(self, kind: str):
        if self.args.dry_run:
            kind = "dry"
        if kind not in self.cache:
            self.cache[kind] = self._make(kind)
        return self.cache[kind]

    def _make(self, kind: str):
        if kind == "dry":
            return L.DryRunLM(self.model)
        if kind == "api":
            from tabagent.tabschema.llm import OpenAIChatLLM

            return OpenAIChatLLM(self.model, temperature=None, max_tokens=512)
        if kind == "ce":
            return L.HFCrossEncoder(self.model, device=self.args.device)
        return L.HFCausalLM(self.model, device=self.args.device)


def run_llm(table, preset: Preset, folds, method, seed, args, backend: Backends):
    task = preset.llm
    if method == "ft-ce" or (method == "ft-slm" and not isinstance(task, L.ListwiseTask | L.ClassifyTask)):
        ce = task.cross_encoder if isinstance(task, L.PointwiseTask) else preset.ce
        if ce is None:
            raise SystemExit(f"no cross-encoder configured for {args.dataset}")
        return L.cross_encoder_scores(table, ce, backend("ce"), folds, seed), asdict(ce)
    if task is None:
        raise SystemExit(f"no {method} prompt is included for {args.dataset}")
    info = {k: v for k, v in asdict(task).items() if not callable(v)}
    if isinstance(task, L.GateTask):
        if method != "fewshot":
            raise SystemExit("the gate has a few-shot (API) baseline only")
        return L.fewshot_gate(table, task, backend("api")), info
    lm = backend("causal")
    if isinstance(task, L.ListwiseTask):
        if method == "fewshot":
            return L.fewshot_listwise(table, task, lm), info
        return L.sft_listwise(table, task, lm, folds, seed), info
    if isinstance(task, L.PointwiseTask):
        return L.fewshot_pointwise(table, task, lm), info
    if method == "fewshot":
        return L.fewshot_classify(table, task, lm, folds), info
    return L.sft_classify(table, task, lm, folds, seed), info


# --------------------------------------------------------------------------- metrics
def per_decision(table: DecisionTable, out: np.ndarray, metric: str) -> pd.Series:
    """Metric per evaluated decision (decisions the method did not score are left out)."""
    if table.spec.kind == "single":
        keep = np.array([o is not None and o == o for o in out])
    else:
        keep = np.isfinite(np.asarray(out, dtype=float))
    ids = table.decision_ids
    whole = pd.Series(keep).groupby(ids.to_numpy()).transform("all").to_numpy()
    sub = table.with_df(table.df[whole])
    return M.per_decision(sub, np.asarray(out)[whole], metric)


def summarize(values: pd.Series, groups: pd.Series) -> dict:
    out = {"value": float(values.mean()), "n_decisions": int(len(values))}
    if len(values) >= 2:
        iv = S.bootstrap_ci(values.to_numpy())
        out.update(ci_low=iv.low, ci_high=iv.high)
    g = values.groupby(groups.reindex(values.index).to_numpy()).mean()
    out.update(group_value=float(g.mean()), n_groups=int(len(g)))
    if len(g) >= 2:
        iv = S.bootstrap_ci(g.to_numpy())
        out.update(group_ci_low=iv.low, group_ci_high=iv.high)
    return out


def evaluate(table: DecisionTable, preset: Preset, outs: dict, args) -> dict:
    ids = table.decision_ids
    group_of = pd.Series(table.df[table.spec.group].to_numpy(), index=ids.to_numpy())
    group_of = group_of[~group_of.index.duplicated()]
    result: dict = {"per_decision": {}, "summary": {}, "per_seed": {}, "slices": {}}
    for metric in preset.metrics:
        runs = {seed: per_decision(table, o, metric) for seed, o in outs.items()}
        mean = pd.concat(runs.values(), axis=1).mean(axis=1).rename(metric)
        result["per_decision"][metric] = {str(k): float(v) for k, v in mean.items()}
        result["summary"][metric] = summarize(mean, group_of)
        result["per_seed"][metric] = {str(s): float(r.mean()) for s, r in runs.items()}
        if preset.slice_col:
            first = pd.Series(table.df[preset.slice_col].to_numpy(), index=ids.to_numpy())
            first = first[~first.index.duplicated()].reindex(mean.index)
            for sl, v in mean.groupby(first.to_numpy()).mean().items():
                result["slices"].setdefault(str(sl), {})[metric] = float(v)
    if table.spec.kind == "single":
        f1 = [M.macro_f1(*_labeled(table, o)) for o in outs.values()]
        result["summary"]["macro_f1"] = {"value": float(np.mean(f1)), "per_seed": f1}
    elif table.spec.kind == "policy":
        result["summary"]["chance_route@1"] = float(M.chance_route(table).mean())
        result["summary"]["oracle_route@1"] = float(M.oracle_route(table).mean())
    else:
        result["summary"]["chance_p@r"] = float(M.chance_p_at_r(table).mean())
    result["decision_group"] = {str(k): str(v) for k, v in group_of.items()}
    return result


def _labeled(table: DecisionTable, preds: np.ndarray):
    keep = np.array([p is not None and p == p for p in preds])
    return table.with_df(table.df[keep]), np.asarray(preds)[keep]


# --------------------------------------------------------------------------- dry run
def prompt_report(lm: L.DryRunLM, prompts_out: str | None) -> dict:
    rec = pd.DataFrame(lm.records)
    report = {"templated": lm.templated, "n_prompts": int(len(rec)), "by_kind": {}}
    for kind, g in rec.groupby("kind") if len(rec) else []:
        report["by_kind"][kind] = {
            "n": int(len(g)),
            "tokens_min": int(g["n_tokens"].min()),
            "tokens_mean": float(g["n_tokens"].mean()),
            "tokens_max": int(g["n_tokens"].max()),
            "examples": g.head(2)[["fold", "prompt", "target"]].to_dict("records"),
        }
    if prompts_out:
        Path(prompts_out).parent.mkdir(parents=True, exist_ok=True)
        rec.to_json(prompts_out, orient="records", lines=True)
    return report


# --------------------------------------------------------------------------- main
def default_out(args) -> Path:
    name = (
        args.dataset.replace("/", "__") + "__" + args.method + (f"+synth{args.synth}" if args.synth else "")
    )
    if args.method == "dsr-ft" and args.recipe != DEFAULT_RECIPE:
        name += "__" + args.recipe
    if args.method not in ("dsr-zs", "dsr-ft"):
        name += "__" + L.resolve_model(args.model).split("/")[-1]
    return ROOT / "results" / "baselines" / f"{name}{'.dryrun' if args.dry_run else ''}.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(PRESETS))
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--model", default=L.DEFAULT_MODEL, help=f"LLM: alias ({', '.join(L.MODELS)}) or HF id")
    p.add_argument("--seeds", type=int, nargs="+", help="training seeds (default: the preset's seed)")
    p.add_argument(
        "--recipe",
        default=DEFAULT_RECIPE,
        choices=sorted(D.RECIPES),
        help="DSR-FT negatives: in-batch (the paper's, default) or hard-negatives",
    )
    p.add_argument("--synth", type=int, default=None, help="DSR-FT Real+synth: TabSynth variants per row")
    p.add_argument("--epochs", type=int, default=None, help="override the DSR-FT epochs")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    p.add_argument("--dry-run", action="store_true", help="LLM methods: build prompts only")
    p.add_argument("--prompts-out", default=None, help="dry run: write every prompt as JSON lines here")
    p.add_argument("--max-groups", type=int, default=None, help="smoke test: keep the first N groups")
    p.add_argument("--download", action="store_true", help="fetch the upstream release if not cached")
    p.add_argument("--accept-license", action="store_true", help="for sources that need an opt-in")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.synth and args.method != "dsr-ft":
        raise SystemExit("--synth applies to dsr-ft only")
    if args.dry_run and args.method.startswith("dsr"):
        raise SystemExit("--dry-run applies to the LLM methods")
    preset = PRESETS[args.dataset]
    table = datasets.load(args.dataset, download=args.download, accept_license=args.accept_license)
    folds = datasets.folds(args.dataset)
    if args.max_groups:
        table = table.subset(table.groups[: args.max_groups])
    default_seed = preset.dsr_seed if args.method.startswith("dsr") else 0
    seeds = [None] if args.method in DETERMINISTIC else (args.seeds or [default_seed])

    t0, outs, config = time.time(), {}, {}
    backend = Backends(args)
    for seed in seeds:
        print(f"{args.dataset} | {args.method} | seed {seed}", flush=True)
        if args.method.startswith("dsr"):
            outs[seed], config = run_dsr(table, preset, folds, args.method, seed or 0, args)
        else:
            outs[seed], config = run_llm(table, preset, folds, args.method, seed or 0, args, backend)
    runtime = time.time() - t0

    result = {
        "dataset": args.dataset,
        "method": args.method + (f"+synth{args.synth}" if args.synth else ""),
        "model": None if args.method.startswith("dsr") else L.resolve_model(args.model),
        "recipe": args.recipe if args.method == "dsr-ft" else None,
        "seeds": seeds,
        "folds": "paper (tabagent.datasets.folds)",
        "config": config,
        "source": preset.source,
        "paper": preset.paper.get(args.method),
        "n_rows": len(table),
        "n_groups": len(table.groups),
        "runtime_s": round(runtime, 1),
        "env": _env(args),
    }
    if args.dry_run:
        result["dry_run"] = prompt_report(backend("dry"), args.prompts_out)
    else:
        result.update(evaluate(table, preset, outs, args))
    out = Path(args.out) if args.out else default_out(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str))
    _print_summary(result, out)
    return result


def _env(args) -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        import transformers

        from experiments.baselines.common import pick_device

        env.update(
            torch=torch.__version__, transformers=transformers.__version__, device=pick_device(args.device)
        )
    except ImportError:
        pass
    return env


def _print_summary(result: dict, out: Path) -> None:
    if "dry_run" in result:
        r = result["dry_run"]
        kinds = ", ".join(f"{k}: {v['n']} (~{v['tokens_mean']:.0f} tok)" for k, v in r["by_kind"].items())
        print(f"dry run: {r['n_prompts']} prompts ({kinds}); templated={r['templated']} -> {out}")
        return
    for metric, s in result["summary"].items():
        if isinstance(s, dict) and "value" in s:
            ci = f" [{s['ci_low']:.3f}, {s['ci_high']:.3f}]" if "ci_low" in s else ""
            n = f" (n={s['n_decisions']})" if "n_decisions" in s else ""
            print(f"  {metric}: {s['value']:.4f}{ci}{n}")
    for sl, vals in result["slices"].items():
        print(f"  {sl}: " + ", ".join(f"{m} {v:.4f}" for m, v in vals.items()))
    print(f"  paper (Table 1): {result['paper']}  ({result['runtime_s']}s) -> {out}")


if __name__ == "__main__":
    main()
