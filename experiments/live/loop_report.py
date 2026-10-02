"""Compare τ²-bench runs of two arms on the same tasks (see ``tau2_loop``): task success paired by task, agent
prompt tokens per call (and the share the provider served from its prompt cache), LLM cost, wall time and
the TabAgent head's per-decision service latency.

    python -m experiments.live.loop_report --arm baseline=RUN_A.json [RUN_A2.json ...] \
        --arm tabagent=RUN_B.json [RUN_B2.json ...] [--out report.json]

The first arm is the reference: paired differences are (other arm - reference), with 95% bootstrap CIs over
tasks (each task's mean over its trials, so trials of one task never count as independent). Simulations that
ended in an infrastructure error (a provider error, say) are excluded from both arms of that task-trial.
Costs are recomputed from each call's tokens at the given prices (``--price``, USD per 1M input, cached
input and output tokens; GPT-4.1 list prices by default) and also read from τ²-bench's own ``cost`` fields.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

GPT41 = (2.0, 0.5, 8.0)  # USD per 1M input, cached input, output tokens


def _sims(paths: list[str]) -> dict[tuple[str, str, int], dict]:
    out = {}
    for p in paths:
        data = json.loads(Path(p).read_text())
        domain = (data.get("info") or {}).get("environment_info", {}).get("domain_name", "")
        for s in data["simulations"]:
            out[(domain, str(s["task_id"]), int(s.get("trial") or 0))] = s
    return out


def _calls(sim: dict) -> list[dict]:
    """The agent's LLM calls (assistant messages with usage; the scripted greeting has none)."""
    return [m for m in sim.get("messages") or [] if m.get("role") == "assistant" and m.get("usage")]


def _cached(m: dict) -> int:
    u = (m.get("raw_data") or {}).get("usage") or {}
    return int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)


def sim_stats(sim: dict, price: tuple[float, float, float]) -> dict[str, Any]:
    calls = _calls(sim)
    prompt = sum(m["usage"].get("prompt_tokens", 0) for m in calls)
    cached = sum(_cached(m) for m in calls)
    completion = sum(m["usage"].get("completion_tokens", 0) for m in calls)
    pin, pcache, pout = price
    users = [m for m in sim.get("messages") or [] if m.get("role") == "user" and m.get("usage")]
    u_prompt = sum(m["usage"].get("prompt_tokens", 0) for m in users)
    u_cached = sum(_cached(m) for m in users)
    u_completion = sum(m["usage"].get("completion_tokens", 0) for m in users)
    filt = [(m.get("raw_data") or {}).get("tabagent_filter") for m in calls]
    filt = [f for f in filt if f]
    return {
        "reward": float((sim.get("reward_info") or {}).get("reward") or 0.0),
        "calls": len(calls),
        "prompt": prompt,
        "cached": cached,
        "completion": completion,
        "agent_cost": ((prompt - cached) * pin + cached * pcache + completion * pout) / 1e6,
        "user_cost": ((u_prompt - u_cached) * pin + u_cached * pcache + u_completion * pout) / 1e6,
        "agent_cost_tau2": float(sim.get("agent_cost") or 0.0),
        "user_cost_tau2": float(sim.get("user_cost") or 0.0),
        "duration_s": float(sim.get("duration") or 0.0),
        "llm_s": sum(float(m.get("generation_time_seconds") or 0.0) for m in calls),
        "decisions": filt,
        "termination": sim.get("termination_reason"),
    }


def _boot(diffs: np.ndarray, reps: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(reps, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(arms: dict[str, list[str]], price: tuple[float, float, float] = GPT41) -> dict[str, Any]:
    names = list(arms)
    runs = {a: _sims(paths) for a, paths in arms.items()}
    infra = {
        a: sorted(k for k, s in r.items() if s.get("termination_reason") == "infrastructure_error")
        for a, r in runs.items()
    }
    keys = set.intersection(*(set(r) for r in runs.values()))
    bad = {k for a in names for k in infra[a]}
    keys = sorted(keys - bad)
    stats = {a: {k: sim_stats(runs[a][k], price) for k in keys} for a in names}
    tasks = sorted({k[:2] for k in keys})

    def per_task(a: str, field: str) -> np.ndarray:
        by: dict[tuple, list[float]] = defaultdict(list)
        for k in keys:
            by[k[:2]].append(stats[a][k][field])
        return np.array([np.mean(by[t]) for t in tasks])

    out: dict[str, Any] = {
        "n_task_trials": len(keys),
        "n_tasks": len(tasks),
        "excluded_infra_errors": {a: [list(k) for k in infra[a]] for a in names},
        "price_per_1M": dict(zip(("input", "cached_input", "output"), price, strict=True)),
        "arms": {},
        "paired_vs_" + names[0]: {},
    }
    for a in names:
        s = list(stats[a].values())
        calls = sum(x["calls"] for x in s)
        dec = [d for x in s for d in x["decisions"]]
        ms = [d["ms"] for d in dec]
        arm = {
            "pass_rate": float(np.mean([x["reward"] == 1.0 for x in s])),
            "pass_rate_by_domain": {
                d: float(np.mean([stats[a][k]["reward"] == 1.0 for k in keys if k[0] == d]))
                for d in sorted({k[0] for k in keys})
            },
            "agent_calls_per_sim": calls / len(s),
            "agent_prompt_tokens_per_call": sum(x["prompt"] for x in s) / calls,
            "agent_uncached_prompt_tokens_per_call": sum(x["prompt"] - x["cached"] for x in s) / calls,
            "agent_cached_share": sum(x["cached"] for x in s) / max(1, sum(x["prompt"] for x in s)),
            "agent_completion_tokens_per_call": sum(x["completion"] for x in s) / calls,
            "agent_cost_per_sim": float(np.mean([x["agent_cost"] for x in s])),
            "user_cost_per_sim": float(np.mean([x["user_cost"] for x in s])),
            "total_llm_cost": float(sum(x["agent_cost"] + x["user_cost"] for x in s)),
            "total_llm_cost_tau2_fields": float(sum(x["agent_cost_tau2"] + x["user_cost_tau2"] for x in s)),
            "sim_wall_s_mean": float(np.mean([x["duration_s"] for x in s])),
            "agent_llm_s_per_call": sum(x["llm_s"] for x in s) / calls,
            "terminations": dict(
                sorted(
                    {t: sum(x["termination"] == t for x in s) for t in {x["termination"] for x in s}}.items()
                )
            ),
        }
        if dec:
            arm["head"] = {
                "decisions": len(dec),
                "service_ms_p50": float(np.percentile(ms, 50)),
                "service_ms_p95": float(np.percentile(ms, 95)),
                "deferred_share": float(np.mean([d["deferred"] for d in dec])),
                "errors": sorted({(d["error"] or "")[:120] for d in dec if d["error"]})[:5],
                "tools_shown_per_call": float(np.mean([len(d["shown"]) for d in dec])),
                "catalog_size": float(np.mean([d["n_tools"] for d in dec])),
                "kept_by_safety_rule_per_call": float(np.mean([len(d["kept_as_used"]) for d in dec])),
                "calls_to_unshown_tools": int(sum(len(d.get("called_unshown") or []) for d in dec)),
            }
        out["arms"][a] = arm
    ref = names[0]
    for a in names[1:]:
        pair: dict[str, Any] = {}
        for field, label in (
            ("reward", "pass_rate_diff"),
            ("agent_cost", "agent_cost_per_sim_diff"),
            ("prompt", "agent_prompt_tokens_per_sim_diff"),
            ("calls", "agent_calls_per_sim_diff"),
            ("duration_s", "sim_wall_s_diff"),
        ):
            mean, lo, hi = _boot(per_task(a, field) - per_task(ref, field))
            pair[label] = {"mean": mean, "ci95": [lo, hi]}
        pt = {a2: per_task(a2, "reward") for a2 in (ref, a)}
        pair["tasks_better"] = int(np.sum(pt[a] > pt[ref]))
        pair["tasks_worse"] = int(np.sum(pt[a] < pt[ref]))
        pair["tasks_same"] = int(np.sum(pt[a] == pt[ref]))
        per_call = {}
        for b in (ref, a):
            s = list(stats[b].values())
            calls = sum(x["calls"] for x in s)
            per_call[b] = sum(x["prompt"] for x in s) / calls
        pair["prompt_tokens_per_call_change"] = per_call[a] / per_call[ref] - 1
        out["paired_vs_" + ref][a] = pair
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--arm",
        action="append",
        nargs="+",
        required=True,
        metavar="NAME=RESULTS",
        help="an arm's name and its τ²-bench results.json files (the first --arm is the reference)",
    )
    ap.add_argument("--price", nargs=3, type=float, default=GPT41, metavar=("IN", "CACHED", "OUT"))
    ap.add_argument("--out", help="write the report as JSON")
    args = ap.parse_args(argv)
    arms: dict[str, list[str]] = {}
    for spec in args.arm:
        name, _, first = spec[0].partition("=")
        arms[name] = [first, *spec[1:]]
    rep = report(arms, tuple(args.price))
    text = json.dumps(rep, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
