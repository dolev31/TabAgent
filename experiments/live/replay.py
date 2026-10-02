"""Live replay: call an onboarded head as an agent scaffold would, on logged runs, through every serving path.

For each successful run of the held-out tasks and each point where the agent is about to call a tool, the
scaffold's state as it existed then (the run-level fields and the messages before that assistant turn) is
sent to the head through

* ``TabAgentHead.decide(run=...)`` (in-process),
* the HTTP service (``tabagent serve``; ``create_app`` with a test client) with ``{"run": ...}``,
* ``integrations.openai_tools.ToolFilter`` (the messages, the tool schemas and the run-level fields),

and checked against the training-time path (``adapter.decision_points`` prefixes, the extractor run as in
``build_table``, the row scored as in training): prefix, feature and score parity, latency per decision
(in-process extractor and sandboxed extractor), deferral and its reasons, and the offline quality of the live
decisions (is the tool the agent called in the top k; with the safety rule that keeps every tool already used
in the conversation). ``--robustness`` perturbs live states (an unknown tool, a missing run-level field, a
very long history, non-ASCII text, empty tool results, ...) and checks every path answers or defers, never
crashes.

    python -m experiments.live.replay --logs DIR --head HEAD_DIR --split split.json --out replay.json \
        --robustness

The head should be fitted on the split's train tasks only (``experiments.live.refit``) for clean held-out
numbers. Outputs (per-decision records and the summary) go to ``--out``; keep them outside the repository.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tabagent.adapters import Tau2Adapter, adapter_from_description, parse_catalog
from tabagent.api import TabAgentHead
from tabagent.integrations.openai_tools import ToolFilter
from tabagent.tabschema.compile import ExtractorRunner
from tabagent.tabschema.orchestrator import rows_to_table

from .common import load_split, scaffold_states, split_tasks

LONG_CHARS = 200_000


# --------------------------------------------------------------------------- helpers
def _same(a: Any, b: Any, tol: float = 1e-9) -> bool:
    """Deep equality with a float tolerance (NaN equals NaN)."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return False
        return (math.isnan(fa) and math.isnan(fb)) or abs(fa - fb) <= tol
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b, strict=True))
    return a == b


def _diff(a: dict, b: dict) -> dict[str, tuple[Any, Any]]:
    keys = sorted(set(a) | set(b))
    return {k: (a.get(k), b.get(k)) for k in keys if not _same(a.get(k), b.get(k))}


def _cand_json(c) -> dict[str, str]:
    return {"id": c.id, "name": c.name, "description": c.description, "args": c.arg_hints}


def _used(prefix: dict) -> list[str]:
    """The tools the agent already called in the prefix (the safety rule keeps them)."""
    steps = prefix.get("steps") or []
    return list(
        dict.fromkeys(t for s in steps if s.get("name") == "assistant" for t in s.get("tool_calls") or [])
    )


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def _forget_contexts() -> None:
    """Drop the memoized context embeddings, so each timed path embeds the decision's new context, as a
    live decision does (the candidates' texts stay memoized: they repeat across decisions)."""
    from tabagent.tabschema import fitted

    for enc in fitted._SHARED.values():
        with enc._lock:
            for key in [k for k in enc.memo if k[0] == "query: "]:
                del enc.memo[key]


def _client(head):
    from fastapi.testclient import TestClient

    from tabagent.serve import create_app

    return TestClient(create_app(head))


def _post(client, body: dict) -> tuple[int, dict]:
    # ensure_ascii keeps lone surrogates (truncated UTF-16 text) encodable, as a JSON client must send them
    r = client.post(
        "/decide", content=json.dumps(body, ensure_ascii=True), headers={"content-type": "application/json"}
    )
    return r.status_code, r.json()


# --------------------------------------------------------------------------- replay
def replay(
    head: TabAgentHead,
    adapter: Any,
    trajs: list,
    target: str,
    k: int,
    gate: float | None = None,
    sandboxed: TabAgentHead | None = None,
    sandbox_every: int = 10,
) -> list[dict]:
    """One record per live decision of ``trajs`` (see the module docstring)."""
    runner = ExtractorRunner(
        head.extractor_code, trust_extractor=head.trust_extractor, sandbox=head._sandbox()
    )
    client = _client(head)
    warm = next((s for t in trajs for s in scaffold_states(adapter, t, target)), None)
    for h in (head, sandboxed):
        if warm is not None and h is not None:  # the first decision loads the text encoder: not timed
            h.decide(run=warm[1], candidates=warm[1].get("tools") or [])
    tf_cache: dict[str, ToolFilter] = {}
    records: list[dict] = []
    n = 0
    for traj in trajs:
        states = scaffold_states(adapter, traj, target)
        if not states:
            continue
        train_feats = runner.run([dp.prefix for dp, _ in states])  # as build_table extracts them
        for (dp, live), tfeat in zip(states, train_feats, strict=True):
            tools = live.get("tools") or []
            cands = list(parse_catalog(tools).values())
            ids = [c.id for c in cands]
            labels = sorted(set(adapter.labels(dp)) & set(ids))
            rec: dict[str, Any] = {"run": traj.id, "group": traj.group, "decision": dp.id, "labels": labels}

            # (a) parity: prefix, features and scores of the live path vs the training path
            lp = head.live_prefix(live)
            rec["prefix_equal"] = lp == dp.prefix
            lfeat, err = head._extract(lp)
            cols = [f.name for f in head.card.features if f.source == "extractor"]
            tfeat_cols = {c: (tfeat or {}).get(c) for c in cols}
            rec["features_equal"] = _same(lfeat, tfeat_cols)
            if not rec["features_equal"]:
                rec["feature_diff"] = {c: list(map(str, v)) for c, v in _diff(lfeat, tfeat_cols).items()}

            _forget_contexts()
            t0 = time.perf_counter()
            out = head.decide(run=live, candidates=cands)
            rec["ms_decide"] = 1e3 * (time.perf_counter() - t0)
            table = rows_to_table(head.card, [dp], [tfeat], adapter)
            train_scores = dict(zip(table.df["candidate"], head.agent.score(table), strict=True))
            rec["scores_equal"] = _same(out["scores"], {c: float(s) for c, s in train_scores.items()})
            rec["max_score_diff"] = max(abs(out["scores"][c] - train_scores[c]) for c in ids)

            # the HTTP service, with the run (the service builds the prefix with the head's adapter)
            _forget_contexts()
            t0 = time.perf_counter()
            # the tool schemas exactly as the scaffold holds them (OpenAI function tools for τ²-bench)
            status, http = _post(client, {"run": live, "candidates": tools})
            rec["ms_http"] = 1e3 * (time.perf_counter() - t0)
            rec["http_equal"] = (
                status == 200
                and http["candidates"] == out["candidates"]
                and _same(http["scores"], out["scores"])
            )

            # the tool filter, with the messages, the tool schemas and the run-level fields
            fields = {key: v for key, v in live.items() if key not in ("messages", "tools")}
            key = json.dumps(fields, sort_keys=True, default=str)
            tf = tf_cache.setdefault(key, ToolFilter(head, k=k, task=fields))
            _forget_contexts()
            t0 = time.perf_counter()
            kept = tf.select(live["messages"], tools)
            rec["ms_toolfilter"] = 1e3 * (time.perf_counter() - t0)
            expect = ids if out["deferred"] else out["candidates"][:k]
            rec["toolfilter_equal"] = kept == expect

            if sandboxed is not None and n % sandbox_every == 0:
                _forget_contexts()
                t0 = time.perf_counter()
                sb = sandboxed.decide(run=live, candidates=cands)
                rec["ms_decide_sandboxed"] = 1e3 * (time.perf_counter() - t0)
                rec["sandboxed_equal"] = sb["candidates"] == out["candidates"] and _same(
                    sb["scores"], out["scores"]
                )
            n += 1

            rec.update(
                ranking=out["candidates"],
                confidence=out["confidence"],
                deferred=out["deferred"],
                error=out["error"],
                gated=gate is not None and out["confidence"] < gate,
                used=_used(lp),
                n_candidates=len(ids),
            )
            records.append(rec)
    return records


# --------------------------------------------------------------------------- quality
def quality(records: list[dict], ks: list[int]) -> dict[str, Any]:
    """Offline quality of the live decisions: Recall@k, the safety-rule recall (top k plus the tools already
    used), the share of runs whose every tool call was covered, and P@R."""
    rows = [r for r in records if r["labels"]]
    out: dict[str, Any] = {"n_decisions": len(rows), "n_without_label_in_catalog": len(records) - len(rows)}
    if not rows:
        return out
    out["p@r"] = float(
        np.mean(
            [len(set(r["ranking"][: len(r["labels"])]) & set(r["labels"])) / len(r["labels"]) for r in rows]
        )
    )
    by_run: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)
    for k in ks:
        rec = [len(set(r["ranking"][:k]) & set(r["labels"])) / len(r["labels"]) for r in rows]
        safe = [
            len((set(r["ranking"][:k]) | set(r["used"])) & set(r["labels"])) / len(r["labels"]) for r in rows
        ]
        shown = [len(set(r["ranking"][:k]) | set(r["used"])) for r in rows]
        runs = [
            all(set(r["labels"]) <= set(r["ranking"][:k]) | set(r["used"]) for r in rs)
            for rs in by_run.values()
        ]
        out[f"k={k}"] = {
            "recall": round(float(np.mean(rec)), 4),
            "recall_with_used": round(float(np.mean(safe)), 4),
            "runs_fully_covered": round(float(np.mean(runs)), 4),
            "tools_shown_mean": round(float(np.mean(shown)), 2),
            "catalog_mean": round(float(np.mean([r["n_candidates"] for r in rows])), 2),
        }
    return out


def summarize(records: list[dict], ks: list[int]) -> dict[str, Any]:
    n = len(records)

    def share(key: str) -> float | None:
        xs = [r[key] for r in records if key in r]
        return round(100 * float(np.mean(xs)), 2) if xs else None

    lat = {}
    for key in ("ms_decide", "ms_http", "ms_toolfilter", "ms_decide_sandboxed"):
        xs = [r[key] for r in records if key in r]
        if xs:
            lat[key] = {"n": len(xs), "p50": round(_pct(xs, 50), 2), "p95": round(_pct(xs, 95), 2)}
    errors = Counter((r["error"] or "").strip().splitlines()[-1][:160] for r in records if r["error"])
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_domain[str(r["group"]).split(":")[0]].append(r)
    return {
        "n_decisions": n,
        "n_runs": len({r["run"] for r in records}),
        "parity_pct": {
            "prefix": share("prefix_equal"),
            "features": share("features_equal"),
            "scores_vs_training_row": share("scores_equal"),
            "http_vs_decide": share("http_equal"),
            "toolfilter_vs_decide": share("toolfilter_equal"),
            "sandboxed_vs_in_process": share("sandboxed_equal"),
        },
        "max_score_diff": max((r["max_score_diff"] for r in records), default=None),
        "latency_ms": lat,
        "deferral": {
            "deferred_pct": share("deferred"),
            "extractor_errors": dict(errors),
            "gated_pct": share("gated"),
        },
        "quality": quality(records, ks),
        "quality_by_domain": {d: quality(rs, ks) for d, rs in sorted(by_domain.items())},
    }


# --------------------------------------------------------------------------- robustness
def _perturbations(live: dict) -> dict[str, tuple[dict, list | None]]:
    """name -> (perturbed run, extra candidates or None)."""
    msgs = live["messages"]
    tools = live.get("tools") or []
    weird = "Je voudrais annuler 🛫 予約をキャンセル אני רוצה לבטל \u200f\u0000 Ω≈ç√ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢"
    unknown_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_x", "name": "frobnicate_widget", "arguments": {"q": 1}, "requestor": "assistant"}
        ],
    }
    unknown_result = {"id": "call_x", "role": "tool", "content": "Error: Tool 'frobnicate_widget' not found"}
    new_tool = {
        "type": "function",
        "function": {
            "name": "frobnicate_widget",
            "description": "A tool the head never saw.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }

    def edit(fn) -> dict:
        r = copy.deepcopy(live)
        r["messages"] = [fn(copy.deepcopy(m)) for m in r["messages"]]
        return r

    def non_ascii(m):
        if m.get("role") == "user":
            m["content"] = f"{m.get('content') or ''} {weird}"
        return m

    def lone_surrogate(m):  # UTF-16 text cut inside a surrogate pair (a JS client truncating an emoji)
        if m.get("role") == "user":
            m["content"] = f"{m.get('content') or ''} \ud83d"
        return m

    def empty_result(m):
        if m.get("role") == "tool":
            m["content"] = ""
        return m

    def none_result(m):
        if m.get("role") == "tool":
            m["content"] = None
        return m

    def odd_shapes(m):
        if m.get("role") == "user":
            m["content"] = [
                {"type": "text", "text": m.get("content") or ""},
                {"type": "image_url", "image_url": {}},
            ]
        for tc in m.get("tool_calls") or []:
            tc["arguments"] = "{not json"
        return m

    no_domain = {key: v for key, v in live.items() if key != "domain"}
    no_tools = {key: v for key, v in live.items() if key != "tools"}
    long = copy.deepcopy(live)
    long["messages"] = msgs * 40 + [{"role": "user", "content": "x " * (LONG_CHARS // 2)}]
    return {
        "unknown_tool_in_history": ({**live, "messages": [*msgs, unknown_call, unknown_result]}, None),
        "unknown_candidate": (live, [*tools, new_tool]),
        "missing_run_level_field": (no_domain, None),
        "missing_tools_field": (no_tools, tools),
        "very_long_history": (long, None),
        "non_ascii": (edit(non_ascii), None),
        "lone_surrogate": (edit(lone_surrogate), None),
        "empty_tool_results": (edit(empty_result), None),
        "null_tool_results": (edit(none_result), None),
        "block_content_and_bad_arguments": (edit(odd_shapes), None),
        "empty_conversation": ({**live, "messages": []}, None),
        "system_prompt_in_messages": (
            {**live, "messages": [{"role": "system", "content": "You are a helpful agent."}, *msgs]},
            None,
        ),
        "only_run_fields": ({key: v for key, v in live.items() if key != "messages"}, None),
    }


def robustness(heads: dict[str, TabAgentHead], states: list[dict], k: int) -> dict[str, Any]:
    """Each perturbation of each live state through every path of every head: answered, deferred (with the
    reason) or crashed (the exception)."""
    out: dict[str, Any] = {}
    clients = {name: _client(h) for name, h in heads.items()}
    for live in states:
        for pname, (run, extra) in _perturbations(live).items():
            tools = extra if extra is not None else (run.get("tools") or [])
            cands = list(parse_catalog(tools).values())
            fields = {key: v for key, v in run.items() if key not in ("messages", "tools")}
            for hname, head in heads.items():
                for path in ("decide", "http", "toolfilter"):
                    t0 = time.perf_counter()
                    try:
                        if path == "decide":
                            res = head.decide(run=run, candidates=cands, k=k)
                        elif path == "http":
                            status, res = _post(
                                clients[hname],
                                {"run": run, "candidates": [_cand_json(c) for c in cands], "k": k},
                            )
                            if status != 200:
                                raise RuntimeError(f"HTTP {status}: {str(res)[:200]}")
                        else:
                            tf = ToolFilter(head, k=k, task=fields)
                            names = tf.select(run.get("messages") or [], tools)
                            res = {**tf.last, "kept": len(names)}
                        outcome = "deferred" if res["deferred"] else "answered"
                        detail = (
                            (res.get("error") or "").strip().splitlines()[-1][:200]
                            if res.get("error")
                            else ""
                        )
                    except Exception as e:  # a crash is the finding
                        outcome, detail = "crashed", f"{type(e).__name__}: {str(e)[:200]}"
                    ms = 1e3 * (time.perf_counter() - t0)
                    p = out.setdefault(pname, {"outcomes": {}, "details": set(), "max_ms": {}})
                    name = f"{hname}/{path}"
                    p["outcomes"].setdefault(name, Counter())[outcome] += 1
                    if detail:
                        p["details"].add(f"{name}: {outcome}: {detail}")
                    p["max_ms"][name] = round(max(ms, p["max_ms"].get(name, 0.0)), 1)
    for p in out.values():
        p["details"] = sorted(p["details"])[:12]
        p["outcomes"] = {name: dict(c) for name, c in p["outcomes"].items()}
    return out


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs", required=True, help="τ²-bench results files (a file or a directory)")
    ap.add_argument("--head", required=True, help="the onboarded head directory")
    ap.add_argument("--split", help="a split JSON ({train, test} task groups); default: a seeded 70/30 split")
    ap.add_argument("--target", help="the decision (default: the head's)")
    ap.add_argument("--k", type=int, default=5, help="the tool filter's k")
    ap.add_argument("--ks", default="1,2,3,4,5,6,7,8,9,10", help="k values for the quality table")
    ap.add_argument("--max-runs", type=int, help="replay at most this many held-out runs")
    ap.add_argument(
        "--sandbox-every", type=int, default=10, help="time the sandboxed path on every n-th decision"
    )
    ap.add_argument("--robustness", type=int, default=0, metavar="N", help="perturb N live states")
    ap.add_argument("--out", required=True, help="output JSON (records and summary)")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    head = TabAgentHead.load(args.head, trust_extractor=True)
    sandboxed = TabAgentHead.load(args.head)  # the default: the extractor in the subprocess sandbox
    target = args.target or head.target
    adapter = adapter_from_description(head.adapter) or Tau2Adapter()
    trajs = adapter.load(args.logs)
    split = load_split(args.split) if args.split else split_tasks(trajs)
    test = set(split["test"])
    runs = [t for t in trajs if t.success and t.group in test][: args.max_runs]
    gates_file = Path(args.head) / "gates.json"
    gate = json.loads(gates_file.read_text()).get("0.9") if gates_file.exists() else None
    ks = [int(x) for x in args.ks.split(",")]

    t0 = time.time()
    records = replay(head, adapter, runs, target, args.k, gate, sandboxed, args.sandbox_every)
    summary = summarize(records, ks)
    summary["target"], summary["head"], summary["minutes"] = (
        target,
        str(args.head),
        round((time.time() - t0) / 60, 1),
    )
    summary["gate_threshold_0.9"] = gate
    if args.robustness:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(records), size=min(args.robustness, len(records)), replace=False)
        by_id = {
            dp.id: live
            for t in runs
            for dp, live in scaffold_states(adapter, t, target)
            if dp.id in {records[i]["decision"] for i in pick}
        }
        summary["robustness"] = robustness(
            {"in_process": head, "sandboxed": sandboxed}, list(by_id.values()), args.k
        )
    Path(args.out).write_text(json.dumps({"summary": summary, "records": records}, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
