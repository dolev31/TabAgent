"""Serving latency of an onboarded head: sandboxed extraction before and after the warm workers, and the HTTP
service under concurrent load.

An onboarded head decides in three stages: build the prefix, run the TabSchema extractor on it (sandboxed
unless trusted), and score the candidates. ``extract`` times the extraction stage alone and one whole
``TabAgentHead.decide`` call per extractor mode:

subprocess  ``SubprocessSandbox``: a fresh ``python -I -S`` interpreter per decision (the serving default
            before the warm pool, and still the onboarding default)
warm        ``WarmSandbox``: persistent isolated interpreters with the extractor compiled once (the serving
            default now)
in-process  ``trust_extractor=True``: the reviewed extractor runs in the serving process

``load`` starts ``tabagent serve`` on the head in a subprocess, waits for ``/ready``, and fires ``POST
/decide`` requests from ``--concurrency`` client threads (one keep-alive connection each). It reports the
throughput, the client-side p50/p95/p99 latency, the deferrals and errors, and the server's mean time per
stage from ``/metrics``, then stops the server with SIGTERM.

Both use a synthetic but representative head unless ``--head`` names a saved one (then pass ``--prefixes``, a
JSONL file of live prefixes, and ``--candidates``, a JSON list): a 30-tool shortlisting decision over chat
prefixes of 2 to 16 steps (about 2 to 6 KB of JSON each), a stdlib extractor of the kind TabSchema writes
(step counts, the last user message, tool calls, parsed tool results, keywords; about 60 lines with the
appended context text), and a CatBoost head with the paper's preset over 16 card features, among them a
lexical overlap and a fitted candidate prior.

    python -m experiments.efficiency.serving extract --n 300
    python -m experiments.efficiency.serving load --modes warm subprocess in-process --concurrency 1 8 32
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tabagent import TabAgent, TabHead
from tabagent.api import TabAgentHead
from tabagent.table import DecisionSpec, DecisionTable
from tabagent.tabschema import Feature, FeatureCard
from tabagent.tabschema.compile import ExtractorRunner, context_text_code
from tabagent.tabschema.orchestrator import CANDIDATE, DECISION, GROUP, LABEL
from tabagent.tabschema.sandbox import SubprocessSandbox
from tabagent.tabschema.warm import WarmSandbox

OUT = Path("results/efficiency")

EXTRACTOR = """
import json
import re
from collections import Counter


def extract_features_for_classification(trajectory_data):
    steps = trajectory_data['steps']
    users = [s['content'] for s in steps if s['name'] == 'user']
    calls = [c for s in steps if s['name'] == 'assistant' for c in (s['tool_calls'] or [])]
    results = [s['content'] for s in steps if s['name'] == 'tool']
    last_user = users[-1] if users else ''
    words = re.findall(r"[a-z]+", " ".join(users).lower())
    counts = Counter(w for w in words if len(w) > 3)
    parsed, failed = 0, 0
    for text in results:
        try:
            payload = json.loads(text)
        except ValueError:
            continue
        parsed += 1
        if isinstance(payload, dict) and 'error' in payload and payload['error']:
            failed += 1
    return {
        "domain": trajectory_data['domain'],
        "goal": trajectory_data['instruction'],
        "last_user_message": last_user[:500],
        "n_steps": len(steps),
        "n_user_turns": len(users),
        "n_tool_calls": len(calls),
        "n_tool_results": parsed,
        "n_tool_errors": failed,
        "last_tool_called": calls[-1] if calls else "none",
        "mentions_numbers": len(re.findall(r"\\b\\d+\\b", " ".join(users))),
        "top_keyword": counts.most_common(1)[0][0] if counts else "",
        "asks_question": int('?' in last_user),
    }


features = extract_features_for_classification(trajectory)
""" + context_text_code("context_text")

TOPICS = {
    "flights": ("search_flights", "book_flight", "cancel_flight", "seat_map", "baggage_rules"),
    "hotels": ("search_hotels", "book_hotel", "cancel_hotel", "hotel_reviews"),
    "mail": ("read_inbox", "send_email", "search_email", "draft_reply"),
    "calendar": ("list_events", "create_event", "move_event", "find_free_slot"),
    "music": ("play_song", "search_music", "create_playlist"),
    "notes": ("create_note", "search_notes", "share_note"),
    "billing": ("get_invoice", "refund_payment", "update_card", "billing_history"),
    "support": ("open_ticket", "escalate_ticket", "ticket_status"),
}
TOOLS = {
    t: f"{t.replace('_', ' ')}: {topic} tool for the user's {topic} requests"
    for topic, ts in TOPICS.items()
    for t in ts
}
CITIES = ("paris", "rome", "tokyo", "lima", "oslo", "cairo", "delhi", "quito", "hanoi", "dakar")


def make_prefix(rng: random.Random) -> tuple[dict, list[str]]:
    """A chat prefix before a tool-shortlisting decision, and the tools the agent then used."""
    topic = rng.choice(list(TOPICS))
    used = rng.sample(TOPICS[topic], k=min(2, len(TOPICS[topic])))
    city = rng.choice(CITIES)
    first, then = used[0].replace("_", " "), used[-1].replace("_", " ")
    goal = f"help me with my {topic} in {city}: {first} and then {then}"
    steps = [
        {"name": "system", "content": "You are a helpful assistant. " * 20, "tool_calls": [], "tool_name": ""}
    ]
    for turn in range(rng.randint(1, 5)):
        ask = (
            f"I need to {rng.choice(TOPICS[topic]).replace('_', ' ')} for {city} on day {rng.randint(1, 28)}"
        )
        steps.append(
            {"name": "user", "content": ask + ("?" if turn % 2 else "."), "tool_calls": [], "tool_name": ""}
        )
        if rng.random() < 0.7:
            tool = rng.choice(TOPICS[rng.choice(list(TOPICS))])
            steps.append({"name": "assistant", "content": "", "tool_calls": [tool], "tool_name": ""})
            result = {
                "ok": True,
                "items": [
                    {"id": j, "city": city, "price": rng.randint(50, 900)} for j in range(rng.randint(1, 6))
                ],
            }
            if rng.random() < 0.15:
                result = {"error": "upstream timeout"}
            steps.append({"name": "tool", "content": json.dumps(result), "tool_calls": [], "tool_name": tool})
        steps.append(
            {
                "name": "assistant",
                "content": f"Sure, let me look at {topic} options in {city}.",
                "tool_calls": [],
                "tool_name": "",
            }
        )
    return {"instruction": goal, "domain": topic, "steps": steps}, used


def build_head(out: Path, n_decisions: int = 400, seed: int = 0, iterations: int | None = None) -> Path:
    """Fit and save the synthetic head (see the module docstring); ``iterations`` overrides the preset's."""
    rng = random.Random(seed)
    data = [make_prefix(rng) for _ in range(n_decisions)]
    feats = ExtractorRunner(EXTRACTOR, trust_extractor=True).run([p for p, _ in data])
    spec = DecisionSpec("ranking", GROUP, LABEL, CANDIDATE, decision=(DECISION,))
    rows = []
    for i, ((_, used), f) in enumerate(zip(data, feats, strict=True)):
        for tool, desc in TOOLS.items():
            rows.append(
                {
                    GROUP: f"t{i}",
                    DECISION: f"t{i}#0",
                    **f,
                    CANDIDATE: tool,
                    "candidate_name": tool,
                    "candidate_description": desc,
                    "candidate_args": "",
                    LABEL: int(tool in used),
                }
            )
    extractor = {
        "domain": "cat",
        "goal": "text",
        "last_user_message": "text",
        "n_steps": "num",
        "n_user_turns": "num",
        "n_tool_calls": "num",
        "n_tool_results": "num",
        "n_tool_errors": "num",
        "last_tool_called": "cat",
        "mentions_numbers": "num",
        "top_keyword": "cat",
        "asks_question": "num",
        "context_text": "text",
    }
    card = FeatureCard(
        "serving_benchmark",
        spec,
        [Feature(name, "state", t, "extractor") for name, t in extractor.items()]
        + [
            Feature("candidate_name", "schema", "cat"),
            Feature("candidate_description", "schema", "text"),
            Feature(
                "lex",
                "schema",
                "num",
                "jaccard",
                {"a": "goal", "b": ["candidate_name", "candidate_description"]},
            ),
            Feature("prior", "dependency", "num", "fitted:candidate_prior", {"fill": "zero"}),
        ],
    )
    agent = TabAgent(
        card, head=TabHead("ranking", preset="paper", **({"iterations": iterations} if iterations else {}))
    ).fit(DecisionTable(pd.DataFrame(rows), spec))
    TabAgentHead(agent, "ToolShortlist", EXTRACTOR).save(out)
    return out


def live_prefixes(n: int, seed: int = 1) -> list[dict]:
    rng = random.Random(seed)
    return [make_prefix(rng)[0] for _ in range(n)]


def _percentiles(ms: list[float]) -> dict[str, float]:
    a = np.asarray(ms)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


def _row(name: str, s: dict) -> str:
    return f"| {name} | {s['n']} | {s['p50']:.2f} | {s['p95']:.2f} | {s['p99']:.2f} | {s['mean']:.2f} |"


# --------------------------------------------------------------------------- extraction before / after
def run_extract(args) -> dict:
    head_dir = (
        Path(args.head)
        if args.head
        else build_head(Path(tempfile.mkdtemp(prefix="tabagent_bench_")) / "head")
    )
    code = (head_dir / "extract.py").read_text()
    prefixes = _load_prefixes(args) or live_prefixes(args.n)
    candidates = _load_candidates(args, head_dir)
    sizes = [len(json.dumps(p)) for p in prefixes]
    out = {
        "prefix_bytes": {"mean": float(np.mean(sizes)), "max": int(max(sizes))},
        "extraction": {},
        "decide": {},
    }

    def timed(fn, items, warmup=5):
        for p in items[:warmup]:
            fn(p)
        ms = []
        for p in items:
            t = time.perf_counter()
            fn(p)
            ms.append((time.perf_counter() - t) * 1000)
        return _percentiles(ms)

    cold = SubprocessSandbox()
    warm = WarmSandbox(size=1)
    t = time.perf_counter()
    warm.warm(code)
    out["warm_start_ms"] = (time.perf_counter() - t) * 1000
    trusted = ExtractorRunner(code, trust_extractor=True)
    out["extraction"]["subprocess"] = timed(lambda p: cold.run(code, p), prefixes)
    out["extraction"]["warm"] = timed(lambda p: warm.run(code, p), prefixes)
    out["extraction"]["in-process"] = timed(lambda p: trusted.run([p]), prefixes)
    warm.close()

    for mode, kw in (
        ("subprocess", {"sandbox": "subprocess"}),
        ("warm", {"sandbox": "warm:1"}),
        ("in-process", {"trust_extractor": True}),
    ):
        head = TabAgentHead.load(head_dir, **kw)
        out["decide"][mode] = timed(lambda p, h=head: h.decide(p, candidates, k=5), prefixes)
        head.close()
    out["load_average"] = os.getloadavg()
    print(
        f"prefixes: {len(prefixes)}, mean {out['prefix_bytes']['mean']:.0f} B of JSON; warm pool start "
        f"{out['warm_start_ms']:.0f} ms; load average {out['load_average'][0]:.1f}"
    )
    print("| stage, mode | n | p50 ms | p95 ms | p99 ms | mean ms |\n|---|---|---|---|---|---|")
    for stage in ("extraction", "decide"):
        for mode, s in out[stage].items():
            print(_row(f"{stage}, {mode}", s))
    return out


# --------------------------------------------------------------------------- HTTP load
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(port: int, path: str) -> tuple[int, str]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        return r.status, r.read().decode()
    finally:
        conn.close()


def _stage_means(metrics: str) -> dict[str, float]:
    sums = dict(re.findall(r'tabagent_decision_seconds_sum\{stage="(\w+)"\} (\S+)', metrics))
    counts = dict(re.findall(r'tabagent_decision_seconds_count\{stage="(\w+)"\} (\S+)', metrics))
    return {k: 1000 * float(sums[k]) / float(counts[k]) for k in sums if float(counts.get(k, 0))}


def run_load_once(
    head_dir: Path, mode: str, concurrency: int, prefixes, candidates, n_requests: int, workers, threads=None
):
    port = _free_port()
    cmd = [sys.executable, "-m", "tabagent.cli", "serve", str(head_dir), "--port", str(port), "--no-drift"]
    cmd += {
        "warm": ["--sandbox", "warm"],
        "subprocess": ["--sandbox", "subprocess"],
        "in-process": ["--trust-extractor"],
    }[mode]
    if mode == "warm" and workers:
        cmd += ["--workers", str(workers)]
    if threads:
        cmd += ["--threads", str(threads)]
    fd, name = tempfile.mkstemp(prefix="tabagent_serve_", suffix=".log")
    os.close(fd)
    log = Path(name)
    with log.open("w") as sink:
        server = subprocess.Popen(cmd, stdout=sink, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 120
        while True:
            try:
                if _get(port, "/ready")[0] == 200:
                    break
            except OSError:
                pass
            if time.monotonic() > deadline or server.poll() is not None:
                raise RuntimeError(f"the server did not become ready; see {log}")
            time.sleep(0.1)
        bodies = [json.dumps({"prefix": p, "candidates": candidates, "k": 5}).encode() for p in prefixes]
        latencies: list[float] = []
        outcome = {"deferred": 0, "errors": 0}
        lock = threading.Lock()
        counter = iter(range(n_requests + 20 * concurrency))

        def client(warmup: int) -> None:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
            mine, done = [], 0
            for i in counter:
                body = bodies[i % len(bodies)]
                t = time.perf_counter()
                conn.request("POST", "/decide", body, {"Content-Type": "application/json"})
                r = conn.getresponse()
                data = r.read()
                ms = (time.perf_counter() - t) * 1000
                done += 1
                if done <= warmup:
                    continue
                mine.append(ms)
                with lock:
                    if r.status != 200:
                        outcome["errors"] += 1
                    elif json.loads(data)["deferred"]:
                        outcome["deferred"] += 1
            conn.close()
            with lock:
                latencies.extend(mine)

        threads = [threading.Thread(target=client, args=(20,)) for _ in range(concurrency)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        metrics = _get(port, "/metrics")[1]
        stats = _percentiles(latencies)
        stats.update(
            mode=mode,
            concurrency=concurrency,
            throughput=len(latencies) / wall,  # warm-up requests excluded; the wall clock includes them
            **outcome,
            server_ms=_stage_means(metrics),
            restarts=float(
                (re.findall(r"^tabagent_extractor_restarts_total (\S+)$", metrics, re.M) or ["0"])[0]
            ),
        )
        return stats
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            code = server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            code = "killed"
        if code not in (0, -signal.SIGTERM):  # uvicorn re-raises SIGTERM after its graceful shutdown
            print(f"  (server exited with {code}; log {log})")


def run_load(args) -> list[dict]:
    head_dir = (
        Path(args.head)
        if args.head
        else build_head(Path(tempfile.mkdtemp(prefix="tabagent_bench_")) / "head")
    )
    prefixes = _load_prefixes(args) or live_prefixes(200)
    candidates = _load_candidates(args, head_dir)
    rows = []
    print(
        "| mode | clients | decisions/s | p50 ms | p95 ms | p99 ms | deferred | errors | "
        "server: extraction / scoring / total ms |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for mode in args.modes:
        for c in args.concurrency:
            n = args.requests if mode != "subprocess" else min(args.requests, args.subprocess_requests)
            s = run_load_once(head_dir, mode, c, prefixes, candidates, n, args.workers, args.threads)
            rows.append(s)
            srv = s["server_ms"]
            print(
                f"| {mode} | {c} | {s['throughput']:.0f} | {s['p50']:.1f} | {s['p95']:.1f} | "
                f"{s['p99']:.1f} | {s['deferred']} | {s['errors']} | {srv.get('extraction', 0):.1f} / "
                f"{srv.get('scoring', 0):.1f} / {srv.get('total', 0):.1f} |",
                flush=True,
            )
    print(f"load average {os.getloadavg()[0]:.1f}")
    return rows


def _load_prefixes(args) -> list[dict] | None:
    if not getattr(args, "prefixes", None):
        return None
    return [json.loads(line) for line in Path(args.prefixes).read_text().splitlines() if line.strip()]


def _load_candidates(args, head_dir: Path) -> list:
    if getattr(args, "candidates", None):
        return json.loads(Path(args.candidates).read_text())
    if args.head:
        raise SystemExit("--head needs --candidates (a JSON list) and --prefixes (a JSONL file of prefixes)")
    return [{"id": t, "name": t, "description": d} for t, d in TOOLS.items()]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "load"):
        p = sub.add_parser(name)
        p.add_argument("--head", help="a saved head (default: build the synthetic one)")
        p.add_argument("--prefixes", help="JSONL of live prefixes for --head")
        p.add_argument("--candidates", help="JSON list of candidates for --head")
        p.add_argument("--out", type=Path, default=OUT / f"serving_{name}.json")
    sub.choices["extract"].add_argument("--n", type=int, default=300, help="timed decisions per mode")
    load = sub.choices["load"]
    load.add_argument("--modes", nargs="+", default=["warm", "subprocess", "in-process"])
    load.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32])
    load.add_argument("--requests", type=int, default=2000, help="timed requests per run")
    load.add_argument("--subprocess-requests", type=int, default=400, help="fewer for the slow cold sandbox")
    load.add_argument("--workers", type=int, help="warm extractor workers (default min(4, CPUs))")
    load.add_argument("--threads", type=int, help="the server's decision threads (default 8)")
    args = ap.parse_args(argv)
    result = run_extract(args) if args.cmd == "extract" else run_load(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
