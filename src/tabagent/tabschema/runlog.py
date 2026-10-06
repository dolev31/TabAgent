"""The run summary of a TabSchema discovery (``run.json``) and its stage-by-stage anatomy
(``tabagent runs show RUN_DIR``).

``run.json`` is written into the run directory as the run progresses (atomically, at every stage boundary and
as decision points are analyzed or audited), for operators and for scripts:

* ``status``: ``running``, ``completed``, ``stopped`` (a spent budget, a provider that stayed down, an
  interruption: rerun with the run directory to resume) or ``failed`` (the pipeline could not produce a card:
  no feasible proposal, nothing accepted, an extractor that never ran, a leaking adapter);
* ``sessions``: every process that worked in the directory, with what it paid;
* ``stages``: per pipeline stage, its status, timing, progress (with an estimate of the time left), what it
  produced, what it dropped, what failed, and its LLM calls, tokens and cost (paid now, and replayed from the
  cache at their first cost);
* ``drops_by_reason``: every dropped decision point, proposal, feature or column, counted by reason;
* ``llm``: the cost meter's totals, and the provider errors and retries of ``llm_calls.jsonl``.

:func:`summarize_run` reads a run directory, from ``run.json`` when present and otherwise rebuilt from the
stage files (``analysis.json``, ``aggregator.json``, ``judges.json``, ``compile.json``, ``audit.json``,
``usefulness.json`` and ``llm_calls.jsonl``), so runs made before ``run.json`` existed can be inspected too;
:func:`render` prints the anatomy.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tabagent.tabschema.runstore import RunStore, atomic_write, read_jsonl, utc_now

logger = logging.getLogger(__name__)

FORMAT = 1
RUN_FILE = "run.json"
#: The pipeline stages, in order, with the LLM stage roles whose calls each one makes.
STAGES = {
    "adapter_check": (),
    "sample": (),
    "analyze": ("analyzer", "extractor", "validator"),
    "aggregate": ("aggregator",),
    "judge": ("judge_relevance", "judge_generality", "judge_impact"),
    "meta_judge": ("meta_judge",),
    "compile": ("extractor", "validator"),
    "audit": ("auditor",),
    "usefulness": (),
    "card": (),
}
#: Drop reasons, by the start of their text (the first match wins), for ``drops_by_reason``.
REASONS = (
    ("constant across", "constant across the onboarding decision points"),
    ("null rate", "too often empty (max_null_rate)"),
    ("not in the accepted feature plan", "not in the accepted feature plan"),
    ("name reserved", "name reserved by the decision table"),
    ("codevalidator: hallucinated", "CodeValidator: hallucinated field"),
    ("codevalidator: plan", "CodeValidator: plan mismatch"),
    ("one of ", "column cap of one feature (max_columns_per_feature)"),
    ("the extractor's columns are capped", "column cap of the extractor (max_extractor_columns)"),
    ("accepted but not produced", "accepted but not produced by the compiled extractor"),
    ("redundant with", "redundant text (the same words as another kept text column)"),
    ("mean length above", "text too long"),
    ("nondeterministic", "verification: nondeterministic"),
    ("reads the run's outcome", "verification: reads outcome fields"),
    ("outputauditor", "OutputAuditor: judged incorrect"),
    ("harmful", "usefulness: harmful (prune_harmful)"),
    ("the metajudge decided on a name", "MetaJudge: decided on a name it was not given"),
    ("the metajudge gave no decision", "MetaJudge: no decision"),
)


def drop_reason(reason: str) -> str:
    """The category of one drop reason (see ``REASONS``); other reasons are kept as they are, clipped."""
    low = str(reason).lower()
    return next((cat for start, cat in REASONS if low.startswith(start)), str(reason)[:80])


def _pid_alive(pid: Any) -> bool:
    """Whether process ``pid`` exists. On Windows ``os.kill(pid, 0)`` would send it a Ctrl+C, so the process
    is looked up (``OpenProcess``) instead; where that cannot be asked, the process counts as alive."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            kernel32.CloseHandle(handle)
            return not ok or code.value == 259  # STILL_ACTIVE
        except Exception:  # noqa: BLE001 - unknown: keep the other session's record as it is
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists, owned by someone else
        return True
    except OSError:
        return False
    return True


def _meter_state(meter: Any) -> dict[str, dict[str, float]]:
    """A snapshot of a cost meter's per-role totals (paid and cached), to diff across a stage."""
    with meter._lock:
        paid = {k: dict(v) for k, v in meter.by_stage.items()}
        cached = {k: dict(v) for k, v in meter.cached.items()}
    return {"paid": paid, "cached": cached}


def _delta(before: dict, after: dict) -> dict[str, Any]:
    """The LLM work between two meter snapshots: paid and cached calls, tokens and cost, per role."""
    out: dict[str, Any] = {"calls": 0, "cached_calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}
    out["cached_usd"] = 0.0
    roles: dict[str, dict[str, float]] = {}
    for kind in ("paid", "cached"):
        for role, now in after[kind].items():
            then = before[kind].get(role, {})
            d = {k: now.get(k, 0) - then.get(k, 0) for k in ("calls", "input_tokens", "output_tokens", "usd")}
            if not d["calls"]:
                continue
            r = roles.setdefault(role, {"calls": 0, "cached_calls": 0})
            if kind == "paid":
                r["calls"] += int(d["calls"])
                out["calls"] += int(d["calls"])
                out["input_tokens"] += int(d["input_tokens"])
                out["output_tokens"] += int(d["output_tokens"])
                out["usd"] += d["usd"]
            else:
                r["cached_calls"] += int(d["calls"])
                out["cached_calls"] += int(d["calls"])
                out["cached_usd"] += d["usd"]
    out["usd"] = round(out["usd"], 6)
    out["cached_usd"] = round(out["cached_usd"], 6)
    out["roles"] = roles
    return out


class RunLog:
    """Writes ``run.json`` for one discovery run (see the module docstring). Thread-safe."""

    def __init__(self, store: RunStore, meter: Any, target: str):
        self.store, self.meter = store, meter
        self._lock = threading.RLock()
        self._last_write = 0.0
        old: dict[str, Any] = {}
        with contextlib.suppress(OSError, ValueError):
            old = json.loads((store.dir / RUN_FILE).read_text())
        sessions = (
            [s for s in old.get("sessions", []) if isinstance(s, dict)] if isinstance(old, dict) else []
        )
        live = [
            s
            for s in sessions
            if s.get("status") == "running"
            and s.get("host") == socket.gethostname()
            and _pid_alive(s.get("pid"))
        ]
        live = [s for s in live if s.get("pid") != os.getpid()]
        if live:
            logger.warning(
                "TabSchema: another process (pid %s) is running in %s; both share its answers (an answer is "
                "paid once) and rewrite its files",
                live[0].get("pid"),
                store.dir,
            )
        for s in sessions:  # a session that ended without saying so was killed
            if s.get("status") == "running" and s not in live:
                s["status"] = "killed"
        self.session = {
            "id": store.session,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started": utc_now(),
            "status": "running",
        }
        self.data: dict[str, Any] = {
            "format": FORMAT,
            "run_id": store.run_id,
            "target": target,
            "status": "running",
            "started": old.get("started", self.session["started"]) if isinstance(old, dict) else utc_now(),
            "updated": utc_now(),
            "sessions": [*sessions, self.session],
            "data": {},
            "stages": {},
            "notes": [],
        }
        self._stage_t0: dict[str, float] = {}
        self._stage_meter: dict[str, dict] = {}
        self.write()

    # ------------------------------------------------------------------ writing
    def write(self, throttle: float = 0.0) -> None:
        """Write ``run.json`` (at most every ``throttle`` seconds). The sessions of other processes working in
        the directory at the same time are merged in from the file, under a lock, so none is lost or left
        stale."""
        with self._lock:
            now = time.monotonic()
            if throttle and now - self._last_write < throttle:
                return
            self._last_write = now
            self.data["updated"] = utc_now()
            self.data["llm"] = self.meter.summary()
            path = self.store.path(RUN_FILE)
            with _locked(path):
                self._merge_sessions(path)
                atomic_write(path, json.dumps(self.data, indent=1, default=str))

    def _merge_sessions(self, path: Path) -> None:
        try:
            disk = json.loads(path.read_text()).get("sessions") or []
        except (OSError, ValueError, AttributeError):
            return
        mine = {s.get("id"): s for s in self.data["sessions"]}
        for s in disk:
            if not isinstance(s, dict) or s.get("id") == self.session["id"]:
                continue
            ours = mine.get(s.get("id"))
            if ours is None or (ours.get("status") == "running" and s.get("status") != "running"):
                mine[s.get("id")] = s
        self.data["sessions"] = sorted(mine.values(), key=lambda s: str(s.get("started", "")))

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self.data[key] = value
        self.write()

    def rec(self, stage: str) -> dict[str, Any]:
        """The record of ``stage`` (created when missing)."""
        with self._lock:
            return self.data["stages"].setdefault(stage, {"status": "pending"})

    def update(self, stage: str, **fields: Any) -> None:
        """Merge ``fields`` into the record of ``stage`` and write."""
        with self._lock:
            self.rec(stage).update(fields)
        self.write()

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        """Run a pipeline stage: its record gets its status, timing and LLM work. An exception marks it
        ``failed`` (or ``stopped`` for a resumable stop) and propagates."""
        with self._lock:
            rec = self.rec(name)
            rec.clear()
            rec.update(status="running", started=utc_now())
            self._stage_t0[name] = time.monotonic()
            self._stage_meter[name] = _meter_state(self.meter)
            self.store.phase = name
        self.write()
        try:
            yield rec
        except BaseException as e:
            self._end(name, stopped_status(e), error=f"{type(e).__name__}: {e}"[:500])
            raise
        self._end(name, "done")

    def _end(self, name: str, status: str, error: str | None = None) -> None:
        with self._lock:
            rec = self.rec(name)
            rec["status"] = status
            rec["ended"] = utc_now()
            rec["seconds"] = round(time.monotonic() - self._stage_t0.get(name, time.monotonic()), 2)
            rec["llm"] = _delta(
                self._stage_meter.get(name, _meter_state(self.meter)), _meter_state(self.meter)
            )
            if error:
                rec["error"] = error
            self.store.phase = None
        self.write()

    def progress(self, stage: str, done: int, total: int, failed: int = 0) -> str:
        """Record progress in ``stage`` (``done`` of ``total`` items, ``failed`` of them failed) with an
        estimate of the time left; returns the progress line for the log."""
        with self._lock:
            rec = self.rec(stage)
            elapsed = time.monotonic() - self._stage_t0.get(stage, time.monotonic())
            left = elapsed / done * (total - done) if done else None
            rec["progress"] = {"done": done, "total": total, "failed": failed}
            rec["eta_s"] = None if left is None else round(left)
        self.write(throttle=1.0 if done < total else 0.0)
        eta = "" if left is None or done >= total else f"; about {_duration(left)} left"
        return f"{done}/{total}" + (f" ({failed} failed)" if failed else "") + eta

    def finish(self, status: str, error: BaseException | None = None, extra: dict | None = None) -> None:
        """Close the session: the run's final ``status``, the error that ended it, the totals."""
        with self._lock:
            self.data["status"] = status
            self.session["status"] = status
            self.session["ended"] = utc_now()
            calls, tin, tout = self.meter.tokens()
            self.session["paid"] = {
                "calls": calls,
                "input_tokens": tin,
                "output_tokens": tout,
                "usd": round(self.meter.total_usd, 6) if self.meter.priced else None,
            }
            if error is not None:
                self.data["error"] = f"{type(error).__name__}: {error}"[:2000]
                self.data["stopped_by"] = type(error).__name__
            else:
                self.data.pop("error", None)
                self.data.pop("stopped_by", None)
            if extra:
                self.data.update(extra)
            self.data["provider_errors"] = provider_errors(self.store.dir)
        self.write()


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """An exclusive lock on ``path`` shared with other processes (POSIX; a no-op elsewhere)."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        yield
        return
    with open(path.parent / f".{path.name}.lock", "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def stopped_status(e: BaseException) -> str:
    """``stopped`` for a stop a rerun resumes from (a budget, a provider that stayed down, an interruption),
    ``failed`` otherwise."""
    from tabagent.tabschema.llm import BudgetExceeded, LLMUnavailable

    resumable = (BudgetExceeded, LLMUnavailable, KeyboardInterrupt, SystemExit)
    return "stopped" if isinstance(e, resumable) else "failed"


def _duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


def provider_errors(run_dir: Path) -> dict[str, Any]:
    """Retries and failed attempts in ``llm_calls.jsonl``: counts by event and by error type."""
    events: Counter = Counter()
    errors: Counter = Counter()
    for r in read_jsonl(Path(run_dir) / "llm_calls.jsonl"):
        ev = r.get("event")
        if ev:
            events[ev] += 1
            errors[str(r.get("error", "")).split(":", 1)[0] or "unknown"] += 1
    return {"events": dict(events), "errors": dict(errors)}


# ---------------------------------------------------------------------------------------------- reading
def _load(d: Path, name: str) -> Any:
    try:
        return json.loads((d / name).read_text())
    except (OSError, ValueError):
        return None


def _calls_by_stage(d: Path, prices: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """LLM work per pipeline stage from ``llm_calls.jsonl``: by the record's ``phase`` when it has one, else
    by its position (the analyzer, extractor and validator calls before the first aggregator call belong to
    the analysis, the later ones to the compile step)."""
    records = read_jsonl(d / "llm_calls.jsonl")
    first_agg = next((i for i, r in enumerate(records) if r.get("stage") == "aggregator"), len(records))
    out: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0, "failed_attempts": 0}
    )
    unpriced = False
    for i, r in enumerate(records):
        role = str(r.get("stage", ""))
        phase = r.get("phase")
        if not phase:
            phase = next((s for s, roles in STAGES.items() if role in roles), role)
            if role in ("extractor", "validator"):
                phase = "analyze" if i < first_agg else "compile"
        rec = out[phase]
        if r.get("event") and r.get("event") != "content_policy":
            rec["failed_attempts"] += 1
            continue
        rec["calls"] += 1
        tin, tout = int(r.get("input_tokens") or 0), int(r.get("output_tokens") or 0)
        rec["input_tokens"] += tin
        rec["output_tokens"] += tout
        price = prices.get(str(r.get("model")))
        if price:
            rec["usd"] += (tin * float(price[0]) + tout * float(price[1])) / 1e6
        elif tin or tout:
            unpriced = True
    for rec in out.values():
        rec["usd"] = None if unpriced else round(rec["usd"], 6)
    return dict(out)


def _rebuild(d: Path) -> dict[str, Any]:
    """A run summary rebuilt from the stage files of a run directory without ``run.json``."""
    cfg = _load(d, "config.json") or {}
    stages: dict[str, dict[str, Any]] = {}
    drops: Counter = Counter()
    analysis = _load(d, "analysis.json")
    if isinstance(analysis, list):
        failed = [a for a in analysis if a.get("error")]
        stages["analyze"] = {
            "status": "done",
            "produced": {
                "points": len(analysis),
                "proposals": sum(len(a.get("proposed") or []) for a in analysis),
                "kept": sum(len(a.get("kept") or []) for a in analysis),
            },
            "failures": dict(Counter(str(a["error"]).split(":", 1)[0] for a in failed)),
        }
        drops["analysis: decision point failed"] += len(failed)
        drops["analysis: proposal flagged by the CodeValidator"] += sum(
            len(a.get("proposed") or []) - len(a.get("kept") or []) for a in analysis if not a.get("error")
        )
    agg = _load(d, "aggregator.json")
    if isinstance(agg, dict):
        stages["aggregate"] = {
            "status": "done",
            "produced": {
                "features": len(agg.get("filtered") or []),
                "rejected": len(agg.get("rejected") or []),
            },
        }
        drops["FeatureAggregator: rejected"] += len(agg.get("rejected") or [])
    judges = _load(d, "judges.json")
    if isinstance(judges, dict):
        decisions = Counter(str(x.get("final_decision")) for x in judges.get("meta") or [])
        stages["judge"] = {"status": "done", "produced": {"judged": len(judges.get("table") or [])}}
        stages["meta_judge"] = {"status": "done", "produced": dict(decisions)}
        accept = set(cfg.get("accept") or ("accept", "conditional"))
        for k, n in decisions.items():
            if k not in accept:
                drops[f"MetaJudge: {k}"] += n
    comp = _load(d, "compile.json")
    if isinstance(comp, dict):
        stages["compile"] = {
            "status": "done",
            "produced": {"columns": len(comp.get("kept") or {}), "attempts": comp.get("attempts")},
        }
        for reason in (comp.get("dropped") or {}).values():
            drops[drop_reason(reason)] += 1
    audit = _load(d, "audit.json")
    if isinstance(audit, dict):
        stages["audit"] = {
            "status": "skipped" if audit.get("skipped") else "done",
            "produced": {"points": audit.get("samples"), "answered": audit.get("answered")},
            **({"note": audit["skipped"]} if audit.get("skipped") else {}),
        }
    use = _load(d, "usefulness.json")
    if isinstance(use, dict):
        verdicts = Counter(str(r.get("verdict")) for r in (use.get("features") or {}).values())
        stages["usefulness"] = {
            "status": "skipped" if use.get("skipped") else "done",
            "produced": dict(verdicts),
            **({"note": use["skipped"]} if use.get("skipped") else {}),
        }
    complete = (d / "card.yaml").exists()
    status = "completed" if complete else ("stopped or failed" if (d / "report.md").exists() else "unknown")
    return {
        "format": 0,
        "run_id": d.name,
        "target": cfg.get("target", "?"),
        "status": status,
        "stages": stages,
        "drops_by_reason": {k: v for k, v in drops.items() if v},
        "rebuilt": True,
    }


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """The summary of a run directory: ``run.json`` (completed with the per-stage LLM work recorded in
    ``llm_calls.jsonl``), or one rebuilt from its stage files. Raises ``FileNotFoundError`` when the directory
    is not a TabSchema run directory."""
    d = Path(run_dir)
    if not d.is_dir() or not any((d / f).exists() for f in (RUN_FILE, "config.json", "llm_calls.jsonl")):
        raise FileNotFoundError(
            f"{d} is not a TabSchema run directory (no run.json, config.json or llm_calls.jsonl)"
        )
    summary = _load(d, RUN_FILE)
    if not isinstance(summary, dict):
        summary = _rebuild(d)
    cfg = _load(d, "config.json") or {}
    calls = _calls_by_stage(d, cfg.get("prices") or {})
    for stage, work in calls.items():
        rec = summary.setdefault("stages", {}).setdefault(stage, {"status": "?"})
        rec.setdefault("llm_log", work)
    summary.setdefault("provider_errors", provider_errors(d))
    summary["files"] = sorted(p.name for p in d.iterdir() if p.is_file() and not p.name.startswith("."))
    summary["dir"] = str(d)
    return summary


def _tokens(n: Any) -> str:
    n = int(n or 0)
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 10_000:
        return f"{n / 1e3:.0f}k"
    return f"{n:,}"


def _usd(x: Any) -> str:
    return "?" if x is None else f"${x:.2f}"


def _produced(rec: dict[str, Any]) -> str:
    p = rec.get("produced") or {}
    return ", ".join(f"{v} {k}" for k, v in p.items() if v not in (None, "", {}))


def render(summary: dict[str, Any]) -> str:
    """The anatomy of a run, stage by stage, as text (``tabagent runs show``)."""
    lines = []
    sessions = summary.get("sessions") or []
    head = f"run {summary.get('run_id')}  target {summary.get('target')}  status {summary.get('status')}"
    if len(sessions) > 1:
        head += f"  ({len(sessions)} sessions: {', '.join(str(s.get('status')) for s in sessions)})"
    lines.append(head)
    if summary.get("rebuilt"):
        lines.append("  (no run.json: rebuilt from the stage files; timings are not known)")
    if summary.get("error"):
        lines.append(f"  ended by {summary['error'][:300]}")
    data = summary.get("data") or {}
    if data:
        lines.append(
            "data: "
            + ", ".join(
                f"{v} {k.replace('_', ' ')}" for k, v in data.items() if not isinstance(v, (dict, list))
            )
        )
    llm = summary.get("llm") or {}
    if llm:
        cached = llm.get("cached") or {}
        line = f"LLM: {_usd(llm.get('total_usd'))} paid in this session, {llm.get('calls', 0)} calls, "
        line += (
            f"{_tokens(llm.get('input_tokens'))} input / {_tokens(llm.get('output_tokens'))} output tokens"
        )
        if cached.get("calls"):
            first = _usd(cached.get("first_paid_usd"))
            line += f"; {cached['calls']} answers replayed from the run directory (first paid {first})"
        lines.append(line)
    paid = [s.get("paid", {}).get("usd") for s in sessions if s.get("paid")]
    if len(paid) > 1 and all(p is not None for p in paid):
        lines.append(f"paid over all sessions: ${sum(paid):.2f}")
    lines += [
        "",
        f"{'stage':<14}{'status':<10}{'time':>9}{'calls':>7}{'cached':>8}{'in':>8}{'out':>8}{'cost':>9}"
        "  produced",
    ]
    order = list(STAGES) + [s for s in summary.get("stages", {}) if s not in STAGES]
    for name in order:
        rec = (summary.get("stages") or {}).get(name)
        if not rec:
            continue
        work = rec.get("llm") or {}
        log = rec.get("llm_log") or {}
        calls = work.get("calls", log.get("calls", 0))
        cost = work.get("usd") if work else log.get("usd")
        seconds = rec.get("seconds")
        time_ = _duration(seconds) if seconds is not None else "-"
        prog = rec.get("progress")
        status = str(rec.get("status", "?"))
        if prog and status == "running":
            status = f"{prog['done']}/{prog['total']}"
        lines.append(
            f"{name:<14}{status:<10}{time_:>9}{calls or 0:>7}{work.get('cached_calls', 0):>8}"
            f"{_tokens(work.get('input_tokens', log.get('input_tokens'))):>8}"
            f"{_tokens(work.get('output_tokens', log.get('output_tokens'))):>8}"
            f"{(_usd(cost) if (calls or cost) else '-'):>9}  {_produced(rec)}"
        )
        extra = []
        if rec.get("failures"):
            extra.append("failed: " + ", ".join(f"{n} x {k}" for k, n in rec["failures"].items()))
        if rec.get("dropped"):
            extra.append("dropped: " + ", ".join(f"{n} {k}" for k, n in rec["dropped"].items()))
        if log.get("failed_attempts"):
            extra.append(
                f"{log['failed_attempts']} failed LLM attempt(s) (retried or raised; see llm_calls.jsonl)"
            )
        if rec.get("note"):
            extra.append(str(rec["note"])[:200])
        if rec.get("error"):
            extra.append(f"error: {rec['error'][:200]}")
        if rec.get("eta_s") is not None and rec.get("status") == "running":
            extra.append(f"about {_duration(rec['eta_s'])} left")
        lines += [f"{'':<14}- {e}" for e in extra]
    drops = summary.get("drops_by_reason") or {}
    if drops:
        lines += ["", "dropped, by reason:"]
        lines += [f"  {n:>5}  {k}" for k, n in sorted(drops.items(), key=lambda kv: -kv[1])]
    pe = summary.get("provider_errors") or {}
    if pe.get("events"):
        lines += [
            "",
            "provider errors: "
            + ", ".join(f"{n} {k}" for k, n in pe["events"].items())
            + " ("
            + ", ".join(f"{k} x{n}" for k, n in pe.get("errors", {}).items())
            + ")",
        ]
    for note in summary.get("notes") or []:
        lines.append(f"note: {note}")
    if summary.get("files"):
        lines += ["", "files: " + " ".join(summary["files"])]
    return "\n".join(lines) + "\n"


def list_runs(root: str | Path) -> list[dict[str, Any]]:
    """One line of facts per run directory under ``root`` (newest first): id, target, status, cost."""
    out = []
    root = Path(root)
    if not root.is_dir():
        return out
    for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            s = summarize_run(d)
        except FileNotFoundError:
            continue
        llm = s.get("llm") or {}
        out.append(
            {
                "run": d.name,
                "target": s.get("target"),
                "status": s.get("status"),
                "updated": s.get("updated", ""),
                "usd": llm.get("total_usd"),
                "dir": str(d),
            }
        )
    return out
