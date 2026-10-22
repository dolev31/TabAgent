"""HTTP decision service for a saved head: the drop-in replacement for the LLM component in a live agent.

    tabagent serve ./my_head --port 8077

``POST /decide`` with ``{"prefix": {...}, "candidates": [...], "k": 9}`` returns ``TabAgentHead.decide``'s
answer: the shortlisted ``candidates``, per-candidate ``scores``, the head's ``confidence``, whether the
decision is ``deferred`` to the original LLM (by the confidence gate, App. N.3, in the loop App. O.2, or
because the extractor failed on the prefix) and the extractor's ``error``, if any. ``prefix`` is the
adapter-normalized prefix; send ``"run"`` instead (the run so far in the format of the training logs, with
its run-level task fields) to have the head build it with its recorded adapter. A single-class head takes no
candidates and answers with ``choice``. A request the head cannot answer as given (``RequestError``: no
candidates for a ranking head, both a prefix and a run, a malformed candidate, ...) is answered with status
400, and a body that does not fit the request schema (``candidates`` not a list, say) with 422; any other
failure of the head is a 500. ``POST /decide_batch`` takes ``{"requests": [...]}`` and answers
``{"results": [...]}``, one answer per request, in order (an invalid one is ``{"status": 400, "detail":
...}``).

With a ``request_timeout``, the extractor gets that long (its warm worker's alarm fires then, and a worker
that ignores it is killed shortly after). A decision still unanswered a grace period later, and every
decision of a batch still unanswered when the batch's deadline passes, is answered deferred, never with an
error status: ``candidates`` lists the request's candidate ids in their input order (``choice`` is None for
a single-class head), ``scores`` is empty, ``confidence`` is None and ``error`` starts with
``TimeoutError``.

Operations: ``GET /health`` (liveness: the process answers), ``GET /ready`` (readiness: the head is loaded and
its warm extractor pool runs at least ``min_ready`` workers and can run the extractor; 503 otherwise, with the
reason), ``GET /version`` (the library version, the paper's citation and the head's manifest: card, decision,
extractor and model hashes, creation time), ``GET /card``, ``GET /metrics`` (Prometheus text: requests,
decisions, deferrals by reason, errors by class, latency histograms per stage, the worker pool, its restarts
and start failures, drift gauges) and ``GET /drift`` (the per-feature drift report of ``tabagent.drift``).
Every decision is logged as one structured record (request id, head id, latency breakdown, deferral and its
reason, error class) without the request's content unless ``log_content``. Requires ``pip install
"tabagent[serve]"``; ``RemoteHead`` is its client, standard library only.
"""

# No `from __future__ import annotations`: FastAPI must see the request model class, not a string.
import asyncio
import contextlib
import datetime as dt
import hashlib
import inspect
import json
import logging
import math
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from tabagent.prometheus import CONTENT_TYPE, Registry

_SERVE_EXTRA = 'serving a head needs the serve extra: pip install "tabagent[serve]"'
LATENCY_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
#: Why a decision is deferred: the confidence gate, an extractor failure (``ExecResult.kind``), the deadline.
DEFER_REASONS = (
    "low_confidence",
    "extractor_error",
    "extractor_timeout",
    "extractor_crash",
    "extractor_broken",
    "sandbox_unavailable",
    "no_extractor",
    "request_timeout",
)
_KIND_CLASS = {
    "error": "extractor_error",
    "timeout": "extractor_timeout",
    "crash": "extractor_crash",
    "broken": "extractor_broken",
    "unavailable": "sandbox_unavailable",
    "no_extractor": "no_extractor",
}

log = logging.getLogger("tabagent.serve")
_ERROR_TYPE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning))\b")


# --------------------------------------------------------------------------- head description
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def head_manifest(head) -> dict[str, Any]:
    """What identifies a served head: its ``id`` (12 hex digits of the model hash, else of the extractor
    hash), decision, card, kind, the extractor's SHA-256, a SHA-256 over the model files' manifest hashes,
    when the model artifact was written and by which tabagent version (None when unknown); and
    ``candidate_function: true`` when its extractor defines the per-candidate function."""
    from tabagent.tabschema.checks import defines_candidate_function

    agent = getattr(head, "agent", None)
    manifest = getattr(agent, "manifest_", None) or {}
    files = manifest.get("sha256") or {}
    model_sha = _sha(json.dumps(sorted(files.items()))) if files else None
    card = getattr(head, "card", None)
    extractor = getattr(card, "extractor", None) or {}
    code = getattr(head, "extractor_code", None)
    extractor_sha = extractor.get("sha256") or (_sha(code) if code else None)
    spec = getattr(card, "spec", None)
    name = getattr(card, "name", None)
    out = {
        "id": (model_sha or extractor_sha or _sha(f"{name}/{getattr(head, 'target', '')}"))[:12],
        "target": getattr(head, "target", None),
        "card": name,
        "kind": getattr(spec, "kind", None),
        "extractor_sha256": extractor_sha,
        "model_sha256": model_sha,
        "created_at": manifest.get("created"),
        "artifact_tabagent_version": manifest.get("tabagent_version"),
    }
    if code and defines_candidate_function(code):  # the extractor scores every candidate too
        out["candidate_function"] = True
    return out


def error_class(error: str | None, kind: str | None = None) -> str | None:
    """The class of an extractor failure: from its ``kind`` (``ExecResult.kind``, reported by
    ``TabAgentHead.decide``'s trace), else, for a head that reports none, from its message. None for no
    error."""
    if not error and not kind:
        return None
    if kind is not None:
        return _KIND_CLASS.get(kind, "extractor_error")
    e = (error or "").lower()
    if "timed out" in e:
        return "extractor_timeout"
    if "has no extractor" in e:
        return "no_extractor"
    if "exited early" in e or "was killed" in e:
        return "extractor_crash"
    return "extractor_error"


def defer_reason(
    answer: dict[str, Any] | None, timed_out: bool = False, kind: str | None = None
) -> str | None:
    """Why an answer is deferred (one of ``DEFER_REASONS``), or None when the head decided."""
    if timed_out:
        return "request_timeout"
    if not answer or not answer.get("deferred"):
        return None
    return error_class(answer.get("error"), kind) or "low_confidence"


def error_type(error: str | None) -> str | None:
    """The exception name an error message starts with (``KeyError``, ...), which carries no request data."""
    m = _ERROR_TYPE.match((error or "").strip())
    return m.group(1) if m else None


def _checked_timeout(value: Any) -> Optional[float]:  # noqa: UP045
    if value is None:
        return None
    t = float(value)
    if not math.isfinite(t) or t <= 0:
        raise ValueError(f"request_timeout must be a positive number of seconds, not {value!r}")
    return t


def candidate_ids(candidates: Any) -> list[str]:
    """The ids of a request's candidates, in their input order ([] when they cannot be read)."""
    from tabagent.api import _as_candidates

    try:
        return [c.id for c in _as_candidates(candidates or [])]
    except (ValueError, KeyError, TypeError):
        return []


# --------------------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    """One JSON object per record: time, level, logger, message, and the record's ``fields`` (a dict passed as
    ``extra={"fields": ...}``). Tracebacks are reduced to their frames and exception type, since exception
    messages may quote request data."""

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": dt.datetime.fromtimestamp(record.created, dt.timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            out.update(fields)
        if record.exc_info and record.exc_info[0] is not None:
            out["exception"] = record.exc_info[0].__name__
            out["traceback"] = "".join(traceback.format_tb(record.exc_info[2]))
        return json.dumps(out, default=str, ensure_ascii=False)


class _DropContentWarnings(logging.Filter):
    """The library's warning about a failed extraction quotes the error; the service logs its class."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.msg).startswith("TabSchema extractor failed on a live prefix")


def configure_logging(fmt: str = "json", level: int = logging.INFO) -> logging.Handler:
    """Log to stderr, one JSON object per line (``fmt="json"``) or plain text, for the service and the
    library (``tabagent`` and ``uvicorn`` loggers). Returns the installed handler."""
    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(_DropContentWarnings())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # the CLI's plain-text handler on `tabagent` would print every record twice, and its WARNING level would
    # drop the per-decision records: the root handler takes over (a more verbose level set by -vv stays)
    lib = logging.getLogger("tabagent")
    lib.handlers[:] = [h for h in lib.handlers if not getattr(h, "tabagent_cli", False)]
    if lib.level == logging.NOTSET or lib.level > level:
        lib.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers[:] = []
        logging.getLogger(name).propagate = True
    return handler


# --------------------------------------------------------------------------- the service
class Service:
    """The serving state of one head: decisions under a deadline, readiness, metrics, logs and drift (see the
    module docstring). ``create_app`` wraps it in a FastAPI application."""

    def __init__(
        self,
        head,
        request_timeout: Optional[float] = None,  # noqa: UP045
        log_content: bool = False,
        drift: Any = True,
        drift_window: int = 500,
        threads: int = 8,
        max_batch: int = 256,
        warm: bool = True,
    ):
        from tabagent.tabschema.warm import WarmSandbox

        self.head, self.log_content, self.max_batch = head, log_content, max_batch
        self.request_timeout = _checked_timeout(request_timeout)
        self.warm_on_start = warm
        self.manifest = head_manifest(head)
        self.kind = self.manifest["kind"] or "ranking"
        box = getattr(head, "extractor_sandbox", None) if getattr(head, "extractor_code", None) else None
        if box is not None and self.request_timeout is not None and hasattr(box, "timeout_s"):
            box.timeout_s = min(box.timeout_s, self.request_timeout)  # the extractor's share of the deadline
            if hasattr(box, "max_timeout_s"):  # nor do a decision's candidates stretch it past the deadline
                box.max_timeout_s = self.request_timeout
        self.sandbox = box
        self.pool = box if isinstance(box, WarmSandbox) else None
        # past the deadline the answer is deferred: the timeout, a warm worker's kill grace, and scoring
        timeout = self.request_timeout
        self.deadline = None if timeout is None else timeout + 1.5 + 0.1 * timeout
        self.traceable = "trace" in inspect.signature(head.decide).parameters
        self.executor = ThreadPoolExecutor(max(1, threads), thread_name_prefix="tabagent-decide")
        self.closing = False
        self.warm_error: Optional[str] = None  # noqa: UP045
        self.drift, self.drift_note = self._drift(drift, drift_window)
        self.registry = Registry()
        self._metrics()

    # ------------------------------------------------------------------ set-up
    def _drift(self, drift: Any, window: int):
        if drift is False or drift is None:
            return None, "drift monitoring is off"
        if not isinstance(drift, bool):  # a DriftMonitor
            return drift, None
        stats = getattr(getattr(self.head, "agent", None), "feature_stats_", None)
        if not stats:
            return None, (
                "this head has no training statistics (feature_stats.json): it was fitted before tabagent "
                "recorded them; refit and save it to monitor drift"
            )
        from tabagent.drift import DriftMonitor

        return DriftMonitor(stats, window=window, min_decisions=min(100, window)), None

    def _metrics(self) -> None:
        r = self.registry
        self.http_requests = r.counter(
            "tabagent_http_requests_total", "HTTP requests by endpoint and status code.", ("endpoint", "code")
        )
        self.decisions = r.counter(
            "tabagent_decisions_total",
            "Decisions answered, by endpoint and outcome (head: the head decided; deferred: to the LLM).",
            ("endpoint", "outcome"),
        )
        self.deferrals = r.counter("tabagent_deferrals_total", "Deferred decisions by reason.", ("reason",))
        self.errors = r.counter(
            "tabagent_errors_total",
            "Errors by class: extractor failures (the decision is deferred), request timeouts, bad requests "
            "(400) and internal errors (500).",
            ("class",),
        )
        self.latency = r.histogram(
            "tabagent_decision_seconds",
            "Decision latency by stage: total, prefix (building it from a run), extraction, scoring.",
            ("stage",),
            LATENCY_BUCKETS,
        )
        self.inflight = r.gauge("tabagent_inflight_decisions", "Decisions a decision thread is computing.")
        r.gauge(
            "tabagent_queued_decisions",
            "Decisions waiting for a free decision thread (--threads).",
            (),
            lambda: self.executor._work_queue.qsize(),
        )
        info = {
            k: str(v) for k, v in self.manifest.items() if k in ("id", "card", "target", "extractor_sha256")
        }
        import tabagent

        info["tabagent_version"] = tabagent.__version__
        labels = tuple(sorted(info))
        r.gauge(
            "tabagent_head_info",
            "The served head (value 1).",
            labels,
            lambda: {tuple(info[k] for k in labels): 1},
        )
        r.gauge(
            "tabagent_ready",
            "1 when the service is ready (see /ready).",
            (),
            lambda: int(self.readiness()[0]),
        )
        r.gauge(
            "tabagent_extractor_broken",
            "1 when the extractor cannot run at all (it does not compile, or breaks the extraction rules).",
            (),
            lambda: int(bool(getattr(self.sandbox, "broken", None))),
        )
        pool = self.pool
        if pool is not None:

            def stat(key):
                return lambda: pool.stats()[key]

            r.gauge(
                "tabagent_extractor_workers",
                "Warm extractor workers by state (size: the configured pool size).",
                ("state",),
                lambda: {
                    (k,): v
                    for k, v in pool.stats().items()
                    if k in ("size", "min_ready", "live", "idle", "busy", "starting")
                },
            )
            r.gauge(
                "tabagent_extractor_degraded",
                "1 when fewer warm extractor workers run than configured (the pool is refilling).",
                (),
                lambda: int(pool.stats()["degraded"]),
            )
            for name, key, text in (
                ("restarts", "restarts", "Extractor workers replaced after a timeout or a crash."),
                (
                    "recycled",
                    "recycled",
                    "Extractor workers replaced after their maximum number of requests or CPU budget.",
                ),
                ("start_failures", "start_failures", "Extractor workers that could not start."),
                ("idle_deaths", "idle_deaths", "Extractor workers found dead while idle (also restarts)."),
                ("timeouts", "timeouts", "Extractions that timed out (in the worker or waiting for one)."),
                ("requests", "requests", "Extractions sent to the warm workers."),
            ):
                counter = r.gauge(f"tabagent_extractor_{name}_total", text, (), stat(key))
                counter.kind = "counter"  # read from the pool's own counter at scrape time
        if self.drift is not None:
            self._drift_metrics()

    def _drift_metrics(self) -> None:
        r, monitor = self.registry, self.drift

        # a scrape reads the background thread's latest evaluation; /drift evaluates the window afresh
        def per_feature(key):
            def read():
                rep = monitor.latest()
                return {
                    (col,): (float(f[key]) if f[key] is not None else None)
                    for col, f in rep["features"].items()
                }

            return read

        r.gauge(
            "tabagent_drift_window_decisions",
            "Decisions in the drift window.",
            (),
            lambda: monitor.latest()["window"],
        )
        r.gauge(
            "tabagent_drift_features_drifting",
            "Features whose drift passes both its practical threshold and its no-drift limit.",
            (),
            lambda: len(monitor.latest()["drifting"]),
        )
        r.gauge(
            "tabagent_feature_psi",
            "Population stability index of each feature.",
            ("feature",),
            per_feature("psi"),
        )
        r.gauge(
            "tabagent_feature_null_rate_shift",
            "Live null rate minus training null rate, per feature.",
            ("feature",),
            per_feature("null_rate_shift"),
        )
        r.gauge(
            "tabagent_feature_unseen_share",
            "Share of live categorical values never seen in training, per feature.",
            ("feature",),
            per_feature("unseen_share"),
        )
        r.gauge(
            "tabagent_feature_drifting",
            "1 when the feature drifts.",
            ("feature",),
            lambda: {(c,): int(f["drifting"]) for c, f in monitor.latest()["features"].items()},
        )

    # ------------------------------------------------------------------ lifecycle
    def readiness(self) -> tuple[bool, Optional[str]]:  # noqa: UP045
        """Whether the service takes decisions, and why not: shutting down, an extractor that cannot run,
        or a warm pool with fewer than ``min_ready`` running workers. A pool that is not warmed at start-up
        (``warm=False``) starts its workers at the first decisions and is ready at once."""
        if self.closing:
            return False, "the service is shutting down"
        broken = getattr(self.sandbox, "broken", None)
        if broken:
            return False, f"the extractor cannot run: {broken.strip().splitlines()[-1]}"
        if self.pool is None or not self.warm_on_start:
            return True, None
        if self.pool.ready:
            return True, None
        s = self.pool.stats()
        reason = f"{s['live']} of {s['size']} extractor workers run (ready needs {s['min_ready']})"
        if s["start_failures"]:
            reason += f"; {s['start_failures']} could not start"
        return False, reason + (f": {self.warm_error}" if self.warm_error else "")

    def warm(self) -> None:
        """Start the head's warm extractor workers (called in the background at start-up)."""
        if getattr(self.head, "extractor_code", None) is None or self.sandbox is None:
            return
        try:
            n = self.head.warm()
        except Exception as e:
            self.warm_error = (
                f"{type(e).__name__}: {str(e).strip().splitlines()[-1] if str(e).strip() else ''}"
            )
            log.error(
                "the extractor workers could not start: %s",
                self.warm_error,
                extra={"fields": {"event": "workers_failed", "error_type": type(e).__name__}},
            )
            return
        log.info("extractor workers started", extra={"fields": {"event": "workers_started", "workers": n}})

    def close(self) -> None:
        """Stop taking decisions, then stop the decision threads, the drift thread and the extractor
        workers."""
        self.closing = True
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.drift is not None and hasattr(self.drift, "close"):
            self.drift.close()
        close = getattr(self.head, "close", None)
        if callable(close):
            close()
        log.info("service stopped", extra={"fields": {"event": "stopped", "head_id": self.manifest["id"]}})

    # ------------------------------------------------------------------ decisions
    def _call(self, req, trace: Optional[dict]) -> dict[str, Any]:  # noqa: UP045
        live = {"run": req.run} if req.run is not None else {}
        if trace is not None:
            live["trace"] = trace
        prefix = req.prefix if req.prefix is not None or req.run is not None else {}
        self.inflight.inc()  # counted while a thread computes it, even past the request's deadline
        try:
            answer = self.head.decide(prefix, req.candidates, k=req.k, defer_below=req.defer_below, **live)
        finally:
            self.inflight.dec()
        frame = trace.pop("frame", None) if trace is not None else None
        if self.drift is not None and frame is not None:
            self.drift.observe(frame)  # queued: the drift thread does the work
        return answer

    def timed_out(self, req, why: str) -> dict[str, Any]:
        """The deferred answer to a decision past its deadline: the response's usual shape, with the request's
        candidate ids in their input order (``choice`` None for a single-class head), empty ``scores`` and
        no ``confidence``."""
        error = f"TimeoutError: {why}; deferred to the LLM"
        base = {"scores": {}, "confidence": None, "deferred": True, "error": error}
        if self.kind == "single":
            return {"choice": None, **base}
        return {"candidates": candidate_ids(req.candidates), **base}

    async def decide(
        self,
        req,
        request_id: str,
        endpoint: str = "decide",
        deadline: Optional[float] = None,  # noqa: UP045
    ) -> tuple[int, Any]:
        """Answer one decision request: (200, answer), (400, message) or (500, message). ``deadline`` (a
        ``time.monotonic`` time) overrides the request's own."""
        from tabagent.api import RequestError

        t0 = time.perf_counter()
        trace: Optional[dict[str, Any]] = {} if self.traceable else None  # noqa: UP045
        status, answer, timed_out, exc = 200, None, False, None
        if deadline is None and self.deadline is not None:
            deadline = time.monotonic() + self.deadline
        try:
            if deadline is not None and deadline <= time.monotonic():
                raise asyncio.TimeoutError
            future = asyncio.get_running_loop().run_in_executor(self.executor, self._call, req, trace)
            budget = None if deadline is None else max(0.0, deadline - time.monotonic())
            answer = await (asyncio.wait_for(future, budget) if budget is not None else future)
        except asyncio.TimeoutError:
            why = f"no answer within the request timeout ({self.request_timeout:g}s plus grace)"
            answer, timed_out = self.timed_out(req, why), True
        except RequestError as e:  # a request the head cannot answer as given
            status, exc = 400, e
        except Exception as e:  # noqa: BLE001 - logged with its class; the client gets a 500
            status, exc = 500, e
        self._record(req, request_id, endpoint, status, answer, timed_out, exc, trace or {}, t0)
        if status == 200:
            return 200, answer
        if status == 400:
            return 400, str(exc)
        return 500, f"internal error ({type(exc).__name__}); request {request_id}"

    def _record(self, req, request_id, endpoint, status, answer, timed_out, exc, trace, t0) -> None:
        total = time.perf_counter() - t0
        stages = {"total": total}
        if not timed_out:
            for stage in ("prefix", "extraction", "scoring"):
                if f"{stage}_s" in trace:
                    stages[stage] = trace[f"{stage}_s"]
        for stage, seconds in stages.items():
            self.latency.observe(seconds, stage=stage)
        kind = None if timed_out else trace.get("error_kind")
        reason = defer_reason(answer, timed_out, kind) if status == 200 else None
        cls = None
        if status == 200:
            self.decisions.inc(endpoint=endpoint, outcome="deferred" if answer.get("deferred") else "head")
            if reason:
                self.deferrals.inc(reason=reason)
            cls = "request_timeout" if timed_out else error_class(answer.get("error"), kind)
        else:
            cls = "bad_request" if status == 400 else "internal"
        if cls:
            self.errors.inc(**{"class": cls})
        error = (answer or {}).get("error") if status == 200 else str(exc)
        fields: dict[str, Any] = {
            "event": "decision",
            "request_id": request_id,
            "head_id": self.manifest["id"],
            "endpoint": endpoint,
            "status": status,
            "latency_ms": {k: round(v * 1000, 3) for k, v in stages.items()},
            "deferred": bool(answer and answer.get("deferred")),
            "defer_reason": reason,
            "error_class": cls,
            "error_type": type(exc).__name__ if exc is not None else error_type(error),
            "n_candidates": len(req.candidates or []),
            "k": req.k,
            "confidence": (answer or {}).get("confidence"),
            "input": "run" if req.run is not None else "prefix",
        }
        if self.log_content:
            fields["error"] = error
            fields["request"] = {"prefix": req.prefix, "run": req.run, "candidates": req.candidates}
            fields["answer"] = answer
        if status == 200:
            level = logging.WARNING if cls else logging.INFO
            outcome = f"deferred:{reason}" if reason else "decided"
        else:
            level, outcome = (logging.WARNING if status == 400 else logging.ERROR), cls
        log.log(
            level,
            f"{endpoint} {status} {outcome}",
            extra={"fields": fields},
            exc_info=exc if status == 500 else None,
        )

    def version(self) -> dict[str, Any]:
        import tabagent
        from tabagent.citation import record

        names = {
            "WarmSandbox": "warm",
            "SubprocessSandbox": "subprocess",
            "DockerSandbox": "docker",
            "RefusingSandbox": "refused",
        }
        mode = "none"
        if getattr(self.head, "extractor_code", None):
            if getattr(self.head, "trust_extractor", False):
                mode = "in-process"
            elif self.sandbox is not None:
                mode = names.get(type(self.sandbox).__name__, type(self.sandbox).__name__)
        return {
            "tabagent_version": tabagent.__version__,
            "head": self.manifest,
            "serving": {
                "extractor": mode,
                "workers": self.pool.size if self.pool is not None else None,
                "request_timeout_s": self.request_timeout,
                "drift": "on" if self.drift is not None else "off",
            },
            "citation": record(),
        }

    def drift_report(self) -> dict[str, Any]:
        if self.drift is None:
            return {"status": "unavailable", "reason": self.drift_note}
        return self.drift.report()


def create_app(
    head,
    *,
    request_timeout: Optional[float] = None,  # noqa: UP045
    log_content: bool = False,
    drift: Any = True,
    drift_window: int = 500,
    threads: int = 8,
    max_batch: int = 256,
    warm: bool = True,
) -> Any:
    """The FastAPI application serving ``head`` (a ``TabAgentHead``).

    ``request_timeout``: seconds the extractor gets per decision, a positive number (see the module
    docstring; None: no deadline beyond the extractor sandbox's own). ``log_content``: also log each
    request's prefix, run, candidates, answer and error message (off by default: they are the agent's data).
    ``drift``: monitor feature drift against the head's training statistics (True), not at all (False), or
    with a given ``tabagent.drift.DriftMonitor``; ``drift_window`` decisions are compared. ``threads``:
    decisions computed at once. ``max_batch``: requests per ``/decide_batch`` call. ``warm``: start the warm
    extractor workers when the application starts (``/ready`` answers 503 until enough of them run); with
    False they start at the first decisions, and the service is ready at once.
    """
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, PlainTextResponse
        from pydantic import BaseModel
    except ImportError as e:
        raise ImportError(_SERVE_EXTRA) from e

    svc = Service(head, request_timeout, log_content, drift, drift_window, threads, max_batch, warm)

    class DecideRequest(BaseModel):
        prefix: Optional[dict[str, Any]] = None  # noqa: UP045
        run: Optional[dict[str, Any]] = None  # noqa: UP045
        candidates: Optional[list[Any]] = None  # noqa: UP045
        k: Optional[int] = None  # noqa: UP045
        defer_below: Optional[float] = None  # noqa: UP045

    class BatchRequest(BaseModel):
        requests: list[DecideRequest]

    @contextlib.asynccontextmanager
    async def lifespan(app):
        if warm:
            threading.Thread(target=svc.warm, daemon=True, name="tabagent-warm").start()
        log.info(
            "serving head %s",
            svc.manifest["id"],
            extra={"fields": {"event": "started", **{f"head_{k}": v for k, v in svc.manifest.items()}}},
        )
        yield
        svc.close()

    app = FastAPI(title="TabAgent decision head", version="1", lifespan=lifespan)
    app.state.service = svc
    routes = {"/health", "/ready", "/version", "/card", "/metrics", "/drift", "/decide", "/decide_batch"}

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        path = request.url.path if request.url.path in routes else "other"
        svc.http_requests.inc(endpoint=path, code=str(response.status_code))
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "target": head.target, "card": head.card.name}

    @app.get("/ready")
    def ready() -> Any:
        ok, reason = svc.readiness()
        stats = svc.sandbox.stats() if hasattr(svc.sandbox, "stats") else None
        body = {"ready": ok, "reason": reason, "head_id": svc.manifest["id"], "workers": stats}
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/version")
    def version() -> dict[str, Any]:
        return svc.version()

    @app.get("/card")
    def card() -> dict[str, Any]:
        return head.card.to_dict()

    @app.get("/metrics")
    def metrics() -> Any:
        return PlainTextResponse(svc.registry.render(), media_type=CONTENT_TYPE)

    @app.get("/drift")
    def drift_report() -> dict[str, Any]:
        return svc.drift_report()

    @app.post("/decide")
    async def decide(req: DecideRequest, request: Request) -> Any:
        status, body = await svc.decide(req, request.state.request_id)
        if status != 200:
            raise HTTPException(status_code=status, detail=body)
        return body

    @app.post("/decide_batch")
    async def decide_batch(batch: BatchRequest, request: Request) -> dict[str, Any]:
        if len(batch.requests) > svc.max_batch:
            raise HTTPException(status_code=400, detail=f"at most {svc.max_batch} requests per batch")
        rid = request.state.request_id
        # one deadline for the whole batch: what is still unanswered then is deferred
        deadline = None if svc.deadline is None else time.monotonic() + svc.deadline
        gate = asyncio.Semaphore(svc.pool.size if svc.pool is not None else 4)

        async def one(i: int, req: DecideRequest) -> Any:
            async with gate:
                status, body = await svc.decide(req, f"{rid}.{i}", "decide_batch", deadline)
            return body if status == 200 else {"status": status, "detail": body}

        return {"results": await asyncio.gather(*(one(i, r) for i, r in enumerate(batch.requests)))}

    return app


class RemoteHead:
    """A head served by ``tabagent serve``, called over HTTP with the standard library (no extra needed).

    ``decide`` takes the arguments of ``TabAgentHead.decide`` and returns the service's answer, so an agent
    whose environment cannot host the head (``ToolFilter("http://...")``, say) uses it like a local one. Send
    ``run=`` (the service builds the prefix with the head's recorded adapter) and JSON candidates (tool
    schemas, ids, ``{"id", "description", "args", "spec"}`` records, or ``Candidate`` objects, which are
    sent as such records, their catalog entries included). When the service cannot be reached, times
    out or fails (status 5xx), the decision is ``deferred`` with the reason in ``error``, as when a local
    extractor fails, in the service's own shape for a decision it could not make (every candidate id in input
    order, empty ``scores``); a request the service rejects (status 4xx) raises ``ValueError``.
    """

    def __init__(self, url: str, timeout: float = 10.0):
        self.url, self.timeout = url.rstrip("/"), timeout

    def decide(
        self,
        prefix: Optional[dict] = None,  # noqa: UP045
        candidates: Optional[list] = None,  # noqa: UP045
        k: Optional[int] = None,  # noqa: UP045
        defer_below: Optional[float] = None,  # noqa: UP045
        *,
        run: Optional[dict] = None,  # noqa: UP045
    ) -> dict[str, Any]:
        """The service's ``POST /decide`` answer (see ``TabAgentHead.decide``)."""
        from tabagent.tabschema.trajectory import Candidate, candidate_dict

        if candidates is not None:  # a Candidate travels as the dict decide reads back to the same candidate
            candidates = [candidate_dict(c) if isinstance(c, Candidate) else c for c in candidates]
        body = {"prefix": prefix, "run": run, "candidates": candidates, "k": k, "defer_below": defer_below}
        data = json.dumps({key: v for key, v in body.items() if v is not None}, default=str).encode()
        request = urllib.request.Request(
            self.url + "/decide", data=data, headers={"content-type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            if 400 <= e.code < 500:
                raise ValueError(f"the head service rejected the request ({e.code}): {detail}") from e
            error = f"the head service failed ({e.code}): {detail}"
        except (OSError, ValueError) as e:  # unreachable, timed out, or an answer that is not JSON
            error = f"the head service could not answer: {type(e).__name__}: {e}"
        # the service's own shape for a decision it could not make: every candidate id, in input order
        return {
            "candidates": candidate_ids(candidates),
            "scores": {},
            "confidence": None,
            "deferred": True,
            "error": error,
        }


def serve(
    path: str,
    host: str = "127.0.0.1",
    port: int = 8077,
    trust_extractor: bool = False,
    sandbox: Any = None,
    redact: Any = None,
    allow_unpinned: bool = False,
    *,
    allow_unchecked_extractor: bool = False,
    workers: Optional[int] = None,  # noqa: UP045
    request_timeout: Optional[float] = 10.0,  # noqa: UP045
    log_format: str = "json",
    log_content: bool = False,
    drift: bool = True,
    drift_window: int = 500,
    threads: int = 8,
    graceful_timeout: float = 30.0,
) -> None:
    """Load the head saved at ``path`` (a directory or ``hf://OWNER/REPO[@REVISION]``) and serve it until
    interrupted (SIGINT or SIGTERM: in-flight decisions finish, up to ``graceful_timeout`` seconds, then the
    workers stop). ``trust_extractor`` runs its extractor in-process instead of in the sandbox (only for an
    extractor you have reviewed); ``sandbox`` (``"warm"``, the default, ``"warm:<N>"``, ``"subprocess"``,
    ``"docker"``, ``"docker:<image>"``) and ``redact`` (a function or ``"module:function"``, applied to every
    ``run`` before its prefix is built) are described in ``TabAgentHead.load``, as are ``allow_unpinned``
    (serve a head whose card pins no hash for its extractor, with a warning) and ``allow_unchecked_extractor``
    (run, in the sandbox, an extractor that fails the extraction rules, with a warning). ``workers`` sizes
    the warm pool (default ``min(4, CPUs)``); the other options are ``create_app``'s."""
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(_SERVE_EXTRA) from e

    from tabagent.api import TabAgentHead

    request_timeout = _checked_timeout(request_timeout)
    if workers is not None:
        if trust_extractor or sandbox not in (None, "warm"):
            raise ValueError(
                f"workers= sizes the warm extractor pool; it does not apply to sandbox={sandbox!r}"
            )
        sandbox = f"warm:{workers}"
    configure_logging(log_format)
    # the library's per-failure warning quotes the extractor's error; the service logs its class instead
    warnings.filterwarnings("ignore", message="the TabSchema extractor failed", category=RuntimeWarning)
    head = TabAgentHead.load(
        path,
        trust_extractor=trust_extractor,
        sandbox=sandbox,
        redact=redact,
        allow_unpinned=allow_unpinned,
        allow_unchecked_extractor=allow_unchecked_extractor,
    )
    app = create_app(
        head,
        request_timeout=request_timeout,
        log_content=log_content,
        drift=drift,
        drift_window=drift_window,
        threads=threads,
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,
        access_log=False,
        timeout_graceful_shutdown=graceful_timeout,
    )
