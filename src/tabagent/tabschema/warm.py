"""Warm sandbox workers: persistent isolated interpreters for the live decisions of a served head (POSIX).

``SubprocessSandbox`` starts a fresh interpreter for every call. That suits onboarding, where one call runs
the extractor on thousands of prefixes, but a live decision runs it on one prefix, and the interpreter
start-up then dominates the decision's latency. ``WarmSandbox`` keeps a small pool of the same isolated
interpreters running, each with the extractor compiled once, and sends them one prefix at a time.

Isolation is that of ``SubprocessSandbox``: each worker is a ``python -I -S`` interpreter (isolated mode, no
site packages) in its own session and temporary directory, with an empty environment. Before it compiles the
code it sets the same file-size, core-dump and (on Linux) address-space limits. Its CPU-time limit applies per
request: before each request the worker sets the soft limit to the CPU time it has used plus ``cpu_s``, under
a hard limit of ``2 * cpu_s`` over its whole life, which the code cannot raise (a worker whose remaining
budget falls below ``cpu_s`` asks to be replaced).

Protocol. Requests and replies are JSON lines on private copies of the worker's stdin and stdout. Inside the
worker, file descriptor 0 then reads ``/dev/null`` and descriptor 1 writes to stderr, so the code can neither
read the next request nor corrupt a reply by printing. The parent reads replies and stderr with non-blocking
reads on raw descriptors, keeps the stderr of the current request as a short tail for its error message, and
accepts only replies that carry the worker's random token and the id of the pending request. The worker's
protocol state lives in a function's local scope, the code runs in a fresh namespace with its own copy of the
builtins, and the worker keeps private references to the JSON functions it uses: code that rebinds
``json.loads`` or a builtin affects only itself.

Failure handling. Each request has two deadlines. Inside the worker a ``SIGALRM`` timer raises
``TimeoutError`` when the code overruns, and the worker survives. The parent waits a short grace period
longer; a worker that has not answered by then (it ignores the alarm, or hangs in C code) is killed with its
whole process group. A worker that dies (``os._exit``, a crash, the memory or CPU limit) is detected as
well. Either way that request fails with an error that says so (``ExecResult.kind``), and after ``warm`` a
background thread keeps the pool at ``size`` workers, retrying a worker that cannot start with exponential
backoff (up to 30 s) and logging each failure. A worker is also replaced, not counted as a restart, after
``max_requests`` requests. Code that does not compile, or (``check_code``) breaks the extraction rules, makes
the pool ``broken`` for it: ``warm`` raises and every run fails at once, without starting interpreters.

Workers exit when their request channel closes (``close()``, garbage collection of the pool, interpreter
exit), and on their own within a second when the parent process dies, even mid-request. A process forked
from the parent (``multiprocessing``, a pre-forking server) drops the workers it inherited, without touching
their locks or streams, and starts its own.

This isolates accidents, not adversaries: see ``SECURITY.md``. A malicious extractor shares its worker with
later requests, so it could observe later prefixes and forge their answers; review the extractor, or use
``SubprocessSandbox`` or ``DockerSandbox`` for code you have not reviewed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import secrets
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tabagent.tabschema.sandbox import ExecResult

logger = logging.getLogger(__name__)

POSIX = sys.platform != "win32"
#: The static checks (``tabagent.tabschema.checks``) that ``check_code`` re-applies before running code.
SECURITY_ISSUES = ("syntax", "import", "call", "dunder", "internals")

_WORKER = r"""
def _tabagent_worker():
    import json, os, signal, sys, threading, time, traceback
    import builtins

    loads, dumps = json.loads, json.dumps  # private references: the code may rebind json's functions
    replies = os.fdopen(os.dup(1), "wb")
    requests = os.fdopen(os.dup(0), "rb")
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)  # the code reads an empty stdin
    os.close(null)
    os.dup2(2, 1)  # and what it prints goes to stderr
    sys.stdout = sys.stderr
    init = loads(requests.readline())
    token, lim, source = init["token"], init["limits"], init["code"]
    del init

    def send(msg):
        msg["token"] = token
        replies.write(dumps(msg, default=str).encode("utf-8") + b"\n")
        replies.flush()

    parent = os.getppid()

    def watch_parent():  # exit when the parent dies, even in the middle of a request
        while True:
            time.sleep(0.5)
            if os.getppid() != parent:
                os._exit(70)

    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        import resource
    except ImportError:
        resource = None

    def setlimit(kind, soft, hard):
        try:
            resource.setrlimit(kind, (soft, hard))
        except (ValueError, OSError):  # a limit the platform does not enforce (macOS: RLIMIT_AS)
            pass

    def cpu_used():
        use = resource.getrusage(resource.RUSAGE_SELF)
        return use.ru_utime + use.ru_stime

    cpu_s, cpu_hard = lim["cpu_s"], None
    if resource is not None:
        setlimit(resource.RLIMIT_FSIZE, lim["file_mb"] << 20, lim["file_mb"] << 20)
        setlimit(resource.RLIMIT_CORE, 0, 0)
        if lim.get("mem_mb"):
            setlimit(resource.RLIMIT_AS, lim["mem_mb"] << 20, lim["mem_mb"] << 20)
        inherited = resource.getrlimit(resource.RLIMIT_CPU)[1]
        cpu_hard = int(cpu_used()) + 1 + 2 * cpu_s  # the worker's whole budget; the code cannot raise it
        if inherited != resource.RLIM_INFINITY:
            cpu_hard = min(cpu_hard, inherited)
        setlimit(resource.RLIMIT_CPU, cpu_hard, cpu_hard)
    try:
        code = compile(source, "<extractor>", "exec")
    except BaseException:
        send({"ready": False, "stage": "compile", "error": traceback.format_exc(limit=8)})
        return
    del source
    alarm = hasattr(signal, "setitimer")
    state = {"limit": None, "fired": False}

    def expire(signum, frame):
        state["fired"] = True
        raise TimeoutError(f"timed out after {state['limit']:.3g}s on this trajectory")

    base = dict(builtins.__dict__)
    send({"ready": True, "pid": os.getpid()})
    while True:
        line = requests.readline()
        if not line:  # the parent closed the channel
            return
        req = loads(line)
        limit = req["timeout"] if alarm else None
        state["limit"], state["fired"] = limit, False
        sys.stdout = sys.stderr
        if cpu_hard is not None:
            setlimit(resource.RLIMIT_CPU, min(int(cpu_used()) + 1 + cpu_s, cpu_hard), cpu_hard)
        ns = {"__name__": "__main__", "__builtins__": dict(base), "trajectory": req["trajectory"]}
        try:
            if limit:
                signal.signal(signal.SIGALRM, expire)  # again each time: the code may have replaced it
                signal.setitimer(signal.ITIMER_REAL, limit)
            try:
                exec(code, ns)
            finally:
                if limit:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            feats = ns.get("features")
            if not isinstance(feats, dict):
                raise TypeError("the code must leave a dict named `features` in its global scope")
            dumps(feats)
            out = {"id": req["id"], "ok": True, "features": feats}
        except BaseException:
            kind = "timeout" if state["fired"] else "error"
            out = {"id": req["id"], "ok": False, "error": traceback.format_exc(limit=8), "kind": kind}
        del ns
        if cpu_hard is not None and cpu_used() + 1 + cpu_s > cpu_hard:
            out["retire"] = True  # less than cpu_s of CPU left for another request
        send(out)


_tabagent_worker()
"""

_EOF = object()
_TAIL = 2000  # characters of a request's stderr kept for its error message
_CHUNK = 1 << 16
_POLL = 0.25  # how often a waiting request checks that its worker is still alive
_MAX_BACKOFF = 30.0  # the longest pause between worker starts after failures or idle deaths
_STABLE = 60.0  # seconds without a failed start or an idle death that reset the backoff
_POOLS: weakref.WeakSet = weakref.WeakSet()  # every pool of this process, to reset in a forked child
_ABANDONED: list = []  # workers a forked child inherited: kept so their objects never close reused fds


def _grace(timeout: float) -> float:
    """How much longer than the worker's own alarm the parent waits before it kills the worker."""
    return 1.0 + 0.1 * timeout


def _checked_timeout(value: Any, name: str = "timeout_s") -> float:
    t = float(value)
    if not math.isfinite(t) or t <= 0:
        raise ValueError(f"{name} must be a positive number of seconds, not {value!r}")
    return t


def _digest(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def static_refusal(code: str) -> str | None:
    """Why ``code`` must not run, by the security-relevant static checks of the compile step (a syntax
    error, an import outside the allowed standard-library modules, a forbidden builtin, an interpreter
    internal), or None when it passes them."""
    from tabagent.tabschema.checks import check_code

    issues = [str(i) for i in check_code(code) if i.kind in SECURITY_ISSUES]
    if not issues:
        return None
    return "extract.py violates the extraction rules, so it is not run: " + "; ".join(issues[:5])


class WorkerStartError(RuntimeError):
    """A sandbox worker did not start (``kind="unavailable"``), or the code cannot run (``"broken"``)."""

    def __init__(self, message: str, kind: str = "unavailable"):
        super().__init__(message)
        self.kind = kind


class ExtractorBrokenError(WorkerStartError):
    """The extractor cannot run at all: it does not compile, or it breaks the extraction rules."""

    def __init__(self, message: str):
        super().__init__(message, "broken")


class RefusingSandbox:
    """The sandbox of an extractor that breaks the extraction rules (``static_refusal``): it never runs the
    code. Every run fails with the reason (``kind="broken"``), ``warm`` raises ``ExtractorBrokenError``, and
    ``broken`` holds the reason."""

    ready = False

    def __init__(self, reason: str):
        self.broken = reason

    def run(self, code: str, trajectory: dict[str, Any], timeout_s: float | None = None) -> ExecResult:
        return ExecResult(False, error=self.broken, kind="broken")

    def run_batch(self, code: str, trajectories: list[dict[str, Any]]) -> list[ExecResult]:
        return [self.run(code, t) for t in trajectories]

    def warm(self, code: str) -> int:
        raise ExtractorBrokenError(self.broken)

    def stats(self) -> dict[str, Any]:
        return {"ready": False, "broken": self.broken}

    def close(self, wait_s: float = 0.0) -> None:
        pass


class _NotSent(Exception):
    """The request could not be sent: the worker had died while idle."""


class _Worker:
    """One persistent sandboxed interpreter running one extractor (see the module docstring). A worker is
    used by one thread at a time; only that thread (its owner) reads, writes or closes its descriptors."""

    def __init__(self, python: str, code: str, limits: dict[str, Any], start_timeout_s: float):
        self.code, self.requests, self.token = code, 0, secrets.token_hex(16)
        self.retire_requested = self.closing = self.dead = self.disposed = False
        self._kill_lock = threading.Lock()
        self.workdir = tempfile.mkdtemp(prefix="tabschema_warm_")
        r_in, w_in = os.pipe()
        r_out, w_out = os.pipe()
        r_err, w_err = os.pipe()
        try:
            self.proc = subprocess.Popen(
                [python, "-I", "-S", "-c", _WORKER],
                stdin=r_in,
                stdout=w_out,
                stderr=w_err,
                cwd=self.workdir,
                env={},
                close_fds=True,
                start_new_session=True,  # its own process group, killed as a whole; no terminal signals
            )
        except BaseException:
            for fd in (r_in, w_in, r_out, w_out, r_err, w_err):
                os.close(fd)
            shutil.rmtree(self.workdir, ignore_errors=True)
            raise
        for fd in (r_in, w_out, w_err):  # the worker's ends
            os.close(fd)
        self.pid = self.proc.pid
        self.fd_in, self.fd_out, self.fd_err = w_in, r_out, r_err
        for fd in (w_in, r_out, r_err):
            os.set_blocking(fd, False)
        self._sel = selectors.DefaultSelector()
        self._sel.register(r_out, selectors.EVENT_READ, "out")
        self._sel.register(r_err, selectors.EVENT_READ, "err")
        self._buf, self._scanned, self._tail = bytearray(), 0, bytearray()
        self._out_open = self._err_open = True
        init = {"token": self.token, "code": code, "limits": limits}
        try:
            rec = self._exchange(json.dumps(init).encode() + b"\n", lambda r: "ready" in r, start_timeout_s)
        except BaseException:
            self.destroy()
            raise
        if rec is None or rec is _EOF or not rec.get("ready"):
            if rec is None:
                error, kind = f"the sandbox worker did not start within {start_timeout_s:g}s", "unavailable"
            elif rec is _EOF:
                error, kind = self.exit_error(), "unavailable"
            else:
                error = (rec.get("error") or "").strip()
                kind = "broken" if rec.get("stage") == "compile" else "unavailable"
            self.destroy()
            cls = ExtractorBrokenError if kind == "broken" else WorkerStartError
            raise (cls(error) if kind == "broken" else cls(f"the sandbox worker could not start: {error}"))
        self.pid = rec.get("pid", self.pid)

    # ------------------------------------------------------------------ channels
    def _reply(self, raw: bytes) -> dict | None:
        """A reply line of this worker (with its token), or None for any other line."""
        try:
            rec = json.loads(raw)
        except ValueError:
            return None
        return rec if isinstance(rec, dict) and rec.get("token") == self.token else None

    def _next_line(self) -> bytes | None:
        """The next complete line of the reply buffer (only the bytes not scanned yet are searched)."""
        idx = self._buf.find(b"\n", self._scanned)
        if idx < 0:
            self._scanned = len(self._buf)
            return None
        line = bytes(self._buf[:idx])
        del self._buf[: idx + 1]
        self._scanned = 0
        return line

    def _read(self, fd: int) -> bytes | None:
        """A chunk from ``fd``: b"" at EOF, None when nothing is ready."""
        try:
            return os.read(fd, _CHUNK)
        except BlockingIOError:
            return None
        except OSError:
            return b""

    def _drain_stderr(self) -> None:
        while self._err_open:
            chunk = self._read(self.fd_err)
            if chunk is None:
                return
            if not chunk:
                self._err_open = False
                with contextlib.suppress(KeyError, ValueError):
                    self._sel.unregister(self.fd_err)
                return
            self._tail += chunk
            if len(self._tail) > 4 * _TAIL:
                del self._tail[: -2 * _TAIL]

    def _exchange(self, payload: bytes | None, match, timeout: float):
        """Write ``payload`` (if any), then return the next reply that ``match`` accepts, ``_EOF`` when the
        worker closed its reply channel, or None at the deadline. Raises ``_NotSent`` when the worker is gone
        before the payload was written."""
        deadline = time.monotonic() + timeout
        pending = memoryview(payload) if payload else None
        if pending is not None:
            self._sel.register(self.fd_in, selectors.EVENT_WRITE, "in")
        try:
            while True:
                while (line := self._next_line()) is not None:
                    rec = self._reply(line)
                    if rec is not None and match(rec):
                        return rec
                if not self._out_open:
                    self._drain_stderr()
                    return _EOF
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                events = self._sel.select(min(remaining, _POLL))
                if not events and not self.alive():  # it died, though a descendant holds its pipes open
                    self._drain_stderr()
                    return _EOF
                for key, _ in events:
                    if key.data == "in":
                        try:
                            pending = pending[os.write(self.fd_in, pending) :]
                        except BlockingIOError:
                            continue
                        except OSError as e:  # EPIPE: the worker is gone
                            raise _NotSent from e
                        if not len(pending):
                            self._sel.unregister(self.fd_in)
                            pending = None
                    elif key.data == "out":
                        chunk = self._read(self.fd_out)
                        if chunk == b"":
                            self._out_open = False
                            self._sel.unregister(self.fd_out)
                        elif chunk:
                            self._buf += chunk
                    else:
                        self._drain_stderr()
        finally:
            if pending is not None:
                with contextlib.suppress(KeyError, ValueError):
                    self._sel.unregister(self.fd_in)

    # ------------------------------------------------------------------ requests
    def alive(self) -> bool:
        """Whether the worker process runs (without reaping it where ``os.waitid`` exists, so its process
        group stays ours). Without it (macOS before Python 3.13), the worker is polled, which reaps it once it
        has exited; its process group, still in use by anything left in it, is killed right away, under the
        kill lock."""
        if self.dead:
            return False
        if hasattr(os, "waitid"):
            try:
                return os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None
            except ChildProcessError:
                return False
        with self._kill_lock:
            if self.proc.returncode is not None:
                return False
            if self.proc.poll() is None:
                return True
            with contextlib.suppress(OSError):  # a group outlives its leader while a member is left
                os.killpg(self.pid, signal.SIGKILL)
            return False

    def call(self, trajectory_json: str, timeout: float) -> tuple[ExecResult, bool]:
        """Run the code on one prefix (already JSON). Returns the result and whether the worker is still
        usable (False after a timeout kill or a crash). Raises ``_NotSent`` when it died while idle."""
        self.requests += 1
        rid = self.requests
        self._tail = bytearray()  # this request's stderr only
        line = f'{{"id": {rid}, "timeout": {timeout!r}, "trajectory": {trajectory_json}}}\n'.encode()
        rec = self._exchange(line, lambda r: r.get("id") == rid, timeout + _grace(timeout))
        if rec is None:
            self.kill()
            if self.closing:
                return ExecResult(False, error="the warm sandbox was closed", kind="unavailable"), False
            error = f"timed out after {timeout:.3g}s on this trajectory; the sandbox worker was killed"
            return ExecResult(False, error=error, kind="timeout"), False
        if rec is _EOF:
            if self.closing:
                return ExecResult(False, error="the warm sandbox was closed", kind="unavailable"), False
            return ExecResult(False, error=self.exit_error(), kind="crash"), False
        self.retire_requested = bool(rec.get("retire"))
        kind = None if rec["ok"] else rec.get("kind", "error")
        return ExecResult(rec["ok"], rec.get("features"), rec.get("error"), kind), True

    def exit_error(self) -> str:
        """Why the worker is gone: its exit status and the tail of its stderr for the current request."""
        self.kill()
        self._drain_stderr()
        tail = self._tail.decode("utf-8", "replace").strip()[-_TAIL:]
        error = f"the sandboxed interpreter exited early (exit status {self.proc.returncode})"
        return error + (f":\n{tail}" if tail else "")

    # ------------------------------------------------------------------ shutdown
    def kill(self) -> None:
        """Kill the worker and its whole process group, and reap it (bounded; any thread may call it). The
        lock keeps one thread from signalling a process group that another has just reaped (whose id the
        system may then reuse)."""
        with self._kill_lock:
            if self.proc.returncode is None:  # not reaped yet: its pid, and so its group, is still ours
                with contextlib.suppress(OSError):
                    os.killpg(self.pid, signal.SIGKILL)
                with contextlib.suppress(OSError):
                    self.proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                self.proc.wait(timeout=5)
            self.dead = True

    def stop(self, wait_s: float = 2.0) -> None:
        """Close the request channel (an idle worker exits at once), then kill what is left (owner only)."""
        if not self.disposed and self.fd_in >= 0:
            with contextlib.suppress(OSError):
                os.close(self.fd_in)
            self.fd_in = -1
        end = time.monotonic() + max(0.0, wait_s)
        while self.alive() and time.monotonic() < end:
            time.sleep(0.01)
        self.destroy()

    def destroy(self) -> None:
        """Kill the worker and close its descriptors (owner only)."""
        self.kill()
        if self.disposed:
            return
        self.disposed = True
        with contextlib.suppress(OSError, ValueError):
            self._sel.close()
        for fd in (self.fd_in, self.fd_out, self.fd_err):
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)
        self.fd_in = self.fd_out = self.fd_err = -1
        shutil.rmtree(self.workdir, ignore_errors=True)

    def abandon(self) -> None:
        """In a forked child: close the inherited pipe ends and forget the worker, which belongs to the
        parent. No lock, wait or stream object is touched (a thread of the parent may have held one)."""
        fds = [self.fd_in, self.fd_out, self.fd_err]
        if sys.platform.startswith("linux"):  # an epoll descriptor is inherited (a kqueue one is not)
            with contextlib.suppress(OSError, ValueError, AttributeError):
                fds.append(self._sel.fileno())
        for fd in fds:
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)
        self.fd_in = self.fd_out = self.fd_err = -1
        self.disposed = self.dead = True
        _ABANDONED.append(self)


def _stop_all(workers: set) -> None:
    """Finalizer of a pool: stop its workers (it must not reference the pool itself)."""
    for w in list(workers):
        if w.disposed:
            continue
        w.destroy()
    workers.clear()


def _after_fork_in_child() -> None:
    for pool in list(_POOLS):
        pool._forget_after_fork()


if POSIX:
    os.register_at_fork(after_in_child=_after_fork_in_child)


def _replenish(ref: weakref.ref) -> None:
    """The background thread of a warm pool: keep ``size`` workers running. A worker that cannot start, or
    workers that die while idle, delay the next start with exponential backoff (0.1 s, doubling up to 30 s,
    reset after ``_STABLE`` seconds without trouble), so an environment that kills them cannot make the pool
    spin. It holds the pool only weakly, so an unreferenced pool is still collected."""
    backoff, retry_at, trouble_at = 0.0, 0.0, 0.0

    def trouble() -> None:
        nonlocal backoff, retry_at, trouble_at
        backoff = min(_MAX_BACKOFF, backoff * 2 or 0.1)
        trouble_at = time.monotonic()
        retry_at = trouble_at + backoff

    while True:
        pool = ref()
        if pool is None:
            return
        if backoff and time.monotonic() - trouble_at > _STABLE:
            backoff = 0.0
        action, arg, died = pool._replenish_action(retry_at)
        for _ in range(died):  # each death doubles the pause: a pool whose workers all die backs off fast
            trouble()
        if action == "exit":
            return
        if action == "wait":
            cond = pool._cond
            del pool
            with cond:
                cond.wait(arg)
            continue
        try:
            worker = pool._spawn(arg)
        except Exception:  # counted and logged by _spawn
            trouble()
            continue
        pool._add_idle(worker)


class WarmSandbox:
    """A pool of persistent sandboxed interpreters for live extraction (see the module docstring).

    Parameters
    ----------
    size: the number of workers, and so of extractions that run at once (default ``min(4, CPUs)``). Workers
        start on demand, up to ``size``; ``warm`` starts them all and keeps the pool full from then on.
    timeout_s: the default per-request timeout; the time spent waiting for a free worker counts against it.
    cpu_s, file_mb, mem_mb: the CPU time per request, the file size and the address space of each worker
        (the CPU and memory limits are enforced on Linux; macOS ignores them).
    python: the interpreter (default ``sys.executable``).
    max_requests: a worker is replaced after this many requests.
    start_timeout_s: how long a worker may take to start and compile the code.
    min_ready: ``ready`` needs this many running workers (a pool with fewer than ``size`` is ``degraded``).
    check_code: refuse code that fails the security-relevant static checks of the compile step
        (``static_refusal``) before any interpreter runs it.

    ``run(code, trajectory)`` and ``run_batch(code, trajectories)`` behave as ``SubprocessSandbox``'s and
    return ``ExecResult``s; they never raise for the code's sake. Each worker holds one extractor: a call
    with other code replaces idle workers with workers for that code, so one pool should serve one
    extractor. ``stats()`` reports the pool's state and counters, and ``close()`` stops it. POSIX only.
    """

    def __init__(
        self,
        size: int | None = None,
        timeout_s: float = 30,
        cpu_s: int = 60,
        file_mb: int = 16,
        python: str | None = None,
        mem_mb: int | None = 2048,
        max_requests: int = 5000,
        start_timeout_s: float = 30,
        min_ready: int = 1,
        check_code: bool = False,
    ):
        if not POSIX:
            raise NotImplementedError("WarmSandbox needs POSIX; use SubprocessSandbox on Windows")
        self.size = max(1, size or min(4, os.cpu_count() or 1))
        self.timeout_s = _checked_timeout(timeout_s)
        self.cpu_s, self.file_mb, self.mem_mb = int(cpu_s), int(file_mb), mem_mb
        self.python = python or sys.executable
        self.max_requests = max(1, max_requests)
        self.start_timeout_s = _checked_timeout(start_timeout_s, "start_timeout_s")
        self.min_ready = max(1, min(min_ready, self.size))
        self.check_code = check_code
        self._pid = os.getpid()
        self._cond = threading.Condition()
        self._idle: list[_Worker] = []
        self._workers: set[_Worker] = set()  # live workers, idle or busy
        self._busy: set[_Worker] = set()
        self._starting = 0
        self._warm_code: str | None = None  # after warm(): keep `size` workers running for this code
        self._broken: dict[str, str] = {}  # code digest -> why it cannot run
        self._checked: set[str] = set()  # code digests that passed (or failed) the static checks
        self._digests: dict[str, str] = {}  # code -> its digest
        self._replenisher: threading.Thread | None = None
        self._closed = False
        self.restarts = 0  # workers replaced after a timeout or a crash (or found dead)
        self.recycled = 0  # workers replaced after max_requests or their CPU budget
        self.started = 0  # workers started
        self.start_failures = 0  # workers that could not start
        self.idle_deaths = 0  # workers found dead while idle (also counted in restarts)
        self.requests = 0
        self.timeouts = 0
        self._finalizer = weakref.finalize(self, _stop_all, self._workers)
        _POOLS.add(self)

    # ------------------------------------------------------------------ public API
    def run(self, code: str, trajectory: dict[str, Any], timeout_s: float | None = None) -> ExecResult:
        """Run ``code`` on one trajectory in a warm worker; ``timeout_s`` overrides the pool's timeout."""
        timeout = self.timeout_s if timeout_s is None else _checked_timeout(timeout_s)
        deadline = time.monotonic() + timeout
        self._check_fork()
        refused = self._refusal(code)
        if refused is not None:
            return refused
        try:
            payload = json.dumps(trajectory, default=str)
        except (TypeError, ValueError) as e:  # as SubprocessSandbox: the payload cannot be sent
            return ExecResult(False, error=f"{type(e).__name__}: {e}", kind="error")
        for attempt in range(2):  # a second worker when the first one had died while idle
            try:
                worker = self._acquire(code, deadline)
            except WorkerStartError as e:  # as SubprocessSandbox: code that cannot start fails its trajectory
                return ExecResult(False, error=str(e), kind=e.kind)
            if worker is _CLOSED:
                return ExecResult(False, error="the warm sandbox is closed", kind="unavailable")
            if worker is None:
                with self._cond:
                    self.timeouts += 1
                return ExecResult(
                    False,
                    error=f"timed out after {timeout:.3g}s waiting for a free sandbox worker "
                    f"(all {self.size} busy)",
                    kind="timeout",
                )
            remaining = max(0.001, deadline - time.monotonic())
            try:
                result, healthy = worker.call(payload, remaining)
            except _NotSent:
                self._release(worker, healthy=False, failed=True)
                if attempt == 0:
                    continue
                return ExecResult(False, error=worker.exit_error(), kind="crash")
            except BaseException:
                self._release(worker, healthy=False, failed=True)
                raise
            if result.kind == "timeout":
                with self._cond:
                    self.timeouts += 1
            self._release(worker, healthy, failed=not healthy and result.kind != "unavailable")
            return result
        raise AssertionError("unreachable")  # pragma: no cover

    def run_batch(self, code: str, trajectories: list[dict[str, Any]]) -> list[ExecResult]:
        """Run ``code`` once per trajectory, spread over the pool's workers; one result per trajectory, in
        order."""
        if len(trajectories) <= 1:
            return [self.run(code, t) for t in trajectories]
        with ThreadPoolExecutor(min(self.size, len(trajectories))) as ex:
            return list(ex.map(lambda t: self.run(code, t), trajectories))

    def warm(self, code: str) -> int:
        """Start workers for ``code`` until the pool has ``size`` of them, and keep it that way from now on:
        a worker that dies, is retired or cannot start is replaced in the background. Returns the number of
        live workers. Raises ``ExtractorBrokenError`` when the code cannot run at all (it does not compile,
        or breaks the extraction rules), and ``WorkerStartError`` when no worker could start."""
        self._check_fork()
        with self._cond:
            if self._closed:
                raise RuntimeError("the warm sandbox is closed")
            self._warm_code = code
        if self._refusal(code) is not None:
            raise ExtractorBrokenError(self._broken[_digest(code)])
        with self._cond:
            missing = self.size - len(self._workers) - self._starting
            self._starting += max(0, missing)
        errors: list[BaseException] = []
        if missing > 0:

            def start(_):
                try:
                    self._add_idle(self._spawn(code))
                except Exception as e:  # counted and logged by _spawn; the replenisher retries
                    errors.append(e)

            with ThreadPoolExecutor(missing) as ex:
                list(ex.map(start, range(missing)))
        with self._cond:
            self._ensure_replenisher()
            live = len(self._workers)
        broken = self._broken.get(_digest(code))
        if broken is not None:
            raise ExtractorBrokenError(broken)
        if live == 0 and errors:
            raise WorkerStartError(f"no sandbox worker could start ({errors[0]})")
        return live

    @property
    def broken(self) -> str | None:
        """Why the extractor this pool was warmed with cannot run at all, or None."""
        code = self._warm_code
        return None if code is None else self._broken.get(_digest(code))

    @property
    def ready(self) -> bool:
        """True after ``warm`` while at least ``min_ready`` workers run and the extractor can run."""
        with self._cond:
            return self._ready_locked()

    def _ready_locked(self) -> bool:
        return (
            self._warm_code is not None
            and not self._closed
            and self.broken is None
            and len(self._workers) >= self.min_ready
        )

    def stats(self) -> dict[str, Any]:
        """The pool's state (``size``, ``min_ready``, ``live``, ``idle``, ``busy``, ``starting``, ``ready``,
        ``degraded``: fewer than ``size`` live workers after ``warm``, ``broken``: why the extractor cannot
        run) and counters (``started``, ``start_failures``, ``idle_deaths``, ``restarts``, ``recycled``,
        ``requests``, ``timeouts``)."""
        with self._cond:
            live = len(self._workers)
            return {
                "size": self.size,
                "min_ready": self.min_ready,
                "live": live,
                "idle": len(self._idle),
                "busy": len(self._busy),
                "starting": self._starting,
                "ready": self._ready_locked(),
                "degraded": self._warm_code is not None and not self._closed and live < self.size,
                "broken": self.broken,
                "started": self.started,
                "start_failures": self.start_failures,
                "idle_deaths": self.idle_deaths,
                "restarts": self.restarts,
                "recycled": self.recycled,
                "requests": self.requests,
                "timeouts": self.timeouts,
            }

    def pids(self) -> list[int]:
        """The process ids of the live workers."""
        with self._cond:
            return sorted(w.pid for w in self._workers)

    def close(self, wait_s: float = 2.0) -> None:
        """Stop the pool: idle workers at once; busy ones may finish their request for ``wait_s`` seconds,
        then they are killed and their requests fail. Later runs fail at once (``kind="unavailable"``)."""
        self._check_fork()
        with self._cond:
            self._closed = True
            idle, self._idle = self._idle, []
            for w in idle:
                self._workers.discard(w)
            self._cond.notify_all()
        for w in idle:  # owned by this thread now
            w.stop(min(wait_s, 1.0))
        end = time.monotonic() + max(0.0, wait_s)
        with self._cond:
            while self._busy and time.monotonic() < end:
                self._cond.wait(end - time.monotonic())
            busy = list(self._busy)
        for w in busy:  # their request threads own them: kill only, the threads clean up
            w.closing = True
            w.kill()
        thread = self._replenisher
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)

    def __enter__(self) -> WarmSandbox:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ pool internals
    def _check_fork(self) -> None:
        if self._pid != os.getpid():
            self._forget_after_fork()

    def _forget_after_fork(self) -> None:
        """In a forked child: drop the parent's workers and threads, and start afresh (lazily)."""
        if self._pid == os.getpid():
            return
        self._pid = os.getpid()
        self._cond = threading.Condition()  # the parent's may have been held by one of its threads
        for w in list(self._workers):
            w.abandon()
        self._workers, self._idle, self._busy = set(), [], set()
        self._starting = 0
        self._replenisher = None
        self._finalizer.detach()
        self._finalizer = weakref.finalize(self, _stop_all, self._workers)

    def _limits(self) -> dict[str, Any]:
        return {"cpu_s": self.cpu_s, "file_mb": self.file_mb, "mem_mb": self.mem_mb}

    def _refusal(self, code: str) -> ExecResult | None:
        """The result of every run of ``code`` when it cannot run at all, else None."""
        key = self._digests.get(code) or self._digests.setdefault(code, _digest(code))
        if self.check_code and key not in self._checked:
            reason = static_refusal(code)  # once per extractor
            if reason is not None:
                self._mark_broken(key, reason)
            self._checked.add(key)
        reason = self._broken.get(key)
        return None if reason is None else ExecResult(False, error=reason, kind="broken")

    def _mark_broken(self, key: str, reason: str) -> None:
        with self._cond:
            first = key not in self._broken
            self._broken[key] = reason
            self._cond.notify_all()
        if first:
            logger.error(
                "the extractor cannot run, so every decision is deferred: %s",
                reason.strip().splitlines()[-1] if reason.strip() else reason,
                extra={"fields": {"event": "extractor_broken"}},
            )

    def _spawn(self, code: str) -> _Worker:
        """Start a worker (the caller has reserved its slot in ``_starting``). Raises ``WorkerStartError``
        (an ``ExtractorBrokenError`` for code that cannot run), whatever the cause: a missing interpreter, a
        failed fork, a slow start."""
        try:
            worker = _Worker(self.python, code, self._limits(), self.start_timeout_s)
        except BaseException as e:
            with self._cond:
                self._starting -= 1
                self.start_failures += 1
                self._cond.notify_all()
            if isinstance(e, ExtractorBrokenError):
                self._mark_broken(_digest(code), str(e))
                raise
            if not isinstance(e, Exception):
                raise
            logger.error(
                "a sandbox worker could not start: %s: %s",
                type(e).__name__,
                e,
                extra={"fields": {"event": "worker_start_failed", "error_type": type(e).__name__}},
            )
            if isinstance(e, WorkerStartError):
                raise
            raise WorkerStartError(f"the sandbox worker could not start: {type(e).__name__}: {e}") from e
        with self._cond:
            self._starting -= 1
            self.started += 1
            closed = self._closed or self._pid != os.getpid()
            if not closed:
                self._workers.add(worker)
            self._cond.notify_all()
        if closed:
            worker.destroy()
            raise WorkerStartError("the warm sandbox is closed")
        return worker

    def _add_idle(self, worker: _Worker) -> None:
        with self._cond:
            if worker in self._workers and not self._closed:
                self._idle.append(worker)
                self._cond.notify_all()
                return
        worker.destroy()

    def _ensure_replenisher(self) -> None:
        """Start the background replenisher if the pool was warmed (lock held)."""
        if self._warm_code is None or self._closed:
            return
        if self._replenisher is None or not self._replenisher.is_alive():
            self._replenisher = threading.Thread(
                target=_replenish, args=(weakref.ref(self),), daemon=True, name="tabagent-warm-pool"
            )
            self._replenisher.start()

    def _replenish_action(self, retry_at: float) -> tuple[str, Any, int]:
        """What the replenisher does next: ("exit", None), ("wait", seconds) or ("spawn", code) with a slot
        reserved, and how many idle workers it found dead (removed, counted and logged first)."""
        dead = []
        with self._cond:
            if self._closed or self._warm_code is None or self._pid != os.getpid() or self.broken:
                return "exit", None, 0
            for w in list(self._idle):
                if not w.alive():
                    self._idle.remove(w)
                    self._workers.discard(w)
                    dead.append(w)
            self.restarts += len(dead)
            self.idle_deaths += len(dead)
            missing = self.size - len(self._workers) - self._starting
            now = time.monotonic()
            if dead or missing <= 0 or now < retry_at:
                wait = 1.0 if missing <= 0 else max(0.01, min(1.0, retry_at - now))
                action: tuple[str, Any] = ("wait", wait)
            else:
                self._starting += 1
                action = ("spawn", self._warm_code)
        for w in dead:
            self._log_idle_death(w)
        return (*action, len(dead))

    def _log_idle_death(self, worker: _Worker) -> None:
        worker.destroy()
        logger.warning(
            "an idle sandbox worker died (exit status %s); it is replaced",
            worker.proc.returncode,
            extra={"fields": {"event": "worker_died_idle", "exit_status": worker.proc.returncode}},
        )

    def _acquire(self, code: str, deadline: float):
        """A worker for ``code``: an idle one, else a new one while the pool has room, else the first one
        released before ``deadline`` (None after it; ``_CLOSED`` once the pool is closed)."""
        stale = []
        try:
            with self._cond:
                while True:
                    if self._closed:
                        return _CLOSED
                    while self._idle:
                        w = self._idle.pop()
                        alive = w.alive()
                        if w.code == code and alive:
                            self._busy.add(w)
                            self.requests += 1
                            return w
                        self._workers.discard(w)  # dead, or another extractor's: replace it
                        self.restarts += not alive  # it died while idle (killed from outside, say)
                        self.idle_deaths += not alive
                        stale.append((w, alive))
                        self._cond.notify_all()
                    if len(self._workers) + self._starting < self.size:
                        self._starting += 1
                        self.requests += 1
                        break
                    self._ensure_replenisher()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(remaining)
        finally:
            for w, alive in stale:
                if alive:
                    w.destroy()
                else:
                    self._log_idle_death(w)
        worker = self._spawn(code)
        with self._cond:
            self._busy.add(worker)
        return worker

    def _release(self, worker: _Worker, healthy: bool, failed: bool) -> None:
        retire = not healthy or worker.requests >= self.max_requests or worker.retire_requested
        with self._cond:
            self._busy.discard(worker)
            closed = self._closed or self._pid != os.getpid()
            if retire or closed:
                self._workers.discard(worker)
                if retire and not closed:
                    if failed:
                        self.restarts += 1
                    else:
                        self.recycled += 1
            else:
                self._idle.append(worker)
            self._cond.notify_all()
        if retire or closed:
            worker.destroy()


_CLOSED = object()
