"""CodeExecutor backend: run LLM-written extraction code in an isolated interpreter (Sec. 3.1; App. D.1).

``SubprocessSandbox`` executes the code in a fresh ``python -I -S`` process (isolated mode, no site
packages, standard library only) in a temporary directory, with an empty environment (so the code cannot
read API keys) and CPU-time, file-size and memory limits (an address-space limit on Linux; elsewhere the
interpreter watches its own peak memory). Each interpreter runs at most ``chunk`` trajectories, and up to
``parallel`` interpreters run at once. On POSIX each trajectory has its own ``timeout_s`` alarm, and the
interpreter a wall-clock limit as a backstop.

One bad trajectory fails alone. Exceptions, ``exit()`` and ``SystemExit`` fail only the trajectory they occur
on and come back as full tracebacks for the retry prompt. When the interpreter itself dies or hangs on a
trajectory (a regular expression that backtracks for minutes, a memory blowup, a crash), that trajectory
fails, with the reason, and a fresh interpreter continues with the next one (at most ``max_restarts`` times in
a row without progress); a trajectory that was still within its own time when its chunk ran out of wall-clock
time is simply run again. The trajectory prefix arrives on stdin as the global ``trajectory``; the code must
leave a ``features`` dict in its global scope, as the CodeExtractor prompt prescribes. Results travel on a
private copy of stdout, so whatever the code prints (to ``sys.stdout`` or file descriptor 1) goes to stderr
and cannot corrupt them; what it prints is kept up to ``print_limit`` bytes per interpreter and the rest
dropped, so code that prints megabytes costs neither memory nor the run.

This isolates accidents, not adversaries: it is not a security boundary. Use ``DockerSandbox`` (no
network, read-only filesystem) when the code or the traces are untrusted.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

_RUNNER = r"""
import json, os, signal, sys, time, traceback
results = os.fdopen(os.dup(1), "w")  # the result channel; the code's own output goes to stderr
os.dup2(2, 1)
payload = json.loads(sys.stdin.read())
for _name, _value in (payload.get("rlimits") or {}).items():  # set here: no code runs between fork and exec
    try:
        import resource

        resource.setrlimit(getattr(resource, _name), (_value, _value))
    except Exception:  # a limit the platform does not enforce (macOS: RLIMIT_AS), or no resource module
        pass
budget = payload.get("mem_budget")
if budget:  # Linux: the extractor's own address space, over what the interpreter and its payload hold now
    try:
        import resource

        with open("/proc/self/statm") as _statm:
            _held = int(_statm.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
        _cap = resource.getrlimit(resource.RLIMIT_AS)[1]
        if _cap == resource.RLIM_INFINITY or _held + budget < _cap:
            resource.setrlimit(resource.RLIMIT_AS, (_held + budget, _held + budget))
    except Exception:  # no /proc or no resource module: the limit set above applies
        pass


class _Capped:
    # what the code prints: kept up to a limit, the rest dropped
    def __init__(self, stream, limit):
        self.stream, self.left = stream, limit

    def write(self, text):
        text = str(text)
        if self.left > 0:
            part = text[: self.left]
            self.stream.write(part)
            self.left -= len(part)
            if self.left <= 0:
                self.stream.write("\n[more printed output dropped]\n")
        return len(text)

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass

    def isatty(self):
        return False


sys.stdout = sys.stderr = _Capped(sys.__stderr__, int(payload.get("print_limit") or 1 << 20))
watch = payload.get("mem_watch")
too_much = "MemoryError: the extractor used more than %d MB" % ((watch or 0) >> 20)
if watch:  # macOS enforces no address-space limit: the interpreter watches its own peak memory
    try:
        import resource, threading
    except ImportError:  # no resource module on this platform: no watch
        watch = None
if watch:
    scale = 1 if sys.platform == "darwin" else 1024
    # the peak with the chunk's payload loaded: the extractor's own growth is measured over it
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale

    def over():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale - base > watch

    def _watch():  # a trajectory still running past the limit: the interpreter dies, and it alone fails
        while True:
            time.sleep(0.05)
            if over():
                sys.__stderr__.write("\n" + too_much + "\n")
                sys.__stderr__.flush()
                os._exit(137)

    threading.Thread(target=_watch, daemon=True).start()
code = compile(payload["code"], "<extractor>", "exec")
limit = payload.get("item_timeout") if hasattr(signal, "setitimer") else None
fired = []


def expire(signum, frame):
    fired.append(1)
    raise TimeoutError(f"timed out after {limit}s on this trajectory")


if limit:
    signal.signal(signal.SIGALRM, expire)
for n, item in enumerate(payload["items"]):
    results.write(json.dumps({"start": n, "t": time.monotonic()}) + "\n")
    results.flush()
    ns = {"__name__": "__main__", "trajectory": item}
    fired.clear()
    try:
        if limit:
            signal.setitimer(signal.ITIMER_REAL, limit)
        try:
            exec(code, ns)
        finally:
            if limit:
                signal.setitimer(signal.ITIMER_REAL, 0)
        feats = ns.get("features")
        if not isinstance(feats, dict):
            raise TypeError("the code must leave a dict named `features` in its global scope")
        json.dumps(feats)
        out = {"ok": True, "features": feats}
    except BaseException:
        out = {"ok": False, "error": traceback.format_exc(limit=8), "kind": "timeout" if fired else "error"}
    if watch and over():  # this trajectory went past the limit and finished: it fails, the next ones go on
        out = {"ok": False, "error": too_much, "kind": "error"}  # in a fresh interpreter (a fresh peak)
        results.write(json.dumps(out) + "\n")
        results.flush()
        os._exit(0)
    results.write(json.dumps(out, default=str) + "\n")
    results.flush()
"""


@dataclass
class ExecResult:
    """The outcome of running extraction code on one trajectory: the features, or the traceback.

    ``kind`` says why a run failed (None when it succeeded): ``error`` (the code raised), ``timeout`` (it ran
    past its timeout), ``crash`` (the interpreter died: a memory blowup, a crash), ``unavailable`` (no
    interpreter could run it) or ``broken`` (the code cannot run at all: it does not compile, or breaks the
    extraction rules)."""

    ok: bool
    features: dict[str, Any] | None = None
    error: str | None = None
    kind: str | None = None


def _parse(stdout: str | bytes | None) -> tuple[list[ExecResult], dict[int, float]]:
    """The runner's result lines (in order) and the start time of each item it began."""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", "replace")
    out: list[ExecResult] = []
    started: dict[int, float] = {}
    for line in (stdout or "").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        if "start" in rec and isinstance(rec.get("start"), int):
            started[rec["start"]] = float(rec.get("t") or 0.0)
        elif isinstance(rec.get("ok"), bool):
            kind_ = None if rec["ok"] else rec.get("kind", "error")
            out.append(ExecResult(rec["ok"], rec.get("features"), rec.get("error"), kind_))
    return out, started


def _results(stdout: str | bytes | None, n: int, error: str, kind: str = "crash") -> list[ExecResult]:
    """One ``ExecResult`` per item from the runner's result lines; items without one failed with ``error``
    and ``kind`` (the interpreter died or timed out before reaching them)."""
    out = _parse(stdout)[0][:n]
    return out + [ExecResult(False, error=error, kind=kind) for _ in range(n - len(out))]


@dataclass
class _Launch:
    """One interpreter's run on a list of items: what it wrote, how it ended, and when."""

    stdout: str
    stderr_tail: str
    returncode: int | None  # None: killed (the wall-clock limit, or an item past its own limit)
    wall_s: float
    ended: float  # time.monotonic() when it ended
    item_overran: bool = False  # killed because the running item was past its own time limit


#: Seconds past ``timeout_s`` after which a trajectory whose alarm could not stop it (a loop in C code, such
#: as a backtracking regular expression) is stopped by killing its interpreter.
GRACE_S = 2.0


def _feed(proc: subprocess.Popen, payload: str) -> None:
    with contextlib.suppress(BrokenPipeError, OSError, ValueError):
        proc.stdin.write(payload)
        proc.stdin.close()


def _drain(stream: Any, got: queue.Queue) -> None:
    with contextlib.suppress(OSError, ValueError):
        for line in stream:
            got.put(line)
    got.put(None)


class SubprocessSandbox:
    """Runs extraction code in a fresh isolated interpreter (see the module docstring).

    ``timeout_s`` bounds the code's run on one trajectory (POSIX; elsewhere it bounds the interpreter), and
    an interpreter runs at most ``chunk`` trajectories within ``timeout_s * max(2, chunk / 10)`` seconds.
    ``parallel`` interpreters run at once (default: up to 4, one per CPU). ``cpu_s``, ``file_mb`` and
    ``mem_mb`` are each interpreter's CPU-time (at least its wall-clock limit), file-size and memory limits.
    ``print_limit`` bytes of what the code prints are kept per interpreter. After a trajectory kills or hangs
    its interpreter, a new one continues with the rest of the chunk; at most ``max_restarts`` times in a row
    without progress (an interpreter that finished trajectories before it stopped made progress). A chunk also
    holds at most ``chunk_bytes`` of prefix JSON, so one interpreter never loads an outsized payload.

    ``mem_mb`` bounds the memory the extractor uses beyond its input: on Linux, an address-space limit of
    ``mem_mb`` over what the interpreter holds once the chunk's payload is parsed; on macOS, a watch of the
    interpreter's peak memory over its peak once the payload is loaded (checked every 50 ms, and after each
    trajectory). Elsewhere it is not enforced. A trajectory still running ``GRACE_S`` (2) seconds past
    ``timeout_s`` is stopped by killing its interpreter.
    """

    def __init__(
        self,
        timeout_s: float = 30,
        cpu_s: int = 60,
        file_mb: int = 16,
        python: str | None = None,
        mem_mb: int | None = 2048,
        chunk: int = 50,
        parallel: int | None = None,
        print_limit: int = 1 << 20,
        max_restarts: int = 5,
        chunk_bytes: int = 32 << 20,
    ):
        self.timeout_s, self.cpu_s, self.file_mb, self.mem_mb = timeout_s, cpu_s, file_mb, mem_mb
        self.python = python or sys.executable
        self.chunk = max(1, chunk)
        self.parallel = max(1, parallel if parallel is not None else min(4, os.cpu_count() or 1))
        self.print_limit, self.max_restarts = print_limit, max(0, max_restarts)
        self.chunk_bytes = max(1, chunk_bytes)

    def _chunks(self, trajectories: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Consecutive chunks of at most ``chunk`` trajectories and ``chunk_bytes`` of JSON (a trajectory
        larger than that alone)."""
        chunks: list[list[dict[str, Any]]] = []
        size = 0
        for t in trajectories:
            n = len(json.dumps(t, default=str))
            if not chunks or len(chunks[-1]) >= self.chunk or (chunks[-1] and size + n > self.chunk_bytes):
                chunks.append([])
                size = 0
            chunks[-1].append(t)
            size += n
        return chunks

    def run_batch(self, code: str, trajectories: list[dict[str, Any]]) -> list[ExecResult]:
        """Run ``code`` once per trajectory, ``chunk`` trajectories (at most ``chunk_bytes`` of JSON) per
        interpreter and ``parallel`` interpreters at once; one ``ExecResult`` per trajectory, in order."""
        chunks = self._chunks(trajectories)
        if len(chunks) <= 1 or self.parallel <= 1:
            return [r for c in chunks for r in self._run_chunk(code, c)]
        with ThreadPoolExecutor(max_workers=min(self.parallel, len(chunks))) as pool:
            return [r for rs in pool.map(lambda c: self._run_chunk(code, c), chunks) for r in rs]

    def _payload(self, code: str, items: list[dict[str, Any]]) -> str:
        watch = (self.mem_mb << 20) if self.mem_mb and sys.platform == "darwin" else None
        budget = (self.mem_mb << 20) if self.mem_mb and sys.platform.startswith("linux") else None
        cpu = max(self.cpu_s, int(self._wall(len(items))) + 1)  # the wall-clock limit ends a slow chunk
        body = json.dumps(items, default=str)
        rlimits = {"RLIMIT_CPU": cpu, "RLIMIT_FSIZE": self.file_mb << 20, "RLIMIT_CORE": 0}
        if self.mem_mb:  # the extractor's memory, plus the interpreter and its parsed payload
            rlimits["RLIMIT_AS"] = (self.mem_mb << 20) + 12 * len(body) + (256 << 20)
        head = json.dumps(
            {
                "code": code,
                "item_timeout": self.timeout_s,
                "print_limit": self.print_limit,
                "mem_watch": watch,
                "mem_budget": budget,
                "rlimits": rlimits,
            },
            default=str,
        )
        return head[:-1] + ', "items": ' + body + "}"

    def _wall(self, n: int) -> float:
        return self.timeout_s * max(2.0, n / 10)

    def _run_chunk(self, code: str, items: list[dict[str, Any]]) -> list[ExecResult]:
        """Run ``code`` on ``items``, a fresh interpreter continuing after a trajectory that killed or hung
        the previous one (see the class docstring)."""
        out: list[ExecResult | None] = [None] * len(items)
        pending = list(range(len(items)))
        restarts = 0
        while pending:
            run = self._launch(code, [items[i] for i in pending])
            results, started = _parse(run.stdout)
            for k, r in enumerate(results[: len(pending)]):
                out[pending[k]] = r
            done = len(results)
            if done >= len(pending):
                break
            if run.item_overran:  # this item ran past its own limit and its alarm could not stop it
                error, kind, blamed = (
                    f"timed out after {self.timeout_s:g}s on this trajectory",
                    "timeout",
                    True,
                )
            elif run.returncode is None:  # the chunk ran out of wall-clock time
                error, kind = f"timed out after {run.wall_s:g}s", "timeout"
                t0 = started.get(done)
                blamed = t0 is not None and run.ended - t0 >= self.timeout_s  # it ran past its own limit
            else:
                error = f"the sandboxed interpreter exited early (exit status {run.returncode})" + (
                    f":\n{run.stderr_tail}" if run.stderr_tail else ""
                )
                kind, blamed = "crash", done in started
            if blamed:
                out[pending[done]] = ExecResult(False, error=error, kind=kind)
                pending = pending[done + 1 :]
            else:  # it never started, or it was within its own time: run it again in a fresh interpreter
                pending = pending[done:]
            # an interpreter that finished or failed trajectories made progress; only restarts in a row that
            # made none count (a chunk of slow but legal trajectories takes several interpreters)
            restarts = 0 if done or blamed else restarts + 1
            if pending and restarts > self.max_restarts:
                note = (
                    f"{error}\n(the interpreter stopped {restarts} times in a row without finishing a "
                    "trajectory; the rest of the chunk was not run)"
                )
                for i in pending:
                    out[i] = ExecResult(False, error=note, kind=kind)
                break
        return [r if r is not None else ExecResult(False, error="not run", kind="crash") for r in out]

    def _command(self) -> list[str]:
        return [self.python, "-I", "-S", "-c", _RUNNER]

    def _launch(self, code: str, items: list[dict[str, Any]]) -> _Launch:
        """One interpreter on ``items``. Its result lines are read as they come: an item still running
        ``GRACE_S`` seconds past its timeout (stuck in C code, out of reach of its alarm) and an interpreter
        past its wall-clock limit are killed. Its stderr goes to a file (bounded by the file-size limit)."""
        wall = self._wall(len(items))
        workdir = tempfile.mkdtemp(prefix="tabschema_")
        lines: list[str] = []
        overran = False
        try:
            with tempfile.TemporaryFile(dir=workdir) as err:
                t0 = time.monotonic()
                proc = subprocess.Popen(
                    self._command(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=err,
                    encoding="utf-8",
                    errors="replace",
                    cwd=workdir,
                    env={},
                )
                feed = threading.Thread(target=_feed, args=(proc, self._payload(code, items)), daemon=True)
                feed.start()
                got: queue.Queue = queue.Queue()
                reader = threading.Thread(target=_drain, args=(proc.stdout, got), daemon=True)
                reader.start()
                item_started: float | None = None
                limit = self.timeout_s + GRACE_S
                while True:
                    try:
                        line = got.get(timeout=0.1)
                    except queue.Empty:
                        line = ""
                    if line is None:  # the interpreter closed its result channel: it is done
                        break
                    now = time.monotonic()
                    if line:
                        lines.append(line)
                        if line.startswith('{"start"'):
                            item_started = now
                        elif line.startswith('{"ok"'):
                            item_started = None
                    if now - t0 > wall or (item_started is not None and now - item_started > limit):
                        overran = now - t0 <= wall
                        proc.kill()
                        break
                code_ = proc.wait()
                reader.join(timeout=5)
                while True:  # lines that arrived after the kill decision
                    try:
                        line = got.get_nowait()
                    except queue.Empty:
                        break
                    if line:
                        lines.append(line)
                ended = time.monotonic()
                timed_out = proc.returncode is not None and (overran or ended - t0 > wall) and code_ < 0
                err.seek(0, os.SEEK_END)
                err.seek(max(0, err.tell() - 4000))
                tail = err.read().decode("utf-8", "replace").strip()[-2000:]
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        if overran:
            return _Launch("".join(lines), tail, None, wall, ended, item_overran=True)
        return _Launch("".join(lines), tail, None if timed_out else code_, wall, ended)

    def run(self, code: str, trajectory: dict[str, Any]) -> ExecResult:
        """Run ``code`` on one trajectory."""
        return self.run_batch(code, [trajectory])[0]


class DockerSandbox(SubprocessSandbox):
    """Same protocol inside ``docker run --network none --read-only`` (requires Docker); ``parallel``
    defaults to 1 container at a time."""

    def __init__(self, image: str = "python:3.12-slim", **kw):
        kw.setdefault("parallel", 1)
        super().__init__(**kw)
        self.image = image

    def _command(self) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--read-only",
            "--memory",
            "1g",
            "--cpus",
            "1",
            self.image,
            "python",
            "-I",
            "-S",
            "-c",
            _RUNNER,
        ]

    def _payload(self, code: str, items: list[dict[str, Any]]) -> str:
        payload = json.loads(super()._payload(code, items))
        payload["mem_watch"] = payload["mem_budget"] = None  # the container's own memory limit applies
        return json.dumps(payload, default=str)

    def _launch(self, code: str, items: list[dict[str, Any]]) -> _Launch:
        wall = self._wall(len(items)) + 30
        try:
            proc = subprocess.run(
                self._command(),
                input=self._payload(code, items),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=wall,
            )
            stdout, code_, tail = proc.stdout, proc.returncode, (proc.stderr or "").strip()[-2000:]
        except subprocess.TimeoutExpired as e:
            stdout, code_, tail = e.stdout, None, ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        return _Launch(stdout or "", tail, code_, wall, time.monotonic())
