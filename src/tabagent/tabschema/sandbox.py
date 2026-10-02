"""CodeExecutor backend: run LLM-written extraction code in an isolated interpreter (Sec. 3.1; App. D.1).

``SubprocessSandbox`` executes the code in a fresh ``python -I -S`` process (isolated mode, no site
packages, standard library only) in a temporary directory, with an empty environment (so the code cannot
read API keys) and CPU-time, file-size and (on Linux) address-space limits. Each interpreter runs at most
``chunk`` trajectories; on POSIX each trajectory has its own ``timeout_s`` alarm, and the interpreter a
wall-clock limit as a backstop, so one slow trajectory fails alone and the finished ones keep their results.
The trajectory prefix arrives on stdin as the global ``trajectory``; the code must leave a ``features`` dict
in its global scope, as the CodeExtractor prompt prescribes. Exceptions, ``exit()`` and ``SystemExit`` fail
only the trajectory they occur on and come back as full tracebacks for the retry prompt. Results travel on a
private copy of stdout, so whatever the code prints (to ``sys.stdout`` or file descriptor 1) goes to stderr
and cannot corrupt them.

This isolates accidents, not adversaries: it is not a security boundary. Use ``DockerSandbox`` (no
network, read-only filesystem) when the code or the traces are untrusted.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

_RUNNER = r"""
import json, os, signal, sys, traceback
results = os.fdopen(os.dup(1), "w")  # the result channel; the code's own output goes to stderr
os.dup2(2, 1)
sys.stdout = sys.stderr
payload = json.loads(sys.stdin.read())
code = compile(payload["code"], "<extractor>", "exec")
limit = payload.get("item_timeout") if hasattr(signal, "setitimer") else None


def expire(signum, frame):
    raise TimeoutError(f"timed out after {limit}s on this trajectory")


if limit:
    signal.signal(signal.SIGALRM, expire)
for item in payload["items"]:
    ns = {"__name__": "__main__", "trajectory": item}
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
        out = {"ok": False, "error": traceback.format_exc(limit=8)}
    results.write(json.dumps(out, default=str) + "\n")
    results.flush()
"""


@dataclass
class ExecResult:
    """The outcome of running extraction code on one trajectory: the features, or the traceback."""

    ok: bool
    features: dict[str, Any] | None = None
    error: str | None = None


def _limits(cpu_s: int, file_mb: int, mem_mb: int | None):
    def apply() -> None:
        try:
            import resource
        except ImportError:  # non-POSIX
            return
        limits = [
            (resource.RLIMIT_CPU, cpu_s),
            (resource.RLIMIT_FSIZE, file_mb << 20),
            (resource.RLIMIT_CORE, 0),
        ]
        if mem_mb:
            limits.append((resource.RLIMIT_AS, mem_mb << 20))
        for kind, value in limits:  # a limit the platform does not enforce (macOS: RLIMIT_AS) is skipped
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(kind, (value, value))

    return apply


def _results(stdout: str | bytes | None, n: int, error: str) -> list[ExecResult]:
    """One ``ExecResult`` per item from the runner's result lines; items without one failed with ``error``
    (the interpreter died or timed out before reaching them)."""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", "replace")
    out: list[ExecResult] = []
    for line in (stdout or "").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("ok"), bool):
            out.append(ExecResult(rec["ok"], rec.get("features"), rec.get("error")))
    out = out[:n]
    return out + [ExecResult(False, error=error) for _ in range(n - len(out))]


class SubprocessSandbox:
    """Runs extraction code in a fresh isolated interpreter (see the module docstring).

    ``timeout_s`` bounds the code's run on one trajectory (POSIX; elsewhere it bounds the interpreter), and
    an interpreter runs at most ``chunk`` trajectories within ``timeout_s * max(2, chunk / 10)`` seconds.
    ``cpu_s``, ``file_mb`` and ``mem_mb`` are each interpreter's CPU-time, file-size and address-space limits
    (the last one is enforced on Linux only; use ``DockerSandbox`` for a memory cap elsewhere).
    """

    def __init__(
        self,
        timeout_s: float = 30,
        cpu_s: int = 60,
        file_mb: int = 16,
        python: str | None = None,
        mem_mb: int | None = 2048,
        chunk: int = 50,
    ):
        self.timeout_s, self.cpu_s, self.file_mb, self.mem_mb = timeout_s, cpu_s, file_mb, mem_mb
        self.python = python or sys.executable
        self.chunk = max(1, chunk)

    def run_batch(self, code: str, trajectories: list[dict[str, Any]]) -> list[ExecResult]:
        """Run ``code`` once per trajectory, ``chunk`` trajectories per interpreter; one ``ExecResult`` per
        trajectory, in order."""
        out: list[ExecResult] = []
        for i in range(0, len(trajectories), self.chunk):
            out += self._run_chunk(code, trajectories[i : i + self.chunk])
        return out

    def _payload(self, code: str, items: list[dict[str, Any]]) -> str:
        return json.dumps({"code": code, "items": items, "item_timeout": self.timeout_s}, default=str)

    def _wall(self, n: int) -> float:
        return self.timeout_s * max(2.0, n / 10)

    def _run_chunk(self, code: str, items: list[dict[str, Any]]) -> list[ExecResult]:
        workdir = tempfile.mkdtemp(prefix="tabschema_")
        limits = _limits(self.cpu_s, self.file_mb, self.mem_mb) if sys.platform != "win32" else None
        try:
            proc = subprocess.run(
                [self.python, "-I", "-S", "-c", _RUNNER],
                input=self._payload(code, items),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                cwd=workdir,
                env={},
                timeout=self._wall(len(items)),
                preexec_fn=limits,
            )
        except subprocess.TimeoutExpired as e:  # keep the results of the items that finished in time
            return _results(e.stdout, len(items), f"timed out after {self._wall(len(items)):g}s")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        tail = (proc.stderr or "").strip()[-2000:]
        error = f"the sandboxed interpreter exited early (exit status {proc.returncode})" + (
            f":\n{tail}" if tail else ""
        )
        return _results(proc.stdout, len(items), error)

    def run(self, code: str, trajectory: dict[str, Any]) -> ExecResult:
        """Run ``code`` on one trajectory."""
        return self.run_batch(code, [trajectory])[0]


class DockerSandbox(SubprocessSandbox):
    """Same protocol inside ``docker run --network none --read-only`` (requires Docker)."""

    def __init__(self, image: str = "python:3.12-slim", **kw):
        super().__init__(**kw)
        self.image = image

    def _run_chunk(self, code: str, items: list[dict[str, Any]]) -> list[ExecResult]:
        cmd = [
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
        try:
            proc = subprocess.run(
                cmd,
                input=self._payload(code, items),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._wall(len(items)) + 30,
            )
        except subprocess.TimeoutExpired as e:
            return _results(e.stdout, len(items), f"timed out after {self._wall(len(items)) + 30:g}s")
        tail = (proc.stderr or "").strip()[-2000:]
        error = f"the container exited early (exit status {proc.returncode})" + (f":\n{tail}" if tail else "")
        return _results(proc.stdout, len(items), error)
