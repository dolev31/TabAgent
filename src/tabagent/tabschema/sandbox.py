"""CodeExecutor backend: run LLM-written extraction code in an isolated interpreter (paper L306, L1210).

``SubprocessSandbox`` executes the code in a fresh ``python -I -S`` process (isolated mode, no site
packages, standard library only) in a temporary directory, with an empty environment (so the code cannot
read API keys), CPU-time and file-size limits, and a wall-clock timeout. The trajectory prefix arrives on
stdin as the global ``trajectory``; the code must leave a ``features`` dict in its global scope, as the
CodeExtractor prompt prescribes. Exceptions come back as full tracebacks for the retry prompt.

This isolates accidents, not adversaries: it is not a security boundary. Use ``DockerSandbox`` (no
network, read-only filesystem) when the code or the traces are untrusted.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

_RUNNER = r"""
import json, sys, traceback
payload = json.loads(sys.stdin.read())
code = compile(payload["code"], "<extractor>", "exec")
for item in payload["items"]:
    ns = {"__name__": "__main__", "trajectory": item}
    try:
        exec(code, ns)
        feats = ns.get("features")
        if not isinstance(feats, dict):
            raise TypeError("the code must leave a dict named `features` in its global scope")
        json.dumps(feats)
        out = {"ok": True, "features": feats}
    except BaseException:
        out = {"ok": False, "error": traceback.format_exc(limit=8)}
    sys.stdout.write(json.dumps(out, default=str) + "\n")
    sys.stdout.flush()
"""


@dataclass
class ExecResult:
    ok: bool
    features: dict[str, Any] | None = None
    error: str | None = None


def _limits(cpu_s: int, file_mb: int):
    def apply() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_mb << 20, file_mb << 20))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ImportError, ValueError, OSError):  # non-POSIX or restricted platform
            pass

    return apply


class SubprocessSandbox:
    def __init__(self, timeout_s: float = 30, cpu_s: int = 60, file_mb: int = 16, python: str | None = None):
        self.timeout_s, self.cpu_s, self.file_mb = timeout_s, cpu_s, file_mb
        self.python = python or sys.executable

    def run_batch(self, code: str, trajectories: list[dict[str, Any]]) -> list[ExecResult]:
        payload = json.dumps({"code": code, "items": trajectories}, default=str)
        workdir = tempfile.mkdtemp(prefix="tabschema_")
        try:
            proc = subprocess.run(
                [self.python, "-I", "-S", "-c", _RUNNER],
                input=payload,
                capture_output=True,
                text=True,
                cwd=workdir,
                env={},
                timeout=self.timeout_s * max(1, len(trajectories) / 50),
                preexec_fn=_limits(self.cpu_s, self.file_mb) if sys.platform != "win32" else None,
            )
        except subprocess.TimeoutExpired:
            return [ExecResult(False, error=f"timed out after {self.timeout_s}s")] * len(trajectories)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        results = [ExecResult(**json.loads(ln)) for ln in lines]
        if len(results) < len(trajectories):  # the interpreter died (CPU limit, crash)
            err = (proc.stderr or "the sandboxed interpreter exited early").strip()[-2000:]
            results += [ExecResult(False, error=err)] * (len(trajectories) - len(results))
        return results

    def run(self, code: str, trajectory: dict[str, Any]) -> ExecResult:
        return self.run_batch(code, [trajectory])[0]


class DockerSandbox(SubprocessSandbox):
    """Same protocol inside ``docker run --network none --read-only`` (requires Docker)."""

    def __init__(self, image: str = "python:3.12-slim", **kw):
        super().__init__(**kw)
        self.image = image

    def run_batch(self, code: str, trajectories: list[dict[str, Any]]) -> list[ExecResult]:
        payload = json.dumps({"code": code, "items": trajectories}, default=str)
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
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.timeout_s * max(1, len(trajectories) / 50) + 30,
            )
        except subprocess.TimeoutExpired:
            return [ExecResult(False, error=f"timed out after {self.timeout_s}s")] * len(trajectories)
        results = [ExecResult(**json.loads(ln)) for ln in proc.stdout.splitlines() if ln.startswith("{")]
        if len(results) < len(trajectories):
            err = (proc.stderr or "container exited early").strip()[-2000:]
            results += [ExecResult(False, error=err)] * (len(trajectories) - len(results))
        return results
