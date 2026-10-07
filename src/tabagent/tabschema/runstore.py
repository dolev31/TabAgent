"""Run directory for a TabSchema onboarding: every intermediate artifact, a content-addressed stage cache,
and the LLM call log. A crashed or interrupted run resumes from the cache; a rerun with identical inputs
makes no new LLM calls.

Every file is written atomically (a temporary file, then a rename), so a process killed at any moment leaves
complete files only; the temporary file it was writing is removed when a run next opens the directory. Cache
entries are also locked across processes (``fcntl`` locks, on POSIX): two runs on one run directory wait for
each other's answer to the same prompt instead of paying for it twice. Each process that opens the directory
is a *session* (``session``), recorded with every logged call.

Run directories hold trajectory text, so they live outside the repository (default
``~/.cache/tabagent/runs``) and must be treated like the trajectories themselves.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tabagent.defaults import cache_root

try:  # POSIX: locks shared with other processes on the same run directory
    import fcntl
except ImportError:  # pragma: no cover - Windows: in-process locks only
    fcntl = None  # type: ignore[assignment]


def utc_now() -> str:
    """The current UTC time, ISO 8601 to the second."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def digest(obj: Any) -> str:
    """SHA-256 of the canonical JSON of ``obj`` (the cache key of a stage call)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


#: The process's umask, read once (reading it means setting it, which is not thread-safe).
UMASK = _umask()


#: A temporary file older than this is removed when a run directory is opened, whoever wrote it.
STALE_TMP_S = 24 * 3600

_writes = threading.Condition()  # announces the end of each atomic write (wait_for_writes)
_writing = 0  # atomic writes in progress in this process


def atomic_write(p: Path | str, text: str) -> None:
    """Write through a temporary file and a rename: readers and resumed runs never see a partial file. The
    file gets the mode a plain ``open`` would give it (``0o666`` less the umask), not the temporary file's
    private ``0o600``. The temporary file, ``.<name>.<pid>.<random>.tmp``, names the process writing it
    (``remove_stale_tmp``)."""
    global _writing
    p = Path(p)
    with _writes:
        _writing += 1
    try:
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.{os.getpid()}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.chmod(tmp, 0o666 & ~UMASK)
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    finally:
        with _writes:
            _writing -= 1
            _writes.notify_all()


def wait_for_writes(timeout: float = 2.0) -> bool:
    """Wait until no atomic write is in progress in this process, for at most ``timeout`` seconds; True when
    none is. An interrupted run waits so for the files its threads are writing (an answer being stored), not
    for the threads still waiting for an LLM answer."""
    with _writes:
        return _writes.wait_for(lambda: _writing == 0, timeout)


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
        except Exception:  # noqa: BLE001 - unknown: counted as alive (its records and files are kept)
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


def remove_stale_tmp(root: Path, max_age_s: float = STALE_TMP_S) -> int:
    """Remove the temporary files under ``root`` that no write will finish: those of a process that is gone
    (killed in the middle of a write) and any older than ``max_age_s``, whoever wrote them (a file named
    before the pid was, ``.<name>.<random>.tmp``, is judged by its age only). The files of live processes on
    this host, this one included, are kept: another process may be writing in the same directory. Returns
    how many were removed."""
    removed, now = 0, time.time()
    for f in root.rglob(".*.tmp"):
        parts = f.name[: -len(".tmp")].rsplit(".", 2)  # ".<name>", "<pid>", "<random>"
        dead = len(parts) == 3 and parts[1].isdigit() and not _pid_alive(parts[1])
        try:
            if dead or now - f.stat().st_mtime > max_age_s:
                f.unlink()
                removed += 1
        except OSError:  # renamed by its writer in the meantime, or not ours to remove
            continue
    return removed


class RunStore:
    """One run directory, ``<root>/<run_id>`` (default root ``$TABAGENT_CACHE/runs``, default id the UTC
    start time). Pass an existing run id to resume a run from its cache."""

    def __init__(self, root: str | Path | None = None, run_id: str | None = None):
        base = Path(root) if root else cache_root() / "runs"
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = base / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        remove_stale_tmp(self.dir)
        self.session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + secrets.token_hex(3)
        self.phase: str | None = None  # the pipeline stage running now (recorded with every logged call)
        self._lock = threading.Lock()
        self._key_locks: dict[Path, threading.Lock] = {}

    def path(self, *parts: str) -> Path:
        """A path inside the run directory (its parent directories are created)."""
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, rel: str, obj: Any) -> Path:
        """Write ``obj`` as JSON to ``rel`` atomically."""
        p = self.path(rel)
        atomic_write(p, json.dumps(obj, indent=1, default=str))
        return p

    def read_json(self, rel: str) -> Any:
        """Read the JSON file ``rel``."""
        return json.loads((self.dir / rel).read_text())

    def write_text(self, rel: str, text: str) -> Path:
        """Write ``text`` to ``rel`` atomically."""
        p = self.path(rel)
        atomic_write(p, text)
        return p

    def cached(self, stage: str, key: Any, fn: Callable[[], Any]) -> Any:
        """Return the stored result for (stage, key), computing and storing it on a miss.

        Stages run on worker threads. Callers of the same key wait for the first one, so a paid LLM call is
        made once; callers of different keys run concurrently.
        """
        p = self._entry(stage, key)
        with self._key_lock(p), self._file_lock(p):
            try:
                return json.loads(p.read_text())["value"]
            except FileNotFoundError:
                pass
            except (ValueError, KeyError, TypeError):  # an unreadable entry is computed again
                pass
            value = fn()
            p.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(p, json.dumps({"key": key, "value": value}, default=str))
            return value

    def peek(self, stage: str, key: Any) -> Any:
        """The stored result for (stage, key), or None (no lock, no computation)."""
        try:
            return json.loads(self._entry(stage, key).read_text())["value"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def forget(self, stage: str, key: Any) -> None:
        """Remove the stored result for (stage, key), if any; the next ``cached`` call computes it again."""
        p = self._entry(stage, key)
        with self._key_lock(p), self._file_lock(p):
            p.unlink(missing_ok=True)

    @contextlib.contextmanager
    def _file_lock(self, p: Path) -> Iterator[None]:
        """An exclusive lock on ``p``'s entry shared with other processes (a no-op without ``fcntl``)."""
        if fcntl is None:
            yield
            return
        lock = p.parent / ".locks" / (p.name + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        with open(lock, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _entry(self, stage: str, key: Any) -> Path:
        return self.dir / "cache" / stage / f"{digest(key)}.json"

    def _key_lock(self, p: Path) -> threading.Lock:
        with self._lock:
            return self._key_locks.setdefault(p, threading.Lock())

    def log_call(self, record: dict[str, Any]) -> None:
        """Append one LLM call record to ``llm_calls.jsonl``, with the time and the session (one write of one
        line in append mode, so concurrent writers never interleave lines)."""
        phase = {"phase": self.phase} if self.phase else {}
        line = json.dumps({"ts": utc_now(), "session": self.session, **phase, **record}, default=str) + "\n"
        with self._lock, self.path("llm_calls.jsonl").open("a") as f:
            f.write(line)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """The records of a JSONL file; a line that does not parse (the last one of a killed writer) is
    skipped."""
    out = []
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out
