"""Run directory for a TabSchema onboarding: every intermediate artifact, a content-addressed stage cache,
and the LLM call log. A crashed or interrupted run resumes from the cache; a rerun with identical inputs
makes no new LLM calls.

Run directories hold trajectory text, so they live outside the repository (default
``~/.cache/tabagent/runs``) and must be treated like the trajectories themselves.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def atomic_write(p: Path, text: str) -> None:
    """Write through a temporary file and a rename: readers and resumed runs never see a partial file."""
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class RunStore:
    def __init__(self, root: str | Path | None = None, run_id: str | None = None):
        default = Path(os.environ.get("TABAGENT_CACHE", Path.home() / ".cache" / "tabagent")) / "runs"
        base = Path(root) if root else default
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = base / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._key_locks: dict[Path, threading.Lock] = {}

    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, rel: str, obj: Any) -> Path:
        p = self.path(rel)
        atomic_write(p, json.dumps(obj, indent=1, default=str))
        return p

    def read_json(self, rel: str) -> Any:
        return json.loads((self.dir / rel).read_text())

    def write_text(self, rel: str, text: str) -> Path:
        p = self.path(rel)
        atomic_write(p, text)
        return p

    def cached(self, stage: str, key: Any, fn: Callable[[], Any]) -> Any:
        """Return the stored result for (stage, key), computing and storing it on a miss.

        Stages run on worker threads. Callers of the same key wait for the first one, so a paid LLM call is
        made once; callers of different keys run concurrently.
        """
        p = self.dir / "cache" / stage / f"{digest(key)}.json"
        with self._lock:
            key_lock = self._key_locks.setdefault(p, threading.Lock())
        with key_lock:
            if p.exists():
                return json.loads(p.read_text())["value"]
            value = fn()
            p.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(p, json.dumps({"key": key, "value": value}, default=str))
            return value

    def log_call(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str) + "\n"
        with self._lock, self.path("llm_calls.jsonl").open("a") as f:
            f.write(line)
