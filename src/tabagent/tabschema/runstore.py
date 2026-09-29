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
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class RunStore:
    def __init__(self, root: str | Path | None = None, run_id: str | None = None):
        default = Path(os.environ.get("TABAGENT_CACHE", Path.home() / ".cache" / "tabagent")) / "runs"
        base = Path(root) if root else default
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = base / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, rel: str, obj: Any) -> Path:
        p = self.path(rel)
        p.write_text(json.dumps(obj, indent=1, default=str))
        return p

    def read_json(self, rel: str) -> Any:
        return json.loads((self.dir / rel).read_text())

    def write_text(self, rel: str, text: str) -> Path:
        p = self.path(rel)
        p.write_text(text)
        return p

    def cached(self, stage: str, key: Any, fn: Callable[[], Any]) -> Any:
        """Return the stored result for (stage, key), computing and storing it on a miss."""
        p = self.dir / "cache" / stage / f"{digest(key)}.json"
        if p.exists():
            return json.loads(p.read_text())["value"]
        value = fn()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"key": key, "value": value}, default=str))
        return value

    def log_call(self, record: dict[str, Any]) -> None:
        with self.path("llm_calls.jsonl").open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
