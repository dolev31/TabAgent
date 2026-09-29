"""Pick the adapter for a file or directory of agent logs by looking at its first records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tabagent.adapters.common import SKIP_FILES, iter_files
from tabagent.adapters.cuga import CUGA_STEPS, CugaAdapter
from tabagent.adapters.openai_messages import ROLES, OpenAIMessagesAdapter, _role, _unwrap
from tabagent.adapters.tau2 import Tau2Adapter
from tabagent.tabschema.trajectory import GenericJsonAdapter

MAX_FILES = 20
FORMATS = """Supported formats:
  * chat messages (OpenAI, Anthropic Messages API format, LangChain), .json or .jsonl, one run per object:
      {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "tool_calls": [...]}, ...],
       "tools": [...], "reward": 1}                                       -> OpenAIMessagesAdapter
  * tau2-bench results: {"info": ..., "tasks": [...], "simulations": [...]}  -> Tau2Adapter
  * CUGA trajectories: {"intent": ..., "task_id": ..., "score": ..., "steps": [{"name": "ShortlisterAgent",
      "prompts": [...], "data": ...}, ...]}                               -> CugaAdapter
  * any log with named steps: {"steps": [{"name": <component>, ...}, ...]}  -> GenericJsonAdapter
Or pass an adapter explicitly (see tabagent.adapters)."""


def _first_record(f: Path) -> Any:
    try:
        if f.suffix == ".jsonl":
            with f.open() as fh:
                line = next((ln for ln in fh if ln.strip()), "")
            return json.loads(line) if line else None
        data = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    return data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else data


def _is_message(m: Any) -> bool:
    if not isinstance(m, dict):
        return False
    m = _unwrap(m)
    return _role(m) in ROLES.values() or m.get("type") in ("function_call", "function_call_output")


def detect_format(record: Any) -> type | None:
    """The adapter class for one parsed record, or None."""
    if not isinstance(record, dict):
        return None
    if isinstance(record.get("simulations"), list):
        return Tau2Adapter
    msgs = record.get("messages")
    if isinstance(msgs, list) and msgs and _is_message(msgs[0]):
        return OpenAIMessagesAdapter
    if _is_message(record) and "content" in record:  # a bare message list
        return OpenAIMessagesAdapter
    steps = record.get("steps")
    if isinstance(steps, list):
        names = {s.get("name") for s in steps if isinstance(s, dict)} - {None, ""}
        if names & CUGA_STEPS:
            return CugaAdapter
        if names:
            return GenericJsonAdapter
    return None


def detect_adapter(path: str | Path, **kwargs: Any):
    """The adapter for the logs at ``path``, chosen from the first files; ``kwargs`` go to its constructor."""
    files = [f for f in iter_files(path) if f.name not in SKIP_FILES][:MAX_FILES]
    seen = []
    for f in files:
        rec = _first_record(f)
        cls = detect_format(rec)
        if cls is not None:
            return cls(**kwargs)
        seen.append(f"{f.name}: {sorted(rec)[:8] if isinstance(rec, dict) else type(rec).__name__}")
    found = "\n".join(f"  {s}" for s in seen[:5]) or "  (no .json or .jsonl files)"
    raise ValueError(f"could not recognize the log format of {path}. First records:\n{found}\n{FORMATS}")
