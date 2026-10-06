"""Pick the adapter for a file or directory of agent logs by looking at its first records."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from tabagent.adapters.cuga import CUGA_STEPS, CugaAdapter
from tabagent.adapters.logfiles import (
    SKIP_FILES,
    LoadNotes,
    MixedFormatsError,
    iter_files,
    read_docs,
    relative,
)
from tabagent.adapters.openai_messages import (
    CALL_ITEMS,
    OUTPUT_ITEMS,
    ROLES,
    OpenAIMessagesAdapter,
    message_role,
    unwrap_message,
)
from tabagent.adapters.tau2 import Tau2Adapter
from tabagent.tabschema.trajectory import GenericJsonAdapter, step_name

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


def format_name(cls: type) -> str:
    """The log format an adapter class reads, as messages name it."""
    return getattr(cls, "format_name", cls.__name__)


def _family(cls: type) -> str:
    if issubclass(cls, Tau2Adapter):
        return "tau2"
    if issubclass(cls, OpenAIMessagesAdapter):
        return "chat"
    if issubclass(cls, GenericJsonAdapter):  # CUGA trajectories are step logs
        return "steps"
    return cls.__qualname__


def compatible(reader: type, found: type) -> bool:
    """Whether an adapter of class ``reader`` reads records detected as ``found``: the same format, where
    CUGA trajectories and other logs with named steps count as one."""
    return _family(reader) == _family(found)


def _first_record(f: Path, root: Path | None = None) -> Any:
    """The first JSON record of a file (the first run of a list), read as the adapters read it; None when the
    file holds none. A JSON Lines file of bare messages (a conversation, one message per line) gives its first
    50 messages as one run."""
    held: list[Any] = []
    for doc in read_docs(root or f.parent, f, LoadNotes()):
        data = doc.data
        if doc.line is not None and isinstance(data, dict) and "messages" not in data and _is_message(data):
            held.append(data)
            if len(held) < 50:
                continue
            break
        if held:
            break
        if isinstance(data, list):
            if data and sum(_is_message(x) for x in data) * 2 >= len(data):  # a bare message list: one run
                return {"messages": data}
            return next((x for x in data if isinstance(x, dict)), data)
        return data
    return {"messages": held} if held else None


def _is_message(m: Any) -> bool:
    if not isinstance(m, dict):
        return False
    m = unwrap_message(m)
    return message_role(m) in ROLES.values() or m.get("type") in (*CALL_ITEMS, *OUTPUT_ITEMS)


def _is_turn(m: Any) -> bool:
    """Whether a message is the agent's: an assistant message or a Responses API call."""
    if not isinstance(m, dict):
        return False
    m = unwrap_message(m)
    return message_role(m) == "assistant" or m.get("type") in CALL_ITEMS


def detect_format(record: Any) -> type | None:
    """The adapter class for one parsed record, or None. A chat run needs an agent turn: messages that are
    only system or user text (a prompt template, a few-shot example) are not a run."""
    if not isinstance(record, dict):
        return None
    if isinstance(record.get("simulations"), list):
        return Tau2Adapter
    msgs = record.get("messages")
    if isinstance(msgs, list) and msgs and any(_is_message(m) for m in msgs[:5]):
        return OpenAIMessagesAdapter if any(_is_turn(m) for m in msgs) else None
    if _is_turn(record) and "content" in record:  # an agent message on its own (a line of a conversation)
        return OpenAIMessagesAdapter
    steps = record.get("steps")
    if isinstance(steps, list):
        names = {step_name(s) for s in steps} - {None}
        if names & CUGA_STEPS:
            return CugaAdapter
        if names:
            return GenericJsonAdapter
    return None


def _census(path: str | Path, limit: int | None) -> tuple[list[tuple[str, type | None, Any]], int]:
    """``(file, format, first record)`` for the first ``limit`` log files (all with None), and how many run
    summaries were passed over."""
    root = Path(path)
    files = iter_files(root)
    summaries = sum(f.name in SKIP_FILES for f in files)
    out = []
    for f in [f for f in files if f.name not in SKIP_FILES][:limit]:
        rec = _first_record(f, root if root.is_dir() else f.parent)
        out.append((relative(root, f), detect_format(rec), rec))
    return out, summaries


def _pick(found: list[tuple[str, type]], path: Any, on_other_formats: str) -> type:
    """The adapter class for the formats found: CUGA over generic steps, else the format most files hold (the
    first in file order on a tie). Several formats raise with ``on_other_formats="error"``."""
    classes = list(dict.fromkeys(cls for _, cls in found))
    if CugaAdapter in classes and GenericJsonAdapter in classes:
        classes.remove(GenericJsonAdapter)
        found = [(f, CugaAdapter if c is GenericJsonAdapter else c) for f, c in found]
    counts = Counter(cls for _, cls in found)
    if len(classes) > 1 and on_other_formats == "error":
        first = {cls: next(f for f, c in found if c is cls) for cls in classes}
        listed = "; ".join(
            f"{format_name(cls)} ({cls.__name__}): {counts[cls]} file(s), e.g. {first[cls]}"
            for cls in classes
        )
        raise MixedFormatsError(
            f"{path} holds runs of more than one log format: {listed}. One onboarding reads one format: "
            "point at a folder or file with one format, pass adapter= for the one you want, or use "
            'on_other_formats="skip" (the default; CLI --on-other-formats skip) to read the most common one '
            "and skip the others with a warning"
        )
    return max(classes, key=lambda cls: counts[cls])  # the first of the most common


def detect_adapter(path: str | Path, *, on_other_formats: str = "skip", **kwargs: Any):
    """The adapter for the logs at ``path``, chosen from the first files; ``kwargs`` go to its constructor.

    The first records of the first files (up to 20, run summaries skipped) decide: CUGA trajectories over
    generic step logs, else the format most of them hold. Runs of another format are skipped by the adapter's
    ``load`` with a warning (``on_other_formats="skip"``), or stop it with ``MixedFormatsError``
    (``"error"``)."""
    if on_other_formats not in ("skip", "error"):
        raise ValueError(f"on_other_formats must be 'skip' or 'error', not {on_other_formats!r}")
    census, summaries = _census(path, MAX_FILES)
    found = [(f, cls) for f, cls, _ in census if cls is not None]
    if found:
        adapter = _pick(found, path, on_other_formats)(**kwargs)
        adapter.on_other_formats = on_other_formats
        return adapter
    seen = [
        f"{f}: {sorted(map(str, rec))[:8] if isinstance(rec, dict) else type(rec).__name__}"
        for f, _, rec in census
    ]
    if seen:
        found_text = "\n".join(f"  {s}" for s in seen[:5])
    elif summaries:
        found_text = f"  (only {summaries} run summaries: results.json, metadata.json)"
    else:
        found_text = "  (no .json or .jsonl files)"
    raise ValueError(f"could not recognize the log format of {path}. First records:\n{found_text}\n{FORMATS}")
