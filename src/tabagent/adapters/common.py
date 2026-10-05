"""Helpers shared by the trajectory adapters: finding log files and parsing tool catalogs into candidates."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tabagent.tabschema.trajectory import Candidate

SKIP_FILES = frozenset({"results.json", "metadata.json"})  # run summaries written next to trajectories
_PARAM_KEYS = ("parameters", "input_schema", "inputSchema", "args_schema")
_SPEC_KEYS = frozenset({"description", "api_name", "function", "properties", *_PARAM_KEYS})


def iter_files(path: str | Path, suffixes: Iterable[str] = (".json", ".jsonl")) -> list[Path]:
    """``path`` itself, or every matching file below the directory (recursive, sorted, no hidden files).
    Symbolic links to directories are followed, each real directory once, so a folder of links to log
    folders reads like the folders themselves."""
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"no such file or directory: {p}")
    sfx = set(suffixes)
    out, seen = [], set()
    for root, dirs, files in os.walk(p, followlinks=True):
        real = os.path.realpath(root)
        if real in seen:  # a link back up the tree, or a second link to the same folder
            dirs[:] = []
            continue
        seen.add(real)
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in files:
            f = Path(root) / name
            if f.suffix in sfx and not name.startswith(".") and f.is_file():
                out.append(f)
    return sorted(out)


def run_id(root: str | Path, f: Path) -> str:
    """Trajectory id that stays unique across a directory tree: the path relative to ``root``, no suffix."""
    root = Path(root)
    rel = f.relative_to(root) if root.is_dir() else Path(f.name)
    return rel.with_suffix("").as_posix()


def _is_spec(d: Any) -> bool:
    return isinstance(d, dict) and (bool(_SPEC_KEYS & d.keys()) or isinstance(d.get("name"), str))


def is_tool_schema(d: Any) -> bool:
    """Whether ``d`` is a tool schema with its parameters (OpenAI, a bare function, the Anthropic Messages API
    format, MCP, CUGA API docs), as opposed to a ``{"name", "description", "args"}`` candidate record."""
    return isinstance(d, dict) and (
        isinstance(d.get("function"), dict) or any(k in d for k in (*_PARAM_KEYS, "properties", "api_name"))
    )


def tool_candidate(spec: dict[str, Any], name: str = "") -> Candidate:
    """One tool schema (OpenAI, Anthropic Messages API format, MCP, CUGA API docs) -> ``Candidate``."""
    f = spec["function"] if isinstance(spec.get("function"), dict) else spec
    params = next((f[k] for k in _PARAM_KEYS if k in f), f if "properties" in f else None)
    if isinstance(params, dict):
        args = [str(a) for a in params.get("properties") or {}]
    elif isinstance(params, list):  # CUGA: [{"name": ..., "type": ..., "required": ...}]
        args = [str(a["name"]) for a in params if isinstance(a, dict) and a.get("name")]
    else:
        args = []
    tid = str(f.get("name") or f.get("api_name") or name)
    return Candidate(tid, tid, str(f.get("description") or ""), ", ".join(args))


def parse_catalog(catalog: Any) -> dict[str, Candidate]:
    """A tool catalog in any common shape -> ``{tool id: Candidate}``.

    Accepts a JSON file path; a list of ``Candidate`` objects, tool names or tool schemas (OpenAI
    ``{"type": "function", "function": {...}}``, bare functions, the Anthropic Messages API format's
    ``input_schema``, MCP ``inputSchema``); a ``{name: description}`` or ``{name: schema}`` mapping; and
    CUGA API docs (``{app: {api_name: {...}}}``). Argument hints are the parameter names, comma-joined.
    """
    if catalog is None:
        return {}
    if isinstance(catalog, (str, Path)):
        return parse_catalog(json.loads(Path(catalog).read_text()))
    if isinstance(catalog, dict) and _is_spec(catalog):
        catalog = [catalog]
    out: dict[str, Candidate] = {}
    if isinstance(catalog, dict):
        for key, v in catalog.items():
            if isinstance(v, str):
                out[str(key)] = Candidate(str(key), str(key), v)
            elif _is_spec(v):
                c = tool_candidate(v, str(key))
                out[c.id] = c
            elif isinstance(v, (dict, list)):  # one level of grouping, e.g. CUGA apps
                out.update(parse_catalog(v))
        return out
    for item in catalog:
        if isinstance(item, Candidate):
            c = item
        elif isinstance(item, str):
            c = Candidate(item, item)
        elif isinstance(item, dict):
            c = tool_candidate(item)
        else:
            continue
        if c.id:
            out[c.id] = c
    return out
