"""Helpers shared by the trajectory adapters: finding log files and parsing tool catalogs into candidates."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tabagent.adapters.logfiles import (  # noqa: F401 - re-exported: the adapters' file helpers
    SKIP_FILES,
    LogParseError,
    iter_files,
    parse_bytes,
    run_id,
)
from tabagent.tabschema.trajectory import Candidate, parse_json, spec_text

_PARAM_KEYS = ("parameters", "input_schema", "inputSchema", "args_schema")
_SPEC_KEYS = frozenset({"description", "api_name", "function", "properties", *_PARAM_KEYS})
#: Keys of a JSON Schema (an object's argument schema), never tool names.
_SCHEMA_KEYS = frozenset(
    {"type", "required", "items", "enum", "additionalProperties", "$schema", "$defs", "title", "default"}
)


def _is_spec(d: Any) -> bool:
    return isinstance(d, dict) and (bool(_SPEC_KEYS & d.keys()) or isinstance(d.get("name"), str))


def is_tool_schema(d: Any) -> bool:
    """Whether ``d`` is a tool schema with its parameters (OpenAI, a bare function, the Anthropic Messages API
    format, MCP, CUGA API docs), as opposed to a ``{"name", "description", "args"}`` candidate record."""
    return isinstance(d, dict) and (
        isinstance(d.get("function"), dict) or any(k in d for k in (*_PARAM_KEYS, "properties", "api_name"))
    )


def _arg_names(params: Any) -> list[str]:
    """Argument names from a parameter schema: the ``properties`` of a JSON Schema, or CUGA's
    ``[{"name": ...}, ...]``; anything malformed gives no names."""
    if isinstance(params, dict):
        props = params.get("properties")
        if isinstance(props, dict):
            return [str(a) for a in props]
        params = props
    if isinstance(params, list):
        return [str(a["name"]) for a in params if isinstance(a, dict) and isinstance(a.get("name"), str)]
    return []


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)


def tool_candidate(spec: dict[str, Any], name: str = "", volatile_keys: Iterable[str] = ()) -> Candidate:
    """One tool schema (OpenAI, Anthropic Messages API format, MCP, CUGA API docs) -> ``Candidate``, the
    schema itself kept as its catalog entry (``Candidate.spec``), without its top-level ``volatile_keys``."""
    f = spec["function"] if isinstance(spec.get("function"), dict) else spec
    params = next((f[k] for k in _PARAM_KEYS if k in f), f if "properties" in f else None)
    raw_name = f.get("name") or f.get("api_name") or name
    tid = raw_name.strip() if isinstance(raw_name, str) else str(raw_name) if raw_name is not None else ""
    return Candidate(
        tid, tid, _text(f.get("description")), ", ".join(_arg_names(params)), spec_text(spec, volatile_keys)
    )


def parse_catalog(catalog: Any, volatile_keys: Iterable[str] = ()) -> dict[str, Candidate]:
    """A tool catalog in any common shape -> ``{tool id: Candidate}``.

    Accepts a JSON file path; a list of ``Candidate`` objects, tool names or tool schemas (OpenAI
    ``{"type": "function", "function": {...}}``, bare functions, the Anthropic Messages API format's
    ``input_schema``, MCP ``inputSchema``); a ``{name: description}`` or ``{name: schema}`` mapping; and
    CUGA API docs (``{app: {api_name: {...}}}``). Argument hints are the parameter names, comma-joined, and a
    schema is kept as the candidate's catalog entry (``Candidate.spec``), without its top-level
    ``volatile_keys``: fields a catalog writes anew in every run (CUGA's ``canary_string``). A tool listed
    twice keeps its last entry; entries without a name are left out.
    """
    if catalog is None:
        return {}
    if isinstance(catalog, (str, Path)):
        p = Path(catalog)
        if not p.is_file():
            raise FileNotFoundError(
                f"tool catalog {str(catalog)[:200]!r}: no such file; pass the path of a JSON file with tool "
                "schemas, or the catalog itself (a list of schemas or a {name: description} mapping)"
            )
        try:
            data = parse_bytes(p.read_bytes())
        except LogParseError as e:
            raise ValueError(f"tool catalog {p}: not valid JSON ({e})") from None
        return parse_catalog(data, volatile_keys)
    if isinstance(catalog, dict) and _is_spec(catalog):
        catalog = [catalog]
    out: dict[str, Candidate] = {}
    if isinstance(catalog, dict):
        for key, v in catalog.items():
            if isinstance(v, str):
                out[str(key)] = Candidate(str(key), str(key), v)
            elif _is_spec(v) or (isinstance(v, dict) and _SCHEMA_KEYS & v.keys()):  # a bare argument schema
                c = tool_candidate(v, str(key), volatile_keys)
                out[c.id] = c
            elif isinstance(v, (dict, list)):  # one level of grouping, e.g. CUGA apps
                out.update(parse_catalog(v, volatile_keys))
        return {k: c for k, c in out.items() if k}
    if not isinstance(catalog, (list, tuple)):
        return {}
    for item in catalog:
        if isinstance(item, Candidate):
            c = item
        elif isinstance(item, str):
            c = Candidate(item.strip(), item.strip())
        elif isinstance(item, dict):
            c = tool_candidate(item, volatile_keys=volatile_keys)
        else:
            continue
        if c.id:
            out[c.id] = c
    return out


def run_catalog(value: Any) -> tuple[dict[str, Candidate], bool]:
    """The catalog a run's own ``tools`` field records, and whether the field could be read as one: a list or
    mapping of tools, or JSON text of one (some loggers write the list as a string). Never a file path."""
    if value is None or value == "" or value == [] or value == {}:
        return {}, True
    if isinstance(value, str):
        value = parse_json(value)
        if value is None:
            return {}, False
    if not isinstance(value, (list, dict)):
        return {}, False
    try:
        return parse_catalog(value), True
    except (TypeError, ValueError, AttributeError):
        return {}, False
