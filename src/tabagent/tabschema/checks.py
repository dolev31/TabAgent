"""Deterministic checks on LLM-written extraction code and its output.

They complement the CodeValidator (an LLM judge, paper L1216) with rules that need no model:

* the code parses, imports only allowed standard-library modules, and calls no dangerous builtins;
* it follows the CodeExtractor contract (``dict['key']`` access, never ``.get()``, L1293);
* every constant key path it reads from the trajectory exists in the union schema of the onboarding
  prefixes (a hallucinated field is the failure the paper's CodeValidator was introduced for);
* it never reads post-hoc outcome fields (scores, rewards, evaluator results);
* its output is a flat dict of JSON scalars.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

ALLOWED_IMPORTS = {
    "json",
    "re",
    "math",
    "collections",
    "typing",
    "statistics",
    "itertools",
    "string",
    "datetime",
    "functools",
    "operator",
    "copy",
}
FORBIDDEN_CALLS = {
    "exit",
    "quit",
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}
ROOTS = ("trajectory_data", "trajectory")


@dataclass(frozen=True)
class Issue:
    kind: str  # syntax | import | call | uses_get | unknown_field | outcome_field | output
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


def _subscript_chain(node: ast.AST) -> tuple[str | None, list[str]]:
    """For ``a['x'][0]['y']`` return ("a", ["x", "[]", "y"]); non-constant indices become "[]"."""
    keys: list[str] = []
    while isinstance(node, ast.Subscript):
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            keys.append(sl.value)
        else:
            keys.append("[]")
        node = node.value
    name = node.id if isinstance(node, ast.Name) else None
    return name, list(reversed(keys))


def key_paths(tree: ast.AST) -> set[tuple[str, ...]]:
    """Constant key paths read from the trajectory, following simple aliases (``steps = t['steps']``) and
    loop variables (``for s in t['steps']``)."""
    aliases: dict[str, tuple[str, ...]] = {r: () for r in ROOTS}
    paths: set[tuple[str, ...]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, keys = _subscript_chain(node.value)
            if name in aliases:
                aliases[node.targets[0].id] = (*aliases[name], *keys)
        elif isinstance(node, (ast.For, ast.comprehension)) and isinstance(node.target, ast.Name):
            name, keys = _subscript_chain(node.iter)
            if name in aliases:
                aliases[node.target.id] = (*aliases[name], *keys, "[]")
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and not isinstance(getattr(node, "parent", None), ast.Subscript):
            name, keys = _subscript_chain(node)
            if name in aliases and keys:
                paths.add((*aliases[name], *keys))
    return paths


def _covered(path: tuple[str, ...], known: set[tuple]) -> bool:
    """A path is valid if it exists in the schema; trailing list indices on scalar lists are tolerated."""
    while path:
        if path in known:
            return True
        if path[-1] != "[]":
            return False
        path = path[:-1]
    return True


def check_code(
    code: str, known_paths: set[tuple] | None = None, outcome_keys: tuple[str, ...] = ()
) -> list[Issue]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [Issue("syntax", f"line {e.lineno}: {e.msg}")]
    for parent in ast.walk(tree):  # mark parents so only outermost subscripts are reported
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        else:
            mods = []
        issues += [Issue("import", m) for m in mods if m not in ALLOWED_IMPORTS]
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in FORBIDDEN_CALLS:
                issues.append(Issue("call", f.id))
            if isinstance(f, ast.Attribute) and f.attr == "get":
                issues.append(Issue("uses_get", f"line {node.lineno}: use dict['key'] instead of .get()"))
    for path in sorted(key_paths(tree)):
        if any(p in outcome_keys for p in path):
            issues.append(Issue("outcome_field", ".".join(path)))
        elif known_paths is not None and not _covered(path, known_paths):
            issues.append(Issue("unknown_field", ".".join(path)))
    return issues


def check_output(features: Any) -> list[Issue]:
    if not isinstance(features, dict):
        return [Issue("output", "features must be a dict")]
    bad = [k for k, v in features.items() if not (v is None or isinstance(v, (str, int, float, bool)))]
    return [Issue("output", f"{k} is not a scalar (flatten it with flatten_json)") for k in bad]
