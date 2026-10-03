"""Deterministic checks on LLM-written extraction code and its output.

They complement the CodeValidator (an LLM judge, paper App. D.1) with rules that need no model:

* the code parses, imports only allowed standard-library modules, calls no dangerous builtins and reaches
  for no interpreter internals (``__builtins__``, ``obj.__class__.__subclasses__()``, a generator's or a
  traceback's frame: ``gi_frame``, ``f_back``, ``f_locals``, ``tb_frame``, ...), the usual ways around the
  first two rules;
* it follows the CodeExtractor contract (``dict['key']`` access, never ``.get()``; App. D.2);
* every constant key path it reads from the trajectory exists in the union schema of the onboarding
  prefixes (a hallucinated field is the failure the paper's CodeValidator was introduced for);
* it never reads the trajectory's top-level post-hoc outcome fields (scores, rewards, evaluator results);
* it never reads the clock (``datetime.now()``, ``date.today()``, ``utcnow()``) or a per-process value
  (``hash()``, ``id()``): a feature must be a function of the prefix alone (the CodeExtractor prompt's
  contract), or it drifts between training and serving (``random``, ``time``, ``uuid``, ``os`` and
  ``secrets`` are not importable at all);
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
#: Calls whose result depends on when or in which process the code runs, not on the prefix: ``now()``,
#: ``today()`` and ``utcnow()`` on ``datetime``/``date`` (resolved through ``import`` aliases), ``hash()`` and
#: ``id()``.
CLOCK_METHODS = {"now", "today", "utcnow"}
CLOCK_CLASSES = {"datetime", "date"}
PROCESS_CALLS = {"hash", "id"}
ROOTS = ("trajectory_data", "trajectory")
ALLOWED_DUNDERS = {"__name__", "__init__"}  # ``if __name__ == "__main__":``, ``type(v).__name__``, classes
#: Attributes of frames, generators, coroutines, tracebacks and code objects: no dunder, but they lead to
#: other code's frames and variables (the interpreter's own included). Refused as attributes, and as string
#: literals (``operator.attrgetter("gi_frame.f_back")``).
FORBIDDEN_ATTRS = {
    *(f"gi_{a}" for a in ("frame", "code", "yieldfrom", "running", "suspended")),
    *(f"cr_{a}" for a in ("frame", "code", "await", "origin", "running", "suspended")),
    *(f"ag_{a}" for a in ("frame", "code", "await", "running")),
    *(f"f_{a}" for a in ("back", "locals", "globals", "builtins", "code", "trace", "lasti", "lineno")),
    *(f"tb_{a}" for a in ("frame", "next", "lasti", "lineno")),
    *(f"co_{a}" for a in ("code", "consts", "names", "varnames", "freevars", "cellvars")),
}
_STRING_DUNDERS_OK = ALLOWED_DUNDERS | {"__main__"}


@dataclass(frozen=True)
class Issue:
    """One rule violation found in extraction code or its output."""

    # syntax | import | call | dunder | internals | uses_get | unknown_field | outcome_field | clock | output
    kind: str
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


def _aliases(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    """Names that hold a part of the trajectory: the roots, and every name whose bindings (``x = t['a']``,
    ``for s in t['steps']``) all resolve to one trajectory path. A name also bound to anything else (parsed
    JSON, a parameter, a tuple unpacking) is not followed, wherever that binding is."""
    binds: dict[str, list[tuple[str | None, list[str]]]] = {}
    tracked: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            binds.setdefault(node.targets[0].id, []).append(_subscript_chain(node.value))
            tracked.add(id(node.targets[0]))
        elif isinstance(node, (ast.For, ast.comprehension)) and isinstance(node.target, ast.Name):
            name, keys = _subscript_chain(node.iter)
            binds.setdefault(node.target.id, []).append((name, [*keys, "[]"]))
            tracked.add(id(node.target))
    other = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and id(n) not in tracked
    }
    other |= {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
    aliases: dict[str, tuple[str, ...]] = {r: () for r in ROOTS}
    for _ in range(len(binds) + 1):  # follow chains of aliases (``steps = t['steps']; for s in steps``)
        changed = False
        for var, bs in binds.items():
            if var in ROOTS:
                continue
            paths = {(*aliases[n], *k) if n in aliases else None for n, k in bs}
            new = next(iter(paths)) if len(paths) == 1 and var not in other else None
            if new != aliases.get(var):
                if new is None:
                    del aliases[var]
                else:
                    aliases[var] = new
                changed = True
        if not changed:
            break
    return aliases


def key_paths(tree: ast.AST) -> set[tuple[str, ...]]:
    """Constant key paths read from the trajectory, following aliases (``steps = t['steps']``) and loop
    variables (``for s in t['steps']``); see :func:`_aliases`."""
    aliases = _aliases(tree)
    paths: set[tuple[str, ...]] = set()
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


def _clock_names(tree: ast.AST) -> set[str]:
    """Names bound to the ``datetime`` module or its ``datetime``/``date`` classes (``import datetime as dt``,
    ``from datetime import datetime``); ``datetime`` and ``date`` themselves count too."""
    names = set(CLOCK_CLASSES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.asname or a.name for a in node.names if a.name.split(".")[0] == "datetime"}
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "datetime":
            names |= {a.asname or a.name for a in node.names}
    return names


def _is_clock(receiver: ast.AST, names: set[str]) -> bool:
    """Whether ``receiver`` (of a ``.now()`` call, say) is ``datetime``, ``date`` or an alias of them, rather
    than an object of the extraction code's own."""
    if isinstance(receiver, ast.Name):
        return receiver.id in names
    if isinstance(receiver, ast.Attribute):
        return receiver.attr in CLOCK_CLASSES and _is_clock(receiver.value, names | CLOCK_CLASSES)
    return False


def check_code(
    code: str, known_paths: set[tuple] | None = None, outcome_keys: tuple[str, ...] = ()
) -> list[Issue]:
    """Static checks of extraction ``code`` (see the module docstring); ``known_paths`` is the union schema
    of the onboarding prefixes and ``outcome_keys`` the fields the code must never read."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [Issue("syntax", f"line {e.lineno}: {e.msg}")]
    for parent in ast.walk(tree):  # mark parents so only outermost subscripts are reported
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]
    issues: list[Issue] = []
    clock = _clock_names(tree)
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
            if isinstance(f, ast.Attribute) and f.attr in CLOCK_METHODS and _is_clock(f.value, clock):
                issues.append(
                    Issue(
                        "clock",
                        f"line {node.lineno}: .{f.attr}() reads the clock; compute features from the "
                        "trajectory only (use its timestamps), or they drift between training and serving",
                    )
                )
            if isinstance(f, ast.Name) and f.id in PROCESS_CALLS:
                issues.append(
                    Issue(
                        "clock",
                        f"line {node.lineno}: {f.id}() differs between processes; compute features from the "
                        "trajectory only",
                    )
                )
        name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else ""
        if name.startswith("__") and name.endswith("__") and name not in ALLOWED_DUNDERS:
            issues.append(Issue("dunder", f"line {node.lineno}: {name}"))
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            issues.append(Issue("internals", f"line {node.lineno}: .{node.attr}"))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 200:
            parts = node.value.split(".")
            reached = [
                p
                for p in parts
                if p in FORBIDDEN_ATTRS
                or (p.startswith("__") and p.endswith("__") and len(p) > 4 and p not in _STRING_DUNDERS_OK)
            ]
            if reached:
                issues.append(Issue("internals", f"line {node.lineno}: the string {node.value!r}"))
    for path in sorted(key_paths(tree)):
        if path and path[0] in outcome_keys:  # a top-level outcome; nested fields of earlier steps are data
            issues.append(Issue("outcome_field", ".".join(path)))
        elif known_paths is not None and not _covered(path, known_paths):
            issues.append(Issue("unknown_field", ".".join(path)))
    return issues


def check_output(features: Any) -> list[Issue]:
    """Issues with one extractor output: it must be a flat dict of JSON scalars."""
    if not isinstance(features, dict):
        return [Issue("output", "features must be a dict")]
    bad = [k for k, v in features.items() if not (v is None or isinstance(v, (str, int, float, bool)))]
    return [Issue("output", f"{k} is not a scalar (flatten it with flatten_json)") for k in bad]
