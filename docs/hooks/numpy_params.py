"""griffe extension: read this package's docstring conventions as NumPy-style sections.

The docstrings list parameters as ``name: description`` under a NumPy-style ``Parameters`` header, with
continuation lines indented, and describe untyped return values as prose under ``Returns``. NumPy style
proper wants ``name`` then the description on indented lines, and a type line under ``Returns``, so read
as they stand the descriptions would become types and be dropped from the API reference. This extension
rewrites those sections before griffe parses them (the parameter types come from the signature), and
turns Sphinx roles (:func:`name`) into plain code spans. It changes only what the documentation shows,
never a docstring in the code."""

from __future__ import annotations

import re

import griffe

_RULE = re.compile(r"^\s*-{3,}\s*$")
_ITEM = re.compile(r"^(?P<indent>\s*)(?P<names>[A-Za-z_*][\w*]*(?:, [A-Za-z_*][\w*]*)*): (?P<rest>\S.*)$")
_BARE = re.compile(r"^(?P<indent>\s*)(?P<names>[a-z_*][\w*]*(?:, [A-Za-z_*][\w*]*)*):\s*$")
_TYPE_DESC = re.compile(r"^(?P<indent>\s*)(?P<type>[A-Za-z_][\w.\[\], |]*): (?P<rest>\S.*)$")
_TYPE_ONLY = re.compile(r"^\s*[A-Za-z_][\w.\[\], |]*\s*$")
_ROLE = re.compile(r":(?:func|class|meth|mod|attr|exc|data|obj):`~?([^`]+)`")
_ITEM_SECTIONS = {"Parameters", "Other Parameters", "Attributes"}
_RESULT_SECTIONS = {"Returns", "Yields"}
_ALL = _ITEM_SECTIONS | _RESULT_SECTIONS | {"Raises", "Examples", "Notes", "See Also", "Warns", "References"}


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _sections(lines: list[str]):
    """(start, body_start, end, title) of each NumPy section: a title line followed by a dashed rule."""
    found = []
    for i in range(len(lines) - 1):
        if lines[i].strip() in _ALL and _RULE.match(lines[i + 1]):
            found.append((i, i + 2, lines[i].strip()))
    if not found:
        return []
    ends = [s[0] for s in found[1:]] + [len(lines)]
    return [(s, b, e, t) for (s, b, t), e in zip(found, ends, strict=True)]


def _items(body: list[str], base: int) -> list[str]:
    """``name: description`` items to NumPy items; a paragraph after them, at the items' indent, becomes a
    continuation of the last item instead of a bogus parameter."""
    out = []
    seen = False
    for line in body:
        at_base = line.strip() and _indent(line) == base
        item = _ITEM.match(line) if at_base else None
        bare = _BARE.match(line) if at_base else None
        if item:
            for name in item["names"].split(", "):  # `llm, model: ...` documents both parameters
                out += [f"{item['indent']}{name}", f"{item['indent']}    {item['rest']}"]
            seen = True
        elif bare:
            out.append(f"{bare['indent']}{bare['names']}")
            seen = True
        elif at_base and seen:
            out.append("    " + line)
        else:
            out.append(line)
    return out


def _result(body: list[str], base: int, annotation: str) -> list[str]:
    """A ``Returns`` section in this package's style (prose, or ``type: description``) to NumPy style."""
    first = next((i for i, line in enumerate(body) if line.strip()), None)
    if first is None:
        return body
    line = body[first]
    nxt = body[first + 1] if first + 1 < len(body) else ""
    if _TYPE_ONLY.match(line) and nxt.strip() and _indent(nxt) > base:
        return body  # `type` then an indented description: NumPy style proper
    pad = " " * base
    m = _TYPE_DESC.match(line)
    if m and " " not in m["type"].strip().replace(", ", ",").replace(" | ", "|"):
        head, rest = m["type"], [f"{pad}    {m['rest']}"]
        tail = body[first + 1 :]
    else:
        head, rest, tail = annotation, [], body[first:]
    tail = [("    " + t if t.strip() and _indent(t) <= base else t) for t in tail]
    return [*body[:first], f"{pad}{head}", *rest, *tail]


def convert(text: str, annotation: str = "object") -> str:
    text = _ROLE.sub(lambda m: f"``{m.group(1)}``", text)
    lines = text.split("\n")
    for _start, body_start, end, title in reversed(_sections(lines)):
        base = _indent(lines[_start])
        body = lines[body_start:end]
        if title in _ITEM_SECTIONS:
            body = _items(body, base)
        elif title in _RESULT_SECTIONS:
            body = _result(body, base, annotation)
        lines[body_start:end] = body
    return "\n".join(lines)


class NameColonParams(griffe.Extension):
    def on_instance(self, *, node, obj, agent=None, **kwargs) -> None:
        doc = getattr(obj, "docstring", None)
        if doc is not None and doc.value:
            returns = getattr(obj, "returns", None)
            doc.value = convert(doc.value, str(returns) if returns else "object")

    def on_package(self, *, pkg, **kwargs) -> None:
        """The three scikit-learn estimators share ``_TabAgentBase``'s constructor, which is where their
        parameters are documented: show that section on each estimator too."""
        sk = pkg.members.get("sklearn")
        base = sk.members.get("_TabAgentBase") if sk is not None else None
        if base is None or base.docstring is None or "Parameters\n" not in base.docstring.value:
            return
        params = base.docstring.value[base.docstring.value.index("Parameters\n") :].rstrip()
        for member in sk.members.values():
            if member.is_alias or not member.is_class or member is base:
                continue
            doc = member.docstring
            inherits = any(str(b).endswith("_TabAgentBase") for b in member.bases)
            if doc is not None and "Parameters\n" not in doc.value and inherits:
                doc.value = f"{doc.value.rstrip()}\n\n{params}\n"
