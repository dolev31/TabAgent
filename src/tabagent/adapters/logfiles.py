"""Reading agent log files: real-world JSON and JSON Lines, read the same way by every adapter.

Logs come from many writers, and a log folder is rarely clean. Every adapter reads its files through
:func:`read_docs`, which

* finds the files (:func:`iter_files`): recursive and sorted, hidden files skipped, linked folders followed
  once, a file reached through two links read once, ``.JSON`` read like ``.json``;
* decodes them: UTF-8 with or without a byte-order mark, UTF-16 and UTF-32 with one; a file that is not
  UTF-8 is read in the locale's encoding or cp1252, with a warning naming the encoding, and bytes no encoding
  reads (or the few damaged bytes of a UTF-8 file) become U+FFFD with a warning that counts them; so does half
  of a surrogate pair (text cut in the middle of a character), which would otherwise fail when the text is
  written out or sent to an LLM provider;
* parses them: ``NaN`` and ``Infinity`` are accepted (Python's reading of them); trailing commas are dropped
  outside strings; a ``.json`` file holding one object per line is read as JSON Lines and a ``.jsonl`` file
  holding one pretty-printed document as JSON; a ``.jsonl`` file is read one line at a time, and only ``\\n``
  ends a line (``\\r\\n`` works, and a U+2028 inside a string does not split it);
* keeps values at most :data:`MAX_DEPTH` levels deep: deeper values are replaced by a placeholder, so that no
  later step of the pipeline runs out of recursion on a pathological log.

Nothing is skipped silently: every file, line or record left out, and every repair, is counted in a
:class:`LoadNotes` with the first places it happened, which the adapter reports as ``warnings`` (and
``tabagent.check_logs`` as a diagnosis). A large input (:data:`LAZY_INPUT_BYTES` and more, or ``lazy=True``)
is not held in memory: a run read from a file of its own or from a JSON Lines line is kept as its place in the
file and the SHA-256 of exactly the bytes it was read from, and read again when it is needed; every read
checks those bytes, so a touched file or a JSON Lines file that grew passes, and any change to the run's own
bytes raises :class:`LogChangedError`.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import locale
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SKIP_FILES = frozenset({"results.json", "metadata.json"})  # run summaries written next to trajectories
SUFFIXES = (".json", ".jsonl")
#: Values nested deeper than this are replaced by :data:`TOO_DEEP` when a log is read.
MAX_DEPTH = 200
TOO_DEEP = f"[nested deeper than {MAX_DEPTH} levels]"
#: From this many bytes of input, runs are read again from their files when needed instead of kept in memory.
LAZY_INPUT_BYTES = 256 * 2**20
#: The source bytes of the parsed runs a lazy load keeps (a few recent runs, so that reading one run several
#: times in a row parses it once).
RUN_CACHE_BYTES = 32 * 2**20
_EXAMPLES = 3

_BOMS = (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff", b"\xff\xfe", b"\xfe\xff")  # UTF-32 and UTF-16
# A string (unrolled: no backtracking stack per character), or a comma before a closing bracket.
_TRAILING_COMMA = re.compile(r'("[^"\\]*(?:\\.[^"\\]*)*")|,(\s*[}\]])')
_SURROGATE_ESCAPE = re.compile(r"\\u[dD][89a-fA-F][0-9a-fA-F]{2}")  # \ud800-\udfff, escaped
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


# --------------------------------------------------------------------------- what reading found
#: code -> (level, what was found and done, how to fix it). An ``error`` stops onboarding; a ``detail`` is
#: shown by ``check_logs`` only, not in the adapter's ``warnings``.
ISSUES: dict[str, tuple[str, str, str]] = {
    "unreadable": ("warning", "could not be read, skipped", "check that they exist and are readable"),
    "empty": ("warning", "empty, skipped", ""),
    "invalid_json": (
        "warning",
        "not valid JSON, skipped",
        "fix the first error shown, or move such files out of the log folder",
    ),
    "invalid_line": (
        "warning",
        "not valid JSON, skipped",
        "each line of a .jsonl file must hold one complete JSON value",
    ),
    "too_deep_to_parse": ("warning", "nested too deeply to parse, skipped", ""),
    "not_utf8": (
        "warning",
        "bytes no encoding could read (damaged bytes of UTF-8 text, or text in neither UTF-8, the locale's "
        "encoding nor cp1252), replaced by U+FFFD (the replacement character)",
        "write the logs as UTF-8",
    ),
    "surrogates": (
        "info",
        "broken Unicode (half of a surrogate pair, usually text cut mid-character), read as U+FFFD",
        "",
    ),
    "trailing_commas": ("info", "trailing commas (JSON allows none), read without them", ""),
    "jsonl_in_json": ("info", "one JSON value per line in a .json file, read as JSON Lines", ""),
    "json_in_jsonl": ("info", "a single JSON document in a .jsonl file, read as JSON", ""),
    "too_deep": (
        "warning",
        f"values nested deeper than {MAX_DEPTH} levels, replaced by a placeholder below that depth",
        "",
    ),
    "summaries": ("detail", "run summaries (results.json, metadata.json), not read as runs", ""),
    "not_a_run": (
        "warning",
        "not a run of this log format, skipped",
        "see the supported formats in the docs (Log formats), or pass an adapter that reads them",
    ),
}
_SINGULAR = {
    "files": "file",
    "runs": "run",
    "steps": "step",
    "simulations": "simulation",
    "JSONL lines": "JSONL line",
    "records": "record",
    "bytes": "byte",
    "calls": "call",
}


@dataclass
class Issue:
    """Something reading the logs found: ``count`` items (files, JSONL lines, runs, steps) and where the first
    ones are. ``text`` replaces the generated message."""

    code: str
    level: str  # error | warning | info | detail
    what: str  # the items counted: "files", "JSONL lines", "runs", ...
    found: str  # what was found and what was done about it
    fix: str = ""
    count: int = 0
    examples: list[str] = field(default_factory=list)
    text: str | None = None

    def message(self, fix: bool = True) -> str:
        """What was found, how many, where the first ones are and (with ``fix``) how to fix it."""
        if self.text is not None:
            return self.text
        unit = self.what if self.count != 1 else _SINGULAR.get(self.what, self.what)
        where = f" (first: {'; '.join(self.examples)})" if self.examples else ""
        hint = f"; {self.fix}" if self.fix and fix else ""
        return f"{self.found}: {self.count:,} {unit}{where}{hint}"


class LoadNotes:
    """The issues one ``load`` found, counted by kind with the first places (see :data:`ISSUES`)."""

    def __init__(self) -> None:
        self.issues: dict[str, Issue] = {}

    def add(
        self,
        code: str,
        where: str = "",
        what: str = "files",
        n: int = 1,
        level: str | None = None,
        found: str | None = None,
        fix: str | None = None,
    ) -> None:
        """Count ``n`` items of kind ``code`` at ``where`` (a file, ``file:line`` or a run id)."""
        lvl, text, hint = ISSUES.get(code, ("warning", code, ""))
        issue = self.issues.get(code)
        if issue is None:
            issue = self.issues[code] = Issue(
                code, level or lvl, what, found or text, hint if fix is None else fix
            )
        issue.count += n
        if where and len(issue.examples) < _EXAMPLES:
            issue.examples.append(where)

    def __contains__(self, code: str) -> bool:
        return code in self.issues

    def merge(self, other: LoadNotes) -> None:
        """Add the issues of ``other``."""
        for code, issue in other.issues.items():
            mine = self.issues.get(code)
            if mine is None:
                self.issues[code] = Issue(**{**vars(issue), "examples": list(issue.examples)})
                continue
            mine.count += issue.count
            mine.examples += issue.examples[: max(0, _EXAMPLES - len(mine.examples))]

    def set_text(self, code: str, text: str, level: str = "warning", n: int = 1) -> None:
        """Record an issue with its own message (replacing an earlier one of the same kind)."""
        self.issues[code] = Issue(code, level, "", "", count=n, text=text)

    def warnings(self, levels: Iterable[str] = ("error", "warning", "info")) -> list[str]:
        """The messages of the issues at these levels, in the order they were first found."""
        keep = set(levels)
        return [i.message() for i in self.issues.values() if i.level in keep]


class MixedFormatsError(ValueError):
    """A log folder holds runs of more than one log format."""


class LogChangedError(OSError):
    """A log file changed (or went away) after it was loaded, so a run read from it again would differ."""


# --------------------------------------------------------------------------- files
def _suffix(f: Path) -> str:
    return f.suffix.lower()


def iter_files(path: str | Path, suffixes: Iterable[str] = SUFFIXES) -> list[Path]:
    """``path`` itself, or every matching file below the directory (recursive, sorted, no hidden files or
    folders; the suffix in any case). Symbolic links to directories are followed, each real directory once,
    so a folder of links to log folders reads like the folders themselves; a file reached twice (a link to a
    file of the tree, say) is read once, at its first path in sorted order."""
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"no such file or directory: {p}")
    sfx = {s.lower() for s in suffixes}
    out = []
    seen: set[str] = set()
    for root, dirs, files in os.walk(p, followlinks=True):
        real = os.path.realpath(root)
        if real in seen:  # a link back up the tree, or a second link to the same folder
            dirs[:] = []
            continue
        seen.add(real)
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in files:
            f = Path(root) / name
            if _suffix(f) in sfx and not name.startswith(".") and f.is_file():
                out.append(f)
    files_seen: set[str] = set()
    unique = []
    for f in sorted(out):
        real = os.path.realpath(f)
        if real not in files_seen:
            files_seen.add(real)
            unique.append(f)
    return unique


def relative(root: str | Path, f: Path) -> str:
    """``f`` below ``root`` (a directory), or its file name when ``root`` is the file."""
    root = Path(root)
    try:
        return (f.relative_to(root) if root.is_dir() else Path(f.name)).as_posix()
    except ValueError:
        return f.name


def run_id(root: str | Path, f: Path) -> str:
    """Trajectory id that stays unique across a directory tree: the path relative to ``root``, no suffix."""
    rel = Path(relative(root, f))
    return rel.with_suffix("").as_posix()


class RunIds:
    """Unique run ids for one ``load``: ``base`` as given, and a second run with the same id (``a.json`` and
    ``a.jsonl`` both give ``a``) gets the file's suffix, then ``~2``, ``~3``, ... (deterministic: files are
    read in sorted order)."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def __call__(self, base: str, f: Path, index: Any = None) -> str:
        tail = "" if index is None else f":{index}"
        rid = f"{base}{tail}"
        if rid in self.seen:
            rid = f"{base}{f.suffix}{tail}"
            k = 2
            while rid in self.seen:
                rid, k = f"{base}{f.suffix}{tail}~{k}", k + 1
        self.seen.add(rid)
        return rid


def use_lazy(files: Iterable[Path], lazy: bool | None) -> bool:
    """Whether a load reads its runs lazily: ``lazy`` when given, else from :data:`LAZY_INPUT_BYTES` of
    input."""
    return total_bytes(files) >= LAZY_INPUT_BYTES if lazy is None else bool(lazy)


def total_bytes(files: Iterable[Path]) -> int:
    n = 0
    for f in files:
        try:
            n += f.stat().st_size
        except OSError:
            continue
    return n


# --------------------------------------------------------------------------- parsing
class LogParseError(ValueError):
    """A file or line that cannot be read as JSON; ``code`` is the :data:`ISSUES` kind."""

    def __init__(self, message: str, code: str = "invalid_json"):
        super().__init__(message)
        self.code = code


def _legacy_encodings() -> list[str]:
    """The encodings tried after UTF-8: the locale's preferred one (when it is not UTF-8), then cp1252."""
    out = []
    try:
        pref = codecs.lookup(locale.getpreferredencoding(False)).name
    except (LookupError, ValueError):
        pref = ""
    if pref and pref not in ("utf-8", "ascii"):
        out.append(pref)
    if "cp1252" not in out:
        out.append("cp1252")
    return out


def _decode(data: bytes, fixes: list[str]) -> str:
    """Bytes as text: UTF-8 (a BOM, UTF-16 and UTF-32 detected), else the first of the locale's encoding and
    cp1252 that reads every byte (noted as ``decoded:<encoding>``), else UTF-8 with U+FFFD for what it cannot
    read (noted as ``not_utf8``, with the count). A UTF-8 file with a few damaged bytes (more valid non-ASCII
    characters than bad bytes, say text cut mid-character) keeps UTF-8 with U+FFFD: reading it as cp1252 would
    garble every other character."""
    enc = json.detect_encoding(data)
    try:
        return data.decode(enc)
    except UnicodeDecodeError:
        pass
    replaced = data.decode(enc, "replace")
    bad = replaced.count("\ufffd") - data.decode(enc, "ignore").count("\ufffd")
    non_ascii = len(replaced) - len(replaced.encode("ascii", "ignore"))
    damaged_utf8 = enc.startswith("utf-8") and non_ascii - bad >= bad
    if enc.startswith("utf-8") and not damaged_utf8:
        for legacy in _legacy_encodings():
            try:
                text = data.decode(legacy)
            except UnicodeDecodeError:
                continue
            fixes.append(f"decoded:{legacy}")
            return text
    fixes.append(f"not_utf8:{bad}")
    return replaced


def _trailing_comma_at(text: str, pos: int) -> bool:
    """Whether a JSON error at ``pos`` is a trailing comma: a comma, then (after whitespace) a closing
    bracket. Python 3.13 points at the comma, earlier versions at the bracket."""
    if not 0 <= pos < len(text):
        return False
    space = " \t\r\n"
    if text[pos] == ",":
        k = pos + 1
        while k < len(text) and text[k] in space:
            k += 1
        return k < len(text) and text[k] in "}]"
    k = pos
    while k < len(text) and text[k] in space:
        k += 1
    if k >= len(text) or text[k] not in "}]":
        return False
    k = pos - 1
    while k >= 0 and text[k] in space:
        k -= 1
    return k >= 0 and text[k] == ","


def _loads(text: str, fixes: list[str]) -> Any:
    try:
        return json.loads(text)
    except RecursionError:
        raise LogParseError("nested too deeply to parse", "too_deep_to_parse") from None
    except json.JSONDecodeError as e:
        first = e
    if _trailing_comma_at(text, first.pos):  # only then: the repair scans the whole text
        repaired = _TRAILING_COMMA.sub(lambda m: m.group(1) or m.group(2), text)
        try:
            value = json.loads(repaired)
        except (ValueError, RecursionError):
            pass
        else:
            fixes.append("trailing_commas")
            return value
    raise LogParseError(f"line {first.lineno}, column {first.colno}: {first.msg}", "invalid_json") from None


def _too_deep(value: Any) -> bool:
    """Whether ``value`` nests containers :data:`MAX_DEPTH` levels deep (one level at a time, without
    recursion)."""
    level, depth = [value], 1
    while level:
        if depth >= MAX_DEPTH:
            return True
        level = [
            c
            for node in level
            for c in (node.values() if type(node) is dict else node)
            if type(c) is dict or type(c) is list
        ]
        depth += 1
    return False


def sanitize(value: Any, fixes: list[str], surrogates: bool = True) -> Any:
    """``value`` with containers deeper than :data:`MAX_DEPTH` replaced by :data:`TOO_DEEP` and, with
    ``surrogates``, unpaired surrogates in strings replaced by U+FFFD; changed in place (a fresh parse),
    noting what changed in ``fixes``. Iterative: never recurses."""
    deep = isinstance(value, (dict, list)) and _too_deep(value)
    if not deep and not surrogates:
        return value
    fix = (lambda s: _LONE_SURROGATE.sub("\ufffd", s)) if surrogates else None
    cut = fixed = 0

    def check(v: Any) -> Any:
        nonlocal fixed
        if fix is not None and isinstance(v, str) and _LONE_SURROGATE.search(v):
            fixed += 1
            return fix(v)
        return v

    value = check(value)
    stack = [(value, 1)] if isinstance(value, (dict, list)) else []
    while stack:
        node, depth = stack.pop()
        items = list(node.items()) if isinstance(node, dict) else list(enumerate(node))
        if isinstance(node, dict) and fix is not None and any(_LONE_SURROGATE.search(k) for k, _ in items):
            node.clear()
            node.update((check(k), v) for k, v in items)
            items = list(node.items())
        for k, v in items:
            if isinstance(v, (dict, list)):
                if depth >= MAX_DEPTH:
                    node[k] = TOO_DEEP
                    cut += 1
                else:
                    stack.append((v, depth + 1))
            elif fix is not None and isinstance(v, str):
                new = check(v)
                if new is not v:
                    node[k] = new
    if cut:
        fixes.append("too_deep")
    if fixed:
        fixes.append("surrogates")
    return value


def _lone_escapes(text: str) -> bool:
    """Whether JSON text escapes half of a surrogate pair (a valid pair, ``\\ud83d\\ude00``, is read as one
    character, and ``\\\\ud83d`` is a backslash and text). One regex scan; the rare text with surrogate
    escapes is then checked escape by escape."""
    first = _SURROGATE_ESCAPE.search(text)
    if first is None:
        return False
    low_at = -1  # where the low half of the last high surrogate must start
    for m in _SURROGATE_ESCAPE.finditer(text, first.start()):
        k = m.start()
        while k > 0 and text[k - 1] == "\\":
            k -= 1
        if (m.start() - k) % 2:  # an escaped backslash, then text
            continue
        if m.group()[3] in "89abAB":  # a high surrogate, \\ud800-\\udbff
            if low_at != -1:
                return True
            low_at = m.end()
        elif m.start() != low_at:
            return True
        else:
            low_at = -1
    return low_at != -1


def parse_bytes(data: bytes, fixes: list[str] | None = None) -> Any:
    """One JSON document from raw bytes, read as described in the module docstring; what had to be repaired is
    appended to ``fixes`` (:data:`ISSUES` codes). Raises :class:`LogParseError`."""
    fixes = [] if fixes is None else fixes
    text = _decode(data, fixes)
    if not text.strip():
        raise LogParseError("empty", "empty")
    value = _loads(text, fixes)
    return sanitize(value, fixes, surrogates=_lone_escapes(text))


# --------------------------------------------------------------------------- documents
def digest(data: bytes) -> bytes:
    """The fingerprint of the bytes a run was read from (SHA-256, truncated to 16 bytes)."""
    return hashlib.sha256(data).digest()[:16]


@dataclass
class Doc:
    """One JSON document read from a log file: the whole file, or one line of a JSON Lines file."""

    file: Path  # absolute, so that a later change of the working directory does not matter
    where: str  # ``rel`` or ``rel:line``, for messages
    line: int | None  # 1-based line of a JSON Lines record
    data: Any
    span: tuple[int, int] | None = None  # (byte offset, length) of a line; None: the whole file
    digest: bytes | None = None  # of exactly the bytes the document was read from
    reloadable: bool = True  # whether ``reload_doc`` can read it again (not a line of a UTF-16 file)


def _note_fixes(notes: LoadNotes, fixes: list[str], where: str, what: str) -> None:
    for code in dict.fromkeys(fixes):
        name, _, arg = code.partition(":")
        if name == "decoded":
            notes.add(
                f"decoded:{arg}",
                where,
                what,
                level="warning",
                found=f"not UTF-8, read as {arg} (the first encoding that reads every byte)",
                fix="write the logs as UTF-8 (or check that the text reads as intended)",
            )
        elif name == "not_utf8":
            notes.add("not_utf8", where, "bytes", n=int(arg or 1))
        else:
            notes.add(code, where, what)


def _split_lines(data: bytes) -> list[bytes]:
    """The lines of a JSON Lines file, each with its newline: only ``\\n`` ends a line."""
    lines = data.split(b"\n")
    return [ln + b"\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def _lines(data: bytes, wide: bool) -> list[bytes]:
    if not wide:
        return _split_lines(data)
    text = data.decode(json.detect_encoding(data), "replace").lstrip("\ufeff")
    return [ln.encode("utf-8", "surrogatepass") for ln in text.split("\n")]


def _jsonl_docs(
    f: Path, rel: str, lines: Iterable[bytes], reloadable: bool, notes: LoadNotes
) -> Iterator[Doc]:
    """The documents of JSON Lines, one at a time (a large file is never held parsed as a whole)."""
    offset = 0
    for i, raw in enumerate(lines, 1):
        start, offset = offset, offset + len(raw)
        if not raw.strip():
            continue
        fixes: list[str] = []
        try:
            value = parse_bytes(raw, fixes)
        except LogParseError as e:
            code = "invalid_line" if e.code == "invalid_json" else e.code
            notes.add(code, f"{rel}:{i}: {e}" if code == "invalid_line" else f"{rel}:{i}", "JSONL lines")
            continue
        _note_fixes(notes, fixes, f"{rel}:{i}", "JSONL lines")
        yield Doc(f, f"{rel}:{i}", i, value, (start, len(raw)), digest(raw), reloadable)


def _opens_a_document(first: bytes) -> bool:
    """Whether a first line opens a pretty-printed JSON document (a lone ``{`` or ``[``)."""
    return first.strip().lstrip(b"\xef\xbb\xbf").strip() in (b"{", b"[")


def _first_line_parses(lines: Iterable[bytes]) -> bool:
    first = next((ln for ln in lines if ln.strip()), b"")
    try:
        parse_bytes(first)
    except LogParseError:
        return False
    return True


def _whole(f: Path, rel: str, data: bytes, code: str, notes: LoadNotes) -> Doc | None:
    """The file as one JSON document (noting ``code``), or None when it is not one."""
    fixes: list[str] = []
    try:
        value = parse_bytes(data, fixes)
    except LogParseError:
        return None
    notes.add(code, rel)
    _note_fixes(notes, fixes, rel, "files")
    return Doc(f, rel, None, value, None, digest(data))


def read_docs(root: str | Path, f: Path, notes: LoadNotes) -> Iterator[Doc]:
    """The JSON documents of one log file (one, or one per line of a JSON Lines file, read line by line),
    noting in ``notes`` whatever was skipped or repaired."""
    rel = relative(root, f)
    f = f.absolute()
    try:
        with f.open("rb") as fh:
            head = fh.read(4)
            wide = head.startswith(_BOMS)  # UTF-16/32: lines are not split on the byte b"\n"
            if _suffix(f) == ".jsonl" and not wide:
                fh.seek(0)
                first = next((ln for ln in fh if ln.strip()), b"")
                if _opens_a_document(first):  # a pretty-printed JSON document in a .jsonl file?
                    doc = _whole(f, rel, f.read_bytes(), "json_in_jsonl", notes)
                    if doc is not None:
                        yield doc
                        return
                fh.seek(0)
                found = False
                for doc in _jsonl_docs(f, rel, fh, True, notes):
                    found = True
                    yield doc
                if not found and not first.strip():
                    notes.add("empty", rel)
                return
        data = f.read_bytes()
    except OSError as e:
        notes.add("unreadable", f"{rel}: {e.strerror or e}")
        return
    if not (data.decode(json.detect_encoding(data), "replace") if wide else data).strip():
        notes.add("empty", rel)
        return
    if _suffix(f) == ".jsonl":  # UTF-16 or UTF-32 JSON Lines: decoded as a whole
        lines = _lines(data, wide)
        if _opens_a_document(next((ln for ln in lines if ln.strip()), b"")):
            doc = _whole(f, rel, data, "json_in_jsonl", notes)
            if doc is not None:
                yield doc
                return
        yield from _jsonl_docs(f, rel, lines, False, notes)
        return
    fixes: list[str] = []
    try:
        value = parse_bytes(data, fixes)
    except LogParseError as e:
        lines = _lines(data, wide) if e.code == "invalid_json" and str(e).endswith("Extra data") else []
        if lines and _first_line_parses(lines):  # JSON Lines in a .json file
            notes.add("jsonl_in_json", rel)
            yield from _jsonl_docs(f, rel, lines, not wide, notes)
            return
        notes.add(e.code, f"{rel}: {e}" if e.code == "invalid_json" else rel)
        return
    _note_fixes(notes, fixes, rel, "files")
    yield Doc(f, rel, None, value, None, digest(data))


#: How a run that is read again from a changed log file is reported.
CHANGED_HINT = (
    "A run of a large input is read again from its file when it is needed: load the logs again once they "
    "stop changing, or read them into memory (lazy=False; CLI --no-lazy). An onboarding rerun with the same "
    "run directory (run_dir=, --run-dir) reuses every LLM answer already paid for."
)


def read_span(doc_file: Path, span: tuple[int, int] | None, expected: bytes | None) -> bytes:
    """Exactly the bytes a document was read from, checked against their fingerprint. Raises
    :class:`LogChangedError` (saying what changed) when the file went away or those bytes differ."""
    where = doc_file if span is None else f"{doc_file} (bytes {span[0]:,} to {span[0] + span[1]:,})"
    try:
        if span is None:
            data = doc_file.read_bytes()
        else:
            with doc_file.open("rb") as fh:
                fh.seek(span[0])
                data = fh.read(span[1])
    except OSError as e:
        raise LogChangedError(
            f"{doc_file}: the log file changed while it was being read: it cannot be read any more "
            f"({e.strerror or e}). {CHANGED_HINT}"
        ) from e
    if expected is not None and digest(data) != expected:
        what = (
            f"it is {len(data):,} bytes long now"
            if span is None
            else "the file is shorter now"
            if len(data) < span[1]
            else "the line the run was read from holds other bytes now"
        )
        raise LogChangedError(
            f"{where}: the log file changed while it was being read: the bytes a run was loaded from are not "
            f"the same any more ({what}). {CHANGED_HINT}"
        )
    return data


def reload_doc(doc_file: Path, span: tuple[int, int] | None, expected: bytes | None) -> Any:
    """The document at ``span`` of ``doc_file`` (the whole file for None), parsed exactly as when it was first
    read, after checking its bytes (:func:`read_span`)."""
    return parse_bytes(read_span(doc_file, span, expected))


class RunCache:
    """The parsed runs one ``load`` read last: at most ``size`` runs and :data:`RUN_CACHE_BYTES` of their
    source bytes, keyed by their place and fingerprint. Every read still checks the bytes on disk
    (:func:`read_span`); the cache only saves parsing them again. It belongs to the trajectories of one load
    and goes away with them."""

    def __init__(self, size: int = 8):
        self.size = size
        self._items: OrderedDict[Any, tuple[int, Any]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: Any, n_bytes: int, parse: Callable[[], Any]) -> Any:
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
                return hit[1]
        value = parse()  # outside the lock: parsing a large run takes a while
        budget = RUN_CACHE_BYTES
        if n_bytes > budget:
            return value
        with self._lock:
            if key not in self._items:
                self._items[key] = (n_bytes, value)
                self._bytes += n_bytes
            while self._items and (len(self._items) > self.size or self._bytes > budget):
                _, (old_bytes, _) = self._items.popitem(last=False)
                self._bytes -= old_bytes
        return value


def lazy_loader(
    doc: Doc, pick: Callable[[Any], Any] | None = None, cache: RunCache | None = None
) -> Callable[[], Any] | None:
    """A function that reads ``doc`` again from its file, checking its bytes every time, and returns a fresh
    copy of the run (``pick`` applied), or None when the document cannot be read again on its own."""
    if not doc.reloadable or doc.digest is None:
        return None
    from tabagent.tabschema.trajectory import json_copy

    f, span, expected = doc.file, doc.span, doc.digest
    cache = cache if cache is not None else RunCache(1)

    def load() -> Any:
        data = read_span(f, span, expected)

        def parse() -> Any:
            value = parse_bytes(data)
            return value if pick is None else pick(value)

        return json_copy(cache.get((str(f), span, expected), len(data), parse))  # edits never reach it

    return load
