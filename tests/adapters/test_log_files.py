"""Reading real-world log files: encodings, damaged JSON, JSON Lines, deep nesting, links and run ids. Every
file, line or record skipped or repaired is counted and reported, never dropped silently."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fake_logs import cuga_trajectory, openai_run

from tabagent.adapters import CugaAdapter, GenericJsonAdapter, OpenAIMessagesAdapter, detect_adapter
from tabagent.adapters import logfiles as lf
from tabagent.adapters.logfiles import LoadNotes, iter_files, parse_bytes, read_docs
from tabagent.tabschema.trajectory import get_json_schema, json_copy

RUN = json.dumps(openai_run(0))


def _docs(tmp_path, name, data):
    f = tmp_path / name
    f.write_bytes(data)
    notes = LoadNotes()
    return [d.data for d in read_docs(tmp_path, f, notes)], notes


def _codes(notes):
    return {code: issue.count for code, issue in notes.issues.items()}


@pytest.mark.parametrize(
    ("data", "fixed"),
    [
        (b"\xef\xbb\xbf" + RUN.encode(), None),  # UTF-8 byte-order mark
        (RUN.encode("utf-16"), None),  # UTF-16 with a BOM (PowerShell's default)
        (json.dumps(openai_run(0), indent=1).replace("\n", "\r\n").encode(), None),  # CRLF
        (RUN[:-1].encode() + b", }", "trailing_commas"),
        (RUN.replace("rome", "rom\xe9").encode("cp1252"), "decoded:cp1252"),
    ],
)
def test_a_run_is_read_whatever_the_encoding_and_small_damage(tmp_path, data, fixed):
    (value,), notes = _docs(tmp_path, "run.json", data)
    assert value["messages"][1]["content"].startswith(openai_run(0)["messages"][1]["content"][:5])
    assert set(_codes(notes)) == ({fixed} if fixed else set())
    if fixed == "decoded:cp1252":  # not UTF-8: read in the first encoding that reads every byte
        assert value["messages"][1]["content"] == "find a flight to rom\xe9 and book it"
        assert "not UTF-8, read as cp1252" in notes.warnings()[0]


def test_nan_and_infinity_are_read(tmp_path):
    (value,), notes = _docs(tmp_path, "x.json", b'{"reward": NaN, "score": Infinity, "steps": []}')
    assert value["reward"] != value["reward"] and value["score"] == float("inf") and not notes.issues


def test_jsonl_bad_lines_are_skipped_and_counted_with_their_place(tmp_path):
    lines = [RUN, "{broken", "", RUN, "[1, 2", "  not json"]
    values, notes = _docs(tmp_path, "runs.jsonl", "\n".join(lines).encode())
    assert len(values) == 2
    issue = notes.issues["invalid_line"]
    assert issue.count == 3 and issue.examples[0].startswith("runs.jsonl:2: line 1, column 2")
    assert "each line of a .jsonl file" in notes.warnings()[0]


def test_a_unicode_line_separator_inside_a_string_does_not_split_a_jsonl_line(tmp_path):
    run = openai_run(0, instruction="first second\x85third\x1cfourth")
    values, notes = _docs(tmp_path, "runs.jsonl", (json.dumps(run, ensure_ascii=False) + "\n").encode())
    assert values == [run] and not notes.issues


def test_jsonl_in_a_json_file_and_a_document_in_a_jsonl_file(tmp_path):
    values, notes = _docs(tmp_path, "runs.json", f"{RUN}\n{RUN}\n".encode())
    assert len(values) == 2 and _codes(notes) == {"jsonl_in_json": 1}
    values, notes = _docs(tmp_path, "pretty.jsonl", json.dumps(openai_run(0), indent=2).encode())
    assert values == [openai_run(0)] and _codes(notes) == {"json_in_jsonl": 1}


def test_unreadable_empty_and_invalid_files_are_reported(tmp_path):
    for name, data in {
        "empty.json": b"  \n",
        "bad.json": b'{"a": 1} trailing',
        "ok.json": RUN.encode(),
    }.items():
        (tmp_path / name).write_bytes(data)
    locked = tmp_path / "locked.json"
    locked.write_bytes(RUN.encode())
    locked.chmod(0)
    try:
        adapter = OpenAIMessagesAdapter()
        trajs = adapter.load(tmp_path)
        readable = os.access(locked, os.R_OK)  # root reads anything
    finally:
        locked.chmod(0o644)
    assert len(trajs) == 1 + readable
    text = "\n".join(adapter.warnings)
    assert "empty, skipped: 1 file (first: empty.json)" in text
    assert "not valid JSON, skipped: 1 file (first: bad.json: line 1, column 10: Extra data)" in text
    assert readable or "could not be read, skipped: 1 file (first: locked.json" in text


def test_values_nested_too_deep_are_cut_never_a_crash(tmp_path):
    deep: dict = {"leaf": "x"}
    for _ in range(400):
        deep = {"d": deep}
    run = {"steps": [{"name": "router", "data": deep}], "task_id": "t1", "score": 1}
    (value,), notes = _docs(tmp_path, "deep.json", json.dumps(run).encode())
    assert _codes(notes) == {"too_deep": 1}
    assert lf.TOO_DEEP in json.dumps(value) and get_json_schema(value) and json_copy(value) == value
    _, notes = _docs(tmp_path, "deeper.json", b"[" * 100_000 + b"]" * 100_000)
    assert _codes(notes) == {"too_deep_to_parse": 1}  # deeper than the parser itself: skipped, not a crash
    ad = GenericJsonAdapter()
    (traj,) = ad.load(tmp_path / "deep.json")
    (dp,) = ad.decision_points(traj, "router")  # every later step works on the cut value
    assert dp.target_step["data"] == value["steps"][0]["data"]


def test_broken_surrogates_become_replacement_characters(tmp_path):
    (value,), notes = _docs(tmp_path, "x.json", b'{"messages": [{"role": "user", "content": "cut \\ud83d"}]}')
    assert value["messages"][0]["content"] == "cut �" and _codes(notes) == {"surrogates": 1}
    value["messages"][0]["content"].encode("utf-8")  # writable and sendable
    (value,), notes = _docs(tmp_path, "y.json", b'{"t": "emoji \\ud83d\\ude00"}')
    assert value["t"] == "emoji \U0001f600" and not notes.issues


def test_parse_bytes_never_raises_anything_but_a_parse_error():
    for data in (b"", b"\xff\xfe\x00", b"{", b"\x00" * 10, b'"\\ud800"', b"nul\x00l"):
        try:
            parse_bytes(data)
        except lf.LogParseError as e:
            assert str(e) and e.code in lf.ISSUES


def test_files_are_found_once_in_any_case_and_through_links(tmp_path):
    logs = tmp_path / "logs"
    (logs / "sub").mkdir(parents=True)
    (logs / "A.JSON").write_text(RUN)
    (logs / "sub" / "b.Jsonl").write_text(RUN)
    try:
        os.symlink(logs / "A.JSON", logs / "sub" / "latest.json")  # a second path to the same file
        os.symlink(logs, logs / "sub" / "loop")
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")
    assert [f.relative_to(logs).as_posix() for f in iter_files(logs)] == ["A.JSON", "sub/b.Jsonl"]


def test_run_ids_stay_unique_when_file_names_collide(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([openai_run(0), openai_run(1)]))  # ids a:0, a:1
    (tmp_path / "a.jsonl").write_text(f"{RUN}\n{RUN}\n")  # ids a:1, a:2 -> a.jsonl:1, a:2
    adapter = OpenAIMessagesAdapter()
    ids = [t.id for t in adapter.load(tmp_path)]
    assert ids == ["a:0", "a:1", "a.jsonl:1", "a:2"] and len(set(ids)) == 4


def test_large_inputs_are_read_lazily_with_the_same_runs(tmp_path, monkeypatch):
    for i in range(4):
        (tmp_path / f"t{i}.json").write_text(json.dumps(cuga_trajectory(f"t{i}", [("phone", ["a"], ["a"])])))
    eager = CugaAdapter().load(tmp_path)
    monkeypatch.setattr(lf, "LAZY_INPUT_BYTES", 0)
    lazy = CugaAdapter().load(tmp_path)
    assert all(t.is_lazy for t in lazy) and not any(t.is_lazy for t in eager)
    assert lazy == eager and [t.raw for t in lazy] == [t.raw for t in eager]
    assert "raw" not in repr(lazy[0]) and "lazy" in repr(lazy[0])
    assert not any(t.is_lazy for t in CugaAdapter().load(tmp_path, lazy=False))  # the switch
    (tmp_path / "t0.json").write_text(json.dumps(cuga_trajectory("t0", [("venmo", ["b"], ["b"])])))
    with pytest.raises(lf.LogChangedError, match="the log file changed while it was being read"):
        _ = lazy[0].raw


def test_detection_reads_past_damaged_files(tmp_path):
    (tmp_path / "a.jsonl").write_text("{broken\n" + RUN + "\n")
    (tmp_path / "results.json").write_text("{}")
    assert type(detect_adapter(tmp_path)) is OpenAIMessagesAdapter
    only = tmp_path / "only"
    only.mkdir()
    (only / "results.json").write_text("{}")
    with pytest.raises(ValueError, match="only 1 run summaries"):
        detect_adapter(only)


# --------------------------------------------------------------------------- review round: H1, H2, L5, L9
def test_a_trailing_comma_repair_on_a_huge_string_stays_small():
    """The repair scans strings with an unrolled pattern (no per-character backtracking stack); it used to
    take about 130 bytes per character of the longest string."""
    import tracemalloc

    data = ('{"a": "' + "x, ]" * 7_500_000 + '", "b": [1, 2,],}').encode()  # a 30 MB string
    tracemalloc.start()
    try:
        fixes: list = []
        value = parse_bytes(data, fixes)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert fixes == ["trailing_commas"] and value["b"] == [1, 2] and len(value["a"]) == 30_000_000
    assert peak < 8 * len(data), f"{peak / 1e6:.0f} MB for a {len(data) / 1e6:.0f} MB file"
    with pytest.raises(lf.LogParseError, match="line 1"):  # damage elsewhere: no repair is attempted
        parse_bytes(('{"a": "' + "x" * 30_000_000 + '", "b": [1 2]}').encode())


def test_a_jsonl_file_whose_first_line_is_cut_is_streamed_not_read_whole(tmp_path, monkeypatch):
    run = openai_run(0)
    run["messages"][1]["content"] = "words " * 50_000
    with (tmp_path / "runs.jsonl").open("w") as fh:
        fh.write('{"messages": [1, 2,\n')  # a line cut by log rotation
        for k in range(40):
            fh.write(json.dumps({**run, "task_id": f"t{k}"}) + "\n")
    read_whole = []
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: read_whole.append(p) or original(p))
    adapter = OpenAIMessagesAdapter()
    trajs = adapter.load(tmp_path, lazy=True)
    assert len(trajs) == 40 and not read_whole
    assert any("runs.jsonl:1" in w for w in adapter.warnings)


def _lazy_cuga(tmp_path, n=3):
    logs = tmp_path / "logs"
    logs.mkdir()
    for i in range(n):
        (logs / f"t{i}.json").write_text(json.dumps(cuga_trajectory(f"t{i}", [("phone", ["p"], ["p"])])))
    return logs, CugaAdapter().load(logs, lazy=True)


def test_a_touched_file_or_a_grown_jsonl_file_reads_the_same_runs(tmp_path):
    logs, trajs = _lazy_cuga(tmp_path)
    before = [t.raw for t in trajs]
    for f in logs.iterdir():
        os.utime(f, None)  # a sync tool or the agent touching the folder
    assert [t.raw for t in trajs] == before
    f = tmp_path / "runs.jsonl"
    f.write_text("".join(json.dumps(openai_run(i)) + "\n" for i in range(3)))
    lines = OpenAIMessagesAdapter().load(f, lazy=True)
    with f.open("a") as fh:  # the agent appends one more run
        fh.write(json.dumps(openai_run(9)) + "\n")
    assert [t.raw for t in lines] == [openai_run(i) for i in range(3)]


def test_any_change_to_a_runs_own_bytes_is_caught_on_every_read(tmp_path):
    logs, trajs = _lazy_cuga(tmp_path)
    assert trajs[0].raw["intent"]  # read once: the parsed run is cached
    f = logs / "t0.json"
    st = f.stat()
    f.write_text(f.read_text().replace("roommate", "ROOMMATE"))  # the same size
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))  # and the old time (rsync -t, cp -p)
    for _ in range(2):  # the cache never hides it
        with pytest.raises(
            lf.LogChangedError, match=r"t0\.json: the log file changed while it was being read"
        ):
            _ = trajs[0].raw
    (logs / "t1.json").unlink()
    with pytest.raises(lf.LogChangedError, match="cannot be read any more"):
        _ = trajs[1].raw
    assert trajs[2].raw["task_id"] == "t2"


def test_lazy_runs_survive_a_change_of_working_directory(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "t0.json").write_text(json.dumps(cuga_trajectory("t0", [("phone", ["p"], ["p"])])))
    monkeypatch.chdir(tmp_path)
    (traj,) = CugaAdapter().load("logs", lazy=True)
    monkeypatch.chdir(tmp_path.parent)
    assert traj.raw["task_id"] == "t0"


def test_encodings_are_tried_in_order_and_named(tmp_path):
    text = json.dumps({"steps": [], "intent": "café crème brûlée"}, ensure_ascii=False)
    (value,), notes = _docs(tmp_path, "cp.json", text.encode("cp1252"))
    assert value["intent"] == "café crème brûlée" and list(_codes(notes)) == ["decoded:cp1252"]
    damaged = text.encode().replace("\u00e9e".encode(), b"\xc3e")  # one character cut: keeps UTF-8
    (value,), notes = _docs(tmp_path, "cut.json", damaged)
    assert "�" in value["intent"] and "café" in value["intent"]
    assert _codes(notes) == {"not_utf8": 1} and "1 byte" in notes.warnings()[0]
    hopeless = b'{"steps": [], "intent": "\x81\x8d\x90"}'  # neither UTF-8 nor cp1252
    (value,), notes = _docs(tmp_path, "bad.json", hopeless)
    assert value["intent"].count("�") == 3 and _codes(notes) == {"not_utf8": 3}
    assert notes.issues["not_utf8"].level == "warning" and "3 bytes" in notes.warnings()[0]


def test_the_run_cache_belongs_to_one_load(tmp_path):
    import gc
    import tracemalloc

    from tabagent.tabschema import trajectory

    assert not hasattr(trajectory, "_RAW_CACHE")  # no process-wide cache
    logs = tmp_path / "logs"
    logs.mkdir()
    for i in range(4):
        run = cuga_trajectory(f"t{i}", [("phone", ["p"], ["p"])])
        run["steps"][0]["data"] = "x" * 2_000_000
        (logs / f"t{i}.json").write_text(json.dumps(run))
    tracemalloc.start()
    try:
        trajs = CugaAdapter().load(logs, lazy=True)
        _ = [t.raw for t in trajs]  # fills the load's cache
        held = tracemalloc.get_traced_memory()[0]
        del trajs, _
        gc.collect()
        after = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    assert held > 6e6 and after < 1e6, (held, after)


def test_the_run_cache_evicts_by_count_and_by_bytes(monkeypatch):
    cache = lf.RunCache(size=2)
    parsed = []

    def parse(k):
        return lambda: parsed.append(k) or {"k": k}

    for k in (1, 2, 3, 1):  # 1 was evicted by 3: parsed again
        assert cache.get(k, 10, parse(k)) == {"k": k}
    assert parsed == [1, 2, 3, 1] and cache.get(3, 10, parse(3)) == {"k": 3} and parsed[-1] == 1
    monkeypatch.setattr(lf, "RUN_CACHE_BYTES", 25)
    small = lf.RunCache(size=8)
    for k in range(4):
        small.get(k, 10, parse(k))
    assert len(small._items) == 2 and small._bytes == 20  # the byte budget
    small.get("huge", 100, parse("huge"))  # larger than the budget: parsed, never kept
    assert "huge" not in small._items
