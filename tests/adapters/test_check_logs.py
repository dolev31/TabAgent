"""``check_logs``: a diagnosis of agent logs before an onboarding, each finding saying what, where and how to
fix it."""

from __future__ import annotations

import json

import pytest
from fake_logs import cuga_trajectory, openai_run, tau2_results, write_openai_logs

from tabagent.adapters import CugaAdapter, GenericJsonAdapter
from tabagent.scanner import check_logs


def test_clean_logs_pass_with_their_decisions_parity_and_leak_check(tmp_path):
    logs = write_openai_logs(tmp_path / "logs", n=12)
    report = check_logs(logs)
    assert report.ok and report.adapter == "OpenAIMessagesAdapter"
    assert (report.runs, report.successful) == (12, 11)
    by_name = {d.name: d for d in report.decisions}
    shortlist = by_name["ToolShortlist"]
    assert shortlist.decision_points == 11 and shortlist.tasks == 4 and not shortlist.labels_outside
    assert shortlist.parity_checked == 11 and not shortlist.parity_failed and shortlist.leak_checked
    text = report.summary()
    assert "live prefixes rebuilt exactly as in training on 11 of 11 checked" in text
    assert "Next: tabagent onboard" in text and "--decision ToolShortlist --dry-run" in text
    assert json.loads(report.to_json())["ok"] is True


def test_problems_say_what_where_and_how_to_fix(tmp_path):
    logs = write_openai_logs(tmp_path / "logs", n=8)
    (logs / "broken.json").write_text('{"messages": [')
    (logs / "old.json.gz").write_bytes(b"\x1f\x8b")
    lines = "\n".join([json.dumps(openai_run(1)), "{oops"])
    (logs / "more.jsonl").write_text(lines)
    report = check_logs(logs)
    assert report.ok  # warnings, not stoppers
    text = " ".join(f.text() for f in report.findings)
    assert report.summary()
    assert "not valid JSON, skipped: 1 file (first: broken.json: line 1, column 15" in text
    assert "more.jsonl:2" in text
    assert ".gz (1)" in text and "decompress them first" in text


def test_a_catalog_that_does_not_name_the_choices_is_a_problem_with_near_misses(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    for i in range(6):
        run = openai_run(i)
        for t in run["tools"]:
            t["function"]["name"] = t["function"]["name"].upper()  # the catalog in another case
        (logs / f"r{i}.json").write_text(json.dumps(run))
    report = check_logs(logs, decision="ToolShortlist")
    assert not report.ok
    (d,) = report.decisions
    assert d.labels_outside == 6 and d.outside[0] == "r0#4 (r0.json): 'book_flight' (offered: 'BOOK_FLIGHT')"
    (err,) = [f for f in report.errors if "not among the offered ones" in f.message]
    assert "use the same names" in err.text() and "r0#4 (r0.json)" in err.message


def test_mixed_formats_no_successes_and_missing_paths_are_errors(tmp_path):
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / "a.json").write_text(json.dumps(openai_run(0)))
    (mixed / "b.json").write_text(json.dumps(tau2_results("airline", [("0", ["t"], 1.0)])))
    report = check_logs(mixed, on_other_formats="error")
    assert not report.ok and "more than one log format" in report.errors[0].message
    skipped = check_logs(mixed)  # by default the other format is skipped with a warning
    assert any("another log format" in f.message and f.level == "warning" for f in skipped.findings)

    failed = write_openai_logs(tmp_path / "failed", n=3, fail_every=1)
    (err,) = check_logs(failed).errors
    assert "none of the 3 runs succeeded" in err.message and "success=" in err.fix

    missing = check_logs(tmp_path / "nowhere")
    assert not missing.ok and missing.errors[0].message.endswith("nowhere: no such file or directory")
    assert missing.errors[0].fix == "check the path" and missing.summary()


def test_cuga_and_generic_logs(tmp_path):
    logs = tmp_path / "cuga"
    logs.mkdir()
    for i in range(4):
        rounds = [("phone", ["phone_search_contacts", "phone_call"], ["phone_search_contacts"])]
        (logs / f"t{i}.json").write_text(json.dumps(cuga_trajectory(f"t{i}", rounds)))
    report = check_logs(logs)
    assert report.ok and [d.name for d in report.decisions] == ["ShortlisterAgent", "TaskAnalyzerAgent"]
    assert all(d.parity_checked == d.decision_points == 4 for d in report.decisions)
    planner = check_logs(logs, adapter=CugaAdapter(), decision="PlanControllerAgent")
    assert planner.decisions[0].candidates_and_labels is False

    generic = check_logs(logs, adapter=GenericJsonAdapter())
    assert generic.ok and not generic.decisions
    assert any("builds no candidates or labels" in f.message for f in generic.findings)


def test_one_task_is_a_problem(tmp_path):
    logs = tmp_path / "one"
    logs.mkdir()
    for i in range(3):
        (logs / f"r{i}.json").write_text(json.dumps(openai_run(0, task_id="same")))
    report = check_logs(logs, decision="NextToolCall")
    assert any("come from one task" in f.message for f in report.errors)


# --------------------------------------------------------------------------- review round: H2, L4
def test_a_log_that_changes_during_the_check_is_reported_as_such(tmp_path, monkeypatch):
    import tabagent.adapters.live as live

    logs = write_openai_logs(tmp_path / "logs", n=6)
    original = live.run_before

    def changing(adapter, traj, dp):  # the agent rewrites a log while the check reads it
        (logs / "run00.json").write_text(json.dumps(openai_run(0, instruction="rewritten")))
        return original(adapter, traj, dp)

    monkeypatch.setattr(live, "run_before", changing)
    report = check_logs(logs, lazy=True)
    (err,) = report.errors
    assert "the log file changed while it was being read" in err.message and "--no-lazy" in err.message
    assert not any("live prefix differs" in f.message for f in report.findings)


def test_the_future_check_reports_a_changed_log_not_a_skipped_point(tmp_path):
    from tabagent.adapters import OpenAIMessagesAdapter
    from tabagent.adapters.logfiles import LogChangedError
    from tabagent.tabschema.verify import check_adapter

    logs = write_openai_logs(tmp_path / "logs", n=6, fail_every=0)
    adapter = OpenAIMessagesAdapter()
    trajs = adapter.load(logs, lazy=True)
    dps = [dp for t in trajs for dp in adapter.decision_points(t, "ToolShortlist")]
    for f in logs.iterdir():
        f.write_text(f.read_text().replace("book", "BOOK"))
    with pytest.raises(LogChangedError, match="the log file changed while it was being read"):
        check_adapter(adapter, trajs, dps, "ToolShortlist", label_of=adapter.labels)


def test_the_parity_sample_is_exactly_the_documented_size(tmp_path):
    logs = write_openai_logs(tmp_path / "logs", n=12, fail_every=0)
    report = check_logs(logs, decision="NextToolCall", sample=5)
    (d,) = report.decisions
    assert d.decision_points == 18 and d.parity_checked == 5
    assert check_logs(logs, decision="NextToolCall", sample=500).decisions[0].parity_checked == 18


def test_a_leak_check_that_checked_nothing_says_so(tmp_path, monkeypatch):
    import tabagent.tabschema.verify as verify

    monkeypatch.setattr(
        verify, "check_adapter", lambda *a, **k: {"checked": 0, "reasons": {"its run is not loaded": 4}}
    )
    report = check_logs(write_openai_logs(tmp_path / "logs", n=6), decision="ToolShortlist")
    (d,) = report.decisions
    assert d.leak_checked == 0 and "its run is not loaded (4)" in d.leak_note
    assert "checked no decision point" in " ".join(report.summary().split())  # wrapped text
    assert any(f.level == "warning" and "checked no decision point" in f.message for f in report.findings)


def test_near_misses_prefer_the_exact_token():
    from tabagent.scanner import _near_miss

    offered = {"phone_get_id", "phone_id", "venmo_id", "spotify_login"}
    assert _near_miss("id", offered) == "phone_id"
    assert _near_miss("login", offered) == "spotify_login"
    assert _near_miss("Phone-Get-ID", offered) == "phone_get_id"
    assert _near_miss("venmo_id", {"id", "venmo_ids"}) == "id"
    assert _near_miss("search", offered) is None
