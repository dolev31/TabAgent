"""Observability of a TabSchema run: ``run.json`` written as the run progresses, the stage-by-stage anatomy of
``tabagent runs show``, ``tabagent runs list``, and the exit status of a resumable stop."""

from __future__ import annotations

import json
import shutil

import pytest
from test_robustness import (
    DEFAULT,
    TARGET,
    FakeLLM,
    adapter,
    discover,
    runs,
    waits,  # noqa: F401 - autouse: backoff is recorded, not slept
)

from tabagent import cli
from tabagent.tabschema.runlog import drop_reason, list_runs, render, summarize_run

PRICES = {"fake": (1.0, 2.0)}


def test_run_json_follows_the_run_as_it_progresses(tmp_path):
    seen = {}
    run = tmp_path / "runs" / "r1"

    def meta(p, n):  # the MetaJudge runs after the judges: read run.json from inside the run
        seen.update(json.loads((run / "run.json").read_text()))
        return DEFAULT["meta_judge"](p, n)

    discover(tmp_path, FakeLLM(meta_judge=meta), prices=PRICES)
    assert seen["status"] == "running" and seen["sessions"][-1]["status"] == "running"
    stages = seen["stages"]
    assert [stages[s]["status"] for s in ("adapter_check", "sample", "analyze", "aggregate", "judge")] == [
        "done"
    ] * 5
    assert stages["meta_judge"]["status"] == "running" and "compile" not in stages
    assert stages["analyze"]["progress"] == {"done": 3, "total": 3, "failed": 0}

    final = json.loads((run / "run.json").read_text())
    assert final["status"] == "completed" and final["sessions"][-1]["status"] == "completed"
    order = [
        "adapter_check",
        "sample",
        "analyze",
        "aggregate",
        "judge",
        "meta_judge",
        "compile",
        "audit",
        "card",
    ]
    assert list(final["stages"]) == order
    assert all(final["stages"][s]["seconds"] >= 0 and "llm" in final["stages"][s] for s in order)
    paid = sum(final["stages"][s]["llm"]["usd"] for s in order)
    assert paid == pytest.approx(final["llm"]["total_usd"])  # every paid call belongs to one stage
    assert final["data"] == {"trajectories": 16, "successful": 16, "decision_points": 16, "tasks": 16}
    assert final["stages"]["analyze"]["llm"]["roles"].keys() >= {"analyzer", "extractor", "validator"}
    assert final["stages"]["compile"]["llm"]["roles"].keys() >= {"extractor", "validator"}
    assert final["sessions"][-1]["paid"]["usd"] == pytest.approx(final["llm"]["total_usd"])
    calls = [json.loads(line) for line in (run / "llm_calls.jsonl").read_text().splitlines()]
    assert {c["phase"] for c in calls} <= set(order) and all(c["session"] and c["ts"] for c in calls)


def test_drops_are_counted_by_reason(tmp_path):
    def meta(p, n):
        items = json.loads(DEFAULT["meta_judge"](p, n))
        decide = {"plan_note": "reject"}
        return json.dumps([{**x, "final_decision": decide.get(x["feature_name"], "accept")} for x in items])

    result = discover(tmp_path, FakeLLM(meta_judge=meta), accept=("accept",))
    drops = json.loads((tmp_path / "runs" / "r1" / "run.json").read_text())["drops_by_reason"]
    assert drops["MetaJudge: reject"] == 1 and "plan_note" not in result.extractor.features
    assert (
        drop_reason("constant across onboarding decision points")
        == "constant across the onboarding decision points"
    )
    assert drop_reason("null rate 0.80 > 0.5") == "too often empty (max_null_rate)"
    assert drop_reason("OutputAuditor: incorrect on 4 of 6") == "OutputAuditor: judged incorrect"


def test_runs_show_prints_the_anatomy_of_a_run(tmp_path, capsys):
    discover(tmp_path, FakeLLM(), prices=PRICES)
    run = tmp_path / "runs" / "r1"
    assert cli.main(["runs", "show", str(run)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"run r1  target {TARGET}  status completed")
    for stage in ("analyze", "aggregate", "judge", "meta_judge", "compile", "audit", "card"):
        assert f"\n{stage}" in out
    assert "LLM: $" in out and "files: " in out and "run.json" in out
    assert cli.main(["runs", "show", str(run), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "completed" and summary["stages"]["analyze"]["produced"]["points"] == 3


def test_runs_show_rebuilds_the_anatomy_of_a_run_without_run_json(tmp_path):
    discover(tmp_path, FakeLLM(), prices=PRICES)
    old = tmp_path / "old"
    shutil.copytree(tmp_path / "runs" / "r1", old)
    (old / "run.json").unlink()
    calls = [json.loads(line) for line in (old / "llm_calls.jsonl").read_text().splitlines()]
    (old / "llm_calls.jsonl").write_text(  # as an earlier release wrote it: no phase, no session
        "\n".join(
            json.dumps({k: v for k, v in c.items() if k not in ("phase", "session", "ts")}) for c in calls
        )
        + '\n{"truncated line'
    )
    summary = summarize_run(old)
    assert summary["rebuilt"] and summary["status"] == "completed"
    stages = summary["stages"]
    assert stages["analyze"]["produced"]["points"] == 3 and stages["aggregate"]["produced"]["features"] == 3
    assert stages["compile"]["llm_log"]["calls"] >= 1 and stages["analyze"]["llm_log"]["calls"] >= 3
    text = render(summary)
    assert "rebuilt from the stage files" in text and "meta_judge" in text


def test_runs_show_names_a_failed_run_and_its_cause(tmp_path, capsys):
    with pytest.raises(Exception, match="MetaJudge accepted none"):
        discover(tmp_path, FakeLLM(meta_judge=lambda p, n: "[]"))
    assert cli.main(["runs", "show", str(tmp_path / "runs" / "r1")]) == 0
    out = capsys.readouterr().out
    assert "status failed" in out and "ended by ExtractionError" in out and "meta_judge" in out


def test_runs_list_and_errors(tmp_path, capsys):
    discover(tmp_path, FakeLLM(), run_id="a", prices=PRICES)
    discover(tmp_path, FakeLLM(), run_id="b", prices=PRICES)
    rows = list_runs(tmp_path / "runs")
    assert {r["run"] for r in rows} == {"a", "b"} and all(r["status"] == "completed" for r in rows)
    assert cli.main(["runs", "list", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert "a " in out and "b " in out and "completed" in out
    assert cli.main(["runs", "show", str(tmp_path)]) == 2
    assert "not a TabSchema run directory" in capsys.readouterr().err
    assert cli.main(["runs", "list", str(tmp_path / "nothing")]) == 0
    assert "no TabSchema run directories" in capsys.readouterr().out


def test_a_resumable_stop_exits_with_status_3(tmp_path, monkeypatch, capsys):
    from tabagent.tabschema import llm as llm_module

    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(llm_module, "auto_llm", lambda model=None: FakeLLM())
    monkeypatch.setattr(cli, "_adapter_for", lambda logs, catalog, on_other_formats="skip": adapter())
    import tabagent

    real = tabagent.onboard
    monkeypatch.setattr(tabagent, "onboard", lambda logs_, *a, **kw: real(runs(), *a, **kw))
    argv = ["onboard", str(logs), "--decision", TARGET, "--run-dir", str(tmp_path / "run")]
    argv += ["--max-cost", "0.000001", "--price-in", "1", "--price-out", "1", "--no-usefulness"]
    assert cli.main(argv) == 3
    err = capsys.readouterr().err
    assert "stopped:" in err and "max_cost_usd" in err and str(tmp_path / "run") in err
    assert json.loads((tmp_path / "run" / "run.json").read_text())["status"] == "stopped"
