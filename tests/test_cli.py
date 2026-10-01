"""The `tabagent` command line: help, version, user errors, dataset listing and building, onboarding."""

import pandas as pd
import pytest
from adapters.fake_logs import write_openai_logs

import tabagent
from tabagent import datasets
from tabagent.cli import main
from tabagent.table import DecisionSpec, DecisionTable


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("scan", "components", "onboard", "serve", "datasets"):
        assert cmd in out


def test_subcommand_help_shows_defaults(capsys):
    with pytest.raises(SystemExit):
        main(["onboard", "--help"])
    out = capsys.readouterr().out
    assert "--trust-extractor" in out and "--price-in" in out and "(default: 10)" in out


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.strip() == f"tabagent {tabagent.__version__}"


def test_onboard_requires_logs():
    with pytest.raises(SystemExit):
        main(["onboard"])


def test_user_errors_are_one_line_with_exit_code_2(capsys):
    assert main(["datasets", "nope/nothing"]) == 2
    err = capsys.readouterr().err.strip()
    assert err.startswith("tabagent datasets: error:") and "unknown dataset" in err and "\n" not in err
    assert "Traceback" not in err


def test_error_hints_name_cli_flags(monkeypatch, capsys):
    def not_cached(name, **kw):
        raise FileNotFoundError(f"{name}: a.json is not cached. Call with download=True to fetch it.")

    monkeypatch.setattr(datasets, "load", not_cached)
    assert main(["datasets", "skillsbench/skill-select", "--download"]) == 2
    err = capsys.readouterr().err
    assert "Rerun with --download" in err and "download=True" not in err


def test_datasets_listing_is_compact(capsys):
    assert main(["datasets"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["name", "kind", "license", "access"]
    assert any(line.startswith("skillsbench/skill-select") for line in lines)
    assert max(len(line) for line in lines) < 120


def test_datasets_name_describes_the_dataset(capsys):
    assert main(["datasets", "llmrouterbench/tau2"]) == 0
    out = capsys.readouterr().out
    assert "paper:" in out and "license:" in out
    assert "tabagent datasets llmrouterbench/tau2 --download --accept-license" in out


def _tiny_table(name, **kw):
    spec = DecisionSpec("ranking", "task", "used", "tool")
    return DecisionTable(pd.DataFrame({"task": ["a", "a"], "tool": ["x", "y"], "used": [1, 0]}), spec)


def test_datasets_all_skips_opt_in_and_local_data(monkeypatch, capsys):
    built = []

    def fake_load(name, download=False, accept_license=False, **kw):
        built.append((name, download, accept_license))
        return _tiny_table(name)

    monkeypatch.setattr(datasets, "load", fake_load)
    monkeypatch.delenv("APPWORLD_ROOT", raising=False)
    assert main(["datasets", "--all", "--download"]) == 0
    out = capsys.readouterr().out
    assert "skipped llmrouterbench/tau2: needs --accept-license" in out
    assert "skipped appworld/subagent-routing: needs a local copy: set APPWORLD_ROOT" in out
    names = {n for n, _, _ in built}
    assert "skillsbench/skill-select" in names and "llmrouterbench/tau2" not in names
    assert all(download for _, download, _ in built)
    assert "built skillsbench/skill-select: 2 rows, 1 tasks" in out

    built.clear()
    monkeypatch.setenv("APPWORLD_ROOT", "/somewhere")
    assert main(["datasets", "--all", "--download", "--accept-license"]) == 0
    assert {"llmrouterbench/tau2", "appworld/subagent-routing"} <= {n for n, _, _ in built}


def test_datasets_all_reports_failures_and_continues(monkeypatch, capsys):
    def flaky(name, **kw):
        if name.startswith("mind2web"):
            raise OSError("network unreachable")
        return _tiny_table(name)

    monkeypatch.setattr(datasets, "load", flaky)
    assert main(["datasets", "--all", "--download"]) == 1
    out = capsys.readouterr().out
    assert "failed mind2web/operation: network unreachable" in out and "built tau2/airline" in out


def test_datasets_argument_conflicts(capsys):
    assert main(["datasets", "tau2/airline", "--all"]) == 2
    assert main(["datasets", "--download"]) == 2


def test_onboard_dry_run_on_fake_logs(tmp_path, capsys):
    write_openai_logs(tmp_path, n=12)
    assert main(["onboard", str(tmp_path), "--dry-run", "--n-analyze", "3"]) == 0
    out = capsys.readouterr().out
    assert "decision points, 3 analyzed by the LLM" in out and "no price for gpt-5.4" in out
    assert main(["onboard", str(tmp_path), "--dry-run", "--price-in", "1", "--price-out", "8"]) == 0
    assert "est. cost $" in capsys.readouterr().out


def test_onboard_budget_without_price_is_a_user_error(tmp_path, monkeypatch, capsys):
    from tabagent.tabschema import llm as llm_module

    write_openai_logs(tmp_path, n=12)
    monkeypatch.setattr(llm_module, "auto_llm", lambda model=None: llm_module.ScriptedLLM(lambda *a: "{}"))
    assert main(["onboard", str(tmp_path), "--max-cost", "5", "--run-dir", str(tmp_path / "run")]) == 2
    err = capsys.readouterr().err
    assert "no price is known for model 'scripted'" in err and "--price-in and --price-out" in err
    assert main(["onboard", str(tmp_path), "--dry-run", "--price-in", "1"]) == 2
    assert "go together" in capsys.readouterr().err
