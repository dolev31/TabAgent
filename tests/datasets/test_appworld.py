"""AppWorld sub-agent routing, on a fabricated AppWorld root (no AppWorld data)."""

import json

import pytest

from tabagent import datasets
from tabagent.datasets import appworld as aw
from tabagent.evaluation.cv import stable_uid
from tabagent.table import DecisionTable

TASKS = {
    "test_normal": {"aaa111_1": "Send $5 to Ann on Venmo.", "aaa111_2": "Send $7 to Bob on Venmo."},
    "test_challenge": {"bbb222_1": "Like every song in my Spotify queue.", "ccc333_1": "Archive old notes."},
}
REPORT = "evaluation/report.md"
GPT4O_REACT = "legacy_react_code_agent/openai/gpt-4o-2024-05-13"


def _report(passed: int, failed: int) -> str:
    return (
        "──── Overall Stats ────\n"
        f"Num Passed Tests : {passed}\nNum Failed Tests : {failed}\nNum Total  Tests : {passed + failed}\n"
    )


def _outcome(system: str, task_id: str) -> tuple[int, int]:
    """(passed, failed): gpt-4o systems solve the normal tasks; everything else fails one of three tests."""
    solved = "gpt-4o" in system and task_id.startswith("aaa")
    return (3, 0) if solved else (2, 1)


def _root(tmp_path, version: str = "0.2.0", skip: str | None = None):
    root = tmp_path / "appworld"
    (root / "data" / "datasets").mkdir(parents=True)
    (root / "data" / "version.txt").write_text(version + "\n")
    for split, tasks in TASKS.items():
        (root / "data" / "datasets" / f"{split}.txt").write_text("\n".join(tasks) + "\n")
        for task_id, instruction in tasks.items():
            spec = root / "data" / "tasks" / task_id / "specs.json"
            spec.parent.mkdir(parents=True)
            spec.write_text(json.dumps({"instruction": instruction, "supervisor": {}}))
            for system in aw.SYSTEMS:
                if skip and system == skip and task_id == "ccc333_1":
                    continue
                report = root / "experiments/outputs" / system / split / "tasks" / task_id / REPORT
                report.parent.mkdir(parents=True)
                report.write_text(_report(*_outcome(system, task_id)))
    return root


def test_systems_catalog():
    assert len(aw.SYSTEMS) == 14 == len(set(aw.SYSTEMS))
    assert sum(s.startswith("legacy_function_calling_agent/openai/") for s in aw.SYSTEMS) == 2


def test_build_dense_table(tmp_path):
    df = aw.build_subagent_routing(root=_root(tmp_path))
    assert len(df) == 4 * 14 and df.groupby("task_uid")["system_id"].nunique().eq(14).all()
    row = df[(df.task_id == "aaa111_2") & (df.system_id == GPT4O_REACT)]
    assert row.success.item() == 1 and row.eval_pass_fraction.item() == 1.0
    assert row.official_split.item() == "test_normal" and row.scenario_id.item() == "aaa111"
    uid = stable_uid("task", "appworld_official", "data-0.2.0_outputs-0.1.3", "aaa111_2")
    assert row.task_uid.item() == uid
    fail = df[df.task_id == "ccc333_1"]
    assert fail.success.eq(0).all() and fail.eval_pass_fraction.eq(2 / 3).all()
    DecisionTable(df, datasets.card("appworld/subagent-routing").spec)


def test_setup_errors_are_actionable(tmp_path, monkeypatch):
    monkeypatch.delenv("APPWORLD_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="appworld download data"):
        aw.build_subagent_routing()
    with pytest.raises(FileNotFoundError, match="experiment-outputs"):
        aw.build_subagent_routing(root=tmp_path / "nowhere")
    with pytest.raises(ValueError, match="0.2.0"):
        aw.build_subagent_routing(root=_root(tmp_path / "old", version="0.1.0"))
    root = _root(tmp_path / "partial", skip=aw.SYSTEMS[0])
    with pytest.raises(FileNotFoundError, match="1 baseline evaluation reports are missing"):
        aw.build_subagent_routing(root=root)
    monkeypatch.setenv("APPWORLD_ROOT", str(_root(tmp_path / "env")))
    assert len(aw.build_subagent_routing()) == 56


def test_report_parsing(tmp_path):
    p = tmp_path / "report.md"
    p.write_text(_report(0, 4))
    assert aw.parse_report(p) == {"passed": 0, "failed": 4, "total": 4}
    p.write_text("no stats here")
    with pytest.raises(ValueError, match="cannot parse"):
        aw.parse_report(p)


def test_card_features(tmp_path):
    df = aw.build_subagent_routing(root=_root(tmp_path))
    card = datasets.card("appworld/subagent-routing").clone()
    for f in card.features:
        if f.source == "fitted:knn_outcome":
            f.args["encoder"] = "hashing"
    table = DecisionTable(df, card.spec)
    X = card.fit(table.subset(table.groups[:3])).transform(table.drop_labels())
    first = X.assign(system=df["system_id"]).groupby("system").first()
    s = "legacy_plan_execute_code_agent/meta/llama-3-70b-chat-hf_together"
    assert tuple(first.loc[s, ["architecture", "backbone", "backbone_family"]]) == (
        "plan_execute_code",
        "llama-3-70b",
        "meta",
    )
    assert first["instr_word_count"].iloc[0] == 6.0
    assert X["prior_success_rank"].min() == 1.0 and X["system_success_on_similar"].between(0, 1).all()


def test_registry_entry():
    info = datasets.info("appworld/subagent-routing")
    assert info.source.redistribution == "access_restricted" and not info.source.manifest
    assert not info.requires_license_acceptance
    folds = datasets.folds("appworld/subagent-routing")
    assert len(folds) == 585 and sorted(set(folds.values())) == list(range(5))
