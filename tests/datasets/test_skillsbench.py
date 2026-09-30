"""SkillsBench builders on a fabricated mini release written in the upstream layout (no real data)."""

import dataclasses
import json
import os
from importlib import resources
from pathlib import Path

import pytest

from tabagent.datasets import skillsbench as sb
from tabagent.datasets._core import Source
from tabagent.evaluation.cv import stable_uid
from tabagent.table import DecisionTable
from tabagent.tabschema import FeatureCard

TASK_REVISION = "27738384b1df694ea2ae466e416f476e94d8fab9"
FOLDS = resources.files("tabagent.datasets") / "folds"
LONG = "# Alpha skill\n\n" + "use   alpha\ttools " * 100  # > 800 characters, irregular whitespace


class MiniSource(Source):
    """The real source ids, with a manifest listing whatever the fabricated mirror holds."""

    @property
    def manifest(self):
        base = Path(os.environ["TABAGENT_SOURCE_MIRROR"]) / self.id / self.release_id / "files"
        return {p.relative_to(base).as_posix(): {} for p in base.rglob("*") if p.is_file()}


def _write(base: Path, rel: str, text: str) -> None:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _run(task, model, mode, passed, reward, verdict="pass"):
    return {
        "task": task,
        "model": model,
        "mode": mode,
        "passed": passed,
        "reward": reward,
        "audit_verdict": verdict,
    }


@pytest.fixture
def source(tmp_path, monkeypatch):
    base = tmp_path / sb.SOURCE.id / sb.SOURCE.release_id / "files"
    skills = {
        "task-a": {"alpha-skill": LONG, "beta": "Beta from task a."},
        "task-b": {"beta": "Beta from task b.", "gamma": "", "delta": "Delta."},
    }
    for task, bundle in skills.items():
        _write(base, f"tasks/tasks/{task}/task.md", f"  Do {task} now.\n\n")
        for skill, md in bundle.items():
            _write(base, f"tasks/tasks/{task}/environment/skills/{skill}/SKILL.md", md)
    _write(base, "tasks/tasks/task-a/environment/skills/beta/README.md", "not the skill file")
    _write(base, "tasks/tasks/task-b/environment/skills/licenses/tool.LICENSE", "MIT")  # not a skill
    _write(base, "tasks/tasks/task-b/environment/skills/nested/docs/SKILL.md", "nested")  # not top level
    g, m = "gemini-3.5-flash", "minimax-m3"
    runs = [
        _run("task-a", g, "with", True, 1.0),
        _run("task-a", g, "with", True, 0.5),
        _run("task-a", g, "with", False, 0.0, "model_capability_failure"),
        _run("task-a", g, "without", False, None, "model_capability_failure"),
        _run("task-a", g, "without", True, 1.0, "infra_or_environment_failure"),  # excluded
        _run("task-b", g, "with", True, 1.0),  # no kept without arm -> no decision
        _run("task-b", g, "without", True, 1.0, "unclear_needs_human_followup"),
        _run("task-a", m, "with", False, 0.0, "model_capability_failure"),
        _run("task-a", m, "without", True, 1.0),
        _run("task-b", m, "without", True, 1.0),
        _run("task-b", m, "with", True, 0.5),
    ]
    _write(base, sb.RUNS, "\n".join(json.dumps(r) for r in runs) + "\n\n")
    monkeypatch.setenv("TABAGENT_SOURCE_MIRROR", str(tmp_path))
    monkeypatch.setenv("TABAGENT_CACHE", str(tmp_path / "cache"))
    return MiniSource(**dataclasses.asdict(sb.SOURCE))


def test_bundle_is_the_top_level_skill_dirs(source):
    assert sb.bundle(source, "task-a") == ["alpha-skill", "beta"]
    assert sb.bundle(source, "task-b") == ["beta", "delta", "gamma"]


def test_skill_select_catalog_labels_and_text(source):
    df = sb.build_skill_select(source)
    assert len(df) == 2 * 4 and df.groupby("task_name").size().tolist() == [4, 4]
    assert df["task_uid"].iloc[0] == stable_uid("task", "skillsbench_official_selected", "task-a", "")
    assert set(df["task_text"]) == {"Do task-a now.", "Do task-b now."}
    rel = df[df.relevant == 1].groupby("task_name")["skill_name"].apply(sorted).to_dict()
    assert rel == {"task-a": ["alpha skill", "beta"], "task-b": ["beta", "delta", "gamma"]}
    desc = df.drop_duplicates("skill_name").set_index("skill_name")["skill_desc"]
    assert (
        desc["beta"] == "Beta from task a."
    )  # the first task by name defines the text; SKILL.md, not README
    assert desc["gamma"] == "gamma"  # empty SKILL.md falls back to the name
    assert desc["alpha skill"] == " ".join(LONG[:800].split())
    assert "  " not in desc["alpha skill"] and len(desc["alpha skill"]) < 800
    DecisionTable(df, FeatureCard.load("skillsbench_skill_select").spec)


def test_gating_arms_labels_and_rewards(source):
    df = sb.build_skill_gating(source)
    assert df["task_name"].unique().tolist() == ["task-a"]  # task-b has no kept without arm
    assert df["mode"].tolist() == ["without", "with"]
    without, with_ = df.iloc[0], df.iloc[1]
    assert with_.passed == 1 and with_.reward == pytest.approx(0.5)  # 2 of 3 kept trials passed
    assert without.passed == 0 and without.reward == 0.0  # the infra failure is not a trial
    assert with_.skill_gain == without.skill_gain == pytest.approx(0.5)
    assert without.supplied_skills_text == "" and without.n_supplied_skills == 0
    assert with_.n_supplied_skills == 2
    assert with_.supplied_skills_text.startswith("alpha skill beta # Alpha skill use alpha tools")
    assert with_.supplied_skills_text.endswith("Beta from task a.")
    table = DecisionTable(df, FeatureCard.load("skillsbench_skill_gating").spec)
    assert table.spec.kind == "policy"


def test_enriched_gating_is_per_backbone(source):
    df = sb.build_skill_gating(source, models=None)
    got = df.groupby(["task_name", "model"])["mode"].apply(list).to_dict()
    assert got == {
        ("task-a", "gemini-3.5-flash"): ["without", "with"],
        ("task-a", "minimax-m3"): ["without", "with"],
        ("task-b", "minimax-m3"): ["without", "with"],
    }
    table = DecisionTable(df, FeatureCard.load("skillsbench_skill_gating_enriched").spec)
    assert table.decision_ids.nunique() == 3


def test_cards_fit_on_the_built_tables(source):
    for name, df in (
        ("skillsbench_skill_select", sb.build_skill_select(source)),
        ("skillsbench_skill_select_noprior", sb.build_skill_select(source)),
        ("skillsbench_skill_gating", sb.build_skill_gating(source)),
        ("skillsbench_skill_gating_enriched", sb.build_skill_gating(source, models=None)),
    ):
        d = FeatureCard.load(name).to_dict()
        for f in d["features"]:
            if "encoder" in f.get("args", {}):
                f["args"]["encoder"] = "hashing"  # offline stand-in for e5
        card = FeatureCard.from_dict(d)
        table = DecisionTable(df, card.spec)
        X = card.fit(table).transform(table.drop_labels())
        assert list(X.columns) == card.feature_names and len(X) == len(df)
        if "mode" in X:
            assert (X.loc[X["mode"] == "without", "skill_task_lex_overlap"] == 0).all()


def test_bundled_folds_cards_and_manifest():
    sizes = {}
    for info in sb.DATASETS:
        folds = json.loads(
            (resources.files("tabagent.datasets") / "folds" / f"{info.folds}.json").read_text()
        )
        assert folds["group_col"] == FeatureCard.load(info.card).spec.group == "task_uid"
        assert set(folds["by_group"].values()) == set(range(5))
        sizes[info.name] = len(folds["by_group"])
    assert sizes == {
        "skillsbench/skill-select": 87,
        "skillsbench/skill-gating": 84,
        "skillsbench/skill-gating-enriched": 87,
    }
    files = sb.SOURCE.manifest
    assert sb.RUNS in files
    assert sum(p.endswith("/task.md") for p in files) == 87
    assert sum(p.endswith("/SKILL.md") for p in files) == 232
    for p, f in files.items():
        assert len(f["sha256"]) == 64 and f["size_bytes"] > 0
        rev = (
            sb.SOURCE.revision if p.startswith("leaderboard/") else "27738384b1df694ea2ae466e416f476e94d8fab9"
        )
        assert f"/{rev}/" in f["url"]


def test_missing_files_ask_for_download(tmp_path, monkeypatch):
    monkeypatch.delenv("TABAGENT_SOURCE_MIRROR", raising=False)
    monkeypatch.setenv("TABAGENT_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="download=True"):
        sb.build_skill_select(sb.SOURCE)
