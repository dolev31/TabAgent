"""SkillsBench (``li2026skillsbench``): skill selection and skill gating (paper Sec. 4.1, App. B.5).

Both decisions are rebuilt from the official v1.1 release: the audited run index of the leaderboard
(``selected_runs.jsonl``, 1,138 executions) and the release-aligned task packages (``task.md`` and the
skills each task ships under ``environment/skills/<skill>/SKILL.md``).

``skillsbench/skill-select``
    Set retrieval over the skill catalog (Table 1: P@R, R@9; Fig. 2 candidate overload; Fig. 3 head
    ablation). One row per (task, catalog skill), 87 tasks x 195 skills. The label says whether the skill is
    in the task's curated bundle. A skill's text is the first 800 characters of its SKILL.md (whitespace
    collapsed), taken from the first task, by name, that ships it. Its card is
    ``skillsbench_skill_select_noprior``; ``skillsbench_skill_select`` adds an in-fold candidate prior.
``skillsbench/skill-gating``
    A policy over two arms per task, ``with`` (activate the task's bundle) and ``without`` (Table 1, gated
    reward against always-activate). Audit-kept runs of one backbone (gemini-3.5-flash), tasks with both
    arms (84). label = a majority of the arm's trials passed; reward = the arm's mean verifier reward.
``skillsbench/skill-gating-enriched``
    The same gate for every audited backbone (gemini-3.5-flash, minimax-m3, opus-4.8); one decision per
    (task, backbone) with both arms observed. ``skill_gain`` (with minus without reward of the pair) is an
    outcome column, used only to fit the per-backbone gain prior. Not reported in the paper.

Runs whose audit verdict is an infrastructure failure or unclear are excluded from every label. Skill texts
always come from the skill's SKILL.md, never from the other Markdown files a skill directory may hold (a
README or reference pages).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import pandas as pd

from tabagent.datasets._core import DatasetInfo, Source
from tabagent.evaluation.cv import stable_uid

SOURCE = Source(
    id="skillsbench_official_selected",
    benchmark="SkillsBench",
    url="https://huggingface.co/datasets/benchflow/skillsbench-leaderboard",
    revision="f104580363a9642563593c475620196ecd36687d",
    release_id="v1.1-hf-f104580363a9642563593c475620196ecd36687d",
    license="Apache-2.0",
    redistribution="redistributable",
    citation="li2026skillsbench",
    notes="Task packages: github.com/benchflow-ai/skillsbench at 27738384b1df694ea2ae466e416f476e94d8fab9.",
)

RUNS = "leaderboard/analysis/skillsbench/v1.1/official-selected/selected_runs.jsonl"
TASKS = "tasks/tasks/"
KEPT_VERDICTS = frozenset({"pass", "model_capability_failure"})  # capability outcomes; infra/unclear dropped
MODES = ("without", "with")  # arm order within a decision
DESC_CHARS, BUNDLE_CHARS = 800, 600


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def task_uid(name: str) -> str:
    """The task id: ``stable_uid("task", <source id>, name, "")``, the key of the pinned fold files."""
    return stable_uid("task", SOURCE.id, name, "")


def read_runs(source: Source, download: bool = False) -> list[dict]:
    """The audited run index (``selected_runs.jsonl``), one dict per execution."""
    with open(source.path(RUNS, download), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read(source: Source, rel: str, download: bool) -> str:
    return source.path(rel, download).read_text(encoding="utf-8", errors="replace")


def bundle(source: Source, task: str) -> list[str]:
    """The skills a task ships: its ``environment/skills/<skill>/`` directories holding a SKILL.md."""
    prefix = f"{TASKS}{task}/environment/skills/"
    rest = [p[len(prefix) :].split("/") for p in source.files(prefix)]
    return sorted(r[0] for r in rest if len(r) == 2 and r[1] == "SKILL.md")


def read_tasks(source: Source, names, download: bool = False) -> dict[str, dict]:
    """``{task: {"text": task.md, "skills": {skill: SKILL.md}}}`` in task-name order."""
    out = {}
    names = sorted(set(names))
    wanted = [f"{TASKS}{n}/task.md" for n in names]
    wanted += [f"{TASKS}{n}/environment/skills/{s}/SKILL.md" for n in names for s in bundle(source, n)]
    source.prefetch(wanted, download)
    for name in names:
        skills = bundle(source, name)
        md = {s: _read(source, f"{TASKS}{name}/environment/skills/{s}/SKILL.md", download) for s in skills}
        out[name] = {"text": _read(source, f"{TASKS}{name}/task.md", download).strip(), "skills": md}
    return out


def _skill_name(skill: str) -> str:
    return skill.replace("-", " ")


# --------------------------------------------------------------------------- skill selection
def skill_select_frame(tasks: dict[str, dict]) -> pd.DataFrame:
    """One row per (task, catalog skill) from ``read_tasks`` output."""
    catalog: dict[str, str] = {}
    for t in tasks.values():  # the first task (by name) shipping a skill defines its text
        for skill, md in t["skills"].items():
            catalog.setdefault(skill, _clean(md[:DESC_CHARS]))
    rows = [
        {
            "task_uid": task_uid(name),
            "task_name": name,
            "task_text": t["text"],
            "skill_name": _skill_name(skill),
            "skill_desc": catalog[skill] or _skill_name(skill),
            "relevant": int(skill in t["skills"]),
        }
        for name, t in tasks.items()
        for skill in sorted(catalog)
    ]
    return pd.DataFrame(rows)


def build_skill_select(source: Source, download: bool = False) -> pd.DataFrame:
    """87 tasks x 195 catalog skills; label = the skill is in the task's curated bundle."""
    names = {r["task"] for r in read_runs(source, download)}
    return skill_select_frame(read_tasks(source, names, download))


# --------------------------------------------------------------------------- skill gating
def bundle_text(skills: dict[str, str]) -> str:
    """The bundle's names followed by the first 600 characters of each SKILL.md, whitespace collapsed."""
    parts = [_skill_name(s) for s in skills] + [md[:BUNDLE_CHARS] for md in skills.values()]
    return _clean(" ".join(parts))


def gating_frame(runs: list[dict], tasks: dict[str, dict], models=None) -> pd.DataFrame:
    """One decision per (task, backbone) observed in both arms; label = majority passed, reward = mean."""
    trials: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        kept = r.get("audit_verdict") in KEPT_VERDICTS and r.get("mode") in MODES and r["task"] in tasks
        if kept and (models is None or r.get("model") in models):
            trials[(r["task"], r["model"], r["mode"])].append(r)
    rows = []
    for name, model in sorted({(t, m) for t, m, _ in trials}):
        arms = [trials.get((name, model, mode), []) for mode in MODES]
        if not all(arms):
            continue
        t = tasks[name]
        reward = [sum(float(r.get("reward") or 0.0) for r in arm) / len(arm) for arm in arms]
        for mode, arm, rw in zip(MODES, arms, reward, strict=True):
            on = mode == "with"
            rows.append(
                {
                    "task_uid": task_uid(name),
                    "task_name": name,
                    "model": model,
                    "task_text": t["text"],
                    "supplied_skills_text": bundle_text(t["skills"]) if on else "",
                    "n_supplied_skills": len(t["skills"]) if on else 0,
                    "mode": mode,
                    "passed": int(sum(bool(r.get("passed")) for r in arm) / len(arm) > 0.5),
                    "reward": rw,
                    "skill_gain": reward[1] - reward[0],
                }
            )
    return pd.DataFrame(rows)


def build_skill_gating(source: Source, download: bool = False, models=("gemini-3.5-flash",)) -> pd.DataFrame:
    """With/without-skills arms per task (and backbone); ``models=None`` keeps every audited backbone."""
    runs = read_runs(source, download)
    return gating_frame(runs, read_tasks(source, {r["task"] for r in runs}, download), models)


DATASETS: list[DatasetInfo] = [
    DatasetInfo(
        name="skillsbench/skill-select",
        source=SOURCE,
        build=build_skill_select,
        card="skillsbench_skill_select_noprior",
        folds="skillsbench_skill_select",
        decision="skill selection (set retrieval)",
        paper="Table 1 (skill selection); Fig. 2a; Fig. 3; App. M, Table 20",
        description="87 SkillsBench tasks x 195 catalog skills; label = skill in the task's curated bundle.",
    ),
    DatasetInfo(
        name="skillsbench/skill-gating",
        source=SOURCE,
        build=build_skill_gating,
        card="skillsbench_skill_gating",
        folds="skillsbench_skill_gating",
        decision="skill gating (policy)",
        paper="Table 1 (skill gating); App. M, Table 20",
        description="84 tasks x {with, without} skills (gemini-3.5-flash); reward = mean verifier reward.",
    ),
    DatasetInfo(
        name="skillsbench/skill-gating-enriched",
        source=SOURCE,
        build=build_skill_gating,
        card="skillsbench_skill_gating_enriched",
        folds="skillsbench_skill_gating_enriched",
        decision="skill gating (policy)",
        paper="not reported in the paper (skill gating over three backbones)",
        description="Gate per (task, backbone) for gemini-3.5-flash, minimax-m3 and opus-4.8.",
        args={"models": None},
    ),
]
