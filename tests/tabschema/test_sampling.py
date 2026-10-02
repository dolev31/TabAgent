"""The FeatureAnalyzer's sample of decision points (``tabagent.tabschema.sampling``)."""

from __future__ import annotations

import pytest

from tabagent.tabschema.orchestrator import TabSchema, TabSchemaConfig
from tabagent.tabschema.sampling import coverage, descriptors, sample_decision_points
from tabagent.tabschema.trajectory import DecisionPoint


def dp(task: str, run: int, step: int, source: str = "a.json", label: str = "x") -> DecisionPoint:
    return DecisionPoint(f"{task}-{run}", step, {"label": label}, {}, task, source)


def runs(tasks: int, per_run: int, sources=("a.json",), labels=("x",)) -> list[DecisionPoint]:
    """``tasks`` tasks with one run each, ``per_run`` decision points per run at steps 2, 4, 6, ..."""
    return [
        dp(f"t{t}", 0, 2 * (k + 1), sources[t % len(sources)], labels[(t + k) % len(labels)])
        for t in range(tasks)
        for k in range(per_run)
    ]


def label(d: DecisionPoint) -> set[str]:
    return {d.prefix["label"]}


def positions(sample: list[DecisionPoint], dps: list[DecisionPoint]) -> set[str]:
    tokens = dict(zip([d.id for d in dps], descriptors(dps), strict=True))
    return {t for d in sample for t in tokens[d.id] if t.startswith("position:")}


def test_tasks_strategy_is_the_earlier_sample():
    dps = runs(6, 3)
    sample = sample_decision_points(dps, 8, seed=7, strategy="tasks")
    assert [d.step_index for d in sample[:6]] == [2] * 6  # the first decision of each run first
    assert len({d.group for d in sample[:6]}) == 6


def test_coverage_reaches_every_position_in_the_run():
    dps = runs(6, 3)
    covered = positions(sample_decision_points(dps, 3, seed=7), dps)
    assert covered == {"position:first", "position:middle", "position:last"}
    assert positions(sample_decision_points(dps, 3, seed=7, strategy="tasks"), dps) == {"position:first"}


def test_coverage_samples_each_task_once_before_repeating():
    dps = runs(5, 4)
    sample = sample_decision_points(dps, 5, seed=1)
    assert len({d.group for d in sample}) == 5
    assert len({d.id for d in sample_decision_points(dps, 20, seed=1)}) == 20  # then the rest, no repeats


def test_coverage_reaches_every_source_and_a_rare_label():
    dps = runs(9, 1, sources=("airline.json", "retail.json", "telecom.json"))
    dps.append(dp("rare", 0, 2, "retail.json", label="refund"))
    sample = sample_decision_points(dps, 4, seed=3, label_of=label)
    assert {d.source for d in sample} == {"airline.json", "retail.json", "telecom.json"}
    assert "rare" in {d.group for d in sample}


def test_long_label_lists_do_not_dominate():
    many = [
        DecisionPoint(f"m{i}", 2, {"l": [f"y{i}_{j}" for j in range(20)]}, {}, f"m{i}", "a") for i in range(3)
    ]
    few = [DecisionPoint(f"f{i}", 2 + 2 * i, {"l": ["z"]}, {}, f"f{i}", "b") for i in range(3)]
    sample = sample_decision_points(many + few, 3, seed=0, label_of=lambda d: d.prefix["l"])
    assert {d.source for d in sample} == {"a", "b"}


def test_sample_is_seeded_and_bounded():
    dps = runs(30, 3, sources=("a", "b"), labels=("x", "y", "z"))
    assert sample_decision_points(dps, 10, seed=5, label_of=label) == sample_decision_points(
        dps, 10, seed=5, label_of=label
    )
    assert sample_decision_points(dps, 0, seed=5) == []
    assert len(sample_decision_points(dps[:4], 10, seed=5)) == 4
    assert sample_decision_points([], 3, seed=5) == []


def test_adapter_without_labels_still_samples():
    def no_labels(_):
        raise NotImplementedError

    dps = runs(4, 2)
    assert len(sample_decision_points(dps, 3, seed=0, label_of=no_labels)) == 3


def test_coverage_report():
    dps = runs(6, 3, sources=("a", "b"), labels=("x", "y"))
    sample = sample_decision_points(dps, 3, seed=7, label_of=label)
    covered = coverage(dps, sample, label)
    assert covered["tasks"] == [3, 6] and covered["positions"] == [3, 3]
    assert covered["sources"][1] == 2 and covered["labels"][1] == 2


def test_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="sample_strategy"):
        sample_decision_points(runs(2, 1), 1, seed=0, strategy="random")
    with pytest.raises(ValueError, match="sample_strategy"):
        TabSchema(llm=object(), config=TabSchemaConfig(sample_strategy="random"))
