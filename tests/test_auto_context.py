"""``auto.select`` tries the winning recipe without the context text (``auto.CONTEXT``) on TabSchema cards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabagent import auto
from tabagent.table import DecisionSpec, DecisionTable
from tabagent.tabschema.card import Feature, FeatureCard

SPEC = DecisionSpec(kind="ranking", group="task", candidate="tool", label="used")


def card(*, context: bool = True, state: bool = True) -> FeatureCard:
    features = [
        Feature("tool", "schema", "cat"),
        Feature("candidate_prior", "dependency", "num", "fitted:candidate_prior", {"key": "tool"}),
    ]
    if state:
        features.append(Feature("n_calls", "state", "num", "extractor"))
    if context:
        features += [
            Feature("context_text", "state", "text", "extractor"),
            Feature(
                "context_candidate_lex", "schema", "num", "jaccard", {"a": ["context_text"], "b": ["tool"]}
            ),
        ]
    return FeatureCard(
        "c", SPEC, features, synth_text=["context_text"] if context else [], extractor={"sha256": "x"}
    )


def table(n_tasks: int = 12) -> DecisionTable:
    rows = [
        {"task": f"t{t}", "tool": tool, "used": int(tool == "a"), "n_calls": t, "context_text": f"task {t}"}
        for t in range(n_tasks)
        for tool in ("a", "b", "c")
    ]
    return DecisionTable(pd.DataFrame(rows), SPEC)


class Scores:
    def __init__(self, value: float):
        self.value = value

    def per_decision(self, metric: str) -> pd.Series:
        return pd.Series([self.value])


@pytest.fixture
def inner(monkeypatch):
    """Replace the inner cross-validation: the score depends only on whether the card has the context text."""
    seen: list[set[str]] = []

    def fake(with_context: float, without: float):
        def cross_validate(agent, tbl, **kw):
            names = {f.name for f in agent.card.features}
            seen.append(names)
            return Scores(with_context if "context_text" in names else without)

        monkeypatch.setattr(auto, "cross_validate", cross_validate)
        return seen

    return fake


def test_context_dropped_when_strictly_better(inner):
    seen = inner(with_context=0.80, without=0.85)
    best, rows = auto.select(table(), card())
    assert best.context is False and best.label().endswith("context=off")
    assert rows["recipe"].iloc[-1].endswith("context=off")
    assert not ({"context_text", "context_candidate_lex"} & seen[-1]) and "n_calls" in seen[-1]


@pytest.mark.parametrize("without", [0.80, 0.75])
def test_context_kept_on_tie_or_loss(inner, without):
    inner(with_context=0.80, without=without)
    best, _ = auto.select(table(), card())
    assert best.context is True and "context" not in best.label()


@pytest.mark.parametrize("kw", [{"context": False}, {"state": False}])
def test_no_extra_trial_without_context_or_other_state(inner, kw):
    seen = inner(with_context=0.80, without=0.99)
    c = card(**kw)
    best, rows = auto.select(table(), c)
    has_text = any(f.type == "text" for f in c.features)  # the other text representation is tried once
    assert best.context is True and len(rows) == len(auto.recipes(c)) + has_text == len(seen)


def test_build_agent_drops_context_and_keeps_the_extractor_pin():
    lean = auto.build_agent(card(), auto.Recipe(context=False)).card
    assert [f.name for f in lean.features] == ["tool", "candidate_prior", "n_calls"]
    assert lean.synth_text == [] and lean.extractor == {"sha256": "x"}
    full = auto.build_agent(card(), auto.Recipe()).card
    assert "context_text" in {f.name for f in full.features}


def test_time_budget_skips_the_extra_trial(inner):
    inner(with_context=0.80, without=0.99)
    best, rows = auto.select(table(), card(), time_budget=0.0)
    assert best.context is True
    assert rows["note"].iloc[-1] == "skipped (time budget)" and np.isnan(rows["p@r"].iloc[-1])
