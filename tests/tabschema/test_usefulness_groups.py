"""Drop-feature usefulness: a feature's columns are dropped together, and p-values have no floor."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from tabagent.table import DecisionSpec, DecisionTable
from tabagent.tabschema.card import Feature, FeatureCard
from tabagent.tabschema.usefulness import ALL_LLM, feature_groups, feature_usefulness, paired_pvalue

SPEC = DecisionSpec(kind="ranking", group="task", candidate="tool", label="used", decision=("decision",))


def table(n_tasks: int = 60, seed: int = 0) -> DecisionTable:
    """The chosen tool is announced by a noisy signal that two columns carry (copies: each makes up for the
    other, as the parts of one feature do); ``noise`` carries nothing."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tasks):
        chosen = rng.integers(4)
        for c in range(4):
            hit = float(c == chosen)
            signal = hit + float(rng.normal(scale=0.35))  # informative, not a copy of the label
            rows.append(
                {
                    "task": f"t{t}",
                    "decision": f"t{t}#0",
                    "tool": f"tool{c}",
                    "used": int(hit),
                    "signal_a": signal,
                    "signal_b": signal,
                    "noise": float(rng.normal()),
                }
            )
    return DecisionTable(pd.DataFrame(rows), SPEC)


def card_for(dropped: frozenset[str]) -> FeatureCard:
    names = [n for n in ("signal_a", "signal_b", "noise") if n not in dropped]
    return FeatureCard(
        "c", SPEC, [Feature("tool", "schema", "cat"), *(Feature(n, "state", "num") for n in names)]
    )


def test_columns_of_one_feature_are_dropped_together():
    t = table()
    one_by_one = feature_usefulness(t, card_for, ["signal_a", "signal_b", "noise"], iterations=30)
    grouped = feature_usefulness(
        t, card_for, {"signal": ["signal_a", "signal_b"], "noise": ["noise"]}, iterations=30
    )
    # Each copy makes up for the other: dropped alone, neither seems to matter.
    assert all(abs(one_by_one["features"][c]["delta"]) < 0.05 for c in ("signal_a", "signal_b"))
    sig = grouped["features"]["signal"]
    assert sig["columns"] == ["signal_a", "signal_b"] and sig["delta"] > 0.3 and sig["verdict"] == "useful"
    assert grouped["features"]["noise"]["verdict"] == "neutral"


def test_paired_pvalue_has_no_resampling_floor():
    rng = np.random.default_rng(0)
    strong = 0.05 + 0.01 * rng.normal(size=200)
    assert paired_pvalue(strong) < 1e-12  # a bootstrap p-value with 2000 draws stops at 1/2001
    assert paired_pvalue(np.zeros(50)) == 1.0 and paired_pvalue(np.full(50, 0.1)) == 0.0
    assert paired_pvalue(np.array([0.3])) == 1.0


def test_feature_groups_follow_the_plan():
    compiled = SimpleNamespace(
        llm_columns=["tools_so_far_get_user", "tools_so_far_book", "intent"],
        features={
            "tools_so_far_get_user": "num",
            "tools_so_far_book": "num",
            "intent": "cat",
            "orphan": "num",
        },
        specs={
            "tools_so_far_get_user": {"feature_name": "tools_so_far"},
            "tools_so_far_book": {"feature_name": "tools_so_far"},
            "intent": {"feature_name": "intent"},
        },
    )
    assert feature_groups(compiled) == {
        "tools_so_far": ["tools_so_far_get_user", "tools_so_far_book"],
        "intent": ["intent"],
        "orphan": ["orphan"],
        ALL_LLM: ["tools_so_far_get_user", "tools_so_far_book", "intent"],
    }
    one = SimpleNamespace(
        llm_columns=["a", "b"],
        features={"a": "num", "b": "num"},
        specs={"a": {"feature_name": "f"}, "b": {"feature_name": "f"}},
    )
    assert ALL_LLM not in feature_groups(one)  # one LLM feature: its own row says it all


def test_the_block_of_llm_features_is_its_own_test():
    t = table()
    groups = {
        "signal": ["signal_a", "signal_b"],
        "noise": ["noise"],
        **{f"pad{i}": ["noise"] for i in range(30)},  # many features: Holm would weaken any single test
        ALL_LLM: ["signal_a", "signal_b", "noise"],
    }
    rec = feature_usefulness(t, card_for, groups, iterations=30)
    block, signal = rec["features"][ALL_LLM], rec["features"]["signal"]
    assert block["verdict"] == "useful" and block["p_holm"] <= signal["p_holm"]
