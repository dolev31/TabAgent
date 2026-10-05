"""Compile caps the columns a planned feature expands into (a dict per row), keeping those that vary most."""

from __future__ import annotations

import pandas as pd
import pytest

from tabagent.tabschema.compile import CompiledExtractor, _cap_columns
from tabagent.tabschema.orchestrator import TabSchemaConfig


def compiled(n_words: int, n_tools: int = 5) -> tuple[CompiledExtractor, pd.DataFrame]:
    rows = []
    for i in range(40):
        row = {f"word_counts_w{j}": int(j == i % n_words) for j in range(n_words)}  # sparse word counts
        row |= {f"tools_used_t{j}": int((i + j) % 2 == 0) for j in range(n_tools)}  # a real multi-hot
        row["intent_type"] = ["read", "write"][i % 2]
        rows.append(row)
    frame = pd.DataFrame(rows)
    result = CompiledExtractor("code", {c: "num" for c in frame.columns})
    for c in frame.columns:
        name = "word_counts" if c.startswith("word") else "tools_used" if c.startswith("tools") else c
        result.specs[c] = {"feature_name": name}
    return result, frame


def test_a_bag_of_words_is_capped_and_real_features_kept():
    result, frame = compiled(n_words=300)
    _cap_columns(result, frame, per_feature=64, total=512)
    words = [c for c in result.features if c.startswith("word")]
    assert len(words) == 64 and {f"tools_used_t{j}" for j in range(5)} <= set(result.features)
    assert "intent_type" in result.features
    assert "max_columns_per_feature" in result.dropped["word_counts_w299"] or len(result.dropped) == 300 - 64
    assert any("word_counts expanded into 300 columns" in n for n in result.notes)


def test_the_total_cap_keeps_the_columns_that_vary_most():
    result, frame = compiled(n_words=40)
    _cap_columns(result, frame, per_feature=64, total=8)
    assert len(result.features) == 8
    assert {f"tools_used_t{j}" for j in range(5)} <= set(result.features)  # they vary most (half the rows)
    assert "intent_type" in result.features
    assert any("capped at 8" in v for v in result.dropped.values())


def test_under_the_caps_nothing_changes():
    result, frame = compiled(n_words=10)
    before = dict(result.features)
    _cap_columns(result, frame, per_feature=64, total=512)
    assert result.features == before and not result.dropped and not result.notes


@pytest.mark.parametrize("field", ["max_columns_per_feature", "max_extractor_columns"])
def test_caps_are_validated(field):
    with pytest.raises(ValueError, match=field):
        TabSchemaConfig(**{field: 0})


def test_the_run_directory_records_the_compile_step(tmp_path):
    from tabagent.tabschema.orchestrator import compile_record

    result, frame = compiled(n_words=100)
    result.attempt_errors = ["KeyError: 'steps'"]
    _cap_columns(result, frame, per_feature=64, total=512)
    rec = compile_record(result)
    assert rec["attempt_errors"] == ["KeyError: 'steps'"]
    assert rec["kept"]["intent_type"] == {"type": "num", "feature": "intent_type"}
    assert sum("max_columns_per_feature" in why for why in rec["dropped"].values()) == 36


def test_the_validator_sees_bounded_sample_outputs():
    from tabagent.tabschema.agents import SAMPLE_OUTPUT_KEYS, sample_output

    wide = {f"w{i}": i for i in range(5000)} | {"text": "x" * 2000}
    shown = sample_output(wide)
    assert len(shown) == SAMPLE_OUTPUT_KEYS + 1 and "4851 more keys" in shown["..."]
    assert sample_output({"t": "y" * 1000})["t"].endswith("...") and sample_output(None) is None
