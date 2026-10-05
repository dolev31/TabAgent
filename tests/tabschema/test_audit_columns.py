"""The output audit judges at most AUDIT_MAX_COLUMNS columns per point, spread over the planned features."""

from __future__ import annotations

from tabagent.tabschema.audit import AUDIT_MAX_COLUMNS, audit_columns, audit_items


def plans(n_words=300, n_tools=30):
    cols = {f"words_w{i}": {"feature_name": "words"} for i in range(n_words)}
    cols |= {f"tools_t{i}": {"feature_name": "tools"} for i in range(n_tools)}
    cols["intent_type"] = {"feature_name": "intent_type"}
    return cols


def test_a_wide_extractor_is_audited_a_share_at_a_time():
    columns = plans()
    values = {c: 0 for c in columns} | {"words_w250": 3, "tools_t7": 1, "intent_type": "write"}
    picked = audit_columns(columns, values)
    assert len(picked) == AUDIT_MAX_COLUMNS == len(audit_items(columns, values))
    assert {"intent_type", "words_w250", "tools_t7"} <= set(picked)  # every feature, set values first
    assert (
        sum(c.startswith("words") for c in picked) <= AUDIT_MAX_COLUMNS // 2 + 1
    )  # round-robin, not 300 words
    assert picked == audit_columns(columns, values)  # deterministic: the same prompt on a rerun


def test_a_narrow_extractor_is_audited_whole():
    columns = {"a": {"feature_name": "a"}, "b": {"feature_name": "b"}}
    assert sorted(audit_columns(columns, {"a": 1, "b": None})) == ["a", "b"]
