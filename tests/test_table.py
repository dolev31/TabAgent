import pandas as pd
import pytest

from tabagent.table import DecisionSpec, DecisionTable, LeakageError

SPEC = DecisionSpec(kind="ranking", group="task", candidate="tool", label="used", eval=("latency",))


def _df():
    return pd.DataFrame(
        {
            "task": ["t1"] * 3 + ["t2"] * 3,
            "tool": ["a", "b", "c"] * 2,
            "used": [1, 0, 1, 0, 1, 0],
            "latency": [1.0] * 6,
            "intent": ["x"] * 6,
        }
    )


def test_spec_validation():
    with pytest.raises(ValueError):
        DecisionSpec(kind="ranking", group="g", label="y")  # ranking needs a candidate
    with pytest.raises(ValueError):
        DecisionSpec(kind="single", group="g", label="y", candidate="c")  # single has none
    with pytest.raises(ValueError):
        DecisionSpec(kind="policy", group="g", label="y", candidate="c")  # policy needs a reward
    with pytest.raises(ValueError):
        DecisionSpec(kind="other", group="g", label="y", candidate="c")


def test_roles():
    assert SPEC.decision_cols == ("task",)
    assert SPEC.outcome_cols == ("used", "latency")


def test_valid_table_and_views():
    t = DecisionTable(_df(), SPEC)
    assert list(t.groups) == ["t1", "t2"]
    assert len(t.subset(["t2"])) == 3
    blind = t.drop_labels()
    assert not blind.has_labels and "used" not in blind.df and "latency" not in blind.df
    with pytest.raises(LeakageError):
        _ = blind.labels


def test_outcome_columns_cannot_be_features():
    t = DecisionTable(_df(), SPEC)
    t.feature_frame(["intent", "tool"])
    with pytest.raises(LeakageError):
        t.feature_frame(["intent", "latency"])


def test_rejects_decision_spanning_groups():
    df = _df().assign(dec="d")
    with pytest.raises(ValueError, match="nest within groups"):
        DecisionTable(df, DecisionSpec("ranking", "task", "used", "tool", decision=("dec",)))


def test_rejects_duplicates_and_nonbinary_labels():
    df = pd.concat([_df(), _df().iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        DecisionTable(df, SPEC)
    with pytest.raises(ValueError, match="binary"):
        DecisionTable(_df().assign(used=[0, 1, 2, 0, 1, 0]), SPEC)


def test_label_free_view_must_not_carry_outcomes():
    with pytest.raises(LeakageError):
        DecisionTable(_df(), SPEC, has_labels=False)
