"""Feature drift: the PSI and the training statistics, the rolling-window monitor, and the statistics' place
in the saved artifact."""

import json
import logging
import math

import numpy as np
import pandas as pd
import pytest

from tabagent import TabAgent, TabHead, load
from tabagent.drift import (
    DriftMonitor,
    bin_groups,
    bin_shares,
    chi2_quantile,
    compare,
    count_tests,
    design_effect,
    encode,
    feature_stats,
    psi,
)

from conftest import make_card, make_table


# --------------------------------------------------------------------------- the math
def test_psi_of_identical_distributions_is_zero():
    assert psi([0.25, 0.25, 0.5], [0.25, 0.25, 0.5]) == 0.0


def test_psi_matches_the_formula():
    # (0.9 - 0.5) ln(0.9 / 0.5) + (0.1 - 0.5) ln(0.1 / 0.5)
    expected = 0.4 * math.log(1.8) + 0.4 * math.log(5)
    assert psi([0.5, 0.5], [0.9, 0.1]) == pytest.approx(expected) == pytest.approx(0.878890, abs=1e-6)
    assert psi([0.5, 0.5], [0.9, 0.1]) == pytest.approx(psi([0.9, 0.1], [0.5, 0.5]))  # symmetric
    assert psi([0.1] * 10, [0.1] * 10) == 0.0


def test_psi_floors_empty_bins_and_needs_the_same_bins():
    value = psi([0.5, 0.5, 0.0], [0.5, 0.0, 0.5])
    assert math.isfinite(value) and value > 1  # a bin emptied and a new one filled: a major shift
    # the floor: (0.5 - 1e-4) ln(0.5 / 1e-4), twice
    assert value == pytest.approx(2 * (0.5 - 1e-4) * math.log(0.5 / 1e-4))
    with pytest.raises(ValueError, match="same bins"):
        psi([0.5, 0.5], [1.0])


def test_bins_are_right_closed_with_open_ends():
    edges = [1.0, 2.0]
    assert bin_shares(np.array([0.5, 1.0, 1.5, 2.0, 9.0]), edges).tolist() == [0.4, 0.4, 0.2]
    assert bin_shares(np.array([]), edges).tolist() == [0.0, 0.0, 0.0]


def test_a_shifted_numeric_feature_has_a_large_psi_and_an_unchanged_one_a_small_one():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    stats = feature_stats(train, {"x": "num"})["features"]["x"]
    same = compare(stats, encode(rng.normal(0, 1, 2000), "num"))
    shifted = compare(stats, encode(rng.normal(1.5, 1, 2000), "num"))
    assert same["psi"] < 0.02 and shifted["psi"] > 0.5
    assert same["null_rate_shift"] == 0 and same["unseen_share"] is None


def test_the_sampling_term_is_the_psi_of_two_samples_of_one_distribution():
    rng = np.random.default_rng(1)
    observed, predicted = [], []
    for _ in range(200):
        stat = feature_stats(pd.DataFrame({"x": rng.normal(size=200)}), {"x": "num"})["features"]["x"]
        c = compare(stat, encode(rng.normal(size=300), "num"))
        observed.append(c["psi"])
        predicted.append(c["psi_sampling"])
    # 9 bins: 8 * (1/200 + 1/300) = 0.067
    assert np.mean(predicted) == pytest.approx(8 * (1 / 200 + 1 / 300), rel=0.15)
    assert np.mean(observed) == pytest.approx(np.mean(predicted), rel=0.3)


# --------------------------------------------------------------------------- training statistics
def _frame():
    """Two decisions of three candidates: a context text, a context number with a gap, a candidate
    category and a candidate score."""
    return pd.DataFrame(
        {
            "context": ["book a flight"] * 3 + ["weather in rome today"] * 3,
            "n_turns": [2.0, 2.0, 2.0, None, None, None],
            "tool": ["search", "book", "weather"] * 2,
            "score": [0.1, 0.5, 0.9, 0.2, 0.4, 0.8],
        }
    )


def test_feature_stats_summarize_each_type_at_its_level():
    types = {"context": "text", "n_turns": "num", "tool": "cat", "score": "num"}
    stats = feature_stats(_frame(), types, decision_ids=[0, 0, 0, 1, 1, 1])
    f = stats["features"]
    assert stats["n_decisions"] == 2 and stats["n_rows"] == 6
    assert {c: f[c]["level"] for c in f} == {
        "context": "decision",  # constant within each decision: counted once per decision
        "n_turns": "decision",
        "tool": "candidate",
        "score": "candidate",
    }
    assert f["context"]["n"] == 2 and f["context"]["length"]["mean"] == pytest.approx((13 + 21) / 2)
    assert f["n_turns"]["null_rate"] == 0.5 and f["n_turns"]["quantiles"]["p50"] == 2.0
    third = pytest.approx(1 / 3, abs=1e-6)
    assert f["tool"]["top"] == [["search", third], ["book", third], ["weather", third]]
    assert f["tool"]["complete"] and sorted(f["tool"]["categories"]) == ["book", "search", "weather"]
    assert sum(f["score"]["freq"]) == pytest.approx(1, abs=1e-5)
    json.dumps(stats)  # plain JSON


def test_unseen_categories_and_null_rates_are_compared():
    tools = ["search"] * 12 + ["book"] * 10 + ["weather"] * 8 + ["rare1", "rare2"]
    frame = pd.DataFrame({"tool": tools, "n_turns": [2.0, None] * 16})
    stats = feature_stats(frame, {"tool": "cat", "n_turns": "num"})["features"]
    assert stats["tool"]["unseen_expected"] == pytest.approx(2 / 32)  # Good-Turing: the two singletons
    tool = compare(stats["tool"], encode(np.array(["search", "email", "email", None]), "cat"))
    assert tool["unseen_share"] == pytest.approx(2 / 3) and tool["null_rate"] == 0.25
    assert tool["psi"] > 0.25  # the 'other' bin (rare in training) now holds two thirds
    turns = compare(stats["n_turns"], encode(np.array([None, None, None, None]), "num"))
    assert turns["null_rate_shift"] == pytest.approx(0.5) and turns["psi"] is None


def test_small_bins_are_merged_before_the_psi():
    stats = feature_stats(pd.DataFrame({"c": ["a"] * 30 + ["b"] * 3 + ["c"] * 2}), {"c": "cat"})["features"][
        "c"
    ]
    assert list(bin_groups(stats)) == [0, 1, 1, 1]  # 'b' and 'c' join the 'other' bin
    assert count_tests(stats) == 3  # psi, null rate, unseen share


def test_the_chi_square_quantiles_are_accurate():
    for p, df, exact in ((0.99, 1, 6.635), (0.99, 2, 9.210), (0.99, 9, 21.666), (0.999, 20, 45.315)):
        assert chi2_quantile(p, df) == pytest.approx(exact, rel=0.02)


def test_the_design_effect_measures_clustering():
    rng = np.random.default_rng(0)
    bins = rng.integers(0, 10, size=5000)
    assert design_effect(bins, np.arange(5000), 10) == pytest.approx(1.0, abs=0.2)  # independent values
    shared = np.repeat(rng.integers(0, 10, size=500), 10)  # ten copies of each value: deff = 10
    assert design_effect(shared, np.repeat(np.arange(500), 10), 10) == pytest.approx(10, rel=0.15)


def _replay(stats, live_frames, **kw):
    monitor = DriftMonitor(stats, background=False, check_every=10**9, **kw)
    for frame in live_frames:
        monitor.observe(frame)
    return monitor.report()


def _iid(n_train, seed, features=30, shift=0.0, window=300):
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(features)]
    stats = feature_stats(
        pd.DataFrame(rng.normal(size=(n_train, features)), columns=cols), dict.fromkeys(cols, "num")
    )
    live = rng.normal(size=(window, features))
    live[:, 0] += shift
    return _replay(stats, [pd.DataFrame([row], columns=cols) for row in live], window=window)


def _clustered(n_train, seed, features=15, candidates=10, shift=0.0, window=300):
    """Candidate-level features that share most of their variance within a decision (ICC about 0.9)."""
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(features)]

    def draw(n):
        x = rng.normal(size=(n, 1, features)) + 0.33 * rng.normal(size=(n, candidates, features))
        return pd.DataFrame(x.reshape(n * candidates, features), columns=cols)

    train = draw(n_train)
    stats = feature_stats(train, dict.fromkeys(cols, "num"), np.repeat(np.arange(n_train), candidates))
    live = draw(window)
    live["f0"] += shift
    frames = [live.iloc[d * candidates : (d + 1) * candidates] for d in range(window)]
    return _replay(stats, frames, window=window)


def _long_tail(n_train, seed, window=300):
    """A decision-level category with a long tail (cities, accounts): most values are rare."""
    rng = np.random.default_rng(seed)
    vocab = np.arange(1, 3001)
    p = 1.0 / vocab**1.1
    p /= p.sum()
    train = pd.DataFrame({"city": [f"c{v}" for v in rng.choice(vocab, size=n_train, p=p)]})
    stats = feature_stats(train, {"city": "cat"})
    frames = [pd.DataFrame({"city": [f"c{v}"]}) for v in rng.choice(vocab, size=window, p=p)]
    return _replay(stats, frames, window=window)


def test_windows_without_drift_rarely_report_any_drifting_feature():
    """The no-drift false-alarm rate per evaluation stays near alpha (0.01), for small training sets, many
    features, candidate-level features clustered within decisions, and long-tail categories."""
    alarms = {
        "independent": sum(bool(_iid(60, s)["drifting"]) for s in range(25)),
        "clustered": sum(bool(_clustered(60, s)["drifting"]) for s in range(25)),
        "long tail": sum(bool(_long_tail(60, s)["drifting"]) for s in range(25)),
    }
    assert all(n <= 1 for n in alarms.values()), alarms  # at most 4% of 25 replays each


def test_real_shifts_are_still_detected():
    assert sum("f0" in _iid(200, s, shift=1.0)["drifting"] for s in range(10)) >= 9
    assert sum("f0" in _clustered(60, s, shift=1.0)["drifting"] for s in range(10)) == 10
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"tool": rng.choice(["a", "b", "c", "d"], size=300), "x": rng.normal(size=300)})
    stats = feature_stats(train, {"tool": "cat", "x": "num"})
    shifted = [pd.DataFrame({"tool": ["brand_new" if i % 2 else "a"], "x": [None]}) for i in range(200)]
    rep = _replay(stats, shifted, window=200)
    assert rep["features"]["tool"]["reasons"] == ["psi", "unseen_categories"]
    assert rep["features"]["x"]["reasons"] == ["null_rate"]


def test_a_head_fitted_on_too_few_decisions_reports_no_drift():
    rng = np.random.default_rng(0)
    stats = feature_stats(pd.DataFrame({"x": rng.normal(size=20)}), {"x": "num"})
    rep = _replay(stats, [pd.DataFrame({"x": [None]})] * 150, window=150)
    assert rep["status"] == "insufficient_training" and rep["drifting"] == []
    assert rep["features"]["x"]["null_rate_shift"] == 1.0  # the numbers are still reported


def test_observe_only_queues_and_the_report_counts_every_decision():
    import time

    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(20)]
    stats = feature_stats(pd.DataFrame(rng.normal(size=(200, 20)), columns=cols), dict.fromkeys(cols, "num"))
    monitor = DriftMonitor(stats, window=100, min_decisions=10)
    frames = [pd.DataFrame(rng.normal(size=(30, 20)), columns=cols) for _ in range(300)]
    t0 = time.perf_counter()
    for f in frames:
        monitor.observe(f)
    per_call = (time.perf_counter() - t0) / len(frames)
    assert per_call < 0.002  # no encoding or evaluation on the request path
    rep = monitor.report()
    assert rep["observed"] == 300 and rep["window"] == 100 and rep["dropped"] == 0
    monitor.close()


# --------------------------------------------------------------------------- the monitor
def _monitor(**kw):
    table = make_table(60)
    agent = TabAgent(make_card(), head=TabHead("ranking", iterations=20)).fit(table)
    X = agent.card_.transform(table)
    decisions = [g for _, g in X.groupby(table.decision_ids.to_numpy(), sort=False)]
    kw = {"window": 40, "min_decisions": 20, "check_every": 10, **kw}
    return DriftMonitor(agent.feature_stats_, **kw), decisions


def test_the_monitor_warms_up_then_reports_no_drift_on_training_like_decisions():
    monitor, decisions = _monitor()
    for d in decisions[:10]:
        monitor.observe(d)
    assert monitor.report()["status"] == "warming" and monitor.report()["drifting"] == []
    for d in decisions[10:40]:
        monitor.observe(d)
    rep = monitor.report()
    assert rep["status"] == "ok" and rep["window"] == 40 and rep["observed"] == 40
    assert all(f["psi"] is not None and f["psi"] < 0.2 for f in rep["features"].values())


def test_shifted_decisions_are_reported_and_logged_once(caplog):
    monitor, decisions = _monitor()
    for d in decisions[:30]:
        monitor.observe(d)
    caplog.set_level(logging.INFO, logger="tabagent.drift")
    for d in decisions[:40]:  # the agent changed: its context text is gone and it offers a new tool
        monitor.observe(d.assign(intent=None, tool="brand_new_tool"))
    rep = monitor.report()
    assert rep["status"] == "drift" and {"intent", "tool"} <= set(rep["drifting"])
    assert rep["features"]["intent"]["reasons"] == ["null_rate"]
    assert rep["features"]["intent"]["null_rate_shift"] == pytest.approx(1.0)
    assert "unseen_categories" in rep["features"]["tool"]["reasons"]
    warned = [r for r in caplog.records if r.levelno == logging.WARNING and r.fields["feature"] == "intent"]
    assert len(warned) == 1 and warned[0].fields["event"] == "feature_drift"  # once, when it starts
    for d in decisions[:40]:  # back to normal: the drift ends
        monitor.observe(d)
    assert monitor.report()["status"] == "ok"
    assert any(getattr(r, "fields", {}).get("event") == "feature_drift_ended" for r in caplog.records)


def test_a_monitor_needs_training_statistics():
    with pytest.raises(ValueError, match="no training statistics"):
        DriftMonitor({})


# --------------------------------------------------------------------------- persistence
def test_the_artifact_keeps_the_statistics_under_its_manifest(tmp_path):
    table = make_table(40)
    agent = TabAgent(make_card(), head=TabHead("ranking", iterations=20)).fit(table)
    assert agent.feature_stats_["n_decisions"] == 40
    out = agent.save(tmp_path / "h")
    assert "feature_stats.json" in json.loads((out / "manifest.json").read_text())["sha256"]
    assert load(out).feature_stats_ == json.loads(json.dumps(agent.feature_stats_))

    stats = json.loads((out / "feature_stats.json").read_text())
    stats["features"]["intent"]["null_rate"] = 0.9
    (out / "feature_stats.json").write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="feature_stats.json does not match"):
        load(out)


def test_an_artifact_without_statistics_still_loads(tmp_path):
    agent = TabAgent(make_card(), head=TabHead("ranking", iterations=20)).fit(make_table(20))
    agent.feature_stats_ = None  # as an artifact written before drift monitoring
    out = agent.save(tmp_path / "old")
    assert not (out / "feature_stats.json").exists() and load(out).feature_stats_ is None
