"""The serving benchmark's synthetic head and its parsing of /metrics (the timed runs are not tests)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from experiments.efficiency import serving
from tabagent.api import TabAgentHead


def test_the_synthetic_head_decides_with_its_extractor_in_every_mode(tmp_path):
    head_dir = serving.build_head(tmp_path / "head", n_decisions=40, iterations=20)
    prefixes = serving.live_prefixes(3)
    candidates = serving._load_candidates(SimpleNamespace(head=None, candidates=None), head_dir)
    assert len(candidates) == len(serving.TOOLS) == 30 and all(len(json.dumps(p)) > 500 for p in prefixes)
    answers = []
    for kw in ({"trust_extractor": True}, {"sandbox": "warm:1"}, {"sandbox": "subprocess"}):
        head = TabAgentHead.load(head_dir, **kw)
        answers.append([head.decide(p, candidates, k=5) for p in prefixes])
        head.close()
    assert answers[0] == answers[1] == answers[2]
    assert all(a["error"] is None and len(a["candidates"]) == 5 for a in answers[0])
    assert (head_dir / "model" / "feature_stats.json").exists()


def test_stage_means_are_read_from_the_metrics():
    text = (
        "# HELP tabagent_decision_seconds Decision latency by stage.\n"
        'tabagent_decision_seconds_sum{stage="total"} 0.5\n'
        'tabagent_decision_seconds_count{stage="total"} 100\n'
        'tabagent_decision_seconds_sum{stage="extraction"} 0.02\n'
        'tabagent_decision_seconds_count{stage="extraction"} 100\n'
    )
    assert serving._stage_means(text) == {"total": 5.0, "extraction": 0.2}
