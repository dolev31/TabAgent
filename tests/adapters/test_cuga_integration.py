"""The CUGA hot swap (``tabagent.integrations.cuga``): CUGA's live run is what the head was trained on."""

from __future__ import annotations

import json
import re

from fake_logs import cuga_trajectory

from tabagent.adapters import CugaAdapter
from tabagent.integrations.cuga import CugaAppMatcher, CugaShortlister, run_so_far

ROUNDS = [
    ("phone", ["search_contacts", "show_contact"], ["search_contacts"]),
    ("venmo", ["create_transaction", "show_balance"], ["create_transaction"]),
]


class Step:
    """A pydantic-like CUGA ``Step``."""

    def __init__(self, d):
        self.d = d

    def model_dump(self):
        return dict(self.d)


class Tracker:
    def __init__(self, raw, upto):
        self.intent, self.dataset_name = raw["intent"], raw["dataset_name"]
        self.steps = [Step(s) for s in raw["steps"][:upto]]
        self.task_id, self.eval, self.score = raw["task_id"], raw["eval"], raw["score"]  # never read


class StubHead:
    def __init__(self, adapter, deferred=False):
        self.adapter, self.deferred, self.calls = adapter, deferred, []

    def decide(self, run=None, candidates=None, k=None, defer_below=None):
        self.calls.append({"prefix": self.adapter.live_prefix(run), "candidates": candidates, "k": k})
        ids = [c.id for c in candidates]
        return {"candidates": ids[: k or 2], "scores": {c: 0.5 for c in ids}, "deferred": self.deferred}


def _available(step):
    text = next(p["value"] for p in step["prompts"] if "Available APIs" in str(p["value"]))
    return json.loads(re.search(r'`Available APIs`:\s*"""(.*?)"""', text, re.S).group(1))


def test_shortlister_sees_the_training_prefix_and_candidates(tmp_path):
    raw = cuga_trajectory("t1", ROUNDS)
    (tmp_path / "t1.json").write_text(json.dumps(raw))
    ad = CugaAdapter()
    (traj,) = ad.load(tmp_path)
    head = StubHead(ad)
    swap = CugaShortlister(head, k=1)
    for dp in ad.decision_points(traj, "ShortlisterAgent"):
        result = swap.shortlist(run_so_far(Tracker(raw, dp.step_index)), _available(dp.target_step))
        call = head.calls[-1]
        assert call["prefix"] == dp.prefix  # live equals training
        assert [c.id for c in call["candidates"]] == [c.id for c in ad.candidates(dp)]
        assert [c.description for c in call["candidates"]] == [c.description for c in ad.candidates(dp)]
        (item,) = result
        assert set(item) == {"name", "relevance_score", "reasoning"}  # CUGA's APIDetails
        assert item["name"] in {c.id for c in ad.candidates(dp)}


def test_app_matcher_sees_the_training_candidates_and_defers(tmp_path):
    raw = cuga_trajectory("t1", ROUNDS)
    (tmp_path / "t1.json").write_text(json.dumps(raw))
    ad = CugaAdapter()
    (traj,) = ad.load(tmp_path)
    (dp,) = ad.decision_points(traj, "TaskAnalyzerAgent")
    offered = [{"name": c.id, "description": c.description} for c in ad.candidates(dp)]
    head = StubHead(ad)
    names = CugaAppMatcher(head).match(run_so_far(Tracker(raw, 0)), offered)
    assert head.calls[-1]["prefix"] == dp.prefix and names == [c.id for c in ad.candidates(dp)][:2]
    assert [c.id for c in head.calls[-1]["candidates"]] == [c.id for c in ad.candidates(dp)]
    shy = CugaAppMatcher(StubHead(ad, deferred=True))
    assert shy.match(run_so_far(Tracker(raw, 0)), offered) is None and shy.last["deferred"]
    assert CugaShortlister(StubHead(ad, deferred=True)).shortlist({"intent": "x", "steps": []}, {}) is None


def test_run_so_far_takes_plain_dicts_and_missing_fields():
    class Bare:
        steps = [{"name": "TaskAnalyzerAgent", "prompts": [], "data": "[]"}]

    assert run_so_far(Bare()) == {"intent": "", "dataset_name": "", "steps": Bare.steps}
