"""Per-candidate features on CUGA's two decisions (fabricated CUGA-shaped runs, scripted LLM): the API
shortlister (ranking) and the app analyzer (multi-label), and the hot swap's live per-candidate values, which
must equal the training rows'."""

from __future__ import annotations

import dataclasses
import json
import re

import numpy as np
import pytest
from fake_logs import cuga_trajectory

import tabagent
from tabagent import auto
from tabagent.adapters import CugaAdapter
from tabagent.integrations.cuga import CugaAppMatcher, CugaShortlister, run_so_far
from tabagent.table import DecisionTable
from tabagent.tabschema import TabSchemaConfig
from tabagent.tabschema.compile import ExtractorRunner
from tabagent.tabschema.llm import ScriptedLLM
from tabagent.tabschema.orchestrator import CANDIDATE, DECISION, extract_rows, rows_to_table

ALL = {
    "phone": ["phone_search_contacts", "phone_show_text_messages", "phone_delete_alarm"],
    "venmo": ["venmo_send_money", "venmo_show_balance"],
}
#: (intent, rounds): what each fabricated task asks, and the APIs each round calls.
TASKS = [
    ("send money to my roommate on venmo", [("venmo", ALL["venmo"], ["send_money"])]),
    ("show my venmo balance", [("venmo", ALL["venmo"], ["show_balance"])]),
    ("search my contacts for mom and show her text messages", [("phone", ALL["phone"], ["search_contacts"])]),
    ("delete the alarm on my phone", [("phone", ALL["phone"], ["delete_alarm"])]),
    (
        "search my phone contacts for my sister and send money on venmo",
        [("phone", ALL["phone"], ["search_contacts"]), ("venmo", ALL["venmo"], ["send_money"])],
    ),
]
#: Per candidate, for both decisions: the goal's words in the candidate's id and in its description, and the
#: required parameters its catalog entry lists (an API's; an app entry has none).
CODE = """
def _goal(trajectory_data):
    return set(trajectory_data['intent'].lower().split())


def extract_features_for_classification(trajectory_data):
    return {"user_goal": trajectory_data['intent']}


def extract_candidate_features(trajectory_data, candidate):
    words = _goal(trajectory_data)
    spec = candidate['spec'] if candidate['spec'] is not None else {}
    params = spec['parameters'] if 'parameters' in spec else []
    return {
        "goal_words_in_id": len(words & set(candidate['id'].lower().split('_'))),
        "goal_words_in_description": len(words & set(candidate['description'].lower().strip('.').split())),
        "required_params": len([p for p in params if 'required' in p and p['required']]),
    }


features = extract_features_for_classification(trajectory)
"""
SPECS = [
    {"feature_name": "user_goal", "feature_type": "text", "description": "the user's goal"},
    {"feature_name": "goal_words_in_id", "feature_type": "numeric", "description": "goal words in the id"},
    {
        "feature_name": "goal_words_in_description",
        "feature_type": "numeric",
        "description": "goal words in the candidate's description",
    },
    {
        "feature_name": "required_params",
        "feature_type": "numeric",
        "description": "the required parameters of the candidate's catalog entry",
    },
]
_NAME = re.compile(r'"feature_name": "([^"]+)"')


def responder(role, prompt, n):
    names = [x for x in dict.fromkeys(_NAME.findall(prompt)) if x != "descriptive name"]
    if role == "analyzer":
        return json.dumps({"potential_features": SPECS})
    if role == "extractor":
        assert (
            "extract_candidate_features(trajectory_data, candidate)" in prompt
        )  # the decision has candidates
        return f"<code>{CODE}</code>"
    if role == "validator":
        return json.dumps({"features": [{"feature_name": x, "verdict": "valid"} for x in names]})
    if role == "aggregator":
        return json.dumps({"filtered_features": SPECS, "rejected_features": []})
    if role.startswith("judge_"):
        return json.dumps({"features": [{"feature_name": x, "score": 5, "confidence": 4} for x in names]})
    if role == "meta_judge":
        return json.dumps([{"feature_name": x, "final_decision": "accept"} for x in names])
    if role == "auditor":
        return json.dumps({"features": [{"feature_name": x, "verdict": "correct"} for x in names]})
    raise AssertionError(role)


@pytest.fixture(scope="module")
def logs(tmp_path_factory):
    d = tmp_path_factory.mktemp("cuga_logs")
    for i in range(15):
        intent, rounds = TASKS[i % len(TASKS)]
        raw = cuga_trajectory(f"t{i}", rounds, intent=intent)
        (d / f"t{i}.json").write_text(json.dumps(raw))
    return d


def onboard(logs, tmp, decision, kind):
    cfg = TabSchemaConfig(n_analyze=3, workers=2, pair_embeddings=False, usefulness=False)
    select = auto.select

    def relative_on(table, card, **kw):
        # the relative features on (the recipe search keeps them only when strictly better, which a toy
        # decides by chance), so the hot swap is checked to compute them as training did
        recipe, rows = select(table, card, **kw)
        return dataclasses.replace(recipe, relative=True), rows

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(auto, "select", relative_on)
        return tabagent.onboard(
            logs,
            decision,
            adapter=CugaAdapter(),
            kind=kind,
            llm=ScriptedLLM(responder, model="scripted"),
            config=cfg,
            run_dir=tmp / f"run_{decision}",
            n_splits=3,
            out=tmp / decision,
            iterations=30,
        )


def training_rows(head, logs, decision):
    adapter = CugaAdapter()
    trajs = [t for t in adapter.load(logs) if t.success]
    dps = [dp for t in trajs for dp in adapter.decision_points(t, decision)]
    feats, per = extract_rows(ExtractorRunner(head.extractor_code), head.card, dps, adapter)
    return adapter, trajs, dps, rows_to_table(head.card, dps, feats, adapter, per).df


class Step:
    def __init__(self, d):
        self.d = d

    def model_dump(self):
        return dict(self.d)


class Tracker:
    """CUGA's ActivityTracker during a run, ``upto`` steps in."""

    def __init__(self, raw, upto):
        self.intent, self.dataset_name = raw["intent"], raw["dataset_name"]
        self.steps = [Step(s) for s in raw["steps"][:upto]]


def traced(head):
    """``head`` with every decision's trace (the feature frame it scored) kept in ``head.traces``."""
    decide = head.decide
    head.traces = []

    def with_trace(*args, **kw):
        trace = {}
        out = decide(*args, **kw, trace=trace)
        head.traces.append(trace)
        return out

    head.decide = with_trace
    return head


def _available(step):
    text = next(p["value"] for p in step["prompts"] if "Available APIs" in str(p["value"]))
    return json.loads(re.search(r'`Available APIs`:\s*"""(.*?)"""', text, re.S).group(1))


@pytest.fixture(scope="module")
def shortlister(logs, tmp_path_factory):
    return onboard(logs, tmp_path_factory.mktemp("shortlister"), "ShortlisterAgent", "ranking")


def test_the_shortlister_hot_swap_scores_the_training_rows_per_candidate_values(logs, shortlister):
    head = shortlister
    per = [f.name for f in head.card.features if f.source == "extractor_candidate"]
    assert per == ["goal_words_in_id", "goal_words_in_description", "required_params"]
    adapter, trajs, dps, table = training_rows(head, logs, "ShortlisterAgent")
    trained = head.agent.features(DecisionTable(table, head.card.spec)).set_index(table.index)
    relative = [c for c in trained.columns if c.endswith(("_rank_in_decision", "_gap_to_best"))]
    assert {f"{c}_rank_in_decision" for c in per} <= set(relative)
    swap = CugaShortlister(traced(head), k=2)
    raws = {t.id: t.raw for t in trajs}
    for dp in dps:
        swap.shortlist(run_so_far(Tracker(raws[dp.trajectory_id], dp.step_index)), _available(dp.target_step))
        frame = head.traces[-1]["frame"]
        rows = table[table[DECISION] == dp.id]
        assert list(rows[CANDIDATE]) == [c.id for c in adapter.candidates(dp)]
        for col in per:  # live equals training, candidate by candidate
            assert list(frame[col]) == list(rows[col]), (dp.id, col)
        # and so do the ranks and gaps within the decision, over the same candidates
        live, train = frame[relative].to_numpy(float), trained.loc[rows.index, relative].to_numpy(float)
        np.testing.assert_allclose(live, train, err_msg=dp.id)
    assert table["goal_words_in_id"].max() >= 2  # "send money ..." names venmo_send_money's words
    assert set(table["required_params"]) == {0, 1, 2}  # read from the catalog entries


def test_a_served_shortlister_scores_as_the_local_one(logs, shortlister):
    """CugaShortlister sends Candidate objects; a RemoteHead sends them as records, catalog entries included,
    so the service scores the same candidate dicts as the local head."""
    from conftest import served

    adapter = CugaAdapter()
    trajs = [t for t in adapter.load(logs) if t.success]
    local = CugaShortlister(shortlister, k=2)
    with served(shortlister) as url:
        remote = CugaShortlister(url, k=2)
        for t in trajs[:3]:
            for dp in adapter.decision_points(t, "ShortlisterAgent"):
                run, apis = run_so_far(Tracker(t.raw, dp.step_index)), _available(dp.target_step)
                assert remote.shortlist(run, apis) == local.shortlist(run, apis)
                assert remote.last["scores"] == pytest.approx(local.last["scores"])


def test_the_app_matcher_is_multilabel_with_per_candidate_features(logs, tmp_path):
    head = onboard(logs, tmp_path, "TaskAnalyzerAgent", "multilabel")
    assert head.card.spec.kind == "multilabel"
    assert {f.name for f in head.card.features if f.source == "extractor_candidate"} == {
        "goal_words_in_id",
        "goal_words_in_description",
    }
    adapter, trajs, dps, table = training_rows(head, logs, "TaskAnalyzerAgent")
    venmo = table[table[CANDIDATE] == "venmo"]
    assert (venmo["goal_words_in_id"] > 0).any() and (venmo["goal_words_in_id"] == 0).any()
    matcher = CugaAppMatcher(traced(head))
    raw = trajs[0].raw
    offered = [{"name": c.id, "description": c.description} for c in adapter.candidates(dps[0])]
    names = matcher.match(run_so_far(Tracker(raw, 0)), offered)
    frame = head.traces[-1]["frame"]
    rows = table[table[DECISION] == dps[0].id]
    for col in ("goal_words_in_id", "goal_words_in_description"):
        assert list(frame[col]) == list(rows[col])
    assert names is None or set(names) <= {c.id for c in adapter.candidates(dps[0])}
