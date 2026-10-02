"""CugaAdapter on fabricated CUGA-shaped trajectories (the API shortlister decision)."""

import json

import pytest
from fake_logs import API_DOCS, cuga_api_docs, cuga_trajectory

from tabagent.adapters import SHORTLISTER, CugaAdapter

PHONE = ["phone_search_contacts", "phone_show_text_messages"]
TWO_ROUNDS = [("phone", PHONE, PHONE), ("venmo", ["venmo_send_money"], ["send_money"])]  # bare name: prefixed


def _write(d, name, traj):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(traj))


def _logs(tmp_path, **kw):
    exp = tmp_path / "logs" / "exp_1"
    _write(exp, "metadata.json", {"task_ids": ["t1", "t2"], "experiment_name": "demo"})
    _write(exp, "results.json", {"t1": {"score": 1}})
    _write(exp, "t1.json", cuga_trajectory("t1", TWO_ROUNDS, **kw))
    _write(exp, "t2.json", cuga_trajectory("t2", TWO_ROUNDS[:1], score=0.0, **kw))
    return tmp_path / "logs"


def test_shortlister_prefix_labels_and_candidates(tmp_path):
    adapter = CugaAdapter()
    trajs = {t.id: t for t in adapter.load(_logs(tmp_path))}
    assert set(trajs) == {"exp_1/t1", "exp_1/t2"}  # recursive; run summaries skipped
    assert trajs["exp_1/t1"].success and not trajs["exp_1/t2"].success and trajs["exp_1/t1"].group == "t1"

    first, second = adapter.decision_points(trajs["exp_1/t1"], SHORTLISTER)
    assert [s["name"] for s in first.prefix["steps"]] == [
        "TaskAnalyzerAgent",
        "PlanControllerAgent",
        "APIPlannerAgent",
    ]
    assert set(first.prefix) == {
        "intent",
        "dataset_name",
        "steps",
    }  # no score, eval, actions_count, nor the task id (the CV group)
    names = {s["name"] for s in second.prefix["steps"]}
    assert SHORTLISTER in names and "CodeAgent" in names  # the first round is history for the second
    assert not names & {"EvaluationResult", "FinalAnswerAgent"} and "FINAL" not in json.dumps(second.prefix)
    assert "venmo_send_money" not in json.dumps(second.prefix)

    assert adapter.labels(first) == {"phone_search_contacts", "phone_show_text_messages"}
    assert adapter.labels(second) == {"venmo_send_money"}

    cands = {c.id: c for c in adapter.candidates(first)}
    assert set(cands) == set(API_DOCS["phone"])
    assert cands["phone_search_contacts"].description == "Search your contacts."
    assert cands["phone_search_contacts"].arg_hints == "query, relationship"
    assert {c.id for c in adapter.candidates(second)} == set(API_DOCS["venmo"])


def test_merged_system_and_human_prompt_layout(tmp_path):
    adapter = CugaAdapter()
    trajs = {t.id: t for t in adapter.load(_logs(tmp_path, merged_prompts=True))}
    first, second = adapter.decision_points(trajs["exp_1/t1"], SHORTLISTER)
    assert adapter.labels(first) == {"phone_search_contacts", "phone_show_text_messages"}
    assert {c.id for c in adapter.candidates(second)} == set(API_DOCS["venmo"])


def test_catalog_fallback_when_the_prompt_has_no_api_block(tmp_path):
    logs = _logs(tmp_path, with_block=False)
    docs = tmp_path / "api_docs.json"
    docs.write_text(json.dumps({app: cuga_api_docs(app) for app in API_DOCS}))

    adapter = CugaAdapter(catalog=docs)
    trajs = {t.id: t for t in adapter.load(logs)}
    first, second = adapter.decision_points(trajs["exp_1/t1"], SHORTLISTER)
    assert {c.id for c in adapter.candidates(first)} == set(API_DOCS["phone"])  # app from the planner's input
    assert {c.id for c in adapter.candidates(second)} == set(API_DOCS["venmo"])
    assert adapter.candidates(first)[0].description.endswith(".")

    seen = CugaAdapter()  # no catalog: the APIs seen for the app in the loaded runs
    trajs = {t.id: t for t in seen.load(logs)}
    first, _ = seen.decision_points(trajs["exp_1/t1"], SHORTLISTER)
    assert {c.id for c in seen.candidates(first)} == {"phone_search_contacts", "phone_show_text_messages"}


def test_components_other_nodes_and_live_prefix(tmp_path):
    adapter = CugaAdapter()
    trajs = adapter.load(_logs(tmp_path))
    comps = adapter.components(trajs)
    assert comps[SHORTLISTER] == 1 and comps["CodeAgent"] == 1 and "EvaluationResult" not in comps

    t1 = next(t for t in trajs if t.id == "exp_1/t1")
    (planner, *_) = adapter.decision_points(t1, "TaskAnalyzerAgent") + adapter.decision_points(
        t1, "PlanControllerAgent"
    )
    with pytest.raises(NotImplementedError, match="labels_fn"):
        adapter.labels(planner)

    custom = CugaAdapter(labels_fn=lambda dp, raw: set(json.loads(dp.target_step["data"])["apps"]))
    custom.load(_logs(tmp_path))
    (dp,) = custom.decision_points(t1, "TaskAnalyzerAgent")
    assert custom.labels(dp) == {"phone", "venmo"} and dp.prefix["steps"] == []

    live = adapter.live_prefix(t1.raw)
    assert not {"score", "eval", "actions_count"} & set(live)
    assert "EvaluationResult" not in {s["name"] for s in live["steps"]}
