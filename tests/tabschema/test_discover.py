"""End-to-end TabSchema discovery on fabricated toy-agent trajectories with a scripted LLM (no network)."""

import json
import re

import numpy as np
import pytest

from tabagent import TabAgent, TabHead, cross_validate
from tabagent.tabschema.agents import FeatureSpec, generate_retry_prompt
from tabagent.tabschema.judges import judge_table, normalize_decision
from tabagent.tabschema.orchestrator import TabSchema, TabSchemaConfig
from tabagent.tabschema.runstore import RunStore
from tabagent.tabschema.trajectory import Candidate, GenericJsonAdapter

TOOLS = {
    "search_flights": "search flights between airports",
    "book_flight": "book a seat on a flight",
    "get_weather": "weather forecast for a city",
    "send_email": "send an email message",
    "read_inbox": "read the latest emails",
    "play_song": "play music",
}
TASKS = [
    ("find a flight to {c} and book it", ["search_flights", "book_flight"]),
    ("what's the weather in {c}", ["get_weather"]),
    ("reply to the email about {c}", ["read_inbox", "send_email"]),
    ("play a song from {c}", ["play_song"]),
]


def write_logs(tmp_path, n=48):
    rng = np.random.default_rng(0)
    for i in range(n):
        intent, used = TASKS[i % len(TASKS)]
        intent = intent.format(c=["rome", "oslo", "lima"][rng.integers(3)])
        traj = {
            "task_id": f"task{i}",
            "intent": intent,
            "score": 1.0 if i % 12 else 0.0,  # a few failed runs are skipped
            "steps": [
                {"name": "TaskAnalyzerAgent", "data": f"apps needed for: {intent}"},
                {
                    "name": "PlanControllerAgent",
                    "data": "plan: " + " then ".join(t.split("_")[0] for t in used),
                },
                {"name": "ShortlisterAgent", "data": ", ".join(used)},
                {"name": "CodeAgent", "calls": used},
            ],
        }
        (tmp_path / f"run{i}.json").write_text(json.dumps(traj))


def adapter():
    return GenericJsonAdapter(
        candidates_fn=lambda dp: [Candidate(t, t.replace("_", " "), d) for t, d in TOOLS.items()],
        labels_fn=lambda dp, raw: {c for s in raw["steps"] if s["name"] == "CodeAgent" for c in s["calls"]},
    )


GOOD_CODE = """
def extract_features_for_classification(trajectory_data):
    steps = trajectory_data['steps']
    plan = [s['data'] for s in steps if s['name'] == 'PlanControllerAgent']
    return {"user_intent": trajectory_data['intent'], "plan_note": plan[-1] if plan else "",
            "n_prior_steps": len(steps)}
features = extract_features_for_classification(trajectory)
"""
HALLUCINATING_CODE = GOOD_CODE.replace(
    '"n_prior_steps": len(steps)', "\"user_mood\": trajectory_data['mood']"
)

SPECS = [
    {
        "feature_name": "user_intent",
        "feature_type": "text",
        "description": "the user's request",
        "extraction_source": "trajectory['intent']",
        "computation": "copy",
        "classification_relevance": "high",
        "rationale": "goal",
    },
    {
        "feature_name": "plan_note",
        "feature_type": "text",
        "description": "latest plan note",
        "extraction_source": "PlanControllerAgent data",
        "computation": "last",
        "classification_relevance": "high",
        "rationale": "names the next tools",
    },
    {
        "feature_name": "n_prior_steps",
        "feature_type": "numeric",
        "description": "steps so far",
        "extraction_source": "len(steps)",
        "computation": "count",
        "classification_relevance": "low",
        "rationale": "phase",
    },
    {
        "feature_name": "user_mood",
        "feature_type": "categorical",
        "description": "mood",
        "extraction_source": "trajectory['mood']",
        "computation": "copy",
        "classification_relevance": "low",
        "rationale": "hallucinated field",
    },
]


def responder(role, prompt, n):
    if role == "analyzer":
        return json.dumps({"potential_features": SPECS})
    if role == "extractor":
        code = GOOD_CODE if "PREVIOUS ATTEMPT FAILED" in prompt else HALLUCINATING_CODE
        return f"<thoughts>t</thoughts><code>{code}</code>"
    if role == "validator":
        names = [s["feature_name"] for s in SPECS]
        return json.dumps(
            {
                "features": [
                    {
                        "feature_name": x,
                        "verdict": "hallucinated_field" if x == "user_mood" else "valid",
                        "evidence": ".",
                    }
                    for x in names
                ]
            }
        )
    if role == "aggregator":
        kept = [s for s in SPECS if s["feature_name"] != "user_mood"]
        return json.dumps({"filtered_features": kept, "rejected_features": []})
    if role.startswith("judge_"):
        return json.dumps(
            {
                "judge_type": role,
                "features": [
                    {
                        "feature_name": s["feature_name"],
                        "score": 5 if s["feature_name"] != "n_prior_steps" else 2,
                        "confidence": 4,
                        "assessment": "ok",
                        "key_factors": [],
                    }
                    for s in SPECS[:3]
                ],
            }
        )
    if role == "meta_judge":
        return (
            "```json\n"
            + json.dumps(
                [
                    {"feature_name": "user_intent", "final_decision": "accept", "meta_score": 5},
                    {"feature_name": "plan_note", "final_decision": "Accept", "meta_score": 5},
                    {"feature_name": "n_prior_steps", "final_decision": "conditional", "meta_score": 3},
                ]
            )
            + "\n```"
        )
    if role == "auditor":
        names = re.findall(r'"feature_name": "([^"]+)"', prompt)
        return json.dumps(
            {"features": [{"feature_name": x, "verdict": "correct", "reason": "."} for x in names]}
        )
    raise AssertionError(role)


def test_discovery_end_to_end(tmp_path):
    from tabagent.tabschema.llm import ScriptedLLM

    logs = tmp_path / "logs"
    logs.mkdir()
    write_logs(logs)
    llm = ScriptedLLM(responder, model="scripted")
    cfg = TabSchemaConfig(
        n_analyze=3,
        workers=2,
        pair_embeddings=False,
        context_text=False,
        last_step_text=False,
        accept=("accept",),
    )
    schema = TabSchema(llm, cfg, adapter(), store=RunStore(tmp_path / "runs", "r1"))
    result = schema.discover(logs, "ShortlisterAgent")

    # the hallucinated feature never survives; the conditional one is excluded by an accept-only policy
    assert set(result.extractor.features) == {"user_intent", "plan_note"}
    assert result.report.n_successful == 44 and result.report.n_decision_points == 44
    assert all(a["attempts"] == 2 for a in result.report.analyzed)  # first code read a missing field
    card = result.card
    assert card.provenance["kind"] == "tabschema-llm" and card.extractor["sha256"] == result.extractor.sha256
    assert {f.name for f in card.features} >= {
        "user_intent",
        "plan_note",
        "candidate_name",
        "candidate_prior",
    }
    assert card.synth_text == ["user_intent", "plan_note"]

    trajs = schema.adapter.load(logs)
    table = schema.build_table(result, trajs, "ShortlisterAgent")
    assert len(table) == 44 * len(TOOLS) and table.df["chosen"].sum() == sum(
        len(TASKS[i % 4][1]) for i in range(48) if i % 12
    )
    res = cross_validate(
        TabAgent(card, head=TabHead("ranking", preset="paper", iterations=60)),
        table,
        n_splits=4,
        seeds=(0,),
    )
    assert res.summary(["p@r"]).loc["p@r", "value"] > 0.9

    saved = result.save(tmp_path / "card")
    assert (saved / "extract.py").read_text() == result.extractor.code and (saved / "discovery.md").exists()

    # resuming the same run makes no new LLM calls
    n_calls = len(llm.calls)
    TabSchema(llm, cfg, adapter(), store=RunStore(tmp_path / "runs", "r1")).discover(logs, "ShortlisterAgent")
    assert len(llm.calls) == n_calls


def test_retry_prompt_carries_the_error_and_the_attempt_number():
    block = generate_retry_prompt(1, "KeyError: 'x'", "print(1)")
    assert "PREVIOUS ATTEMPT FAILED:" in block and "Attempt #2" in block and "KeyError" in block
    assert generate_retry_prompt(0, "e", "c") == ""


def test_judge_table_joins_analyzer_view():
    verdicts = {"relevance": [{"feature_name": "Plan Note", "score": 5}], "generality": [], "impact": []}
    table = judge_table(verdicts, [FeatureSpec("plan note", description="d", classification_relevance="r")])
    assert table[0]["judge_relevance_score"] == 5 and table[0]["analyzer_description"] == "d"
    assert normalize_decision("Revise") == "conditional" and normalize_decision("REJECT") == "reject"


def test_no_decision_points_is_an_error(tmp_path):
    from tabagent.tabschema.llm import ScriptedLLM

    logs = tmp_path / "logs"
    logs.mkdir()
    write_logs(logs, 4)
    with pytest.raises(ValueError, match="no occurrences"):
        TabSchema(
            ScriptedLLM(responder), TabSchemaConfig(), adapter(), store=RunStore(tmp_path / "r")
        ).discover(logs, "MissingAgent")
