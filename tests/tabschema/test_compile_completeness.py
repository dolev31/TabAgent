"""Accepted features the compiled extractor does not produce are asked for in the repair round: the
CodeExtractor adds each one, or names the feature that already carries it, or declines it with a reason; the
reasons are recorded in compile.json and the report (fabricated toy runs, scripted LLM)."""

from __future__ import annotations

import json

from test_robustness import FakeLLM, discover, spec, tagged

from tabagent.tabschema.compile import NOT_PRODUCED, declared_reason, not_extracted

PROPOSALS = [
    spec("user_intent", "text"),
    spec("plan_tokens", "text", description="the tokens of the latest plan note"),
    spec("intent_tokens", "text", description="the tokens of the user's request"),
    spec("run_reward", "numeric", description="the reward of the run"),
]
FIRST = """
def extract_features_for_classification(trajectory_data):
    return {"user_intent": trajectory_data['intent']}
features = extract_features_for_classification(trajectory)
"""
REPAIRED = """
NOT_EXTRACTED = {
    "intent_tokens": "merged into user_intent",
    "run_reward": "declined: the reward exists only after the run ends",
}


def extract_features_for_classification(trajectory_data):
    plan = [s['data'] for s in trajectory_data['steps'] if s['name'] == 'PlanControllerAgent']
    return {"user_intent": trajectory_data['intent'], "plan_tokens": plan[-1] if plan else ""}
features = extract_features_for_classification(trajectory)
"""


def fake(repaired: str = REPAIRED) -> FakeLLM:
    def extractor(p, n):
        return tagged(repaired if "The code does not produce these accepted features" in p else FIRST)

    return FakeLLM(
        analyzer=lambda p, n: json.dumps({"potential_features": PROPOSALS}),
        aggregator=lambda p, n: json.dumps({"filtered_features": PROPOSALS}),
        extractor=extractor,
    )


def test_the_repair_round_adds_missing_features_and_records_the_others(tmp_path):
    llm = fake()
    ext = discover(tmp_path, llm).extractor
    asked = [p for p in llm.prompts("extractor") if "The code does not produce these accepted features" in p]
    assert len(asked) == 1  # one repair call, with every missing feature's plan
    assert all(f'"feature_name": "{n}"' in asked[0] for n in ("plan_tokens", "intent_tokens", "run_reward"))
    assert set(ext.features) == {"user_intent", "plan_tokens"}  # added, under its name, as plain text
    assert ext.not_produced == {
        "intent_tokens": "merged into user_intent",
        "run_reward": "declined: the reward exists only after the run ends",
    }
    record = json.loads((tmp_path / "runs" / "r1" / "compile.json").read_text())
    assert record["dropped"]["intent_tokens"] == "merged into user_intent"
    assert record["not_produced"]["count"] == 2
    assert record["not_produced"]["by_reason"] == {"declined": 1, "merged into another feature": 1}
    assert record["per_decision"] == ["user_intent", "plan_tokens"] and record["per_candidate"] == []
    report = (tmp_path / "runs" / "r1" / "report.md").read_text()
    assert "### Accepted features without a column (2: declined: 1; merged into another feature: 1)" in report
    drops = json.loads((tmp_path / "runs" / "r1" / "run.json").read_text())["drops_by_reason"]
    assert drops["merged into another feature (repair round)"] == 1
    assert drops["declined by the CodeExtractor (repair round)"] == 1


def test_a_repair_that_produces_nothing_new_is_not_kept(tmp_path):
    ext = discover(tmp_path, fake(repaired=FIRST)).extractor
    assert set(ext.features) == {"user_intent"}
    assert any("did not reduce the 3 flagged feature(s)" in n for n in ext.notes)
    assert ext.not_produced == dict.fromkeys(("plan_tokens", "intent_tokens", "run_reward"), NOT_PRODUCED)
    report = (tmp_path / "runs" / "r1" / "report.md").read_text()
    assert "### Accepted features without a column (3: not produced: 3)" in report


def test_not_extracted_is_read_from_the_code_and_normalized():
    assert not_extracted(REPAIRED) == {
        "intent_tokens": "merged into user_intent",
        "run_reward": "declined: the reward exists only after the run ends",
    }
    assert not_extracted("NOT_EXTRACTED = make()\nfeatures = {}") == {} and not_extracted("def (:") == {}
    assert declared_reason("Merged into  user_intent") == "merged into user_intent"
    assert declared_reason("Declined - no such field") == "declined: - no such field"
    assert declared_reason("it repeats user_intent") == "declined: it repeats user_intent"
