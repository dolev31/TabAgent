"""TabSchema discovery end to end on chat-message logs read by OpenAIMessagesAdapter (scripted LLM)."""

import json

from fake_logs import TASKS, TOOLS, write_openai_logs

from tabagent.adapters import TOOL_SHORTLIST, detect_adapter
from tabagent.tabschema.llm import ScriptedLLM
from tabagent.tabschema.orchestrator import TabSchema, TabSchemaConfig
from tabagent.tabschema.runstore import RunStore

CODE = """
def extract_features_for_classification(trajectory_data):
    steps = trajectory_data['steps']
    users = [s['content'] for s in steps if s['name'] == 'user']
    first = users[0] if users else ""
    return {"user_request": first, "request_words": len(first.split()), "n_user_turns": len(users)}
features = extract_features_for_classification(trajectory)
"""
SPECS = [
    {"feature_name": "user_request", "feature_type": "text", "description": "the first user message"},
    {"feature_name": "request_words", "feature_type": "numeric", "description": "words in the request"},
    {"feature_name": "n_user_turns", "feature_type": "numeric", "description": "user turns so far"},
]


def responder(role, prompt, n):
    if role == "analyzer":
        assert (
            '"name": "ToolShortlist"' in prompt and "FINAL ANSWER" not in prompt and "result of" not in prompt
        )
        return json.dumps({"potential_features": SPECS})
    if role == "extractor":
        return f"<code>{CODE}</code>"
    if role == "validator":
        return json.dumps(
            {"features": [{"feature_name": s["feature_name"], "verdict": "valid"} for s in SPECS]}
        )
    if role == "aggregator":
        return json.dumps({"filtered_features": SPECS, "rejected_features": []})
    if role.startswith("judge_"):
        scores = [{"feature_name": s["feature_name"], "score": 5, "confidence": 4} for s in SPECS]
        return json.dumps({"judge_type": role, "features": scores})
    if role == "meta_judge":
        return json.dumps([{"feature_name": s["feature_name"], "final_decision": "accept"} for s in SPECS])
    raise AssertionError(role)


def test_discovery_on_chat_message_logs(tmp_path):
    logs = write_openai_logs(tmp_path / "logs", n=24)
    adapter = detect_adapter(logs)
    cfg = TabSchemaConfig(n_analyze=2, workers=2, pair_embeddings=False)
    schema = TabSchema(ScriptedLLM(responder), cfg, adapter, store=RunStore(tmp_path / "runs", "r1"))
    result = schema.discover(logs, TOOL_SHORTLIST)

    assert set(result.extractor.features) == {"user_request", "request_words"}
    assert "constant" in result.report.dropped["n_user_turns"]  # every fabricated run has two user turns
    assert result.report.n_successful == 21 and result.report.n_decision_points == 21
    table = schema.build_table(result, adapter.load(logs), TOOL_SHORTLIST)
    assert len(table) == 21 * len(TOOLS)
    expected = sum(len(TASKS[i % len(TASKS)][1]) for i in range(24) if i % 8 != 7)
    assert table.df["chosen"].sum() == expected
    row = table.df[table.df["candidate"] == "search_flights"].iloc[0]
    assert row["candidate_args"] == "origin, destination, date" and row["request_words"] == 8
