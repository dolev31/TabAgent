"""TabSchema foundations: trajectories, LLM layer, sandbox, deterministic checks, prompts, run store."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tabagent.tabschema import checks, prompts
from tabagent.tabschema.llm import (
    BudgetExceeded,
    CostMeter,
    LLMResponse,
    RecordingLLM,
    ReplayLLM,
    ScriptedLLM,
    auto_llm,
    extract_code,
    extract_json,
)
from tabagent.tabschema.runstore import RunStore
from tabagent.tabschema.sandbox import SubprocessSandbox
from tabagent.tabschema.trajectory import (
    GenericJsonAdapter,
    field_paths,
    flatten_json,
    get_json_schema,
    union_paths,
)

TRAJ = {
    "intent": "return the blender and venmo my roommate",
    "task_id": "t1",
    "score": 1.0,
    "eval": {"passed": True},
    "steps": [
        {"name": "TaskAnalyzerAgent", "data": "apps: amazon, venmo"},
        {
            "name": "PlanControllerAgent",
            "data": "next: shortlist amazon apis",
            "prompts": [{"role": "generation", "value": "thoughts"}],
        },
        {"name": "ShortlisterAgent", "data": "amazon_show_orders, amazon_initiate_return"},
        {"name": "CodeAgent", "data": "..."},
        {"name": "EvaluationResult", "data": "pass"},
    ],
}


# --------------------------------------------------------------------------- trajectories
def test_decision_points_are_truncated_and_redacted(tmp_path):
    (tmp_path / "run1.json").write_text(json.dumps(TRAJ))
    (tmp_path / "results.json").write_text("{}")
    ad = GenericJsonAdapter()
    trajs = ad.load(tmp_path)
    assert [t.id for t in trajs] == ["run1"] and trajs[0].success and trajs[0].group == "t1"
    (dp,) = ad.decision_points(trajs[0], "ShortlisterAgent")
    assert dp.step_index == 2 and dp.target_step["name"] == "ShortlisterAgent"
    assert [s["name"] for s in dp.prefix["steps"]] == ["TaskAnalyzerAgent", "PlanControllerAgent"]
    assert "score" not in dp.prefix and "eval" not in dp.prefix and dp.prefix["intent"]


def test_json_schema_matches_the_original_pipeline():
    schema = get_json_schema({"a": {"b": 1}, "c": [{"d": "x"}], "e": [], "f": 2.5})
    assert schema == {
        "a": {"type": "dict", "properties": {"b": {"type": "int"}}},
        "c": {"type": "list", "items": {"d": {"type": "str"}}},
        "e": {"type": "list", "items": "unknown (empty array)"},
        "f": {"type": "float"},
    }
    assert flatten_json({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a_b": 1, "a_c_d": 2, "e": 3}
    assert ("steps", "[]", "prompts", "[]", "role") in union_paths([TRAJ])
    assert ("c", "[]", "d") in field_paths({"c": [{"d": 1}]})


# --------------------------------------------------------------------------- LLM layer
def test_extract_json_and_code():
    assert extract_json('noise ```json\n{"a": 1}\n``` tail') == {"a": 1}
    assert extract_json("Here: [1, 2] and more text") == [1, 2]
    assert extract_json('{"a": {"b": 2}} trailing words') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")
    assert extract_code("<thoughts>x</thoughts><code>\nprint(1)\n</code>") == "print(1)"
    assert extract_code("```python\nx = 1\n```") == "x = 1"


def test_record_then_replay(tmp_path):
    live = ScriptedLLM(lambda role, prompt, i: f"{role}:{i}", model="gpt-x")
    rec = RecordingLLM(live, tmp_path / "session.jsonl")
    assert rec.complete("prompt A", role="analyzer").text == "analyzer:0"
    replay = ReplayLLM(tmp_path / "session.jsonl", model="gpt-x")
    assert replay.complete("prompt A", role="analyzer").text == "analyzer:0"
    with pytest.raises(KeyError):
        replay.complete("an unrecorded prompt", role="analyzer")


def test_cost_meter_budget():
    meter = CostMeter(prices={"m": (1.0, 2.0)}, max_cost_usd=0.01)
    meter.record("analyzer", LLMResponse("x", "m", 1000, 1000))
    assert meter.total_usd == pytest.approx(0.003)
    with pytest.raises(BudgetExceeded):
        meter.record("judges", LLMResponse("x", "m", 10_000, 10_000))


def test_auto_llm_needs_credentials(monkeypatch):
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        auto_llm()


# --------------------------------------------------------------------------- sandbox
CODE_OK = """
def extract_features_for_classification(trajectory_data):
    return {"n_steps": len(trajectory_data['steps']), "intent": trajectory_data['intent']}
features = extract_features_for_classification(trajectory)
"""


def test_sandbox_runs_code_on_each_trajectory():
    box = SubprocessSandbox(timeout_s=20)
    res = box.run_batch(CODE_OK, [{"steps": [1, 2], "intent": "a"}, {"steps": [], "intent": "b"}])
    assert [r.features for r in res] == [{"n_steps": 2, "intent": "a"}, {"n_steps": 0, "intent": "b"}]


def test_sandbox_reports_tracebacks_and_missing_features():
    box = SubprocessSandbox(timeout_s=20)
    bad = box.run(CODE_OK, {"steps": []})  # no 'intent' key
    assert not bad.ok and "KeyError" in bad.error
    none = box.run("x = 1", {"steps": []})
    assert not none.ok and "features" in none.error


def test_sandbox_has_no_secrets_and_times_out(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "x" * 30)
    r = SubprocessSandbox(timeout_s=60).run("import os\nfeatures = {'env': len(os.environ)}", {})
    assert r.ok and r.features["env"] <= 2  # empty environment (a platform may inject one variable)
    slow = SubprocessSandbox(timeout_s=2).run("while True:\n    pass", {})
    assert not slow.ok and ("timed out" in slow.error or "exited" in slow.error or slow.error)


# --------------------------------------------------------------------------- deterministic checks
def test_checks_flag_hallucinated_and_outcome_fields():
    known = union_paths([{"intent": "x", "steps": [{"name": "a", "data": "d"}]}])
    code = (
        "import re\n"
        "steps = trajectory_data['steps']\n"
        "for s in trajectory_data['steps']:\n"
        "    n = s['name']\n"
        "x = trajectory_data['intent']\n"
        "y = trajectory_data['user_goal']\n"
        "z = trajectory_data['score']\n"
    )
    kinds = {(i.kind, i.detail) for i in checks.check_code(code, known, outcome_keys=("score",))}
    assert ("unknown_field", "user_goal") in kinds and ("outcome_field", "score") in kinds
    assert not any(d in ("intent", "steps.[].name") for k, d in kinds if k == "unknown_field")


def test_checks_enforce_the_extractor_contract():
    issues = checks.check_code("import os\nv = trajectory_data.get('a')\nopen('f')\n")
    kinds = {i.kind for i in issues}
    assert {"import", "uses_get", "call"} <= kinds
    assert checks.check_code("def (:")[0].kind == "syntax"
    assert checks.check_output({"a": 1, "b": "x"}) == []
    assert checks.check_output({"a": {"nested": 1}})[0].kind == "output"


# --------------------------------------------------------------------------- prompts and run store
def test_prompts_render_every_placeholder():
    fields = {
        "feature_analyzer": dict(
            trajectory_json="{}",
            trajectory_schema="{}",
            target_name="T",
            target_json="{}",
            trajectory_file="f.json",
            analyzer_hints="",
        ),
        "code_extractor": dict(
            potential_features="[]",
            trajectory_schema="{}",
            features_schema="{}",
            target_name="T",
            retry_block="",
        ),
        "feature_aggregator": dict(potential_features="[]", features_schema="{}"),
        "judge_relevance": dict(filtered_features="[]", features_schema="{}"),
        "judge_generality": dict(filtered_features="[]", features_schema="{}"),
        "judge_impact": dict(filtered_features="[]", features_schema="{}"),
        "meta_judge": dict(
            relevance_schema="{}", generality_schema="{}", impact_schema="{}", all_judge_features="[]"
        ),
        "code_validator": dict(
            target_name="T", feature_specs="[]", trajectory_schema="{}", code="x=1", sample_output="[]"
        ),
    }
    assert set(fields) == set(prompts.PROMPTS)
    for name, f in fields.items():
        out = prompts.render(name, **f)
        assert "${" not in out, name


def test_runstore_cache_and_log(tmp_path):
    store = RunStore(tmp_path, run_id="r1")
    calls = []
    assert store.cached("stage", {"a": 1}, lambda: calls.append(1) or {"v": 2}) == {"v": 2}
    assert store.cached("stage", {"a": 1}, lambda: calls.append(1) or {"v": 3}) == {"v": 2}
    assert calls == [1]
    store.log_call({"stage": "x"})
    assert (tmp_path / "r1" / "llm_calls.jsonl").read_text().count("\n") == 1


def test_runstore_cache_is_thread_safe(tmp_path):
    # Threads that request one key never read a half-written entry, and the paid call runs once.
    store = RunStore(tmp_path, run_id="r1")
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.05)
        return {"text": "x" * 100_000}

    with ThreadPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(lambda _: store.cached("llm", {"k": 1}, slow), range(16)))
    assert calls == [1] and all(o == {"text": "x" * 100_000} for o in out)
    assert not list((tmp_path / "r1" / "cache").rglob("*.tmp"))


# Pinned hashes of the prompt files. The seven paper prompts reproduce App. C word for word (verified against
# tabagent.tex); any edit must be deliberate and re-verified against the paper before updating a hash.
PAPER_PROMPT_SHA256 = {
    "feature_analyzer": "11233a14c78bbf9c7faaa163ef01b851b96abc7b87f0f5ebd964892f936a1eb6",
    "code_extractor": "4216f68b43b8b5ed088b03d5596857cf2aaabdd9ba9806015d24cbd8e047c8d4",
    "feature_aggregator": "9e4adc4eb03385a3b2a4666ec8ebdbf774611126e2496ed24f8c237f6491728d",
    "judge_relevance": "f9a02d00f9b75077f9be05b02791f5e1b432b8b3ce61870761659b418673e788",
    "judge_generality": "35796538e8c2b2af6d2c5ada7ef5bfd303be4b141ad23b41929eb98328762c88",
    "judge_impact": "a7221a62baf74ec714bce08e09da4322a30a351bcc477ef74b72bb1d21d5c8e3",
    "meta_judge": "c33677663ca48eeacbed1ebb6faaab702ebc75d7861af00063eb76ae90a90e70",
    "code_validator": "fdf390b3d21fb6f67e608f362e2015d4d8f18fd8d8d10357ec001e452c2f3109",
}


def test_prompts_are_pinned():
    assert {n: prompts.sha256(n) for n in prompts.PROMPTS} == PAPER_PROMPT_SHA256
