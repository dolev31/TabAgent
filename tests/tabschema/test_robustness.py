"""Robustness of TabSchema discovery and onboarding under adversarial conditions.

Every test drives the real pipeline (``TabSchema.discover``, ``tabagent.onboard``, the sandbox, the adapters
and the LLM backends) with scripted fakes: no network, no credentials, no real model. The conditions are the
ones a real onboarding run meets: malformed, truncated or hostile LLM answers; extraction code that fails,
misbehaves or tries to escape; odd agent logs; crashes and resumes, concurrent runs, budgets, redaction and
determinism; provider quirks; and the training-inference parity of the compiled extractor.
"""

from __future__ import annotations

import copy
import json
import re
import sys
import threading
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

import tabagent
from tabagent import DecisionSpec, DecisionTable, Feature, FeatureCard, TabAgent, TabHead
from tabagent.adapters import NEXT_TOOL_CALL, TOOL_SHORTLIST, CugaAdapter, OpenAIMessagesAdapter
from tabagent.api import DryRun, TabAgentHead
from tabagent.tabschema import TabSchema, TabSchemaConfig, Trajectory, checks, prompts
from tabagent.tabschema.agents import (
    CodeExecutor,
    CodeExtractor,
    FeatureSpec,
    LLMFormatError,
    normalize_name,
    premerge,
)
from tabagent.tabschema.compile import ExtractionError, ExtractorRunner, compile_extractor
from tabagent.tabschema.judges import normalize_decision
from tabagent.tabschema.llm import (
    AZURE_API_VERSION,
    USER_TURN,
    BudgetExceeded,
    CostMeter,
    LiteLLM,
    LLMResponse,
    LLMTruncated,
    OpenAIChatLLM,
    auto_llm,
    extract_code,
    extract_json,
)
from tabagent.tabschema.runstore import RunStore
from tabagent.tabschema.sandbox import SubprocessSandbox
from tabagent.tabschema.trajectory import (
    TASK_KEYS,
    Candidate,
    GenericJsonAdapter,
    get_json_schema,
    union_paths,
    union_schema,
)

TARGET = "ShortlisterAgent"
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
    ("what is the weather in {c}", ["get_weather"]),
    ("reply to the email about {c}", ["read_inbox", "send_email"]),
    ("play a song from {c}", ["play_song"]),
]
CITIES = ["rome", "oslo", "lima"]
ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "OPENAI_API_VERSION",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
)


@pytest.fixture
def no_credentials(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


# --------------------------------------------------------------------------- toy agent logs
def raw_run(i: int) -> dict:
    """One fabricated successful run of a toy multi-agent system (no real agent data)."""
    template, used = TASKS[i % len(TASKS)]
    intent = template.format(c=CITIES[i % len(CITIES)])
    return {
        "task_id": f"task{i}",
        "intent": intent,
        "score": 1.0,
        "steps": [
            {"name": "TaskAnalyzerAgent", "data": f"apps needed for: {intent}"},
            {"name": "PlanControllerAgent", "data": "plan: " + " then ".join(t.split("_")[0] for t in used)},
            {"name": TARGET, "data": ", ".join(used)},
            {"name": "CodeAgent", "calls": used},
            {"name": "EvaluationResult", "data": "pass"},
        ],
    }


def runs(n: int = 16, make=raw_run) -> list[Trajectory]:
    return [Trajectory(f"run{i}", make(i), True, f"task{i}", f"run{i}.json") for i in range(n)]


def tool_candidates(dp) -> list[Candidate]:
    return [Candidate(t, t.replace("_", " "), d) for t, d in TOOLS.items()]


def used_tools(dp, raw) -> set[str]:
    steps = raw.get("steps") if isinstance(raw.get("steps"), list) else []
    return {c for s in steps if isinstance(s, dict) and s.get("name") == "CodeAgent" for c in s["calls"]}


def adapter(**kw) -> GenericJsonAdapter:
    return GenericJsonAdapter(**{"candidates_fn": tool_candidates, "labels_fn": used_tools, **kw})


def chat_run(i: int) -> dict:
    """One fabricated OpenAI chat run that calls its task's tools one turn at a time."""
    template, used = TASKS[i % len(TASKS)]
    intent = template.format(c=CITIES[i % len(CITIES)])
    msgs = [
        {"role": "system", "content": "You are a travel and email assistant."},
        {"role": "user", "content": intent},
        {"role": "assistant", "content": "Let me check."},
        {"role": "user", "content": "yes please"},
    ]
    for k, t in enumerate(used):
        call = {"id": f"c{k}", "type": "function", "function": {"name": t, "arguments": "{}"}}
        msgs += [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": f"c{k}", "content": f"result of {t}: ok"},
        ]
    tools = [{"type": "function", "function": {"name": t, "description": d}} for t, d in TOOLS.items()]
    msgs.append({"role": "assistant", "content": f"done: {intent}"})
    return {"task_id": f"task{i}", "messages": msgs, "tools": tools, "reward": 1.0}


# --------------------------------------------------------------------------- the scripted fake LLM
#: The features the fake analyzer proposes, with the expression the fake CodeExtractor computes each from.
EXPRESSIONS = {
    "user_intent": ("text", "trajectory_data['intent']"),
    "plan_note": ("text", "plan[-1] if plan else ''"),
    "intent_words": ("numeric", "len(trajectory_data['intent'].split())"),
}


def spec(name: str, ftype: str = "text", **kw) -> dict:
    return {
        "feature_name": name,
        "feature_type": ftype,
        "description": f"the {name}",
        "extraction_source": "trajectory",
        "computation": "copy",
        "classification_relevance": "high",
        "rationale": "predictive",
        **kw,
    }


PROPOSALS = [spec(n, t) for n, (t, _) in EXPRESSIONS.items()]
_NAME = re.compile(r'"feature_name": "((?:[^"\\]|\\.)*)"')


def names_in(prompt: str) -> list[str]:
    """The feature names a stage prompt lists (the templates' own ``descriptive name`` example excluded)."""
    found = [json.loads(f'"{m}"') for m in _NAME.findall(prompt)]
    return [n for n in dict.fromkeys(found) if n != "descriptive name"]


def by_name(name: str) -> dict:
    return next((p for p in PROPOSALS if p["feature_name"] == name), spec(name))


def code_for(names, extra: str = "") -> str:
    """Extraction code for the planned ``names`` that have a known expression, in the prompt's structure."""
    keys = dict.fromkeys(normalize_name(n) for n in names)
    items = ", ".join(f"{k!r}: {EXPRESSIONS[k][1]}" for k in keys if k in EXPRESSIONS)
    return (
        "def extract_features_for_classification(trajectory_data):\n"
        "    steps = trajectory_data['steps']\n"
        "    plan = [s['data'] for s in steps if s['name'] == 'PlanControllerAgent']\n"
        f"{extra}"
        f"    return {{{items}}}\n"
        "features = extract_features_for_classification(trajectory)\n"
    )


def tagged(code: str) -> str:
    return f"<thoughts>t</thoughts><code>{code}</code>"


DEFAULT = {
    "analyzer": lambda p, n: json.dumps({"potential_features": PROPOSALS}),
    "extractor": lambda p, n: tagged(code_for(names_in(p))),
    "validator": lambda p, n: json.dumps(
        {"features": [{"feature_name": x, "verdict": "valid", "evidence": "."} for x in names_in(p)]}
    ),
    "aggregator": lambda p, n: json.dumps(
        {"filtered_features": [by_name(x) for x in names_in(p)], "rejected_features": []}
    ),
    "judge": lambda p, n: json.dumps(
        {"features": [{"feature_name": x, "score": 5, "confidence": 4} for x in names_in(p)]}
    ),
    "meta_judge": lambda p, n: json.dumps(
        [{"feature_name": x, "final_decision": "accept", "meta_score": 5} for x in names_in(p)]
    ),
    "auditor": lambda p, n: json.dumps(
        {"features": [{"feature_name": x, "verdict": "correct", "reason": "."} for x in names_in(p)]}
    ),
}


class FakeLLM:
    """A scripted LLM: one handler per stage role, ``handler(prompt, n) -> answer`` (``n`` counts the role's
    earlier calls), or ``(answer, finish_reason)``. ``judge`` answers for the three judges unless a
    ``judge_<dimension>`` handler is given. Every prompt it receives is recorded in ``calls``."""

    def __init__(self, model: str = "fake", **handlers):
        self.model = model
        self.handlers = {**DEFAULT, **handlers}
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def complete(self, system: str, *, role: str = "") -> LLMResponse:
        with self._lock:
            n = sum(r == role for r, _ in self.calls)
            self.calls.append((role, system))
        key = role if role in self.handlers else role.split("_")[0]
        out = self.handlers[key](system, n)
        text, finish = out if isinstance(out, tuple) else (out, "stop")
        return LLMResponse(text, self.model, len(system) // 4, len(text) // 4, 0.0, finish)

    def roles(self) -> Counter:
        return Counter(r for r, _ in self.calls)

    def prompts(self, role: str) -> list[str]:
        return [p for r, p in self.calls if r == role]


def config(**kw) -> TabSchemaConfig:
    # context_text=False: these tests pin the LLM-written features (the context text has its own tests);
    # usefulness=False: the usefulness estimate (cross-validated CatBoost fits) has its own tests
    return TabSchemaConfig(
        **{
            "n_analyze": 3,
            "workers": 2,
            "pair_embeddings": False,
            "timeout_s": 30,
            "context_text": False,
            "usefulness": False,
            **kw,
        }
    )


def schema_for(tmp_path, llm, run_id: str = "r1", ad=None, **kw) -> TabSchema:
    return TabSchema(llm, config(**kw), ad or adapter(), store=RunStore(tmp_path / "runs", run_id))


def discover(tmp_path, llm, trajs=None, run_id: str = "r1", ad=None, **kw):
    return schema_for(tmp_path, llm, run_id, ad, **kw).discover(runs() if trajs is None else trajs, TARGET)


def with_bad_feature(p, n):
    """Analyzer answer: the usual proposals, plus ``bad_feature`` on the weather tasks."""
    extra = [spec("bad_feature")] if "weather" in p else []
    return json.dumps({"potential_features": PROPOSALS + extra})


def down(p, n):
    raise ConnectionError("provider unavailable")


# =========================================================================== 1. LLM answers
def test_json_wrapped_in_prose_and_fences_or_with_trailing_commas_is_parsed(tmp_path):
    llm = FakeLLM(
        analyzer=lambda p, n: (
            "Here is my analysis:\n```json\n"
            + json.dumps({"potential_features": PROPOSALS})
            + "\n```\nLet me know!"
        ),
        aggregator=lambda p, n: "Sure. " + DEFAULT["aggregator"](p, n)[:-1] + ",} Hope this helps.",
        meta_judge=lambda p, n: "```\n" + DEFAULT["meta_judge"](p, n).replace("}]", "},\n]") + "\n```",
    )
    assert set(discover(tmp_path, llm).extractor.features) == set(EXPRESSIONS)
    assert extract_json('{"a": [1, 2,], "b": {"c": 3,},}') == {"a": [1, 2], "b": {"c": 3}}


def test_truncated_json_is_reasked_once_with_a_repair_instruction(tmp_path):
    good = json.dumps({"potential_features": PROPOSALS})
    llm = FakeLLM(analyzer=lambda p, n: good if "could not be parsed" in p else good[: len(good) // 2])
    result = discover(tmp_path, llm)
    assert set(result.extractor.features) == set(EXPRESSIONS)
    assert sum("could not be parsed" in p for p in llm.prompts("analyzer")) == 3  # one repair per point


def test_answers_cut_at_the_output_limit_raise_a_clear_error(tmp_path):
    def cut(p, n):
        return DEFAULT["aggregator"](p, n)[:50], "length"

    with pytest.raises(LLMTruncated, match="output limit.*max_tokens"):
        discover(tmp_path, FakeLLM(aggregator=cut))

    def cut_weather(p, n):  # one decision point's truncated analysis loses only that point
        return (with_bad_feature(p, n)[:40], "length") if "weather" in p else DEFAULT["analyzer"](p, n)

    result = discover(tmp_path, FakeLLM(analyzer=cut_weather), run_id="r2", n_analyze=16, workers=4)
    lost = [a for a in result.report.analyzed if a["error"]]
    assert len(lost) == 4 and all("LLMTruncated" in a["error"] for a in lost)
    assert set(result.extractor.features) == set(EXPRESSIONS)


@pytest.mark.parametrize(
    "answer",
    ["", "   ", "I cannot help with that.", '{"potential_features": [', "null", "42", '"text"', "[]"],
    ids=["empty", "blank", "refusal", "truncated", "null", "number", "string", "empty-list"],
)
def test_unusable_analyzer_answers_end_in_a_clear_error(tmp_path, answer):
    with pytest.raises(RuntimeError, match="no feasible feature proposals .*analysis.json"):
        discover(tmp_path, FakeLLM(analyzer=lambda p, n: answer))


@pytest.mark.parametrize(
    "payload",
    [
        {"potential_features": None},
        {"potential_features": 5},
        {"potential_features": {"user_intent": "text"}},
        {"features": PROPOSALS},
        [None, 3, "user_intent", [1], {"feature_name": None}],
        {"potential_features": [{"feature_name": "x", "support": "many"}, {"feature_name": 7}]},
    ],
    ids=["null-list", "number-list", "dict-list", "wrong-key", "junk-items", "wrong-field-types"],
)
def test_wrong_shapes_on_some_decision_points_only_lose_those_points(tmp_path, payload):
    def analyzer(p, n):
        return json.dumps(payload) if "weather" in p else DEFAULT["analyzer"](p, n)

    result = discover(tmp_path, FakeLLM(analyzer=analyzer), n_analyze=16, workers=4)
    assert set(result.extractor.features) == set(EXPRESSIONS)


def test_provider_errors_on_some_decision_points_only_lose_those_points(tmp_path):
    def flaky(p, n):
        if "weather" in p:
            raise ConnectionError("context length exceeded")
        return DEFAULT["analyzer"](p, n)

    result = discover(tmp_path, FakeLLM(analyzer=flaky), n_analyze=16, workers=4)
    lost = [a for a in result.report.analyzed if a["error"]]
    assert len(lost) == 4 and all("ConnectionError: context length" in a["error"] for a in lost)
    assert "4 failed, see analysis.json" in result.report.to_markdown()
    assert set(result.extractor.features) == set(EXPRESSIONS)
    with pytest.raises(RuntimeError, match="first error: ConnectionError: provider unavailable"):
        discover(tmp_path, FakeLLM(analyzer=down), run_id="r2")  # every point fails: the cause is named


def test_feature_fields_of_the_wrong_type_are_coerced():
    nameless, many, named, numeric = (
        FeatureSpec.from_dict(d)
        for d in (
            {"feature_name": None, "description": "x"},
            {"feature_name": "a", "support": "many", "description": None},
            {"name": "b", "feature_type": 3},
            {"feature_name": 7},
        )
    )
    assert nameless.feature_name == "" and not nameless.key  # a null name is no name, never "None"
    assert many.support == 1 and many.description == ""
    assert named.feature_name == "b" and named.feature_type == "3"
    assert numeric.feature_name == "7"


@pytest.mark.parametrize(
    "role, answer",
    [
        ("validator", '{"features": null}'),
        ("validator", '[{"verdict": "valid"}, 7, "x"]'),
        ("aggregator", json.dumps(PROPOSALS)),  # a bare list instead of the requested object
        ("judge_relevance", '{"features": null}'),
        ("judge_relevance", '{"features": {"user_intent": 5}}'),
        ("judge_generality", "[]"),
        (
            "meta_judge",
            json.dumps({"decisions": [{"feature_name": "user_intent", "final_decision": "accept"}]}),
        ),
    ],
)
def test_wrong_shapes_from_later_stages_never_crash(tmp_path, role, answer):
    if role == "meta_judge":  # decisions under another key: only what they cover is accepted
        assert set(discover(tmp_path, FakeLLM(meta_judge=lambda p, n: answer)).extractor.features) == {
            "user_intent"
        }
        return
    result = discover(tmp_path, FakeLLM(**{role: lambda p, n: answer}))
    assert set(result.extractor.features) == set(EXPRESSIONS)


def test_an_aggregator_or_meta_judge_that_keeps_nothing_is_a_clear_error(tmp_path):
    empty = '{"filtered_features": null, "rejected_features": null}'
    llm = FakeLLM(aggregator=lambda p, n: empty)
    with pytest.raises(ExtractionError, match="FeatureAggregator kept none of the 3 proposals"):
        discover(tmp_path, llm)
    assert not {"judge_relevance", "meta_judge"} & set(llm.roles())  # no judge is paid for nothing
    with pytest.raises(ExtractionError, match="MetaJudge accepted none"):
        discover(tmp_path, FakeLLM(meta_judge=lambda p, n: '{"features": null}'), run_id="r2")


def test_verdicts_are_read_tolerantly(tmp_path):
    def validator(p, n):
        verdicts = {"user_intent": "Valid.", "plan_note": "VALID - reads the plan step"}
        items = [{"feature_name": x, "verdict": verdicts.get(x, "Hallucinated field")} for x in names_in(p)]
        return json.dumps({"features": items})

    result = discover(tmp_path, FakeLLM(validator=validator))
    assert all(a["kept"] == ["user_intent", "plan_note"] for a in result.report.analyzed)
    assert set(result.extractor.features) == {"user_intent", "plan_note"}


def test_an_unusable_validator_answer_leaves_the_deterministic_checks_in_charge(tmp_path):
    def validator(p, n):  # garbage for the compile step's review (three sample outputs), fine otherwise
        return "no idea" if p.count('"user_intent":') > 1 else DEFAULT["validator"](p, n)

    result = discover(tmp_path, FakeLLM(validator=validator))
    assert set(result.extractor.features) == set(EXPRESSIONS)
    assert any("CodeValidator" in n for n in result.report.notes)


def test_enormous_and_non_ascii_answers(tmp_path):
    huge = "x" * 2_000_000
    noise = [spec(f"noise_{i}", description=huge if i == 0 else "🙂 ünïcödé 特徴 " * 5) for i in range(30)]
    llm = FakeLLM(
        analyzer=lambda p, n: json.dumps({"potential_features": PROPOSALS + noise}, ensure_ascii=False)
    )
    result = discover(tmp_path, llm)
    assert set(result.extractor.features) == set(EXPRESSIONS)
    assert result.extractor.dropped["noise_0"] == "accepted but not produced by the compiled extractor"
    assert any("🙂 ünïcödé 特徴" in p for p in llm.prompts("aggregator"))  # sent as is, not \\u-escaped
    assert FeatureCard.load(result.save(tmp_path / "card")).feature_names == result.card.feature_names


def test_whole_run_stages_work_in_batches(tmp_path):
    many = [spec(f"feature_{i:02d}") for i in range(60)]
    llm = FakeLLM(analyzer=lambda p, n: json.dumps({"potential_features": PROPOSALS + many}))
    result = discover(tmp_path, llm)
    roles = llm.roles()
    assert roles["aggregator"] == 2  # 63 proposals, 40 per call
    assert roles["judge_relevance"] == roles["judge_impact"] == roles["meta_judge"] == 3  # 25 per call
    assert max(len(names_in(p)) for p in llm.prompts("meta_judge")) <= 25
    assert len(result.report.judged) == 63 and set(result.extractor.features) == set(EXPRESSIONS)


def test_duplicate_and_colliding_names_merge_into_one_feature(tmp_path):
    variants = ["User Intent", "user_intent", "user-intent", "USER  INTENT!"]
    llm = FakeLLM(
        analyzer=lambda p, n: json.dumps({"potential_features": [spec(v) for v in variants] + PROPOSALS[1:]}),
        aggregator=lambda p, n: json.dumps(  # the aggregator itself repeats a feature
            {"filtered_features": [spec("user intent"), *(by_name(x) for x in names_in(p))]}
        ),
    )
    result = discover(tmp_path, llm)
    keys = [normalize_name(n) for n in result.report.aggregated]
    assert len(keys) == len(set(keys)) == 3
    merged = premerge([FeatureSpec(v) for v in variants])
    assert len(merged) == 1 and merged[0].support == 4
    assert len(result.card.feature_names) == len(set(result.card.feature_names))
    assert set(result.extractor.features) == set(EXPRESSIONS)


def test_unicode_feature_names_and_keys_stay_distinct(tmp_path):
    assert normalize_name("意図") != normalize_name("計画") and normalize_name("🙂 Mood") == "mood"

    def make(i):
        raw = raw_run(i)
        raw["意図"] = raw.pop("intent") + " 🙂"
        return raw

    code = (
        "steps = trajectory['steps']\n"
        "plan = [s['data'] for s in steps if s['name'] == 'PlanControllerAgent']\n"
        "features = {'意図': trajectory['意図'], '計画': plan[-1] if plan else ''}\n"
    )
    llm = FakeLLM(
        analyzer=lambda p, n: json.dumps({"potential_features": [spec("意図"), spec("計画")]}),
        extractor=lambda p, n: tagged(code),
    )
    trajs = runs(16, make)
    schema = schema_for(tmp_path, llm, ad=adapter(task_keys=(*TASK_KEYS, "意図")))
    result = schema.discover(trajs, TARGET)
    assert {"意図", "計画"} <= set(result.report.aggregated)
    assert {"意図", "計画"} == set(result.extractor.features)
    assert any("意図" in p for p in llm.prompts("analyzer"))
    card = FeatureCard.load(result.save(tmp_path / "card"))
    assert {"意図", "計画"} <= set(card.feature_names)
    table = schema.build_table(result, trajs, TARGET)
    assert table.df["意図"].str.endswith("🙂").all()


def test_extractor_columns_cannot_shadow_decision_table_columns(tmp_path):
    reserved = ["task", "decision_id", "candidate", "chosen", "candidate_name", "candidate_prior"]
    code = (
        "features = {"
        + ", ".join(f"{c!r}: trajectory['intent'] + {c!r}" for c in reserved)
        + ", 'user_intent': trajectory['intent']}\n"
    )
    llm = FakeLLM(
        analyzer=lambda p, n: json.dumps(
            {"potential_features": [spec(c) for c in [*reserved, "user_intent"]]}
        ),
        extractor=lambda p, n: tagged(code),
    )
    trajs = runs()
    schema = schema_for(tmp_path, llm)
    result = schema.discover(trajs, TARGET)
    assert set(result.extractor.features) == {"user_intent"}
    assert all("reserved" in result.extractor.dropped[c] for c in reserved)
    table = schema.build_table(result, trajs, TARGET)
    assert set(table.df["task"]) == {t.group for t in trajs}  # the CV groups are intact


def test_unplanned_columns_are_dropped_and_types_follow_the_data(tmp_path):
    code = (
        "intent = trajectory['intent']\n"
        "features = {\n"
        "    'user_intent': intent,\n"
        "    'user_intent_word_count': len(intent.split()),\n"
        "    'intent_category': intent + ' and some more words to make a long free text value',\n"
        "    'has_booking': True if 'book' in intent else None,\n"
        "    'whatever_extra': len(intent),\n"
        "}\n"
    )
    plans = [
        FeatureSpec("user_intent", "text"),
        FeatureSpec("intent_category", "categorical"),
        FeatureSpec("has_booking", "computed"),
    ]
    dps = [dp for t in runs() for dp in adapter().decision_points(t, TARGET)]
    compiled = compile_extractor(
        plans,
        dps,
        CodeExtractor(FakeLLM(extractor=lambda p, n: tagged(code))),
        None,
        TARGET,
        known_paths=union_paths(dp.prefix for dp in dps),
        context_text=False,
    )
    assert compiled.dropped["whatever_extra"] == "not in the accepted feature plan"
    assert compiled.features == {
        "user_intent": "text",
        "user_intent_word_count": "num",  # a sub-key of a text feature is typed by its data
        "intent_category": "text",  # a "categorical" plan of long, mostly distinct values is free text
        "has_booking": "num",  # a presence flag (True or missing) is kept, not dropped as constant
    }


# =========================================================================== 2. extraction code
def test_code_without_tags_or_fenced_inside_the_tags_is_recovered(tmp_path):
    def extractor(p, n):
        code = code_for(names_in(p))
        return tagged(f"```python\n{code}```") if "between <code> and </code> tags" in p else "Here:\n" + code

    assert set(discover(tmp_path, FakeLLM(extractor=extractor)).extractor.features) == set(EXPRESSIONS)
    assert extract_code("<code>\n```python\nfeatures = {}\n```\n</code>") == "features = {}"


def test_code_never_tagged_counts_as_a_failed_attempt(tmp_path):
    def extractor(p, n):
        return "I would compute the features like this." if "bad_feature" in p else DEFAULT["extractor"](p, n)

    result = discover(
        tmp_path, FakeLLM(analyzer=with_bad_feature, extractor=extractor), n_analyze=16, workers=4
    )
    bad = [a for a in result.report.analyzed if "bad_feature" in a["proposed"]]
    assert bad and all(a["attempts"] == 4 and not a["kept"] and "no code" in a["error"] for a in bad)
    assert set(result.extractor.features) == set(EXPRESSIONS)


def test_syntax_errors_stop_after_exactly_three_retries(tmp_path):
    def extractor(p, n):
        return tagged("def broken(:\n    pass\n") if "bad_feature" in p else DEFAULT["extractor"](p, n)

    llm = FakeLLM(analyzer=with_bad_feature, extractor=extractor)
    result = discover(tmp_path, llm, n_analyze=16, workers=4)
    bad = [a for a in result.report.analyzed if "bad_feature" in a["proposed"]]
    assert len(bad) == 4  # the weather tasks: the whole proposal of each is infeasible
    assert all(a["attempts"] == 4 and a["kept"] == [] for a in bad)
    assert all("infeasible after 4 attempts" in a["error"] and "syntax" in a["error"] for a in bad)
    # the first attempt and three retries; the other weather points reuse these cached answers
    asked = [p for p in llm.prompts("extractor") if "bad_feature" in p]
    assert len(asked) == 4
    assert [sum(f"Attempt #{k}" in p for p in asked) for k in (2, 3, 4, 5)] == [1, 1, 1, 0]
    last = next(p for p in asked if "Attempt #4" in p)  # every earlier error is fed back (App. D.1)
    assert all(f"[attempt {k}]" in last for k in (1, 2, 3))
    assert "bad_feature" not in result.report.aggregated
    assert set(result.extractor.features) == set(EXPRESSIONS)


def failing_on(condition: str):
    def extractor(p, n):
        return tagged(code_for(names_in(p), f"    if {condition}:\n        raise ValueError('boom')\n"))

    return extractor


def test_code_failing_on_a_few_decision_points_degrades_those_rows(tmp_path):
    llm = FakeLLM(extractor=failing_on("trajectory_data['intent'] == 'play a song from lima'"))  # run 11
    trajs = runs()
    schema = schema_for(tmp_path, llm, compile_min_success=0.9)
    result = schema.discover(trajs, TARGET)
    assert set(result.extractor.features) == set(EXPRESSIONS)
    with pytest.warns(RuntimeWarning, match="failed on 1 of 16"):
        table = schema.build_table(result, trajs, TARGET)
    rows = table.df[table.df["task"] == "task11"]
    assert len(rows) == len(TOOLS) and rows["user_intent"].isna().all()


def test_code_failing_on_many_decision_points_is_a_clear_error(tmp_path):
    llm = FakeLLM(extractor=failing_on("'lima' in trajectory_data['intent']"))
    with pytest.raises(ExtractionError, match=r"after 4 attempts: 5/16 trajectories failed") as err:
        discover(tmp_path, llm, n_analyze=16, workers=4, compile_min_success=0.9)
    assert "rerun with that run directory" in str(err.value)


@pytest.mark.parametrize(
    "value",
    [
        "trajectory_data['intent'].split()",
        "set(trajectory_data['intent'])",
        "object()",
    ],
    ids=["list", "set", "object"],
)
def test_non_scalar_outputs_are_retried(tmp_path, value):
    def extractor(p, n):
        code = code_for(names_in(p))
        if "PREVIOUS ATTEMPT FAILED" not in p:
            code = code.replace("return {", f"return {{'tags': {value}, ")
        return tagged(code)

    result = discover(tmp_path, FakeLLM(extractor=extractor))
    assert set(result.extractor.features) == set(EXPRESSIONS) and result.extractor.attempts == 2


def test_one_slow_trajectory_fails_alone():
    code = "if trajectory['slow']:\n    while True:\n        pass\nfeatures = {'x': trajectory['i']}\n"
    items = [{"slow": i == 3, "i": i} for i in range(7)]
    t0 = time.monotonic()
    res = SubprocessSandbox(timeout_s=1, chunk=3).run_batch(code, items)
    assert time.monotonic() - t0 < 15
    assert [r.ok for r in res] == [i != 3 for i in range(7)] and "timed out" in res[3].error
    assert [r.features for r in res if r.ok] == [{"x": i} for i in range(7) if i != 3]
    attempt = CodeExecutor(SubprocessSandbox(timeout_s=1)).run("while True:\n    pass\n", [{}])
    assert not attempt.ok and "timed out" in attempt.error  # fed back to the CodeExtractor's retry


def test_a_hang_that_ignores_the_alarm_keeps_the_finished_results():
    code = (
        "import signal\n"
        "if trajectory['hang']:\n"
        "    signal.signal(signal.SIGALRM, signal.SIG_IGN)\n"
        "    while True:\n"
        "        pass\n"
        "features = {'x': trajectory['i']}\n"
    )
    items = [{"hang": i == 2, "i": i} for i in range(4)]
    res = SubprocessSandbox(timeout_s=1, chunk=4).run_batch(code, items)
    assert [r.ok for r in res] == [True, True, False, False] and "timed out" in res[2].error
    assert [r.features for r in res[:2]] == [{"x": 0}, {"x": 1}]


ESCAPES = {
    "open a file": "features = {'x': open('/etc/hosts').read()}",
    "pathlib": "import pathlib\nfeatures = {'x': pathlib.Path('/etc/hosts').read_text()}",
    "environment": "import os\nfeatures = {'x': os.environ['OPENAI_API_KEY']}",
    "network": "import urllib.request\nfeatures = {'x': urllib.request.urlopen('http://example.com').read()}",
    "socket": "import socket\nfeatures = {'x': socket.gethostname()}",
    "subprocess": "from subprocess import run\nfeatures = {'x': 1}",
    "dunder import": "features = {'x': __import__('os').getcwd()}",
    "builtins table": "features = {'x': __builtins__['open']('/etc/hosts').read()}",
    "class walk": "features = {'x': len(().__class__.__base__.__subclasses__())}",
    "exit": "exit(0)",
    "eval": "features = eval('{}')",
}


@pytest.mark.parametrize("code", ESCAPES.values(), ids=ESCAPES.keys())
def test_escape_attempts_are_rejected_before_running(code, monkeypatch):
    ran = []
    box = SubprocessSandbox(timeout_s=20)
    monkeypatch.setattr(box, "run_batch", lambda *a: ran.append(1) or [])
    attempt = CodeExecutor(box).run(code, [{"intent": "x", "steps": []}])
    assert not attempt.ok and "violates the extraction rules" in attempt.error and not ran


def test_the_static_checks_have_no_false_alarms_on_parsed_data_and_nested_fields():
    known = union_paths([{"steps": [{"name": "a", "data": "{}", "result": {"success": True}}]}])
    code = (
        "import json\n"
        "out = {}\n"
        "for s in trajectory_data['steps']:\n"
        "    out['n'] = s['name']\n"
        "for s in json.loads(trajectory_data['steps'][0]['data'])['items']:\n"  # `s` now holds parsed JSON
        "    out['m'] = s['price']\n"
        "out['ok'] = trajectory_data['steps'][0]['result']['success']\n"  # a pre-decision tool result
        "out['kind'] = type(out).__name__\n"
        "features = out\n"
    )
    assert checks.check_code(code, known, ("score", "success")) == []
    flagged = checks.check_code(
        "x = trajectory['score']\ny = trajectory['steps'][0]['nme']", known, ("score",)
    )
    assert {(i.kind, i.detail) for i in flagged} == {
        ("outcome_field", "score"),
        ("unknown_field", "steps.[].nme"),
    }


def test_sandbox_never_sees_api_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "t" * 30)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a" * 30)
    code = "import os\nfeatures = {'keys': ','.join(k for k in os.environ if 'KEY' in k or 'TOKEN' in k)}"
    r = SubprocessSandbox(timeout_s=20).run(code, {})
    assert r.ok and r.features == {"keys": ""}


def test_exit_calls_fail_their_item_and_keep_results_aligned():
    code = (
        "import os\n"
        "mode = trajectory['mode']\n"
        "if mode == 'exit':\n    exit(3)\n"
        "if mode == 'raise':\n    raise SystemExit(4)\n"
        "if mode == 'interrupt':\n    raise KeyboardInterrupt\n"
        "if mode == 'os_exit':\n    os._exit(5)\n"
        "features = {'mode': mode}\n"
    )
    modes = ["ok", "exit", "raise", "interrupt", "ok", "os_exit", "ok"]
    res = SubprocessSandbox(timeout_s=20).run_batch(code, [{"mode": m} for m in modes])
    assert [r.ok for r in res] == [True, False, False, False, True, False, False]
    assert res[0].features == res[4].features == {"mode": "ok"}
    assert "SystemExit" in res[2].error and "exited early" in res[6].error


def test_printing_code_cannot_corrupt_the_result_channel():
    code = (
        "import os\n"
        "print(trajectory)\n"
        'print(\'{"ok": true, "features": {"forged": 1}}\')\n'
        "print('{not json')\n"
        'os.write(1, b\'{"ok": false, "error": "forged"}\\n\')\n'
        "features = {'n': trajectory['n']}\n"
        "print(features, '🙂')\n"
    )
    res = SubprocessSandbox(timeout_s=20, chunk=3).run_batch(code, [{"n": i} for i in range(5)])
    assert [r.features for r in res] == [{"n": i} for i in range(5)]


def test_recursion_bomb_fails_cleanly():
    r = SubprocessSandbox(timeout_s=20).run("def f(n):\n    return f(n + 1)\nfeatures = {'x': f(0)}", {})
    assert not r.ok and "RecursionError" in r.error


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is enforced on Linux only")
def test_memory_bomb_is_capped():
    r = SubprocessSandbox(timeout_s=30, mem_mb=512).run(
        "x = bytearray(4 * 1024 ** 3)\nfeatures = {'n': len(x)}", {}
    )
    assert not r.ok and "MemoryError" in r.error


# =========================================================================== 3. judges
def test_meta_judge_cannot_add_or_resurrect_features(tmp_path):
    def aggregator(p, n):
        names = names_in(p)
        return json.dumps(
            {
                "filtered_features": [by_name(x) for x in names if x != "plan_note"],
                "rejected_features": [by_name("plan_note")],
            }
        )

    def meta(p, n):
        names = [*EXPRESSIONS, "magic_feature"]
        return json.dumps([{"feature_name": x, "final_decision": "accept"} for x in names])

    result = discover(tmp_path, FakeLLM(aggregator=aggregator, meta_judge=meta))
    assert result.report.rejected_by_aggregator == ["plan_note"]
    assert set(result.extractor.features) == {"user_intent", "intent_words"}
    assert not {"plan_note", "magic_feature"} & set(result.card.feature_names)
    assert {"plan_note", "magic_feature"} <= set(result.report.dropped)  # reported, not silently ignored


def test_features_renamed_by_the_aggregator_are_reported(tmp_path):
    def aggregator(p, n):
        return json.dumps({"filtered_features": [spec("user_goal"), by_name("plan_note")]})

    result = discover(tmp_path, FakeLLM(aggregator=aggregator))
    assert any("user_goal" in note for note in result.report.notes)  # judged without the analyzer's view
    assert result.report.dropped["user_goal"] == "accepted but not produced by the compiled extractor"
    assert set(result.extractor.features) == {"plan_note"}


def test_conditional_decisions_follow_the_accept_policy(tmp_path):
    decisions = {"plan_note": "Accept (conditional)", "intent_words": "REJECT", "user_intent": "Accept."}

    def meta(p, n):
        return json.dumps([{"feature_name": x, "final_decision": decisions[x]} for x in names_in(p)])

    strict = discover(tmp_path, FakeLLM(meta_judge=meta), run_id="strict", accept=("accept",))
    default = discover(tmp_path, FakeLLM(meta_judge=meta), run_id="default")
    everything = discover(
        tmp_path, FakeLLM(meta_judge=meta), run_id="all", accept=("accept", "conditional", "reject")
    )
    assert set(strict.extractor.features) == {"user_intent"}
    assert set(default.extractor.features) == {"user_intent", "plan_note"}  # conditional ones are kept
    assert set(everything.extractor.features) == {"user_intent", "plan_note", "intent_words"}
    assert {j["feature_name"]: j["final_decision"] for j in strict.report.judged} == {
        "plan_note": "conditional",
        "intent_words": "reject",
        "user_intent": "accept",
    }
    assert [normalize_decision(v) for v in ("conditionally accept", "Accept unconditionally", "Revise")] == [
        "conditional",
        "accept",
        "conditional",
    ]


def test_disagreeing_and_silent_judges(tmp_path):
    def low(p, n):
        return json.dumps({"features": [{"feature_name": x, "score": 1} for x in names_in(p)]})

    def partial(p, n):
        return json.dumps({"features": [{"feature_name": "user_intent", "score": 5}]})

    llm = FakeLLM(judge_relevance=low, judge_generality=lambda p, n: '{"features": []}', judge_impact=partial)
    result = discover(tmp_path, llm)
    rows = {j["feature_name"]: j for j in result.report.judged}
    assert rows["user_intent"]["impact"] == 5 and rows["plan_note"]["impact"] is None
    assert all(r["relevance"] == 1 and r["generality"] is None for r in rows.values())
    assert set(result.extractor.features) == set(EXPRESSIONS)


def test_every_feature_rejected_is_a_clear_error_with_a_report(tmp_path):
    def meta(p, n):
        return json.dumps([{"feature_name": x, "final_decision": "reject"} for x in names_in(p)])

    with pytest.raises(ExtractionError, match="MetaJudge accepted none of the 3") as err:
        discover(tmp_path, FakeLLM(meta_judge=meta))
    assert "conditional" in str(err.value) and "judges.json" in str(err.value)
    run = tmp_path / "runs" / "r1"
    assert (run / "judges.json").exists() and "Judged features" in (run / "report.md").read_text()


def test_an_unparseable_answer_is_not_replayed_on_resume(tmp_path):
    with pytest.raises(LLMFormatError, match="aggregator"):
        discover(tmp_path, FakeLLM(aggregator=lambda p, n: "garbage"))
    good = FakeLLM()
    schema = schema_for(tmp_path, good)  # the same run directory
    result = schema.discover(runs(), TARGET)
    assert set(result.extractor.features) == set(EXPRESSIONS)
    assert good.roles()["aggregator"] == 1 and "analyzer" not in good.roles()
    assert schema.meter.tokens()[0] == len(good.calls)  # only the new calls are paid


# =========================================================================== 4. data
def test_zero_successful_trajectories_fail_before_any_llm_call(tmp_path, no_credentials):
    failed = [Trajectory(t.id, t.raw, False, t.group, t.source) for t in runs(4)]
    llm = FakeLLM()
    with pytest.raises(ValueError, match="no successful trajectories"):
        discover(tmp_path, llm, failed)
    for decision in (TARGET, None):  # no LLM and no credentials: the data error comes first
        with pytest.raises(ValueError, match="no successful trajectories"):
            tabagent.onboard(failed, decision, adapter=adapter(), run_dir=tmp_path / "o")
    assert llm.calls == []


def test_too_few_decision_points_or_tasks_fail_before_any_llm_call(tmp_path):
    llm = FakeLLM()
    with pytest.raises(ValueError, match="at least 2 decision points"):
        discover(tmp_path, llm, runs(1))
    one_task = [Trajectory(t.id, t.raw, True, "the-only-task") for t in runs(4)]
    with pytest.raises(ValueError, match="one task group"):  # grouped CV needs a second task to hold out
        tabagent.onboard(one_task, TARGET, adapter=adapter(), llm=llm, run_dir=tmp_path / "o")
    dry = tabagent.onboard(one_task, TARGET, adapter=adapter(), dry_run=True)
    assert "one task group" in dry.problems[0] and "stop before any LLM call" in str(dry)
    assert llm.calls == []


def test_missing_candidates_or_labels_fail_before_any_llm_call(tmp_path):
    llm = FakeLLM()
    none = adapter(candidates_fn=lambda dp: [])
    with pytest.raises(ValueError, match="has candidates"):
        tabagent.onboard(runs(), TARGET, adapter=none, llm=llm, run_dir=tmp_path / "o")
    no_fn = GenericJsonAdapter(labels_fn=used_tools)
    with pytest.raises(NotImplementedError, match="candidates_fn"):
        tabagent.onboard(runs(), TARGET, adapter=no_fn, llm=llm, run_dir=tmp_path / "o")
    other_names = adapter(labels_fn=lambda dp, raw: {"tool_not_in_catalog"})
    with pytest.raises(ValueError, match="same names"):
        tabagent.onboard(runs(), TARGET, adapter=other_names, llm=llm, run_dir=tmp_path / "o")
    logs = tmp_path / "logs"  # CUGA-like logs without an API catalog: auto-detected, nothing to choose from
    logs.mkdir()
    for i in range(8):
        (logs / f"run{i}.json").write_text(json.dumps(raw_run(i)))
    with pytest.raises(ValueError, match="has candidates"):
        tabagent.onboard(logs, TARGET, llm=llm, run_dir=tmp_path / "o")
    assert llm.calls == []


def test_decision_points_without_candidates_leave_no_rows(tmp_path):
    def some(dp):  # the weather decisions offer no candidates
        return [] if "weather" in dp.prefix["intent"] else tool_candidates(dp)

    schema = schema_for(tmp_path, FakeLLM(), ad=adapter(candidates_fn=some))
    table = schema.build_table(schema.discover(runs(), TARGET), runs(), TARGET)
    assert table.df["decision_id"].nunique() == 12


def test_single_class_decisions_with_several_labels_are_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="one label per decision"):
        tabagent.onboard(
            runs(), TARGET, adapter=adapter(), kind="single", llm=FakeLLM(), run_dir=tmp_path / "o"
        )
    schema = schema_for(tmp_path, FakeLLM())
    result = schema.discover(runs(), TARGET, kind="single")
    with pytest.raises(ValueError, match="one label per decision"):
        schema.build_table(result, runs(), TARGET)


def test_candidates_that_change_across_runs(tmp_path):
    def used(dp):
        return set(TASKS[int(dp.trajectory_id[3:]) % len(TASKS)][1])

    def varying(dp):
        i = int(dp.trajectory_id[3:])
        offered = [c for k, c in enumerate(tool_candidates(dp)) if (k + i) % 5 or c.id in used(dp)]
        return [*offered, Candidate(f"tool_of_run_{i}", f"tool of run {i}", "a tool only this run offers")]

    trajs = runs()
    schema = schema_for(tmp_path, FakeLLM(), ad=adapter(candidates_fn=varying))
    table = schema.build_table(schema.discover(trajs, TARGET), trajs, TARGET)
    sets = table.df.groupby("decision_id")["candidate"].apply(frozenset)
    assert sets.nunique() == len(sets)  # every decision has its own candidate set
    assert (table.df.groupby("decision_id")["chosen"].sum() >= 1).all()


def test_very_long_traces_reach_the_analyzer_clipped(tmp_path):
    def make(i):
        raw = raw_run(i)
        raw["steps"][0]["data"] += " lorem ipsum" * 100_000  # 1.2 MB of step text
        return raw

    llm = FakeLLM()
    result = discover(tmp_path, llm, runs(16, make))
    assert set(result.extractor.features) == set(EXPRESSIONS)  # the extractor still sees the whole prefix
    assert all(len(p) < 450_000 and "more characters]" in p for p in llm.prompts("analyzer"))
    dry = tabagent.onboard(runs(16, make), TARGET, adapter=adapter(), dry_run=True, n_analyze=3)
    assert dry.est_input_tokens < 3 * 150_000  # the estimate counts the clipped prompts


def test_fields_of_later_steps_reach_the_schema_prompts(tmp_path):
    def make(i):
        raw = raw_run(i)
        extra = [{"name": "Tool", "data": "x"}] * 3 + [
            {"name": "Tool", "data": "y", "result": {"status": "ok"}}
        ]
        raw["steps"][2:2] = extra  # a field that only a late pre-decision step has
        return raw

    prefix = adapter().decision_points(runs(1, make)[0], TARGET)[0].prefix
    assert "status" not in json.dumps(get_json_schema(prefix)) and "status" in json.dumps(
        union_schema([prefix])
    )
    llm = FakeLLM()
    discover(tmp_path, llm, runs(16, make))
    assert all("'status'" in p for p in [*llm.prompts("extractor"), *llm.prompts("validator")])


def test_malformed_and_mixed_trajectories(tmp_path):
    odd = [
        Trajectory("nosteps", {"task_id": "x1", "intent": "a"}, True, "x1"),
        Trajectory("nullsteps", {"task_id": "x2", "steps": None}, True, "x2"),
        Trajectory("dictsteps", {"task_id": "x3", "steps": {"name": TARGET}}, True, "x3"),
        Trajectory("junksteps", {"task_id": "x4", "steps": [TARGET, 3, None, [TARGET]]}, True, "x4"),
        Trajectory(
            "mixed",
            {
                "task_id": "x5",
                "intent": "b",
                "steps": ["note", {"name": "Plan", "data": "p"}, {"name": TARGET}],
            },
            True,
            "x5",
        ),
    ]
    ad = adapter()
    dps = [dp for t in odd for dp in ad.decision_points(t, TARGET)]
    assert [(dp.trajectory_id, dp.step_index) for dp in dps] == [("mixed", 2)]
    assert dps[0].prefix["steps"] == ["note", {"name": "Plan", "data": "p"}]
    assert ad.components(odd) == {"Plan": 1, TARGET: 1}
    result = discover(tmp_path, FakeLLM(), runs() + odd[:4])
    assert result.report.n_decision_points == 16 and set(result.extractor.features) == set(EXPRESSIONS)


def test_the_task_id_never_reaches_a_prefix_and_code_reading_it_is_refused():
    ad = adapter(task_keys=None)  # even when every other top-level field is kept
    dps = [dp for t in runs() for dp in ad.decision_points(t, TARGET)]
    assert all("task_id" not in dp.prefix and "intent" in dp.prefix for dp in dps)
    attempt = CodeExecutor(known_paths=union_paths(dp.prefix for dp in dps)).run(
        "features = {'task': trajectory['task_id']}", [dps[0].prefix]
    )
    assert not attempt.ok and "unknown_field: task_id" in attempt.error


def test_a_saved_adapter_rebuilds_the_same_prefixes():
    from tabagent.adapters import adapter_from_description, describe_adapter

    for ad in (adapter(task_keys=("intent", "extra")), CugaAdapter(task_keys=("intent",))):
        raw = {**raw_run(2), "extra": "known before the decision", "final_answer": "x"}
        again = adapter_from_description(json.loads(json.dumps(describe_adapter(ad))))
        assert again.live_prefix(raw) == ad.live_prefix(raw)


def test_generic_logs_outcomes_prefix_fields_and_files(tmp_path):
    logs = tmp_path / "logs"
    (logs / "nested").mkdir(parents=True)
    runs_ = {
        "no_outcome": {k: v for k, v in raw_run(0).items() if k != "score"},
        "failed_flag": {**raw_run(1), "success": False, "final_answer": "booked flight X"},
        "zero_reward": {**{k: v for k, v in raw_run(2).items() if k != "score"}, "reward": 0},
        "passed_word": {**raw_run(3), "success": "passed"},
        "odd_score": {**raw_run(4), "score": "n/a"},
        "dict_score": {**raw_run(5), "score": {"value": 1}},
        "nested/run_in_dir": raw_run(6),
        "my_results_run": raw_run(7),  # a run whose file name mentions results
    }
    for name, raw in runs_.items():
        (logs / f"{name}.json").write_text(json.dumps(raw))
    (logs / "broken.json").write_text('{"steps": [')
    (logs / "list.json").write_text("[1, 2]")
    (logs / "results.json").write_text(json.dumps({"steps": [], "summary": 1}))  # a run summary
    ad = GenericJsonAdapter()
    trajs = {t.id: t for t in ad.load(logs)}
    assert set(trajs) == set(runs_)  # recursive; summaries and unreadable files skipped
    assert {k for k, t in trajs.items() if not t.success} == {
        "failed_flag",
        "zero_reward",
        "odd_score",
        "dict_score",
    }
    assert any("1 of 8 runs have no outcome field" in w for w in ad.warnings)
    (dp,) = ad.decision_points(trajs["failed_flag"], TARGET)
    assert set(dp.prefix) == {"intent", "steps"}  # never final_answer, nor the task id (the CV group)
    assert any("final_answer" in w for w in ad.warnings)
    keep_all = GenericJsonAdapter(task_keys=None)
    (dp,) = keep_all.decision_points(trajs["failed_flag"], TARGET)
    assert "final_answer" in dp.prefix and "success" not in dp.prefix
    (cuga,) = CugaAdapter().decision_points(
        Trajectory("c", {**raw_run(8), "final_answer": "x"}, True), TARGET
    )
    assert set(cuga.prefix) == {"intent", "steps"}


# =========================================================================== 5. operations
class CountingSandbox(SubprocessSandbox):
    """A sandbox that counts the trajectories it runs code on."""

    n_run = 0

    def run_batch(self, code, trajectories):
        CountingSandbox.n_run += len(trajectories)
        return super().run_batch(code, trajectories)


def test_onboard_runs_the_generated_code_in_the_sandbox_it_is_given(tmp_path):
    with pytest.raises(ValueError, match="outside any sandbox"):
        tabagent.onboard(
            runs(), TARGET, adapter=adapter(), llm=FakeLLM(), trust_extractor=True, sandbox="docker"
        )
    box = CountingSandbox(timeout_s=30)
    head = tabagent.onboard(
        runs(),
        TARGET,
        adapter=adapter(),
        llm=FakeLLM(),
        config=config(),
        sandbox=box,
        n_splits=2,
        iterations=20,
    )
    used = CountingSandbox.n_run
    assert used >= 16 and head.sandbox is box  # discovery and the decision table ran in it
    head.decide({"intent": "what is the weather in rome", "steps": []}, list(TOOLS), k=1)
    assert CountingSandbox.n_run == used + 1  # and so does the head


def test_a_column_of_earlier_tool_calls_feeds_the_co_usage_statistic(tmp_path):
    def make(i):
        raw = raw_run(i)
        earlier = TASKS[i % len(TASKS)][1][:1] if i % 2 else []  # half the runs called a tool before
        raw["steps"].insert(2, {"name": "EarlierTools", "calls": earlier})
        return raw

    code = (
        "calls = [c for s in trajectory['steps'] if s['name'] == 'EarlierTools' for c in s['calls']]\n"
        "features = {'tools_called_so_far': ', '.join(calls), 'user_intent': trajectory['intent']}\n"
    )
    llm = FakeLLM(
        analyzer=lambda p, n: json.dumps({"potential_features": [spec("tools_called_so_far"), PROPOSALS[0]]}),
        extractor=lambda p, n: tagged(code),
    )
    trajs = runs(16, make)
    schema = schema_for(tmp_path, llm)
    result = schema.discover(trajs, TARGET)
    (co,) = [f for f in result.card.features if f.source == "fitted:co_usage"]
    assert co.args["context"] == "tools_called_so_far"
    assert result.extractor.quality["user_intent"].get("candidate_share", 0) < 0.8  # free text is no context
    table = schema.build_table(result, trajs, TARGET)
    fitted = TabAgent(result.card, head=TabHead("ranking", iterations=20)).fit(table)
    assert "co_usage" in fitted.card_._fitted


def test_crash_then_resume_reuses_every_paid_call(tmp_path):
    prices = {"fake": (1.0, 2.0)}
    with pytest.raises(ConnectionError):
        discover(tmp_path, FakeLLM(judge_impact=down), prices=prices)
    assert "Judged features" in (tmp_path / "runs" / "r1" / "report.md").read_text()  # a partial report
    second = FakeLLM()
    schema = schema_for(tmp_path, second, prices=prices)
    resumed = schema.discover(runs(), TARGET)
    roles = second.roles()
    assert not {"analyzer", "aggregator", "judge_relevance", "judge_generality"} & set(roles)
    assert roles["judge_impact"] == 1 and roles["meta_judge"] == 1
    assert schema.meter.tokens()[0] == len(second.calls)  # cached answers cost nothing now
    fresh = discover(tmp_path, FakeLLM(), run_id="fresh", prices=prices)
    assert resumed.card.to_dict() == fresh.card.to_dict()  # the card records the cost of all its answers
    assert resumed.card.provenance["cost_usd"] > 0


def test_onboard_reruns_resume_from_the_run_directory(tmp_path):
    run_dir = tmp_path / "run"
    kw = dict(adapter=adapter(), config=config(), run_dir=run_dir, n_splits=2, iterations=20)
    with pytest.raises(ConnectionError):
        tabagent.onboard(runs(), TARGET, llm=FakeLLM(meta_judge=down), **kw)
    assert (run_dir / "config.json").exists()  # run_dir is the run directory itself
    first = FakeLLM()
    head = tabagent.onboard(runs(), TARGET, llm=first, **kw)
    # the rest came from the cache (the auditor runs after the MetaJudge, so it was never reached)
    assert set(first.roles()) == {"meta_judge", "extractor", "validator", "auditor"}
    again = FakeLLM()
    assert tabagent.onboard(runs(), TARGET, llm=again, **kw).card.to_dict() == head.card.to_dict()
    assert again.calls == []  # a complete run directory answers everything


def test_concurrent_runs_on_one_run_directory(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as pool:  # two stores on one directory, as two processes have
        a, b = pool.map(lambda _: discover(tmp_path, FakeLLM()), range(2))
    assert a.card.to_dict() == b.card.to_dict() and a.extractor.code == b.extractor.code
    run = tmp_path / "runs" / "r1"
    assert not list(run.rglob("*.tmp"))
    for f in run.rglob("*.json"):
        json.loads(f.read_text())
    assert all(json.loads(line) for line in (run / "llm_calls.jsonl").read_text().splitlines())
    assert FeatureCard.load(run / "card.yaml").to_dict() == a.card.to_dict()


def test_budget_stop_is_clean_and_resumable(tmp_path):
    prices = {"fake": (1.0, 1.0)}  # USD per 1M tokens
    full = FakeLLM()
    reference = schema_for(tmp_path, full, run_id="full", workers=1, prices=prices)
    reference.discover(runs(), TARGET)
    budget = reference.meter.total_usd / 2
    capped = FakeLLM()
    with pytest.raises(BudgetExceeded, match="would exceed max_cost_usd.*resume") as err:
        discover(tmp_path, capped, run_id="b", workers=4, max_cost_usd=budget, prices=prices)
    run = tmp_path / "runs" / "b"
    assert str(run) in str(err.value) and "LLM cost" in (run / "report.md").read_text()
    assert 0 < len(capped.calls) < len(full.calls)
    spent = sum(len(p) // 4 + len(DEFAULT_TEXT(r, p)) // 4 for r, p in capped.calls) / 1e6
    assert spent <= budget  # no call started that could cross the budget, even with four in flight
    resumed = FakeLLM()
    result = discover(tmp_path, resumed, run_id="b", workers=1, prices=prices)
    assert set(result.extractor.features) == set(EXPRESSIONS)
    assert len(capped.calls) + len(resumed.calls) == len(full.calls)  # every paid answer was kept


def DEFAULT_TEXT(role: str, prompt: str) -> str:
    """The default fake answer to a prompt (for recomputing what a call cost)."""
    return DEFAULT[role if role in DEFAULT else "judge"](prompt, 0)


def test_cost_meter_is_thread_safe_and_reports_cached_answers():
    meter = CostMeter(prices={"m": (1.0, 1.0)})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: meter.record("analyzer", LLMResponse("x", "m", 10, 10)), range(800)))
    assert meter.tokens() == (800, 8000, 8000) and meter.total_usd == pytest.approx(0.016)
    meter.record_cached("judge_impact", LLMResponse("x", "m", 1_000_000, 0))
    assert meter.summary()["cached"] == {
        "calls": 1,
        "input_tokens": 1_000_000,
        "output_tokens": 0,
        "first_paid_usd": 1.0,
    }
    assert "1 more answers reused from the run directory (paid earlier: $1.00)" in meter.describe("m")


def test_dry_run_makes_no_llm_call_and_needs_no_credentials(tmp_path, monkeypatch, no_credentials):
    monkeypatch.setenv("TABAGENT_CACHE", str(tmp_path / "cache"))

    def boom(p, n):
        raise AssertionError("dry_run called the LLM")

    llm = FakeLLM(**{role: boom for role in DEFAULT})
    dry = tabagent.onboard(runs(), TARGET, adapter=adapter(), llm=llm, dry_run=True, redact=lambda r: r)
    assert isinstance(dry, DryRun) and dry.n_decision_points == 16 and llm.calls == [] and not dry.problems
    small = tabagent.onboard(runs(), TARGET, adapter=adapter(), dry_run=True, n_analyze=2)
    large = tabagent.onboard(runs(), TARGET, adapter=adapter(), dry_run=True, n_analyze=8)
    assert (
        small.est_input_tokens < large.est_input_tokens and small.est_output_tokens < large.est_output_tokens
    )
    assert not (tmp_path / "cache").exists()  # no run directory either
    chats = [Trajectory(f"chat{i}", chat_run(i), True, f"task{i}") for i in range(12)]
    assert tabagent.onboard(chats, dry_run=True).components == {TOOL_SHORTLIST: 12}  # format detected


PERSONAL = "Jane Roe 4711"


def scrub(raw: dict) -> dict:
    """The test's redaction function: removes ``PERSONAL`` from a run."""
    return json.loads(json.dumps(raw).replace(PERSONAL, "[REDACTED]"))


def test_redaction_applies_before_any_text_reaches_the_llm(tmp_path):
    def make(i):
        raw = json.loads(json.dumps(raw_run(i)).replace("rome", f"rome {PERSONAL}"))
        raw["steps"][2]["data"] += f" ({PERSONAL})"  # the decision step, which the analyzer reads
        raw["steps"][3]["note"] = PERSONAL  # a later step
        return raw

    llm = FakeLLM()
    out = tmp_path / "head"
    head = tabagent.onboard(
        runs(16, make),
        TARGET,
        adapter=adapter(),
        llm=llm,
        config=config(),
        redact=scrub,
        run_dir=tmp_path / "run",
        n_splits=2,
        iterations=20,
        out=out,
    )
    assert isinstance(head, TabAgentHead) and llm.calls
    assert not any(PERSONAL in p for _, p in llm.calls)
    assert any("[REDACTED]" in p for p in llm.prompts("analyzer"))
    for f in [*(tmp_path / "run").rglob("*"), *out.rglob("*")]:
        if f.is_file() and f.suffix in (".json", ".jsonl", ".yaml", ".md", ".py"):
            assert PERSONAL not in f.read_text(), f
    meta = json.loads((out / "head.json").read_text())  # the head records how its prefixes were built
    assert meta["adapter"]["class"] == "GenericJsonAdapter" and meta["redact"].endswith(":scrub")
    live = {
        "task_id": "live",
        "intent": f"what is the weather in {PERSONAL}",
        "steps": raw_run(1)["steps"][:2],
    }
    assert PERSONAL not in json.dumps(head.live_prefix(live)) and "task_id" not in head.live_prefix(live)


def test_same_inputs_and_answers_give_a_byte_identical_card(tmp_path):
    a = discover(tmp_path / "a", FakeLLM(), run_id="one", workers=4)
    b = discover(tmp_path / "b", FakeLLM(), run_id="two", workers=1)
    pa, pb = a.save(tmp_path / "out_a"), b.save(tmp_path / "out_b")
    assert (pa / "card.yaml").read_bytes() == (pb / "card.yaml").read_bytes()
    assert (pa / "extract.py").read_bytes() == (pb / "extract.py").read_bytes()
    assert (pa / "extractor_tests.json").read_bytes() == (pb / "extractor_tests.json").read_bytes()
    assert str(tmp_path) not in (pa / "card.yaml").read_text()  # no local paths in a shareable card
    assert str(tmp_path) not in (pa / "extractor_tests.json").read_text()
    assert set(a.card.provenance["prompts_sha256"]) == set(prompts.PROMPTS)  # the validator's prompt too


# =========================================================================== 6. LLM backends
class FakeLiteLLMModule:
    """Stands in for the ``litellm`` package: records every ``completion`` call; refuses ``max_tokens`` above
    ``limit`` with Anthropic's error message."""

    def __init__(self, limit: int = 64_000):
        self.calls: list[dict] = []
        self.limit = limit

    def completion(self, **kw):
        self.calls.append(kw)
        if kw["max_tokens"] > self.limit:
            raise RuntimeError(f"max_tokens: {kw['max_tokens']} > {self.limit}, the maximum output tokens")
        choice = SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1))


def fake_openai_client(answers):
    """A client whose ``chat.completions.create`` records its kwargs and replays ``answers`` (an exception is
    raised, a string is the answer text)."""
    calls: list[dict] = []

    def create(**kw):
        calls.append(kw)
        a = answers[len(calls) - 1]
        if isinstance(a, Exception):
            raise a
        choice = SimpleNamespace(message=SimpleNamespace(content=a), finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2))

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def test_litellm_sends_a_user_turn_and_a_per_stage_output_budget(monkeypatch, no_credentials):
    fake = FakeLiteLLMModule()
    monkeypatch.setitem(sys.modules, "litellm", fake)
    lite = LiteLLM("anthropic/claude-sonnet-4-5")
    assert lite.complete("PROMPT", role="analyzer").text == "ok"
    lite.complete("PROMPT", role="meta_judge")
    assert fake.calls[0]["messages"] == [
        {"role": "system", "content": "PROMPT"},
        {"role": "user", "content": USER_TURN},
    ]
    assert [c["max_tokens"] for c in fake.calls] == [16_000, 32_000]
    small = FakeLiteLLMModule(limit=8192)  # an older model: asked again at its own limit
    monkeypatch.setitem(sys.modules, "litellm", small)
    assert LiteLLM("anthropic/claude-3-5-sonnet").complete("PROMPT", role="aggregator").text == "ok"
    assert [c["max_tokens"] for c in small.calls] == [32_000, 8192]
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")  # a provider prefix wins over an OpenAI key
    assert isinstance(auto_llm("anthropic/claude-sonnet-4-5"), LiteLLM)


def test_openai_backend_adapts_to_the_server_and_auto_llm_picks_it(monkeypatch, no_credentials):
    openai = pytest.importorskip("openai")  # the tabschema extra
    import httpx

    oa = OpenAIChatLLM("small-model", base_url="http://localhost:9/v1")  # a keyless compatible server
    assert oa.client.api_key == "EMPTY"
    too_large = openai.BadRequestError(
        "max_tokens is too large: 32000. This model supports at most 16384 completion tokens",
        response=httpx.Response(400, request=httpx.Request("POST", "http://localhost:9/v1")),
        body=None,
    )
    oa.client, calls = fake_openai_client([too_large, "fine"])
    assert oa.complete("PROMPT", role="aggregator").text == "fine"
    assert [c["max_completion_tokens"] for c in calls] == [32_000, 16_384]  # the model's own limit
    assert calls[1]["messages"][-1] == {"role": "user", "content": USER_TURN}

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    plain = auto_llm("openai/gpt-x")
    assert isinstance(plain, OpenAIChatLLM) and plain.model == "gpt-x"
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9/v1")
    served = auto_llm("meta-llama/Llama-3.1-8B-Instruct")  # a slash in a served model's name
    assert isinstance(served, OpenAIChatLLM) and served.client.api_key == "EMPTY"
    monkeypatch.delenv("OPENAI_BASE_URL")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    azure = auto_llm("my-deployment")
    assert type(azure.client).__name__ == "AzureOpenAI" and azure.model == "my-deployment"
    assert azure.client._custom_query == {"api-version": AZURE_API_VERSION}


# =========================================================================== 7. training-inference parity
VIEW = (
    "import json\n"
    "features = {'view': json.dumps(trajectory, sort_keys=True), 'n_steps': len(trajectory['steps'])}\n"
)


def _same_features(train_prefix: dict, live_prefix: dict) -> dict:
    """The extractor's features for both prefixes, sandboxed and in-process; asserts they all agree."""
    outs = [ExtractorRunner(VIEW, trust).run([train_prefix, live_prefix]) for trust in (False, True)]
    assert outs[0] == outs[1] and outs[0][0] == outs[0][1] and outs[0][0] is not None
    return outs[0][0]


@pytest.mark.parametrize("make_adapter", [adapter, CugaAdapter], ids=["generic", "cuga"])
def test_step_logs_parity_and_no_post_decision_reads(make_adapter):
    ad = make_adapter()
    raw = raw_run(0)
    (dp,) = ad.decision_points(Trajectory("r0", raw, True, "t0"), TARGET)
    live = {k: v for k, v in raw.items() if k != "score"}  # at decision time there is no outcome yet
    live["steps"] = raw["steps"][: dp.step_index]
    feats = _same_features(dp.prefix, ad.live_prefix(live))
    assert feats["n_steps"] == 2 and "CodeAgent" not in feats["view"]

    later = copy.deepcopy(raw)  # change the decision, everything after it and every outcome field
    later["steps"][dp.step_index]["data"] = "play_song"
    later["steps"][dp.step_index + 1 :] = [{"name": "CodeAgent", "calls": ["play_song"]}]
    later.update(score=0.0, eval={"passed": False}, reward=-1, outcome="fail", final_answer="done")
    (dp2,) = ad.decision_points(Trajectory("r0", later, True, "t0"), TARGET)
    assert _same_features(dp2.prefix, dp.prefix) == feats


@pytest.mark.parametrize("with_task_id", [True, False])
@pytest.mark.parametrize("target", [TOOL_SHORTLIST, NEXT_TOOL_CALL])
def test_chat_logs_parity_and_no_post_decision_reads(target, with_task_id):
    granularity = "task" if target == TOOL_SHORTLIST else "step"
    ad = OpenAIMessagesAdapter(granularity)
    run = chat_run(0)
    if not with_task_id:
        run.pop("task_id")
    traj = Trajectory("logs/run0", run, True, "t0", "run0.json")
    dp = ad.decision_points(traj, target)[-1]
    calls = [i for i, m in enumerate(run["messages"]) if m.get("tool_calls")]
    live = {k: v for k, v in run.items() if k != "reward"}
    live["messages"] = run["messages"][: calls[0] if target == TOOL_SHORTLIST else calls[-1]]
    feats = _same_features(dp.prefix, ad.live_prefix(live, target))

    later = copy.deepcopy(run)
    later["messages"][-1]["content"] = "a different final answer"
    later.update(reward=0.0, answer="secret gold answer")
    dp2 = OpenAIMessagesAdapter(granularity).decision_points(Trajectory("logs/run0", later, True), target)[-1]
    assert _same_features(dp2.prefix, dp.prefix) == feats


def test_in_process_and_sandboxed_extractors_see_the_same_prefix():
    prefix = {"steps": [{"name": "a", "data": ("x", "y")}], 1: "an int key", "when": datetime(2026, 1, 2)}
    code = "import json\nfeatures = {'view': json.dumps(trajectory, sort_keys=True)}\n"
    assert ExtractorRunner(code).run([prefix]) == ExtractorRunner(code, trust_extractor=True).run([prefix])
    p = {"steps": [1, 2]}
    out = ExtractorRunner("trajectory['steps'].clear()\nfeatures = {'n': 0}\n", trust_extractor=True).run([p])
    assert out == [{"n": 0}] and p == {"steps": [1, 2]}  # the caller's live prefix is never changed
    with pytest.warns(RuntimeWarning, match="failed on 1 of 1"):
        assert ExtractorRunner("features = [1]", trust_extractor=True).run([p]) == [None]


def _ranking_head(code: str) -> TabAgentHead:
    spec_ = DecisionSpec("ranking", "task", "chosen", "candidate", decision=("decision_id",))
    rows = []
    for i in range(24):
        intent, used = TASKS[i % len(TASKS)][0].format(c=CITIES[i % 3]), TASKS[i % len(TASKS)][1]
        for t in TOOLS:
            rows.append(
                {
                    "task": f"t{i}",
                    "decision_id": f"t{i}#0",
                    "intent": intent,
                    "n_words": len(intent.split()),
                    "candidate": t,
                    "candidate_name": t,
                    "chosen": int(t in used),
                }
            )
    card = FeatureCard(
        "toy",
        spec_,
        [
            Feature("intent", "state", "text", "extractor"),
            Feature("n_words", "state", "num", "extractor"),
            Feature("candidate_name", "schema", "cat"),
        ],
    )
    agent = TabAgent(card, head=TabHead("ranking", iterations=20)).fit(
        DecisionTable(pd.DataFrame(rows), spec_)
    )
    return TabAgentHead(agent, TARGET, code, trust_extractor=True)


def test_decide_degrades_like_training_when_the_extractor_fails_or_omits_a_column():
    code = (
        "features = {'intent': trajectory['intent']}\n"
        "if 'skip' not in trajectory:\n"
        "    features['n_words'] = len(trajectory['intent'].split())\n"
    )
    head = _ranking_head(code)
    ok = head.decide({"intent": "what is the weather in rome"}, list(TOOLS), k=1)
    assert ok["candidates"] == ["get_weather"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        omitted = head.decide({"intent": "what is the weather in rome", "skip": 1}, list(TOOLS), k=1)
    assert len(omitted["candidates"]) == 1
    with pytest.warns(RuntimeWarning, match="failed on 1 of 1"):
        failed = head.decide({"no_intent": 1}, list(TOOLS), k=2)
    assert len(failed["candidates"]) == 2 and set(failed["scores"]) == set(TOOLS)


# =========================================================================== content filters and stage models
def filtered(p, n):
    return ("", "content_filter")


def test_content_filtered_answers_are_not_retried_on_the_same_model(tmp_path):
    llm = FakeLLM(extractor=filtered)
    with pytest.raises(Exception, match="content filter") as err:
        discover(tmp_path, llm)
    assert "stage_models" in str(err.value) and "extractor" in str(err.value)
    # one call per analyzed decision point: no code-tag re-ask and no retries with the same model
    assert llm.roles()["extractor"] == config().n_analyze


def test_a_fallback_model_answers_when_the_content_filter_stops_one(tmp_path):
    primary, backup = FakeLLM("primary", extractor=filtered), FakeLLM("backup")
    schema = TabSchema(
        primary,
        config(fallback_model="backup"),
        adapter(),
        store=RunStore(tmp_path / "runs", "r1"),
        stage_llms={"backup": backup},
    )
    result = schema.discover(runs(), TARGET)
    assert set(result.extractor.features) == set(EXPRESSIONS)
    assert backup.roles()["extractor"] == primary.roles()["extractor"]  # each filtered answer, asked once
    calls = [json.loads(line) for line in (schema.store.dir / "llm_calls.jsonl").read_text().splitlines()]
    assert {c.get("fallback_from") for c in calls if c["model"] == "backup"} == {"primary"}


def test_stage_models_route_each_stage_to_its_model(tmp_path):
    main, coder, judge = FakeLLM("main"), FakeLLM("coder"), FakeLLM("judge-model")
    schema = TabSchema(
        main,
        config(stage_models={"extractor": "coder", "judges": "judge-model"}),
        adapter(),
        store=RunStore(tmp_path / "runs", "r1"),
        stage_llms={"coder": coder, "judge-model": judge},
    )
    result = schema.discover(runs(), TARGET)
    assert set(coder.roles()) == {"extractor"}
    assert set(judge.roles()) == {"judge_relevance", "judge_generality", "judge_impact"}
    assert not ({"extractor"} | set(judge.roles())) & set(main.roles())
    assert result.card.provenance["models"]["extractor"] == "coder"
    assert result.card.provenance["models"]["meta_judge"] == "main"


def test_stage_models_are_checked_before_any_call(tmp_path):
    with pytest.raises(ValueError, match="unknown stages"):
        TabSchema(FakeLLM(), config(stage_models={"coder": "x"}), adapter())
    with pytest.raises(ValueError, match="other"):  # a budget needs a price for every model the run may use
        TabSchema(
            FakeLLM("main"),
            config(stage_models={"extractor": "other"}, max_cost_usd=5.0, prices={"main": (1.0, 1.0)}),
            adapter(),
            stage_llms={"other": FakeLLM("other")},
        )


def test_cli_parses_stage_models_prices_and_a_catalog(tmp_path):
    from tabagent import cli

    assert cli._pairs(["extractor=azure/gpt-6-sol", "judges = m"], "--stage-model") == {
        "extractor": "azure/gpt-6-sol",
        "judges": "m",
    }
    assert cli._model_prices(["azure/gpt-6-sol=2,10"]) == {"azure/gpt-6-sol": (2.0, 10.0)}
    with pytest.raises(ValueError, match="KEY=VALUE"):
        cli._pairs(["extractor"], "--stage-model")
    with pytest.raises(ValueError, match="two numbers"):
        cli._model_prices(["m=2"])
    logs = tmp_path / "logs"
    logs.mkdir()
    run = {
        "messages": [
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": "{}"}}]},
        ],
        "reward": 1,
    }
    (logs / "run.json").write_text(json.dumps(run))
    catalog = tmp_path / "tools.json"
    catalog.write_text(
        json.dumps({"get_weather": "Current weather for a city", "send_email": "Send an email"})
    )
    ad = cli._adapter_for(str(logs), str(catalog))
    traj = ad.load(str(logs))[0]
    dp = ad.decision_points(traj, TOOL_SHORTLIST)[0]
    descriptions = {c.id: c.description for c in ad.candidates(dp)}
    assert descriptions["get_weather"] == "Current weather for a city" and "send_email" in descriptions


def test_dry_run_prices_each_stage_at_its_model(tmp_path):
    from tabagent.api import _estimate

    trajs = runs()
    dps = [dp for t in trajs for dp in adapter().decision_points(t, TARGET)][:3]
    one = _estimate(dps, {"a": (1.0, 1.0)}, "a")
    split = _estimate(dps, {"a": (1.0, 1.0), "b": (10.0, 10.0)}, "a", stage_models={"extractor": "b"})
    assert one[:2] == split[:2] and split[2] > one[2]  # same work, the code costs more on "b"
    assert _estimate(dps, {"a": (1.0, 1.0)}, "a", stage_models={"extractor": "b"})[2] is None  # no price


# =========================================================================== the appended context text
def test_every_card_gets_the_context_text_and_its_similarity(tmp_path):
    from tabagent.tabschema.compile import context_texts

    trajs = runs()
    result = discover(tmp_path, FakeLLM(), trajs, context_text=True)
    feats = {f.name: f for f in result.card.features}
    assert feats["context_text"].type == "text" and feats["context_text"].source == "extractor"
    assert result.extractor.specs["context_text"]["added_by"] == "tabagent"  # marked as not LLM-written
    assert "context_candidate_lex" in feats and "context_text" in feats["context_candidate_lex"].args["a"]
    assert "context_text" in result.card.synth_text  # TabSynth paraphrases it
    # the appended code is part of extract.py: the same text in the sandbox and in-process, for every prefix
    dps = [dp for t in trajs for dp in adapter().decision_points(t, TARGET)]
    sandboxed = ExtractorRunner(result.extractor.code).run([dp.prefix for dp in dps])
    in_process = ExtractorRunner(result.extractor.code, trust_extractor=True).run([dp.prefix for dp in dps])
    expected = context_texts([dp.prefix for dp in dps])
    assert [f["context_text"] for f in sandboxed] == [f["context_text"] for f in in_process] == expected


def test_the_context_text_skips_system_prompts_and_keeps_both_ends_of_long_histories():
    from tabagent.tabschema.compile import context_texts

    prefix = {
        "task": "book a flight",
        "steps": [
            {"name": "system", "content": "POLICY " * 50},
            {"name": "user", "content": "first request"},
            *({"name": "assistant", "content": f"step {i} " + "x" * 200} for i in range(40)),
            {"name": "user", "content": "latest message"},
        ],
    }
    (text,) = context_texts([prefix])
    assert "POLICY" not in text
    assert text.startswith("book a flight") and "first request" in text and text.endswith("latest message")
    assert len(text) <= 4000 + len("\n...\n")


def test_the_context_text_never_shadows_an_llm_column_and_is_skipped_when_constant(tmp_path):
    from tabagent.tabschema.compile import CompiledExtractor, _add_context_text

    prefixes = [{"steps": [{"name": "user", "content": f"request {i}"}]} for i in range(4)]
    res = CompiledExtractor("features = {}", {"context_text": "cat"})
    _add_context_text(res, prefixes, {"context_text"})  # the LLM's extractor already has a context_text
    assert res.features["context_text"] == "cat" and res.features["tabagent_context_text"] == "text"
    assert "features['tabagent_context_text']" in res.code
    flat = CompiledExtractor("features = {}", {"a": "num"})
    _add_context_text(flat, [{"steps": [{"name": "Planner", "data": 1}]}] * 4, {"a"})  # no text anywhere
    assert set(flat.features) == {"a"} and "no context text" in flat.notes[0]
    off = discover(tmp_path, FakeLLM(), context_text=False)
    assert "context_text" not in off.extractor.features


def test_likely_duplicate_proposals_reach_the_aggregator_in_the_same_batch():
    from tabagent.tabschema.agents import similar_together

    names = [f"filler_{i}" for i in range(38)] + ["has_user_id", "order_total"] + ["user_id_present"]
    specs = [FeatureSpec.from_dict(spec(n)) for n in names]
    batches = similar_together(specs, 40)
    where = {s.feature_name: i for i, b in enumerate(batches) for s in b}
    assert where["has_user_id"] == where["user_id_present"]  # side by side, for the aggregator to decide
    assert all(len(b) <= 40 for b in batches) and sum(map(len, batches)) == len(specs)
    assert [s.feature_name for s in batches[0]][:3] == ["filler_0", "filler_1", "filler_2"]  # order kept
    big = [FeatureSpec.from_dict(spec(f"x_{i}")) for i in range(90)]
    assert [len(b) for b in similar_together(big, 40)] == [40, 40, 10]


def test_a_column_that_is_false_or_missing_is_constant_but_a_presence_flag_is_kept():
    from tabagent.tabschema.compile import _is_flag

    assert _is_flag(pd.Series([True, None, None, True]))
    assert _is_flag(pd.Series([1, None, 1, None]))
    assert not _is_flag(pd.Series([0.0, None, 0.0, None]))  # "no prior tool" where the key was written
    assert not _is_flag(pd.Series([False, None, None, False]))
    assert not _is_flag(pd.Series([True, False, None]))  # two values: an ordinary column
    assert not _is_flag(pd.Series([True, True]))  # never missing: constant


def test_nested_dict_outputs_are_flattened_not_retried(tmp_path):
    """The CodeExtractor prompt asks for flatten_json on every dict; TabAgent applies it to the output, so a
    nested dict is parent_child columns, the same at training and serving, and costs no retry."""
    calls = []

    def extractor(p, n):
        calls.append(p)
        return tagged(
            code_for(names_in(p)).replace("return {", "return {'user_intent_parts': {'n_words': 2}, ")
        )

    result = discover(tmp_path, FakeLLM(extractor=extractor))
    assert not any("PREVIOUS ATTEMPT FAILED" in p for p in calls)
    assert "user_intent_parts_n_words" in result.extractor.features or "user_intent_parts_n_words" in str(
        result.extractor.dropped
    )
    assert "_tabagent_flatten" in result.extractor.code


def test_compile_repairs_features_the_validator_flags(tmp_path):
    """At compile, the CodeValidator's objections go back to the CodeExtractor once (the retry prompt); the
    repaired code is kept because it runs and fewer accepted features are flagged."""
    state = {"compiling": False}

    def meta(p, n):  # the MetaJudge runs right before compile
        state["compiling"] = True
        return json.dumps([{"feature_name": x, "final_decision": "accept"} for x in names_in(p)])

    def extractor(p, n):
        code = code_for(names_in(p))
        return tagged(code + "\n# repaired" if "CodeValidator found" in p else code)

    def validator(p, n):
        flag = state["compiling"] and "# repaired" not in p
        return json.dumps(
            {
                "features": [
                    {
                        "feature_name": x,
                        "verdict": "plan_mismatch" if flag and x == "plan_note" else "valid",
                        "evidence": "it reads the first plan, the plan says the last",
                    }
                    for x in names_in(p)
                ]
            }
        )

    llm = FakeLLM(meta_judge=meta, extractor=extractor, validator=validator)
    result = discover(tmp_path, llm)
    assert "plan_note" in result.extractor.features and "# repaired" in result.extractor.code
    assert any("repair round fixed 1 of 1" in n for n in result.extractor.notes)
    repair_prompt = next(p for p in llm.prompts("extractor") if "CodeValidator found" in p)
    assert "plan_note: plan_mismatch. it reads the first plan" in repair_prompt

    state["compiling"] = False
    off = discover(
        tmp_path,
        FakeLLM(meta_judge=meta, extractor=extractor, validator=validator),
        run_id="off",
        compile_repair=False,
    )
    assert "plan_note" not in off.extractor.features
