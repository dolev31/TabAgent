"""Chaos tests: every LLM stage of TabSchema meets every kind of bad answer and provider failure.

Each test drives the real pipeline (``TabSchema.discover``, the sandbox, the run directory) with a scripted
fake LLM that misbehaves at one stage: malformed, partial, truncated (``finish_reason`` ``length``),
empty, huge, non-ASCII, duplicate-name and wrong-schema answers, prose instead of JSON, code outside its tags,
content filters, prompts refused as too long, and provider errors (timeouts, 429 with ``Retry-After``, 5xx,
reset connections). Whatever happens, the run must end in a card or a clear error, within a bounded number of
calls, with what went wrong recorded in the run directory: nothing silently wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_robustness import (
    DEFAULT,
    EXPRESSIONS,
    PROPOSALS,
    TARGET,
    FakeLLM,
    adapter,
    code_for,
    config,
    discover,
    names_in,
    runs,
    spec,
    waits,  # noqa: F401 - autouse: backoff is recorded, not slept
)

from tabagent.tabschema import TabSchema
from tabagent.tabschema.llm import (
    LLMContentFiltered,
    LLMFormatError,
    LLMResponse,
    LLMTruncated,
    LLMUnavailable,
    RetryPolicy,
    is_content_policy_error,
    is_prompt_too_long,
    is_timeout,
    is_transient,
    response_of,
    retry_after,
)
from tabagent.tabschema.runstore import RunStore, read_jsonl

STAGES = (
    "analyzer",
    "extractor",
    "validator",
    "aggregator",
    "judge_relevance",
    "judge_generality",
    "judge_impact",
    "meta_judge",
    "auditor",
)


class ProviderError(Exception):
    """A provider's HTTP error, as the OpenAI and LiteLLM clients raise them (``status_code``,
    ``response``)."""

    def __init__(self, status: int, message: str = "", headers: dict | None = None):
        super().__init__(message or f"HTTP {status}")
        self.status_code = status
        self.response = SimpleNamespace(status_code=status, headers=headers or {})


class ContentPolicyViolationError(Exception):
    """LiteLLM's name for a prompt the provider's filter refuses (status 400)."""

    status_code = 400


# --------------------------------------------------------------------------- faults
def answer(text: str, finish: str = "stop"):
    return lambda p, n: (text, finish)


def raising(make):
    def handler(p, n):
        raise make()

    return handler


def first(k: int, bad, role: str):
    """``bad`` for the first ``k`` calls of the role, the default answer after."""

    def handler(p, n):
        return bad(p, n) if n < k else DEFAULT[role if role in DEFAULT else "judge"](p, n)

    return handler


def huge(role: str):
    """An enormous answer for the role: thousands of items, or megabytes of text."""

    def handler(p, n):
        if role == "analyzer":
            return json.dumps({"potential_features": PROPOSALS + [spec(f"f{i}") for i in range(5000)]})
        if role == "extractor":
            return "<thoughts>" + "x" * 3_000_000 + "</thoughts><code>" + code_for(names_in(p)) + "</code>"
        if role == "aggregator":
            invented = [spec(f"invented_{i}") for i in range(3000)]
            return json.dumps({"filtered_features": [*map(spec, names_in(p)), *invented]})
        if role == "meta_judge":
            names = [*names_in(p), *(f"invented_{i}" for i in range(3000))]
            return json.dumps([{"feature_name": x, "final_decision": "accept"} for x in names])
        items = [
            {"feature_name": x, "verdict": "valid", "score": 5, "evidence": "e" * 200_000}
            for x in names_in(p)
        ]
        return json.dumps({"features": items})

    return handler


def duplicated(role: str):
    """Every item twice: the first time as the default answer gives it, then contradicting it."""

    def handler(p, n):
        base = DEFAULT[role if role in DEFAULT else "judge"](p, n)
        if role == "extractor":
            return base
        data = json.loads(base)
        items = data if isinstance(data, list) else next(v for v in data.values() if isinstance(v, list))
        flipped = [
            {**x, "score": 1, "final_decision": "reject", "verdict": "incorrect"}
            if isinstance(x, dict)
            else x
            for x in items
        ]
        if isinstance(data, list):
            return json.dumps(items + flipped)
        key = next(k for k, v in data.items() if isinstance(v, list))
        return json.dumps({**data, key: items + flipped})

    return handler


def unicode(role: str):
    def handler(p, n):
        base = DEFAULT[role if role in DEFAULT else "judge"](p, n)
        return "🙂 ünïcödé 特徴 réponse:\n" + base + "\n— ✓ 完了 ​﻿"

    return handler


FAULTS = {
    "malformed": lambda role: answer('{"features": [{"feature_name": "x", "verdict": }'),
    "partial": lambda role: lambda p, n: (DEFAULT[role if role in DEFAULT else "judge"](p, n)[:40], "stop"),
    "truncated": lambda role: (
        lambda p, n: (DEFAULT[role if role in DEFAULT else "judge"](p, n)[:40], "length")
    ),
    "empty": lambda role: answer(""),
    "prose": lambda role: answer("I reviewed the features carefully; they all look reasonable to me."),
    "wrong_schema": lambda role: answer('{"result": {"ok": true}, "items": "all of them"}'),
    "filtered": lambda role: answer("", "content_filter"),
    "policy_error": lambda role: raising(lambda: ContentPolicyViolationError("content_filter: refused")),
    "too_long": lambda role: raising(
        lambda: ProviderError(400, "This model's maximum context length is 8192")
    ),
    "rate_limited": lambda role: raising(lambda: ProviderError(429, "slow down", {"retry-after": "7"})),
    "timeout": lambda role: raising(lambda: TimeoutError("read timed out")),
    "server_error": lambda role: raising(lambda: ProviderError(503, "overloaded")),
    "reset": lambda role: raising(lambda: ConnectionResetError("connection reset by peer")),
    "auth": lambda role: raising(lambda: ProviderError(401, "invalid api key")),
    "huge": huge,
    "duplicates": duplicated,
    "unicode": unicode,
}
TRANSIENT = ("rate_limited", "server_error", "reset", "empty")
#: timed out twice: handled like an answer cut at the output limit
UNUSABLE = (
    "malformed",
    "partial",
    "truncated",
    "timeout",
    "prose",
    "wrong_schema",
    "filtered",
    "policy_error",
)
HARMLESS = ("huge", "duplicates", "unicode")


def evidence(run: Path) -> str:
    """Everything the run directory records about what went wrong, as one string."""
    parts = []
    for name in ("run.json", "analysis.json", "aggregator.json", "judges.json", "compile.json", "audit.json"):
        f = run / name
        if f.exists():
            parts.append(f.read_text())
    for r in read_jsonl(run / "llm_calls.jsonl"):
        if r.get("event") or r.get("finish_reason") not in ("stop", "", None):
            parts.append(json.dumps(r))
    if (run / "report.md").exists():
        parts.append((run / "report.md").read_text())
    return "\n".join(parts)


def failure_recorded(run: Path, role: str) -> bool:
    """Whether the run directory names a failure of ``role``'s calls (an event, an error, a note)."""
    summary = json.loads((run / "run.json").read_text())
    stages = json.dumps(summary.get("stages", {}))
    failed_calls = [
        r for r in read_jsonl(run / "llm_calls.jsonl") if r.get("stage") == role and r.get("event")
    ]
    keywords = ("failures", "events", "error", "note", "skipped", "unaggregated")
    return (
        bool(failed_calls) or (role in stages and any(k in stages for k in keywords)) or role in evidence(run)
    )


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    """The calls and card of a clean run."""
    base = tmp_path_factory.mktemp("clean")
    llm = FakeLLM()
    result = discover(base, llm)
    return llm.roles(), result


# --------------------------------------------------------------------------- the matrix
def expected(role: str, fault: str):
    """What a run must do when ``role`` always meets ``fault``: ``"card"`` (degrade, recorded) or an
    exception type (a clear error)."""
    if fault in TRANSIENT:
        return LLMUnavailable
    if fault == "auth":
        return ProviderError  # not retried, at any stage (the audit included): the run stops and says so
    if fault in HARMLESS:
        return "card"
    if role in ("validator", "auditor") or role.startswith("judge_"):
        return "card"  # a validator without verdicts, an audit skipped, a judge without scores: recorded
    if role == "aggregator":
        return LLMContentFiltered if fault in ("filtered", "policy_error") else LLMFormatError
    if role == "meta_judge":
        return LLMContentFiltered if fault in ("filtered", "policy_error") else LLMFormatError
    return RuntimeError  # analyzer, extractor: every point fails, and the error names the first cause


@pytest.mark.parametrize("fault", list(FAULTS))
@pytest.mark.parametrize("role", STAGES)
def test_every_stage_meets_every_fault(tmp_path, role, fault, baseline, waits):  # noqa: F811 - the imported fixture
    clean, reference = baseline
    llm = FakeLLM(**{role: FAULTS[fault](role)})
    want = expected(role, fault)
    run = tmp_path / "runs" / "r1"
    if want == "card":
        result = discover(tmp_path, llm)
        assert result.extractor.features and set(result.extractor.features) <= set(EXPRESSIONS)
        if fault not in HARMLESS:
            assert failure_recorded(run, role), f"{role} {fault}: nothing recorded"
        summary = json.loads((run / "run.json").read_text())
        assert summary["status"] == "completed"
    else:
        with pytest.raises(want) as err:
            discover(tmp_path, llm)
        notes = " ".join(getattr(err.value, "__notes__", []))
        if sys.version_info >= (3, 11):  # a provider's own error carries the run directory as a note
            assert str(run) in str(err.value) + notes or isinstance(err.value, RuntimeError)
        summary = json.loads((run / "run.json").read_text())
        assert summary["status"] in ("failed", "stopped") and summary["error"]
        assert (run / "report.md").exists()
        if want is LLMUnavailable:
            assert summary["status"] == "stopped"
            retries = [r for r in read_jsonl(run / "llm_calls.jsonl") if r.get("event") == "retry"]
            if fault != "empty":  # an empty answer is logged as a call, then retried
                assert retries and all(r["stage"] == role for r in retries)
            assert len(waits) >= RetryPolicy().attempts - 1  # backed off before every retry
            if fault == "rate_limited":
                assert 7.0 in waits  # the provider's Retry-After
    # bounded cost: at most ten times the clean run's calls, whatever the fault
    assert sum(llm.roles().values()) <= 10 * sum(clean.values()) + 3 * RetryPolicy().attempts


@pytest.mark.parametrize("fault", TRANSIENT)
@pytest.mark.parametrize(
    "role", ["analyzer", "aggregator", "judge_impact", "meta_judge", "extractor", "auditor"]
)
def test_a_transient_failure_is_retried_and_the_run_ends_as_a_clean_one(
    tmp_path,
    role,
    fault,
    baseline,
    waits,  # noqa: F811 - the imported fixture
):
    _, reference = baseline
    llm = FakeLLM(**{role: first(2, FAULTS[fault](role), role)})
    result = discover(tmp_path, llm)
    assert result.card.to_dict()["features"] == reference.card.to_dict()["features"]
    calls = read_jsonl(tmp_path / "runs" / "r1" / "llm_calls.jsonl")
    if fault != "empty":
        retries = [r for r in calls if r.get("event") == "retry"]
        assert len(retries) >= 1 and retries[0]["stage"] == role and retries[0]["attempt"] == 1
    assert waits and all(w >= 1 for w in waits)  # never a wait under a second


@pytest.mark.parametrize("role", ["analyzer", "aggregator", "meta_judge", "extractor"])
def test_a_timeout_is_retried_once(tmp_path, role, baseline, waits):  # noqa: F811 - the imported fixture
    """A call that times out once is retried; twice in a row on one prompt is handled like a truncated
    answer (the batch is split, the point skipped): a long answer would time out every time."""
    _, reference = baseline
    llm = FakeLLM(**{role: first(1, FAULTS["timeout"](role), role)})
    result = discover(tmp_path, llm)
    assert result.card.to_dict()["features"] == reference.card.to_dict()["features"]
    calls = [r for r in read_jsonl(tmp_path / "runs" / "r1" / "llm_calls.jsonl") if r.get("event")]
    assert [c["event"] for c in calls] == ["retry"] and calls[0]["stage"] == role


def test_a_batch_that_times_out_twice_is_split_like_a_truncated_one(tmp_path):
    def aggregator(p, n):  # the full batch of three times out every time; halves answer
        if len(names_in(p)) >= 3:
            raise TimeoutError("read timed out")
        return DEFAULT["aggregator"](p, n)

    result = discover(tmp_path, FakeLLM(aggregator=aggregator))
    assert set(result.extractor.features) == set(EXPRESSIONS)
    events = json.loads((tmp_path / "runs" / "r1" / "run.json").read_text())["stages"]["aggregate"]
    assert "aggregator: split" in events["events"]


def test_a_long_retry_after_or_an_exhausted_quota_stops_at_once(tmp_path, waits):  # noqa: F811
    hours = FakeLLM(aggregator=raising(lambda: ProviderError(429, "slow down", {"retry-after": "7200"})))
    with pytest.raises(LLMUnavailable, match="asked to wait 7200s"):
        discover(tmp_path, hours)
    assert hours.roles()["aggregator"] == 1 and not waits  # no hot loop, no hours of waiting
    quota = FakeLLM(aggregator=raising(lambda: ProviderError(429, "insufficient_quota: check your plan")))
    with pytest.raises(ProviderError, match="insufficient_quota"):
        discover(tmp_path, quota, run_id="r2")
    assert quota.roles()["aggregator"] == 1  # not transient: never retried
    zero = FakeLLM(aggregator=first(1, raising(lambda: ProviderError(429, "x", {"retry-after": "0"})), "a"))
    discover(tmp_path, zero, run_id="r3")
    assert waits == [1.0]  # a Retry-After of 0 still waits a second


@pytest.mark.parametrize("fault", ["malformed", "truncated", "prose", "filtered", "too_long", "timeout"])
@pytest.mark.parametrize("role", ["aggregator", "judge_generality", "meta_judge"])
def test_a_batch_whose_answer_fails_is_asked_again_in_halves(tmp_path, role, fault):
    """A whole-run stage answers for a batch of features; when that answer fails, each half is asked again,
    and the run completes. Here only the full batch of three fails."""

    def handler(p, n):
        if len(names_in(p)) >= 3:
            return FAULTS[fault](role)(p, n)
        return DEFAULT[role if role in DEFAULT else "judge"](p, n)

    llm = FakeLLM(**{role: handler})
    result = discover(tmp_path, llm)
    assert set(result.extractor.features) == set(EXPRESSIONS)
    events = json.loads((tmp_path / "runs" / "r1" / "run.json").read_text())
    assert "split" in json.dumps(events["stages"])
    assert llm.roles()[role] <= 2 * (1 + 2 + 4)  # the batch, its halves and quarters, each re-asked once


def test_analyzer_prompts_too_long_are_clipped_harder_then_the_point_is_lost(tmp_path):
    sizes = []

    def analyzer(p, n):
        sizes.append(len(p))
        if "weather" in p and len(p) > 1_500:
            raise ProviderError(400, "prompt is too long: 200000 tokens > 8192 maximum")
        return DEFAULT["analyzer"](p, n)

    result = discover(tmp_path, FakeLLM(analyzer=analyzer), n_analyze=8, max_prompt_chars=4_000)
    lost = [a for a in result.report.analyzed if a["error"]]
    assert lost and all("LLMPromptTooLong" in a["error"] for a in lost)
    events = (tmp_path / "runs" / "r1" / "run.json").read_text()
    assert "clipped" in events  # the prompt was sent again with the prefix fit to half the budget
    assert set(result.extractor.features) == set(EXPRESSIONS)


def test_the_rate_limit_wait_follows_retry_after_and_is_capped(waits):  # noqa: F811 - the imported fixture
    assert retry_after(ProviderError(429, headers={"retry-after": "7"})) == 7.0
    assert retry_after(ProviderError(429, headers={"retry-after-ms": "1500"})) == 1.5
    assert retry_after(ProviderError(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})) == 0.0
    assert retry_after(ProviderError(429)) is None
    policy = RetryPolicy()
    assert policy.wait(1, 10_000) == policy.max_retry_after_s  # a provider cannot stall the run for hours
    assert policy.wait(1, 0) == policy.wait(1, 0.01) == 1.0  # nor make it spin
    assert all(1.5 <= policy.wait(1) <= 2.5 for _ in range(50))  # 2 s +-25%
    assert all(policy.wait(k) <= policy.max_s * 1.25 for k in range(1, 20))


def test_provider_errors_are_sorted():
    assert all(
        is_transient(e)
        for e in (ProviderError(429), ProviderError(503), TimeoutError(), ConnectionResetError())
    )
    assert not is_transient(ProviderError(429, "You exceeded your current quota: insufficient_quota"))
    assert is_timeout(TimeoutError()) and is_timeout(ProviderError(408))
    assert not is_timeout(ConnectionResetError()) and not is_timeout(ProviderError(503))
    assert not any(
        is_transient(e) for e in (ProviderError(400), ProviderError(401), ValueError("x"), KeyError("y"))
    )
    assert is_prompt_too_long(ProviderError(400, "maximum context length is 128000 tokens"))
    assert not is_prompt_too_long(ProviderError(400, "unsupported parameter"))
    assert is_content_policy_error(ContentPolicyViolationError("refused"))
    assert is_content_policy_error(
        ProviderError(400, "The response was filtered due to the content management policy")
    )
    openai = pytest.importorskip("openai")
    import httpx

    request = httpx.Request("POST", "http://localhost:9/v1")
    limited = openai.RateLimitError(
        "rate", response=httpx.Response(429, headers={"retry-after": "3"}, request=request), body=None
    )
    assert is_transient(limited) and retry_after(limited) == 3.0
    assert is_transient(openai.APIConnectionError(request=request))
    assert is_transient(openai.APITimeoutError(request=request))
    bad = openai.BadRequestError("bad", response=httpx.Response(400, request=request), body=None)
    assert not is_transient(bad)


def test_litellm_errors_are_sorted():
    litellm = pytest.importorskip("litellm")
    ex = litellm.exceptions
    assert is_transient(ex.RateLimitError("r", llm_provider="x", model="m"))
    assert is_transient(ex.Timeout("t", model="m", llm_provider="x"))
    assert is_transient(ex.ServiceUnavailableError("s", llm_provider="x", model="m"))
    assert is_prompt_too_long(ex.ContextWindowExceededError("too long", model="m", llm_provider="x"))
    assert is_content_policy_error(ex.ContentPolicyViolationError("no", model="m", llm_provider="x"))
    assert not is_transient(ex.BadRequestError("b", model="m", llm_provider="x"))


def test_odd_provider_responses_become_answers_the_stages_can_judge():
    empty = response_of(SimpleNamespace(choices=[], usage=None), "m", 0.1)
    assert empty.text == "" and empty.finish_reason == "empty"
    refusal = SimpleNamespace(
        message=SimpleNamespace(content=None, refusal="I can't help"), finish_reason="stop"
    )
    assert response_of(SimpleNamespace(choices=[refusal], usage=None), "m", 0).finish_reason == "refusal"
    parts = SimpleNamespace(message=SimpleNamespace(content=[{"type": "text", "text": "a"}, {"text": "b"}]))
    usage = SimpleNamespace(prompt_tokens=None, completion_tokens=3)
    r = response_of(SimpleNamespace(choices=[parts], usage=usage), "m", 0)
    assert r.text == "ab" and r.input_tokens == 0 and r.output_tokens == 3


@pytest.mark.parametrize(
    "style, attempts",
    [("py_fence", 1), ("thoughts_mention_tag", 1), ("bare", 2)],
)
def test_code_outside_its_tags_is_recovered_or_counted_as_a_failed_attempt(tmp_path, style, attempts):
    def extractor(p, n):
        code = code_for(names_in(p))
        if "between <code> and </code> tags" in p:
            return "Sure, here it is:\n" + code  # still no tags and no fence: a failed attempt
        if "PREVIOUS ATTEMPT FAILED" in p:
            return f"<code>{code}</code>"
        return {
            "py_fence": f"Here:\n```py\n{code}```\nDone.",
            "thoughts_mention_tag": f"<thoughts>I put it in <code> tags</thoughts>\n<code>{code}</code>",
            "bare": "Here is the code:\n" + code,
        }[style]

    result = discover(tmp_path, FakeLLM(extractor=extractor))
    assert set(result.extractor.features) == set(EXPRESSIONS)
    assert {a["attempts"] for a in result.report.analyzed} == {attempts}


def test_a_prompt_the_filter_refuses_goes_to_the_fallback_model_once_and_is_recorded(tmp_path):
    primary, backup = FakeLLM("primary", aggregator=FAULTS["policy_error"]("aggregator")), FakeLLM("backup")
    schema = TabSchema(
        primary,
        config(fallback_model="backup"),
        adapter(),
        store=RunStore(tmp_path / "runs", "r1"),
        stage_llms={"backup": backup},
    )
    result = schema.discover(runs(), TARGET)
    assert set(result.extractor.features) == set(EXPRESSIONS)
    calls = read_jsonl(tmp_path / "runs" / "r1" / "llm_calls.jsonl")
    assert any(c.get("event") == "content_policy" and c["model"] == "primary" for c in calls)
    assert any(c.get("fallback_from") == "primary" and c["stage"] == "aggregator" for c in calls)


def test_meter_reservations_use_the_model_tokenizer():
    from tabagent.tabschema.llm import CostMeter, chars_per_token

    assert chars_per_token("aws/claude-opus-5-5") < 3 < chars_per_token("gpt-5.4")
    meter = CostMeter(prices={"anthropic/claude-x": (1.0, 0.0)}, max_cost_usd=100.0)
    est = meter.reserve("analyzer", "anthropic/claude-x", 230_000)
    assert est == pytest.approx(0.1, rel=0.01)  # 100k tokens at 2.3 characters per token
    meter.release(est)
    for _ in range(3):  # measured on the run's own calls: 5 characters per token here
        meter.record("analyzer", LLMResponse("x", "anthropic/claude-x", 10_000, 0), prompt_chars=50_000)
    assert meter.chars_per_token("anthropic/claude-x") == pytest.approx(5.0)


def test_truncated_and_filtered_answers_are_kept_and_replayed_without_paying(tmp_path):
    """A paid answer is never paid for twice: a truncated or filtered answer is stored with its finish reason,
    and a resumed run meets the same outcome from the run directory."""
    with pytest.raises(LLMTruncated):
        discover(tmp_path, FakeLLM(meta_judge=FAULTS["truncated"]("meta_judge")))
    # the run stopped on the truncation: those answers are dropped so that a rerun (with a larger output
    # limit, say) asks again; everything else is replayed
    again = FakeLLM()
    discover(tmp_path, again)
    assert set(again.roles()) == {"meta_judge", "extractor", "validator", "auditor"}
    assert again.roles()["meta_judge"] == 1
