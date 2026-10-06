"""LLM access for TabSchema: provider-agnostic, recordable, replayable, and metered.

Every TabSchema stage sends its prompt as a system message (paper App. D), followed by a one-line user turn
(``USER_TURN``) because some providers (Anthropic) reject a request without a non-system message. Any
OpenAI-compatible endpoint works (OpenAI, Azure OpenAI, vLLM, Ollama, a LiteLLM proxy) through
``OpenAIChatLLM``; other providers work through LiteLLM (``pip install "tabagent[llm]"``). ``auto_llm``
picks a backend from the model name and the environment, so onboarding needs nothing but an API key.

The stages that answer for many features at once (``WHOLE_RUN_ROLES``: the FeatureAggregator, the three judges
and the MetaJudge) get a larger output budget than the per-decision-point stages. An answer that stops at the
output limit is incomplete, and raises ``LLMTruncated`` instead of being parsed.

``RecordingLLM`` stores every response; ``ReplayLLM`` serves them back without network access (a replay miss
is an error, never a silent live call). This is how the discovery pipeline runs deterministically in tests.
``CostMeter`` counts the tokens of every call and prices them when the model's price is known, so the
one-time onboarding cost (Sec. 3.1) can be measured and capped.

Provider failures are sorted here (:func:`is_transient`, :func:`is_prompt_too_long`,
:func:`is_content_policy_error`): a rate limit, a timeout, a dropped connection or a server error is retried
by the TabSchema stage that made the call, with exponential backoff that honors ``Retry-After``
(:class:`RetryPolicy`); a filtered prompt follows the content-filter path; a prompt the model cannot take is
answered by splitting the work or clipping the prompt; anything else is not retried. The backends themselves
retry nothing transient, so every attempt is visible in the run directory's ``llm_calls.jsonl``.
"""

from __future__ import annotations

import email.utils
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

DEFAULT_MODEL = "gpt-5.4"  # the model the paper runs TabSchema on (App. D)
#: The user turn sent after every stage prompt (the prompt itself is the system message).
USER_TURN = "Follow the instructions above."
#: Stages whose answer covers every feature of the run, and so get ``whole_run_max_tokens``.
WHOLE_RUN_ROLES = frozenset(
    {"aggregator", "judge_relevance", "judge_generality", "judge_impact", "meta_judge"}
)
#: Model-name prefixes that ``auto_llm`` sends to LiteLLM (``anthropic/claude-...``) rather than to an
#: OpenAI-compatible endpoint.
LITELLM_PROVIDERS = frozenset(
    {
        "anthropic",
        "azure_ai",
        "bedrock",
        "cohere",
        "deepseek",
        "fireworks_ai",
        "gemini",
        "groq",
        "huggingface",
        "mistral",
        "ollama",
        "openrouter",
        "together_ai",
        "vertex_ai",
        "xai",
    }
)
AZURE_API_VERSION = "2024-10-21"  # the default api-version for classic Azure OpenAI endpoints


@dataclass
class LLMResponse:
    """One completion: its text, the model that produced it, token counts, wall-clock latency, why it
    stopped (``length`` when it hit the output limit) and the output limit it was asked with (``max_tokens``,
    when the backend knows it)."""

    text: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str = ""
    max_tokens: int | None = None


class LLM(Protocol):
    """What TabSchema needs from a model: a name and one system-prompt completion."""

    model: str

    def complete(self, system: str, *, role: str = "") -> LLMResponse: ...


class BudgetExceeded(RuntimeError):
    """Raised when onboarding would exceed the configured ``max_cost_usd``."""


class MissingCredentials(RuntimeError):
    """Raised by ``auto_llm`` when no LLM provider key is set in the environment."""


class LLMFormatError(ValueError):
    """The model's answer could not be parsed after the allowed repair attempts."""


class LLMTruncated(LLMFormatError):
    """The model stopped at its output-token limit, so its answer is incomplete."""


class LLMContentFiltered(LLMFormatError):
    """The provider's content filter stopped the model's answer (``finish_reason`` ``content_filter``): the
    same prompt to the same model would be stopped again, so it is not retried on that model."""


class LLMPromptTooLong(LLMFormatError):
    """The provider refused the prompt as longer than the model's context window. A stage that batches its
    work asks again for each half of the batch; one that reads a trajectory clips it harder."""


class LLMTimedOut(LLMFormatError):
    """A call timed out twice: the answer takes longer than the request timeout (``request_timeout``). It is
    handled like an answer cut at the output limit: a whole-run stage asks again for each half of the batch,
    and a decision point is skipped. The provider may have billed the timed-out attempts."""


class LLMUnavailable(RuntimeError):
    """The provider kept failing with transient errors (rate limits, dropped connections, server errors)
    through every retry of :class:`RetryPolicy`, or asked to wait longer than the policy allows. Every answer
    paid so far is in the run directory: rerunning with it resumes. (A failed attempt is usually not billed;
    one that timed out may have been.)"""


#: HTTP statuses that mean "try again later": request timeouts, conflicts, rate limits and server errors.
TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 524, 529})
#: Exception class names (anywhere in the class's MRO) that providers and HTTP clients raise for transient
#: failures without an HTTP status (connection errors, client-side timeouts).
TRANSIENT_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "Timeout",
        "TimeoutException",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",
        "RateLimitError",
        "ServiceUnavailableError",
        "InternalServerError",
        "OverloadedError",
    }
)
_TOO_LONG = re.compile(
    r"context[ _-]?(length|window)|maximum context|prompt is too long|too many (input )?tokens|"
    r"input is too long|reduce the length of the messages|exceeds? the (model's )?(maximum|context)"
)
_POLICY = re.compile(r"content[ _-]?(filter|policy|management)|responsible ?ai|safety system")


def status_code(e: BaseException) -> int | None:
    """The HTTP status of a provider error (on the error or its ``response``), or None."""
    for obj in (e, getattr(e, "response", None)):
        for attr in ("status_code", "status", "http_status"):
            v = getattr(obj, attr, None)
            if isinstance(v, int) and not isinstance(v, bool):
                return v
    return None


def is_prompt_too_long(e: BaseException) -> bool:
    """Whether a provider error says the prompt exceeds the model's context window (a 400 or 413, or
    LiteLLM's ``ContextWindowExceededError``)."""
    if isinstance(e, LLMPromptTooLong):
        return True
    if any(c.__name__ == "ContextWindowExceededError" for c in type(e).__mro__):
        return True
    return status_code(e) in (400, 413) and bool(_TOO_LONG.search(str(e).lower()))


def is_content_policy_error(e: BaseException) -> bool:
    """Whether a provider error is its content filter refusing the prompt (LiteLLM's
    ``ContentPolicyViolationError``; an Azure OpenAI 400 with code ``content_filter``)."""
    if any(c.__name__ == "ContentPolicyViolationError" for c in type(e).__mro__):
        return True
    code = str(getattr(e, "code", "") or "").lower()
    return status_code(e) == 400 and ("content_filter" in code or bool(_POLICY.search(str(e).lower())))


#: Exception class names of a request that ran out of time (not of a connection that could not be made).
TIMEOUT_NAMES = frozenset({"APITimeoutError", "Timeout", "TimeoutException", "ReadTimeout", "WriteTimeout"})


def is_timeout(e: BaseException) -> bool:
    """Whether a failed call ran out of time waiting for the answer (``TimeoutError``, a client's read
    timeout, or HTTP 408). A long answer can do this every time, so it is retried once and then handled like a
    truncated answer (``LLMTimedOut``)."""
    if isinstance(e, TimeoutError) and not isinstance(e, ConnectionError):
        return True
    if any(c.__name__ in TIMEOUT_NAMES for c in type(e).__mro__):
        return True
    return status_code(e) == 408


def is_transient(e: BaseException) -> bool:
    """Whether a failed LLM call is worth retrying unchanged: a rate limit, a timeout, a dropped or reset
    connection, an overloaded or failing server (an HTTP status in ``TRANSIENT_STATUS``, or one of the
    connection errors in ``TRANSIENT_NAMES``). A bad request, a refused prompt, an authentication error, a
    budget stop, or a 429 for an exhausted quota (``insufficient_quota``: retrying cannot help) is not."""
    if isinstance(e, (BudgetExceeded, LLMFormatError, LLMUnavailable)):
        return False
    if "insufficient_quota" in f"{getattr(e, 'code', '')} {e}".lower():
        return False
    status = status_code(e)
    if status is not None:
        return status in TRANSIENT_STATUS
    if isinstance(e, (ConnectionError, TimeoutError)):  # ConnectionResetError, socket timeouts, ...
        return True
    return any(c.__name__ in TRANSIENT_NAMES for c in type(e).__mro__)


def retry_after(e: BaseException) -> float | None:
    """Seconds a provider asked to wait before the next call (``Retry-After``, in seconds or as an HTTP date,
    or ``retry-after-ms``), or None."""
    headers = getattr(getattr(e, "response", None), "headers", None) or getattr(e, "headers", None)
    if not headers or not hasattr(headers, "get"):
        return None

    def get(name: str) -> Any:
        return headers.get(name) or headers.get(name.title())

    try:
        ms = get("retry-after-ms")
        if ms is not None:
            return max(0.0, float(ms) / 1000)
        value = get("retry-after")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            when = email.utils.parsedate_to_datetime(str(value))
            return max(0.0, when.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return None


@dataclass(frozen=True)
class RetryPolicy:
    """How a stage retries a call that failed transiently (:func:`is_transient`): up to ``attempts`` tries in
    all, waiting ``base_s * 2 ** k`` seconds (at most ``max_s``, with +-25% jitter) or what the provider's
    ``Retry-After`` asks, never less than ``min_s``, and never more than ``max_total_s`` seconds of waiting
    for one call. A provider that asks to wait longer than ``max_retry_after_s`` stops the run at once, and a
    call that times out is retried ``timeout_retries`` times (then ``LLMTimedOut``). Otherwise the stage
    raises ``LLMUnavailable`` when the attempts run out."""

    attempts: int = 6
    base_s: float = 2.0
    max_s: float = 60.0
    min_s: float = 1.0
    max_retry_after_s: float = 300.0
    max_total_s: float = 900.0
    timeout_retries: int = 1

    def wait(self, attempt: int, hint: float | None = None) -> float:
        """Seconds to wait after failed attempt number ``attempt`` (1-based), given the provider's hint (at
        least ``min_s``, so a ``Retry-After: 0`` cannot make a hot loop)."""
        if hint is not None:
            return max(self.min_s, min(hint, self.max_retry_after_s))
        backoff = min(self.max_s, self.base_s * 2 ** (attempt - 1))
        return max(self.min_s, backoff * (0.75 + 0.5 * random.random()))


def chat_messages(system: str) -> list[dict[str, str]]:
    """The messages of one stage call: the prompt as the system message, then ``USER_TURN``."""
    return [{"role": "system", "content": system}, {"role": "user", "content": USER_TURN}]


def is_truncated(finish_reason: Any) -> bool:
    """Whether a provider's finish reason means the output-token limit was hit."""
    return str(finish_reason or "").lower() in ("length", "max_tokens")


def is_content_filtered(finish_reason: Any) -> bool:
    """Whether a provider's finish reason means its content filter (or a refusal) stopped the answer."""
    return str(finish_reason or "").lower() in ("content_filter", "content_filtered", "refusal", "safety")


#: The default request timeout: long enough for the longest answer of the stage at ``TOKENS_PER_S`` output
#: tokens per second plus ``TIMEOUT_OVERHEAD_S`` for the prompt, and never under ``MIN_TIMEOUT_S``: 520 s for
#: a 16,000-token answer, 920 s for 32,000 (``request_timeout``).
TOKENS_PER_S, TIMEOUT_OVERHEAD_S, MIN_TIMEOUT_S = 40.0, 120.0, 300.0


def request_timeout(max_tokens: int | None) -> float:
    """Seconds one request may take before it times out, for an answer of up to ``max_tokens`` tokens:
    ``max(300, 120 + max_tokens / 40)``."""
    return max(MIN_TIMEOUT_S, TIMEOUT_OVERHEAD_S + (max_tokens or 0) / TOKENS_PER_S)


def output_limit(llm: Any, role: str) -> int | None:
    """The output-token limit ``llm`` asks for in stage ``role`` (``whole_run_max_tokens`` for the whole-run
    stages, else ``max_tokens``), or None when it has none."""
    for attr in ("whole_run_max_tokens", "max_tokens") if role in WHOLE_RUN_ROLES else ("max_tokens",):
        v = getattr(llm, attr, None)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    return None


_OUTPUT_LIMIT = re.compile(r"(?:at most|maximum(?: value)?(?: is| of)?|up to|>)\s*(\d+)")


def lower_output_limit(error: str, current: int) -> int | None:
    """The model's own output-token limit when a provider error says ``current`` exceeds it (OpenAI: "supports
    at most 16384 completion tokens"; Anthropic: "max_tokens: 32000 > 8192"), else None."""
    low = error.lower()
    m = _OUTPUT_LIMIT.search(low) if "token" in low else None
    return int(m.group(1)) if m and 0 < int(m.group(1)) < current else None


# --------------------------------------------------------------------------- backends
class OpenAIChatLLM:
    """Chat Completions against OpenAI or any OpenAI-compatible server (``base_url``).

    ``max_tokens`` bounds the answers of the per-decision-point stages and ``whole_run_max_tokens`` those of
    ``WHOLE_RUN_ROLES``; a model with a lower output limit gets its own limit (read from the provider's
    error), and a server that does not know ``max_completion_tokens`` gets ``max_tokens``. ``azure_endpoint``
    (with ``api_version``) selects a classic Azure OpenAI resource, where ``model`` is the deployment name. A
    server that needs no key (vLLM, Ollama) works with ``base_url`` alone.

    Transient failures (rate limits, timeouts, server errors) are raised, not retried: the TabSchema stage
    that makes the call retries them (``RetryPolicy``) and records every attempt. ``max_retries`` > 0 adds
    retries inside this backend, for use outside TabSchema. ``timeout`` bounds one request in seconds
    (default: ``request_timeout`` of the stage's output limit).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float | None = 0.1,
        max_tokens: int = 16_000,
        whole_run_max_tokens: int = 32_000,
        timeout: float | None = None,
        max_retries: int = 0,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
    ):
        from openai import AzureOpenAI, OpenAI  # extra: tabagent[tabschema]

        self.model = model
        self.temperature, self.max_retries = temperature, max_retries
        self.max_tokens, self.whole_run_max_tokens = max_tokens, whole_run_max_tokens
        self.timeout = timeout
        client_timeout = timeout or request_timeout(max(max_tokens, whole_run_max_tokens))
        key = os.environ.get(api_key_env)
        if azure_endpoint:
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=key,
                api_version=api_version or os.environ.get("OPENAI_API_VERSION") or AZURE_API_VERSION,
                timeout=client_timeout,
                max_retries=0,
            )
        else:
            base = base_url or os.environ.get("OPENAI_BASE_URL")
            # a keyless OpenAI-compatible server still needs a placeholder key for the client
            self.client = OpenAI(
                api_key=key or ("EMPTY" if base else None),
                base_url=base,
                timeout=client_timeout,
                max_retries=0,
            )

    def _params(self, role: str = "") -> dict[str, Any]:
        limit = self.whole_run_max_tokens if role in WHOLE_RUN_ROLES else self.max_tokens
        p: dict[str, Any] = {"max_completion_tokens": limit}
        if self.temperature is not None:
            p["temperature"] = self.temperature
        return p

    @staticmethod
    def _adapt(params: dict[str, Any], error: str) -> bool:
        """Adjust ``params`` to a provider's ``BadRequestError``; False when the error is not about them."""
        low = error.lower()
        if "temperature" in low and "temperature" in params:  # some reasoning models reject it
            params.pop("temperature")
            return True
        unknown = any(w in low for w in ("unrecognized", "unsupported", "not supported", "unknown"))
        if unknown and "max_completion_tokens" in params and "max_completion_tokens" in low:
            params["max_tokens"] = params.pop("max_completion_tokens")  # an older server
            return True
        key = "max_completion_tokens" if "max_completion_tokens" in params else "max_tokens"
        limit = lower_output_limit(error, params[key])
        if limit:  # the model's own output limit is lower
            params[key] = limit
            return True
        return False

    def complete(self, system: str, *, role: str = "") -> LLMResponse:
        import openai

        params = self._params(role)
        adaptations = retries = 0
        while True:
            t0 = time.monotonic()
            limit = params.get("max_completion_tokens", params.get("max_tokens"))
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=chat_messages(system),
                    timeout=self.timeout or request_timeout(limit),
                    **params,
                )
            except openai.BadRequestError as e:
                adaptations += 1
                if adaptations <= 4 and self._adapt(params, str(e)):  # each fix removes one parameter problem
                    continue
                raise
            except Exception as e:
                if retries >= self.max_retries or not is_transient(e):
                    raise
                retries += 1
                time.sleep(RetryPolicy().wait(retries, retry_after(e)))
                continue
            out = response_of(r, self.model, time.monotonic() - t0)
            out.max_tokens = limit
            return out


def response_of(r: Any, model: str, latency_s: float) -> LLMResponse:
    """An ``LLMResponse`` from an OpenAI-style completion, tolerating what proxies return on errors: no
    choices (finish reason ``empty``), no usage, content as a list of parts, and a refusal in place of content
    (finish reason ``refusal``, a content filter)."""
    choices = getattr(r, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, list):  # content parts: keep the text ones
        content = "".join(
            str(p.get("text", "")) if isinstance(p, dict) else str(getattr(p, "text", "") or "")
            for p in content
        )
    finish = str(getattr(choice, "finish_reason", "") or "") if choice is not None else "empty"
    refusal = getattr(message, "refusal", None)
    if not content and isinstance(refusal, str) and refusal.strip():
        finish = "refusal"
    usage = getattr(r, "usage", None)
    return LLMResponse(
        content if isinstance(content, str) else str(content or ""),
        model,
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
        latency_s,
        finish,
    )


class LiteLLM:
    """Any provider supported by LiteLLM (Anthropic, Gemini, Bedrock, ...), keyed by standard env vars.
    ``max_tokens``, ``whole_run_max_tokens`` and ``timeout`` are as in ``OpenAIChatLLM``. Transient failures
    are raised for the calling stage to retry: LiteLLM's own retries are off, unless a global
    ``litellm.num_retries`` turns them back on for some providers (a warning says so, once)."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float | None = 0.1,
        max_tokens: int = 16_000,
        whole_run_max_tokens: int = 32_000,
        timeout: float | None = None,
    ):
        import litellm  # extra: tabagent[llm]

        self._litellm, self.model = litellm, model
        self.temperature, self.max_tokens, self.whole_run_max_tokens = (
            temperature,
            max_tokens,
            whole_run_max_tokens,
        )
        self.timeout = timeout

    def complete(self, system: str, *, role: str = "") -> LLMResponse:
        t0 = time.monotonic()
        limit = self.whole_run_max_tokens if role in WHOLE_RUN_ROLES else self.max_tokens
        kw: dict[str, Any] = {
            "max_tokens": limit,
            "timeout": self.timeout or request_timeout(limit),
            "num_retries": 0,
        }
        _warn_hidden_retries(self._litellm)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        try:
            r = self._litellm.completion(model=self.model, messages=chat_messages(system), **kw)
        except Exception as e:  # a model with a lower output limit than asked for: ask again at its limit
            limit = lower_output_limit(str(e), kw["max_tokens"])
            if limit is None:
                raise
            kw["max_tokens"] = limit
            r = self._litellm.completion(model=self.model, messages=chat_messages(system), **kw)
        out = response_of(r, self.model, time.monotonic() - t0)
        out.max_tokens = kw["max_tokens"]
        return out


_HIDDEN_RETRIES_WARNED = []


def _warn_hidden_retries(litellm: Any) -> None:
    """Warn once when a global ``litellm.num_retries`` is set: LiteLLM then retries some failures itself (its
    wrapper reads ``num_retries or litellm.num_retries``, so our 0 cannot turn it off), out of TabSchema's
    record and on top of its retries."""
    n = getattr(litellm, "num_retries", None)
    if n and not _HIDDEN_RETRIES_WARNED:
        _HIDDEN_RETRIES_WARNED.append(n)
        import logging

        logging.getLogger(__name__).warning(
            "litellm.num_retries=%s is set globally: LiteLLM retries some failed calls itself, unrecorded "
            "and on top of TabSchema's retries; set litellm.num_retries = 0 to leave retries to TabSchema",
            n,
        )


def auto_llm(model: str | None = None, timeout: float | None = None) -> LLM:
    """Pick a backend for ``model`` from its name and the environment:

    1. a model named ``<provider>/<model>`` for a LiteLLM provider (``anthropic/claude-sonnet-4-5``,
       ``gemini/...``; see ``LITELLM_PROVIDERS``) goes to LiteLLM, unless ``OPENAI_BASE_URL`` is set (an
       OpenAI-compatible server, whose model names may contain a slash);
    2. ``AZURE_OPENAI_ENDPOINT`` selects a classic Azure OpenAI resource (key ``AZURE_OPENAI_API_KEY``,
       ``OPENAI_API_VERSION``; the model is the deployment name);
    3. ``OPENAI_API_KEY`` or ``OPENAI_BASE_URL`` (a server that needs no key) selects ``OpenAIChatLLM``; an
       ``openai/`` prefix is dropped;
    4. otherwise LiteLLM with the default model of the provider whose key is set (Anthropic, Gemini, Mistral).

    ``timeout`` bounds one request in seconds (default: scaled with the stage's output limit,
    ``request_timeout``).
    """
    provider = model.split("/", 1)[0] if model and "/" in model else ""
    if provider in LITELLM_PROVIDERS and not os.environ.get("OPENAI_BASE_URL"):
        return LiteLLM(model, timeout=timeout)
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return OpenAIChatLLM(
            model or DEFAULT_MODEL,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key_env="AZURE_OPENAI_API_KEY",
            timeout=timeout,
        )
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        name = model[len("openai/") :] if model and model.startswith("openai/") else model
        return OpenAIChatLLM(name or DEFAULT_MODEL, timeout=timeout)
    providers = {
        "ANTHROPIC_API_KEY": "anthropic/claude-sonnet-4-5",
        "GEMINI_API_KEY": "gemini/gemini-2.5-pro",
        "MISTRAL_API_KEY": "mistral/mistral-large-latest",
    }
    for env, default in providers.items():
        if os.environ.get(env):
            return LiteLLM(model or default, timeout=timeout)
    raise MissingCredentials(
        "No LLM credentials found. Set OPENAI_API_KEY (plus OPENAI_BASE_URL for any OpenAI-compatible "
        "server, or OPENAI_BASE_URL alone for one that needs no key), AZURE_OPENAI_ENDPOINT with "
        "AZURE_OPENAI_API_KEY, or ANTHROPIC_API_KEY / GEMINI_API_KEY / MISTRAL_API_KEY with "
        '`pip install "tabagent[llm]"`.'
    )


# --------------------------------------------------------------------------- record / replay / scripted
def call_key(model: str, role: str, system: str) -> str:
    """The recording key of one call: a SHA-256 of the model, the stage role and the prompt."""
    return hashlib.sha256(f"{model}\x1f{role}\x1f{system}".encode()).hexdigest()


class RecordingLLM:
    """Wrap a live LLM and append every exchange to a JSONL session file."""

    def __init__(self, inner: LLM, path: str | Path):
        self.inner, self.model, self.path = inner, inner.model, Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # stages call the LLM from worker threads

    def complete(self, system: str, *, role: str = "") -> LLMResponse:
        r = self.inner.complete(system, role=role)
        rec = {
            "key": call_key(self.model, role, system),
            "role": role,
            "model": r.model,
            "text": r.text,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "finish_reason": r.finish_reason,
        }
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return r


class ReplayLLM:
    """Serve recorded responses; raise on any prompt that was not recorded."""

    def __init__(self, path: str | Path, model: str = DEFAULT_MODEL):
        self.model = model
        self.responses: dict[str, dict] = {}
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                self.responses[rec["key"]] = rec

    def complete(self, system: str, *, role: str = "") -> LLMResponse:
        rec = self.responses.get(call_key(self.model, role, system))
        if rec is None:
            raise KeyError(f"no recorded response for stage {role!r}; re-record the session")
        return LLMResponse(
            rec["text"],
            rec.get("model", self.model),
            rec.get("input_tokens", 0),
            rec.get("output_tokens", 0),
            finish_reason=rec.get("finish_reason", ""),
        )


class ScriptedLLM:
    """Test double: ``responder(role, prompt, call_index) -> text``."""

    def __init__(self, responder: Callable[[str, str, int], str], model: str = "scripted"):
        self.responder, self.model = responder, model
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, *, role: str = "") -> LLMResponse:
        n = sum(1 for r, _ in self.calls if r == role)
        self.calls.append((role, system))
        text = self.responder(role, system, n)
        return LLMResponse(text, self.model, len(system) // 4, len(text) // 4)


# --------------------------------------------------------------------------- cost
#: USD per 1M (input, output) tokens by model name. Empty: prices change, so pass current ones
#: (``prices=`` or ``--price-in/--price-out``) to have costs computed and a budget enforced.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {}
#: Output tokens a stage's next answer is assumed to take before the stage has answered once (the mean
#: answer of the per-decision-point stages in real runs; the whole-run stages answer at a similar length per
#: batch).
DEFAULT_ANSWER_TOKENS = 4000
#: Characters per input token of TabSchema's prompts (indented JSON and prose), used until a run has measured
#: its own models. Claude: the rendered prompt length over the provider's billed count on real runs, 2.33 to
#: 2.74 (2.3, the low end, so budgets are not under-reserved). Other models: 3.9 is unverified against billed
#: usage; it was measured offline with OpenAI's o200k tokenizer (GPT-4o and later), 3.7 to 4.4 on the same
#: prompts.
CHARS_PER_TOKEN = {"claude": 2.3}
DEFAULT_CHARS_PER_TOKEN = 3.9


def chars_per_token(model: str) -> float:
    """The prior characters-per-token ratio of ``model``'s tokenizer on TabSchema prompts (see
    ``CHARS_PER_TOKEN``)."""
    name = str(model).lower()
    return next((r for family, r in CHARS_PER_TOKEN.items() if family in name), DEFAULT_CHARS_PER_TOKEN)


def _totals() -> defaultdict:
    return defaultdict(lambda: defaultdict(float))


@dataclass
class CostMeter:
    """Token counts and cost of every LLM call, per TabSchema stage, with an optional budget.

    ``prices`` maps a model name to USD per 1M (input, output) tokens. Tokens are always counted; the cost
    is known only when every call's model has a price. With ``max_cost_usd`` no call starts whose estimated
    cost (its prompt, plus the stage's mean answer so far) would take the spent and in-flight cost past the
    budget: :meth:`reserve` raises ``BudgetExceeded`` instead. Since an unpriced call could never be
    checked, :meth:`require_price` refuses to start without a price for the model. Answers a run directory
    serves from its cache are counted apart, at the cost they had when first paid (``cached``). The meter is
    thread-safe.
    """

    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    max_cost_usd: float | None = None
    by_stage: dict[str, dict[str, float]] = field(default_factory=_totals)
    by_model: dict[str, dict[str, float]] = field(default_factory=_totals)
    unpriced: set[str] = field(default_factory=set)
    cached: dict[str, dict[str, float]] = field(default_factory=_totals)
    reserved_usd: float = 0.0
    _lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)

    def require_price(self, model: str) -> None:
        """Raise ``ValueError`` if a budget is set but ``model`` has no price (checked before any call)."""
        if self.max_cost_usd is not None and model not in self.prices:
            raise ValueError(
                f"max_cost_usd={self.max_cost_usd} cannot be enforced: no price is known for model "
                f"{model!r}. Pass prices={{{model!r}: (usd_per_1m_input, usd_per_1m_output)}} "
                "(CLI: --price-in and --price-out), or drop the budget."
            )

    def check(self) -> None:
        """Raise ``BudgetExceeded`` if the budget is already spent (called before every LLM call)."""
        if self.max_cost_usd is not None and self.total_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"onboarding cost ${self.total_usd:.2f} exceeded max_cost_usd={self.max_cost_usd}"
            )

    def reserve(self, stage: str, model: str, prompt_chars: int) -> float:
        """Reserve the estimated cost of one call before it is made and return it (pass it to
        :meth:`release` once the call is over). Raises ``BudgetExceeded`` when the call could take the total
        past ``max_cost_usd``; without a budget or a price nothing is reserved."""
        if self.max_cost_usd is None or model not in self.prices:
            return 0.0
        pin, pout = self.prices[model]
        with self._lock:
            s = self.by_stage.get(stage)
            answer = s["output_tokens"] / s["calls"] if s and s["calls"] else DEFAULT_ANSWER_TOKENS
            est = (prompt_chars / self.chars_per_token(model) * pin + answer * pout) / 1e6
            if self.total_usd + self.reserved_usd + est > self.max_cost_usd:
                raise BudgetExceeded(
                    f"onboarding cost ${self.total_usd:.4f} (${self.reserved_usd:.4f} more in flight) plus "
                    f"the next {stage} call (about ${est:.4f}) would exceed max_cost_usd={self.max_cost_usd}"
                )
            self.reserved_usd += est
            return est

    def release(self, est: float) -> None:
        """Release a reservation made by :meth:`reserve`."""
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - est)

    def cost(self, r: LLMResponse) -> float | None:
        """USD cost of one response, or None when its model has no price."""
        if r.model not in self.prices:
            return None
        pin, pout = self.prices[r.model]
        return (r.input_tokens * pin + r.output_tokens * pout) / 1e6

    def chars_per_token(self, model: str) -> float:
        """Characters per input token of ``model``: measured on this run's calls once they hold 20,000
        tokens, else the prior (:func:`chars_per_token`)."""
        m = self.by_model.get(model)
        if m and m.get("prompt_chars") and m.get("measured_tokens", 0) >= 20_000:
            return max(1.0, m["prompt_chars"] / m["measured_tokens"])
        return chars_per_token(model)

    def record(self, stage: str, r: LLMResponse, check: bool = True, prompt_chars: int = 0) -> None:
        """Add one response to ``stage``'s totals; raise ``BudgetExceeded`` past ``max_cost_usd`` (with
        ``check=False`` that is left to the checks before the next call). ``prompt_chars``, the prompt's
        length, calibrates the model's characters per token for later reservations."""
        with self._lock:
            s, m = self.by_stage[stage], self.by_model[r.model]
            for t in (s, m):
                t["calls"] += 1
                t["input_tokens"] += r.input_tokens
                t["output_tokens"] += r.output_tokens
            if prompt_chars and r.input_tokens:
                m["prompt_chars"] += prompt_chars
                m["measured_tokens"] += r.input_tokens
            usd = self.cost(r)
            if usd is None:
                self.unpriced.add(r.model)
                if self.max_cost_usd is not None:
                    raise BudgetExceeded(f"cannot enforce max_cost_usd: no price for model {r.model!r}")
                return
            s["usd"] += usd
            m["usd"] += usd
        if check:
            self.check()

    def record_cached(self, stage: str, r: LLMResponse) -> None:
        """Count an answer served from a run directory's cache (free now; ``r`` carries its first cost)."""
        with self._lock:
            s = self.cached[stage]
            s["calls"] += 1
            s["input_tokens"] += r.input_tokens
            s["output_tokens"] += r.output_tokens
            usd = self.cost(r)
            s["usd"] += usd or 0.0
            s["unpriced"] += usd is None

    @property
    def priced(self) -> bool:
        """True when every recorded call had a price, so ``total_usd`` is the full cost."""
        return not self.unpriced

    @property
    def total_usd(self) -> float:
        """Cost of the priced calls (the full cost when ``priced``)."""
        return float(sum(s["usd"] for s in self.by_stage.values()))

    @property
    def cached_usd(self) -> float | None:
        """What the cached answers cost when they were first paid (None when a model had no price)."""
        if any(s["unpriced"] for s in self.cached.values()):
            return None
        return float(sum(s["usd"] for s in self.cached.values()))

    def tokens(self) -> tuple[int, int, int]:
        """Total (calls, input tokens, output tokens) of the calls paid in this run."""
        tot = [sum(s[k] for s in self.by_stage.values()) for k in ("calls", "input_tokens", "output_tokens")]
        return int(tot[0]), int(tot[1]), int(tot[2])

    def describe(self, model: str = "") -> str:
        """One line for reports: the cost (or why it is unknown), the token totals and the cached answers."""
        calls, tin, tout = self.tokens()
        tokens = f"{calls} calls, {tin:,} input / {tout:,} output tokens"
        n_cached = int(sum(s["calls"] for s in self.cached.values()))
        reused = ""
        if n_cached:
            first = self.cached_usd
            paid = f"${first:.2f}" if first is not None else "an unknown cost"
            reused = f"; {n_cached} more answers reused from the run directory (paid earlier: {paid})"
        if not self.priced or (calls and model and model not in self.prices):
            missing = ", ".join(sorted(self.unpriced or {model}))
            return f"cost unknown (no price for {missing}; {tokens}){reused}"
        if len(self.by_model) > 1:  # each model's share, so a multi-model run is not credited to one
            label = "".join(
                f"{name} ${t['usd']:.2f} in {int(t['calls'])} calls; "
                for name, t in sorted(self.by_model.items())
            )
        else:
            label = f"{model or next(iter(self.by_model), '')}; " if model or self.by_model else ""
        return f"${self.total_usd:.2f} ({label}{tokens}){reused}"

    def summary(self) -> dict[str, Any]:
        """JSON-able totals: cost (None when unknown), token counts, per-stage details and the cached answers
        (``cached``, with what they cost when first paid)."""
        calls, tin, tout = self.tokens()
        cached = {
            k: int(sum(s[k] for s in self.cached.values()))
            for k in ("calls", "input_tokens", "output_tokens")
        }
        return {
            "total_usd": self.total_usd if self.priced else None,
            "unpriced_models": sorted(self.unpriced),
            "calls": calls,
            "input_tokens": tin,
            "output_tokens": tout,
            "stages": {k: dict(v) for k, v in self.by_stage.items()},
            "models": {k: dict(v) for k, v in self.by_model.items()},
            "cached": {**cached, "first_paid_usd": self.cached_usd},
        }


# --------------------------------------------------------------------------- parsing
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def extract_json(text: str) -> Any:
    """JSON from a fenced block, the whole text, or the longest JSON value that starts at the text's first
    ``{`` or first ``[`` (prose around it is ignored). Trailing commas, which the paper's prompt templates
    themselves contain, are tolerated."""

    def parse(candidate: str) -> tuple[Any, int] | None:
        for variant in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
            try:
                return json.JSONDecoder().raw_decode(variant)
            except (json.JSONDecodeError, TypeError):
                continue
        return None

    m = _FENCE.search(text)
    whole = text.strip()
    for c in ([m.group(1)] if m else []) + [whole]:
        got = parse(c)
        if got is not None and (c is not whole or got[1] == len(c.rstrip())):
            return got[0]
    brace, bracket = text.find("{"), text.find("[")
    # a "[" after the first "{" may sit inside a broken object: it is never returned in the object's place
    starts = [i for i in (brace, bracket) if i >= 0 and (i == brace or brace < 0 or bracket < brace)]
    found = [got for i in starts if (got := parse(text[i:]))]
    if found:
        return max(found, key=lambda g: g[1])[0]  # the longest value ("Note [1]: {...}" gives the object)
    raise ValueError("no JSON found in the model response")


_FENCE_BLOCK = re.compile(r"```[ \t]*([\w+.-]*)[ \t]*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """Code between ``<code>`` tags (the CodeExtractor output format; a Markdown fence inside the tags is
    removed), else from a Markdown fence (``python``, ``py`` or unlabeled; the longest one). When several
    ``<code>`` blocks appear (the thoughts mention the tag, say), the longest one is the code."""
    # each block runs from the last <code> before a </code> (an unclosed last block, cut short, counts too)
    segments = text.split("</code>")
    blocks = [seg.rsplit("<code>", 1)[1].strip() for seg in segments if "<code>" in seg]
    blocks = [b for b in blocks if b]
    if blocks:
        code = max(blocks, key=len)
        fenced = (
            re.match(r"```[\w+.-]*[ \t]*\n?(.*?)(?:```|\Z)", code, re.S) if code.startswith("```") else None
        )
        return fenced.group(1).strip() if fenced else code
    fences = [
        body.strip()
        for lang, body in _FENCE_BLOCK.findall(text)
        if lang.lower() in ("", "python", "py", "python3") and body.strip()
    ]
    if fences:
        return max(fences, key=len)
    raise ValueError("no code found in the model response")
