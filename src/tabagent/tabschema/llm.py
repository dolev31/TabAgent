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
"""

from __future__ import annotations

import hashlib
import json
import os
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
    """One completion: its text, the model that produced it, token counts, wall-clock latency and why it
    stopped (``length`` when it hit the output limit)."""

    text: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str = ""


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


def chat_messages(system: str) -> list[dict[str, str]]:
    """The messages of one stage call: the prompt as the system message, then ``USER_TURN``."""
    return [{"role": "system", "content": system}, {"role": "user", "content": USER_TURN}]


def is_truncated(finish_reason: Any) -> bool:
    """Whether a provider's finish reason means the output-token limit was hit."""
    return str(finish_reason or "").lower() in ("length", "max_tokens")


def is_content_filtered(finish_reason: Any) -> bool:
    """Whether a provider's finish reason means its content filter (or a refusal) stopped the answer."""
    return str(finish_reason or "").lower() in ("content_filter", "content_filtered", "refusal", "safety")


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
        timeout: float = 300,
        max_retries: int = 4,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
    ):
        from openai import AzureOpenAI, OpenAI  # extra: tabagent[tabschema]

        self.model = model
        self.temperature, self.max_retries = temperature, max_retries
        self.max_tokens, self.whole_run_max_tokens = max_tokens, whole_run_max_tokens
        key = os.environ.get(api_key_env)
        if azure_endpoint:
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=key,
                api_version=api_version or os.environ.get("OPENAI_API_VERSION") or AZURE_API_VERSION,
                timeout=timeout,
            )
        else:
            base = base_url or os.environ.get("OPENAI_BASE_URL")
            # a keyless OpenAI-compatible server still needs a placeholder key for the client
            self.client = OpenAI(api_key=key or ("EMPTY" if base else None), base_url=base, timeout=timeout)

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
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=chat_messages(system), **params
                )
            except openai.BadRequestError as e:
                if self._adapt(params, str(e)):
                    continue
                raise
            except (
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.InternalServerError,
                openai.APITimeoutError,
            ):
                if attempt == self.max_retries:
                    raise
                time.sleep(min(60, 2**attempt))
                continue
            usage = r.usage
            return LLMResponse(
                r.choices[0].message.content or "",
                self.model,
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
                time.monotonic() - t0,
                str(r.choices[0].finish_reason or ""),
            )
        raise RuntimeError(f"{self.model}: no answer after {self.max_retries + 1} attempts")


class LiteLLM:
    """Any provider supported by LiteLLM (Anthropic, Gemini, Bedrock, ...), keyed by standard env vars.
    ``max_tokens`` and ``whole_run_max_tokens`` are as in ``OpenAIChatLLM``."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float | None = 0.1,
        max_tokens: int = 16_000,
        whole_run_max_tokens: int = 32_000,
    ):
        import litellm  # extra: tabagent[llm]

        self._litellm, self.model = litellm, model
        self.temperature, self.max_tokens, self.whole_run_max_tokens = (
            temperature,
            max_tokens,
            whole_run_max_tokens,
        )

    def complete(self, system: str, *, role: str = "") -> LLMResponse:
        t0 = time.monotonic()
        kw: dict[str, Any] = {
            "max_tokens": self.whole_run_max_tokens if role in WHOLE_RUN_ROLES else self.max_tokens
        }
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
        u = getattr(r, "usage", None)
        return LLMResponse(
            r.choices[0].message.content or "",
            self.model,
            getattr(u, "prompt_tokens", 0) or 0,
            getattr(u, "completion_tokens", 0) or 0,
            time.monotonic() - t0,
            str(getattr(r.choices[0], "finish_reason", "") or ""),
        )


def auto_llm(model: str | None = None) -> LLM:
    """Pick a backend for ``model`` from its name and the environment:

    1. a model named ``<provider>/<model>`` for a LiteLLM provider (``anthropic/claude-sonnet-4-5``,
       ``gemini/...``; see ``LITELLM_PROVIDERS``) goes to LiteLLM, unless ``OPENAI_BASE_URL`` is set (an
       OpenAI-compatible server, whose model names may contain a slash);
    2. ``AZURE_OPENAI_ENDPOINT`` selects a classic Azure OpenAI resource (key ``AZURE_OPENAI_API_KEY``,
       ``OPENAI_API_VERSION``; the model is the deployment name);
    3. ``OPENAI_API_KEY`` or ``OPENAI_BASE_URL`` (a server that needs no key) selects ``OpenAIChatLLM``; an
       ``openai/`` prefix is dropped;
    4. otherwise LiteLLM with the default model of the provider whose key is set (Anthropic, Gemini, Mistral).
    """
    provider = model.split("/", 1)[0] if model and "/" in model else ""
    if provider in LITELLM_PROVIDERS and not os.environ.get("OPENAI_BASE_URL"):
        return LiteLLM(model)
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return OpenAIChatLLM(
            model or DEFAULT_MODEL,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key_env="AZURE_OPENAI_API_KEY",
        )
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        name = model[len("openai/") :] if model and model.startswith("openai/") else model
        return OpenAIChatLLM(name or DEFAULT_MODEL)
    providers = {
        "ANTHROPIC_API_KEY": "anthropic/claude-sonnet-4-5",
        "GEMINI_API_KEY": "gemini/gemini-2.5-pro",
        "MISTRAL_API_KEY": "mistral/mistral-large-latest",
    }
    for env, default in providers.items():
        if os.environ.get(env):
            return LiteLLM(model or default)
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
#: Output tokens a stage's next answer is assumed to take before the stage has answered once.
DEFAULT_ANSWER_TOKENS = 1500


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
            est = (prompt_chars / 4 * pin + answer * pout) / 1e6
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

    def record(self, stage: str, r: LLMResponse, check: bool = True) -> None:
        """Add one response to ``stage``'s totals; raise ``BudgetExceeded`` past ``max_cost_usd`` (with
        ``check=False`` that is left to the checks before the next call)."""
        with self._lock:
            s, m = self.by_stage[stage], self.by_model[r.model]
            for t in (s, m):
                t["calls"] += 1
                t["input_tokens"] += r.input_tokens
                t["output_tokens"] += r.output_tokens
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


def extract_code(text: str) -> str:
    """Code between ``<code>`` tags (the CodeExtractor output format; a Markdown fence inside the tags is
    removed), else from a python fence."""
    if "<code>" in text:
        code = text.split("<code>", 1)[1].split("</code>", 1)[0].strip()
        fenced = re.match(r"```[\w+-]*[ \t]*\n?(.*?)```", code, re.S) if code.startswith("```") else None
        return fenced.group(1).strip() if fenced else code
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    raise ValueError("no code found in the model response")
