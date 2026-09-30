"""LLM access for TabSchema: provider-agnostic, recordable, replayable, and metered.

Every TabSchema stage sends its prompt as a system message (paper App. D). Any OpenAI-compatible
endpoint works (OpenAI, Azure OpenAI, vLLM, Ollama, a LiteLLM proxy) through ``OpenAIChatLLM``; other
providers work through LiteLLM (``pip install "tabagent[llm]"``). ``auto_llm`` picks a backend from the
environment, so onboarding needs nothing but an API key.

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


@dataclass
class LLMResponse:
    """One completion: its text, the model that produced it, token counts and wall-clock latency."""

    text: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class LLM(Protocol):
    """What TabSchema needs from a model: a name and one system-prompt completion."""

    model: str

    def complete(self, system: str, *, role: str = "") -> LLMResponse: ...


class BudgetExceeded(RuntimeError):
    """Raised when onboarding would exceed the configured ``max_cost_usd``."""


class MissingCredentials(RuntimeError):
    """Raised by ``auto_llm`` when no LLM provider key is set in the environment."""


# --------------------------------------------------------------------------- backends
class OpenAIChatLLM:
    """Chat Completions against OpenAI or any OpenAI-compatible server (``base_url``)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float | None = 0.1,
        max_tokens: int = 7000,
        timeout: float = 300,
        max_retries: int = 4,
    ):
        from openai import OpenAI  # extra: tabagent[tabschema]

        self.model = model
        self.temperature, self.max_tokens, self.max_retries = temperature, max_tokens, max_retries
        self.client = OpenAI(
            api_key=os.environ.get(api_key_env),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            timeout=timeout,
        )

    def _params(self) -> dict[str, Any]:
        p: dict[str, Any] = {"max_completion_tokens": self.max_tokens}
        if self.temperature is not None:
            p["temperature"] = self.temperature
        return p

    def complete(self, system: str, *, role: str = "") -> LLMResponse:
        import openai

        params = self._params()
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "system", "content": system}], **params
                )
            except openai.BadRequestError as e:  # some reasoning models reject temperature
                if "temperature" in str(e) and "temperature" in params:
                    params.pop("temperature")
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
            )
        raise RuntimeError("unreachable")


class LiteLLM:
    """Any provider supported by LiteLLM (Anthropic, Gemini, Bedrock, ...), keyed by standard env vars."""

    def __init__(self, model: str, *, temperature: float | None = 0.1, max_tokens: int = 7000):
        import litellm  # extra: tabagent[llm]

        self._litellm, self.model = litellm, model
        self.temperature, self.max_tokens = temperature, max_tokens

    def complete(self, system: str, *, role: str = "") -> LLMResponse:
        t0 = time.monotonic()
        kw = {"max_tokens": self.max_tokens}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        r = self._litellm.completion(model=self.model, messages=[{"role": "system", "content": system}], **kw)
        u = getattr(r, "usage", None)
        return LLMResponse(
            r.choices[0].message.content or "",
            self.model,
            getattr(u, "prompt_tokens", 0) or 0,
            getattr(u, "completion_tokens", 0) or 0,
            time.monotonic() - t0,
        )


def auto_llm(model: str | None = None) -> LLM:
    """Pick a backend from the environment: an OpenAI-compatible endpoint when ``OPENAI_API_KEY`` (and
    optionally ``OPENAI_BASE_URL``) is set, otherwise LiteLLM for the provider whose key is present."""
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIChatLLM(model or DEFAULT_MODEL)
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
        "server), "
        'or ANTHROPIC_API_KEY / GEMINI_API_KEY / MISTRAL_API_KEY with `pip install "tabagent[llm]"`.'
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
            rec["text"], rec.get("model", self.model), rec.get("input_tokens", 0), rec.get("output_tokens", 0)
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


@dataclass
class CostMeter:
    """Token counts and cost of every LLM call, per TabSchema stage, with an optional budget.

    ``prices`` maps a model name to USD per 1M (input, output) tokens. Tokens are always counted; the cost
    is known only when every call's model has a price. With ``max_cost_usd`` the meter raises
    ``BudgetExceeded`` once the cost exceeds it; since an unpriced call could never trip that check,
    :meth:`require_price` refuses to start without a price for the model.
    """

    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    max_cost_usd: float | None = None
    by_stage: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )
    unpriced: set[str] = field(default_factory=set)

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

    def cost(self, r: LLMResponse) -> float | None:
        """USD cost of one response, or None when its model has no price."""
        if r.model not in self.prices:
            return None
        pin, pout = self.prices[r.model]
        return (r.input_tokens * pin + r.output_tokens * pout) / 1e6

    def record(self, stage: str, r: LLMResponse) -> None:
        """Add one response to ``stage``'s totals; raise ``BudgetExceeded`` past ``max_cost_usd``."""
        s = self.by_stage[stage]
        s["calls"] += 1
        s["input_tokens"] += r.input_tokens
        s["output_tokens"] += r.output_tokens
        usd = self.cost(r)
        if usd is None:
            self.unpriced.add(r.model)
            if self.max_cost_usd is not None:
                raise BudgetExceeded(f"cannot enforce max_cost_usd: no price for model {r.model!r}")
            return
        s["usd"] += usd
        self.check()

    @property
    def priced(self) -> bool:
        """True when every recorded call had a price, so ``total_usd`` is the full cost."""
        return not self.unpriced

    @property
    def total_usd(self) -> float:
        """Cost of the priced calls (the full cost when ``priced``)."""
        return float(sum(s["usd"] for s in self.by_stage.values()))

    def tokens(self) -> tuple[int, int, int]:
        """Total (calls, input tokens, output tokens)."""
        tot = [sum(s[k] for s in self.by_stage.values()) for k in ("calls", "input_tokens", "output_tokens")]
        return int(tot[0]), int(tot[1]), int(tot[2])

    def describe(self, model: str = "") -> str:
        """One line for reports: the cost (or why it is unknown) and the token totals."""
        calls, tin, tout = self.tokens()
        tokens = f"{calls} calls, {tin:,} input / {tout:,} output tokens"
        if not self.priced or (calls and model and model not in self.prices):
            missing = ", ".join(sorted(self.unpriced or {model}))
            return f"cost unknown (no price for {missing}; {tokens})"
        label = f"{model}; " if model else ""
        return f"${self.total_usd:.2f} ({label}{tokens})"

    def summary(self) -> dict[str, Any]:
        """JSON-able totals: cost (None when unknown), token counts and per-stage details."""
        calls, tin, tout = self.tokens()
        return {
            "total_usd": self.total_usd if self.priced else None,
            "unpriced_models": sorted(self.unpriced),
            "calls": calls,
            "input_tokens": tin,
            "output_tokens": tout,
            "stages": {k: dict(v) for k, v in self.by_stage.items()},
        }


# --------------------------------------------------------------------------- parsing
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def extract_json(text: str) -> Any:
    """JSON from a fenced block, the whole text, or the first balanced object/array in the text."""
    m = _FENCE.search(text)
    candidates = [m.group(1)] if m else []
    candidates.append(text.strip())
    for start in (text.find("{"), text.find("[")):
        if start >= 0:
            candidates.append(text[start:])
    for c in candidates:
        try:
            return json.loads(c)
        except (json.JSONDecodeError, TypeError):
            pass
        try:  # tolerate trailing prose after a JSON value
            return json.JSONDecoder().raw_decode(c)[0]
        except (json.JSONDecodeError, TypeError):
            continue
    raise ValueError("no JSON found in the model response")


def extract_code(text: str) -> str:
    """Code between ``<code>`` tags (the CodeExtractor output format), else from a python fence."""
    if "<code>" in text:
        return text.split("<code>", 1)[1].split("</code>", 1)[0].strip()
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    raise ValueError("no code found in the model response")
