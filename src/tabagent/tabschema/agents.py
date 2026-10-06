"""TabSchema's feature-extraction agents (paper Sec. 3.1; App. D).

FeatureAnalyzer   proposes features from one decision point (App. D.2, FeatureAnalyzer prompt)
CodeExtractor     writes the extraction code for them (App. D.2, CodeExtractor prompt)
CodeExecutor      runs it in the sandbox, with up to three retries that feed errors back (App. D.1)
CodeValidator     an LLM check that the executed code reads only real fields and follows the plan (App. D.1)
FeatureAggregator merges the proposals of many trajectories and drops unobtainable ones (App. D.2)
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from tabagent.tabschema import checks, prompts
from tabagent.tabschema.llm import (
    LLM,
    CostMeter,
    LLMContentFiltered,
    LLMFormatError,
    LLMPromptTooLong,
    LLMResponse,
    LLMTimedOut,
    LLMTruncated,
    LLMUnavailable,
    RetryPolicy,
    extract_code,
    extract_json,
    is_content_filtered,
    is_content_policy_error,
    is_prompt_too_long,
    is_timeout,
    is_transient,
    is_truncated,
    output_limit,
    retry_after,
    status_code,
)
from tabagent.tabschema.runstore import RunStore
from tabagent.tabschema.sandbox import MAX_CANDIDATE_ERRORS, ExecResult, SubprocessSandbox
from tabagent.tabschema.trajectory import (
    DecisionPoint,
    clip_for_prompt,
    clip_strings,
    get_json_schema,
    union_schema,
)

logger = logging.getLogger(__name__)

SPEC_FIELDS = (
    "feature_name",
    "feature_type",
    "description",
    "extraction_source",
    "computation",
    "classification_relevance",
    "rationale",
)
JSON_REPAIR = (
    "\n\nYour previous answer could not be parsed. Return only the requested JSON, with no other text."
)
#: Characters a feature plan's name and its other text fields keep (a longer value is clipped with a note), so
#: one runaway answer cannot inflate every later prompt that lists the features.
NAME_CHARS, FIELD_CHARS = 200, 2_000
#: Proposals one FeatureAnalyzer answer contributes at most (real answers propose 13 to 16 on average); the
#: rest are dropped and recorded, so a runaway answer cannot multiply the cost of the whole-run stages.
MAX_PROPOSALS = 60
#: Times a batch of a whole-run stage (aggregator, judges, MetaJudge) is halved when its answer is unusable,
#: truncated, filtered or its prompt too long: at most 2 ** SPLIT_DEPTH sub-batches, about SPLIT_DEPTH + 1
#: times the batch's input tokens in the worst case.
SPLIT_DEPTH = 2


#: Set when the run is interrupted (Ctrl+C): every backoff wait ends at once instead of running out its time.
STOP = threading.Event()


def _wait(seconds: float) -> None:
    """Wait ``seconds`` of backoff, unless the run is interrupted (``STOP``): then raise KeyboardInterrupt."""
    if STOP.wait(seconds):
        raise KeyboardInterrupt("TabSchema was interrupted")


_sleep = _wait  # one place for tests to replace


def short(e: BaseException, limit: int = 300) -> str:
    """``Type: message`` of an exception, on one line, clipped."""
    text = " ".join(str(e).split())
    return f"{type(e).__name__}: {text}"[:limit]


def _clip(value: str, limit: int) -> str:
    return value if len(value) <= limit else f"{value[:limit]} ...[{len(value) - limit} more characters]"


@dataclass
class FeatureSpec:
    """One proposed feature, in the fields of the FeatureAnalyzer's JSON answer, plus its support (the
    number of decision points that proposed it)."""

    feature_name: str
    feature_type: str = "computed"
    description: str = ""
    extraction_source: str = ""
    computation: str = ""
    classification_relevance: str = ""
    rationale: str = ""
    trajectory_file: str = ""
    support: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureSpec:
        """A spec from one JSON item of a model answer: text fields are coerced to strings, null fields
        count as absent (a null name is no name), and an unreadable ``support`` counts as 1."""
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and v is not None}
        if "feature_name" not in known:
            known["feature_name"] = d["name"] if d.get("name") is not None else ""
        try:
            support = int(known.pop("support", 1))
        except (TypeError, ValueError, OverflowError):
            support = 1
        text = {
            k: str(v)[:NAME_CHARS] if k == "feature_name" else _clip(str(v), FIELD_CHARS)
            for k, v in known.items()
        }
        return cls(**text, support=support)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        return normalize_name(self.feature_name)


def normalize_name(name: str) -> str:
    """Lowercase ``name`` and collapse each run of characters other than letters and digits (of any script)
    to ``_`` (the key features match on)."""
    return re.sub(r"[\W_]+", "_", str(name).lower()).strip("_")


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def json_items(value: Any) -> list[dict[str, Any]]:
    """The objects of a JSON list from a model answer; anything else (null, a number, a string, a mapping)
    holds none."""
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def in_parallel(fn, items: list, workers: int = 1) -> list:
    """``[fn(x) for x in items]``, run on up to ``workers`` threads; results keep the order of ``items``. An
    interruption (Ctrl+C) returns at once: waiting threads are woken (``STOP``) and not waited for."""
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    pool = ThreadPoolExecutor(max_workers=min(workers, len(items)))
    try:
        out = list(pool.map(fn, items))
    except KeyboardInterrupt:
        shutdown_interrupted(pool)
        raise
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return out


def shutdown_interrupted(pool: ThreadPoolExecutor) -> None:
    """Stop a pool after an interruption without waiting for its threads: queued work is cancelled, and
    threads waiting in a backoff wake up and stop (``STOP``)."""
    STOP.set()
    pool.shutdown(wait=False, cancel_futures=True)


class Stage:
    """Base of the LLM-backed stages: one prompt in, one answer out, metered and optionally cached.
    ``workers`` bounds the concurrent calls of a stage that splits its work into batches.

    A call that fails transiently (a rate limit, a dropped connection, a server error, an empty answer) is
    retried under ``retry`` and raises ``LLMUnavailable`` when the retries run out; a call that times out is
    retried once and then raises ``LLMTimedOut`` (handled like a truncated answer); a prompt the model refuses
    as too long raises ``LLMPromptTooLong``; a filtered answer (or a prompt the provider's filter refuses)
    goes once to the ``fallback`` model and otherwise raises ``LLMContentFiltered``; an answer cut at the
    output limit raises ``LLMTruncated``. Every attempt is recorded in the run directory's
    ``llm_calls.jsonl``, and every paid answer is cached, including truncated and filtered ones, so a resumed
    run meets the same outcome without paying for it again. ``events`` lists what went wrong in this stage's
    calls (unusable answers, split batches, clipped prompts), for the run's record.
    """

    role = "stage"
    workers = 1
    retry = RetryPolicy()

    def __init__(
        self,
        llm: LLM,
        meter: CostMeter | None = None,
        store: RunStore | None = None,
        json_retries: int = 1,
        fallback: LLM | None = None,
    ):
        self.llm, self.meter, self.store, self.json_retries = llm, meter or CostMeter(), store, json_retries
        self.fallback = fallback  # asked once when the provider's content filter stops ``llm``'s answer
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

    def note(self, event: str, **fields: Any) -> None:
        """Record what happened in this stage (``events``; thread-safe)."""
        with self._events_lock:
            self.events.append({"stage": self.role, "event": event, **fields})

    def _key(self, prompt: str, llm: LLM | None = None) -> dict[str, str]:
        """The cache key of ``prompt`` asked of ``llm`` (default the stage's model); a fallback model's answer
        is kept under its own key, next to the filtered answer of the stage's model."""
        key = {
            "role": self.role,
            "model": (llm or self.llm).model,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        if llm is not None and llm is not self.llm:
            key["fallback_from"] = self.llm.model
        return key

    def _has_fallback(self) -> bool:
        return self.fallback is not None and self.fallback.model != self.llm.model

    def unusable(self, error: LLMFormatError, prompts_: list[str]) -> LLMFormatError:
        """``error``, marked with the prompts whose answers could not be used: if it stops the run, those
        answers are dropped from the run directory (``forget_cause``) so that a rerun asks for them again,
        while every answer a run used, or skipped without stopping, is replayed as it was."""
        error.forget = [(self, list(prompts_))]  # type: ignore[attr-defined]
        return error

    def call(self, prompt: str) -> str:
        """One LLM call. With a run store, calls are content-addressed: a resumed or repeated run reuses
        stored answers instead of paying for them again (the meter counts them as cached, at the cost they
        had when first paid). A filtered answer goes to the ``fallback`` model, whether it is new or replayed
        (a fallback added after a filtered stop is asked on the rerun); the fallback's answer is stored under
        its own key. A replayed answer cut at the output limit is asked again when the model's output limit
        has been raised since. Raises ``LLMTruncated`` or ``LLMContentFiltered`` for such an answer."""
        value = self._answer(self.llm, prompt)
        if is_content_filtered(value.get("finish_reason")) and self._has_fallback():
            value = self._answer(self.fallback, prompt, fallback_from=self.llm.model)
        self._check(value, prompt)
        return str(value.get("text") or "")

    def _answer(self, llm: LLM, prompt: str, fallback_from: str = "") -> dict[str, Any]:
        """``llm``'s stored answer to ``prompt``, or a paid one (stored)."""
        limit = output_limit(llm, self.role)
        if self.store is None:
            return _stored(self._ask(llm, prompt, fallback_from), limit)
        paid: list[LLMResponse] = []

        def fetch() -> dict[str, Any]:
            r = self._ask(llm, prompt, fallback_from)
            paid.append(r)
            return _stored(r, limit)

        key = self._key(prompt, llm if fallback_from else None)
        value = self.store.cached("llm", key, fetch)
        if not isinstance(value, dict):  # a cache entry written before usage was stored with the answer
            value = {"text": value}
        then = value.get("max_tokens")
        if not paid and is_truncated(value.get("finish_reason")) and limit and then and limit > then:
            self.note("asked_again", reason=f"output limit raised from {then} to {limit} tokens")
            self.store.forget("llm", key)  # cut at a lower output limit than the model has now: ask again
            value = self.store.cached("llm", key, fetch)
        if not paid:
            self.meter.record_cached(
                self.role,
                LLMResponse(
                    str(value.get("text") or ""),
                    value.get("model", llm.model),
                    value.get("input_tokens", 0),
                    value.get("output_tokens", 0),
                ),
            )
        return value

    def _check(self, value: dict[str, Any], prompt: str) -> None:
        finish = value.get("finish_reason", "")
        model, out = value.get("model", self.llm.model), value.get("output_tokens", 0)
        if is_content_filtered(finish):
            stage = "judges" if self.role.startswith("judge_") else self.role
            also = f" (the fallback {self.fallback.model} was asked too)" if self._has_fallback() else ""
            raise LLMContentFiltered(
                f"{self.role}: the provider's content filter stopped {model}'s answer "
                f"(finish_reason={finish!r}, {out} tokens){also}. The same prompt to the same model would be "
                "stopped again: give this stage another model with "
                f"TabSchemaConfig(stage_models={{{stage!r}: ...}}) (CLI --stage-model {stage}=MODEL) or a "
                "fallback_model (--fallback-model MODEL)"
            )
        if is_truncated(finish):
            raise self.unusable(
                LLMTruncated(
                    f"{self.role}: the answer stopped at the model's output limit ({out} tokens), so it is "
                    "incomplete; raise the backend's max_tokens (whole_run_max_tokens for the aggregator, "
                    "the judges and the MetaJudge): a rerun then asks again"
                ),
                [prompt],
            )

    def forget(self, prompts: list[str]) -> None:
        """Drop the stored answers to ``prompts``, so a rerun asks again instead of replaying an answer that
        could not be used: the fallback's answer when the stage's model was filtered (that filtered answer
        is kept: the same prompt would be stopped again), else the stage's model's answer."""
        if self.store is None:
            return
        for p in prompts:
            primary = self.store.peek("llm", self._key(p))
            filtered = isinstance(primary, dict) and is_content_filtered(primary.get("finish_reason"))
            if filtered and self._has_fallback():
                self.store.forget("llm", self._key(p, self.fallback))
            else:
                self.store.forget("llm", self._key(p))

    def _ask(self, llm: LLM, prompt: str, fallback_from: str = "") -> LLMResponse:
        """One paid call to ``llm``: budgeted, metered, logged, and retried while it fails transiently (a
        timeout is retried ``retry.timeout_retries`` times, then raised as ``LLMTimedOut``; an empty answer,
        a proxy's error, is retried like a transient failure)."""
        waited = 0.0
        attempt = timeouts = 0
        while True:
            attempt += 1
            self.meter.check()  # a spent budget stops every stage before its next paid call
            est = self.meter.reserve(self.role, llm.model, len(prompt))  # and so does one about to be
            t0 = time.monotonic()
            try:
                r = llm.complete(prompt, role=self.role)
            except Exception as e:  # sorted below: retried, refused, filtered, or raised as it is
                self.meter.release(est)
                latency = time.monotonic() - t0
                if is_content_policy_error(e):  # the provider refused the prompt itself: a filtered answer
                    r = LLMResponse("", llm.model, 0, 0, latency, "content_filter")
                    self._log(llm, r, fallback_from, attempt, len(prompt), error=e)
                    return r
                too_long, timed_out = is_prompt_too_long(e), is_timeout(e)
                transient = timed_out or is_transient(e)
                timeouts += timed_out
                hint = retry_after(e) if transient else None
                if hint is not None and hint > self.retry.max_retry_after_s:
                    self._log_failure(llm, e, "error", attempt, len(prompt), latency, 0.0)
                    raise LLMUnavailable(
                        f"{self.role}: {llm.model} asked to wait {hint:.0f}s before the next call (more than "
                        f"{self.retry.max_retry_after_s:.0f}s): rerun later with the same run directory to "
                        "resume"
                    ) from e
                wait = self.retry.wait(attempt, hint) if transient else 0.0
                spent = timed_out and timeouts > self.retry.timeout_retries
                last = (
                    not transient
                    or spent
                    or attempt >= self.retry.attempts
                    or waited + wait > self.retry.max_total_s
                )
                event = "prompt_too_long" if too_long else ("retry" if not last else "error")
                self._log_failure(llm, e, event, attempt, len(prompt), latency, 0.0 if last else wait)
                if too_long:
                    raise LLMPromptTooLong(
                        f"{self.role}: {llm.model} refused the prompt ({len(prompt):,} characters) as longer "
                        f"than its context window ({short(e, 200)})"
                    ) from e
                if not transient:
                    raise
                if spent:
                    raise LLMTimedOut(
                        f"{self.role}: {llm.model} timed out {timeouts} times on this prompt "
                        f"({short(e, 160)}): its answer takes longer than the request timeout "
                        "(TabSchemaConfig.request_timeout_s, --request-timeout), so it is handled like an "
                        "answer cut at the output limit. The provider may have billed the timed-out attempts"
                    ) from e
                if last:
                    raise LLMUnavailable(
                        f"{self.role}: {llm.model} failed {attempt} times in a row ({short(e, 200)}), after "
                        f"{waited:.0f}s of backoff; rerun with the same run directory to resume (a failed "
                        "attempt is usually not billed; one that timed out may have been)"
                    ) from e
                self._backoff(llm, short(e, 160), attempt, wait)
                waited += wait
                continue
            self.meter.release(est)
            # A paid answer is kept (and cached) even when it crosses the budget; the check before the next
            # call stops the run, and a resumed run does not pay for this one again.
            self.meter.record(self.role, r, check=False, prompt_chars=len(prompt))
            self._log(llm, r, fallback_from, attempt, len(prompt))
            if _empty(r):  # no choices, or nothing said: a proxy's error, never stored
                wait = self.retry.wait(attempt)
                if attempt >= self.retry.attempts or waited + wait > self.retry.max_total_s:
                    raise LLMUnavailable(
                        f"{self.role}: {llm.model} returned an empty answer {attempt} times in a row "
                        f"(finish_reason={r.finish_reason!r}); rerun with the same run directory to resume"
                    )
                self._backoff(llm, f"an empty answer (finish_reason={r.finish_reason!r})", attempt, wait)
                waited += wait
                continue
            return r

    def _backoff(self, llm: LLM, what: str, attempt: int, wait: float) -> None:
        logger.warning(
            "TabSchema %s: %s on %s (attempt %d of %d); retrying in %.1fs",
            self.role,
            what,
            llm.model,
            attempt,
            self.retry.attempts,
            wait,
        )
        _sleep(wait)

    def _log(
        self,
        llm: LLM,
        r: LLMResponse,
        fallback_from: str,
        attempt: int,
        prompt_chars: int,
        error: BaseException | None = None,
    ) -> None:
        if self.store is None:
            return
        self.store.log_call(
            {
                "stage": self.role,
                "model": r.model or llm.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_s": round(r.latency_s, 3),
                "finish_reason": r.finish_reason,
                "prompt_chars": prompt_chars,
                **({"attempt": attempt} if attempt > 1 else {}),
                **({"fallback_from": fallback_from} if fallback_from else {}),
                **({"event": "content_policy", "error": short(error)} if error is not None else {}),
            }
        )

    def _log_failure(
        self,
        llm: LLM,
        e: BaseException,
        event: str,
        attempt: int,
        prompt_chars: int,
        latency: float,
        wait: float,
    ) -> None:
        if self.store is None:
            return
        self.store.log_call(
            {
                "stage": self.role,
                "model": llm.model,
                "event": event,
                "error": short(e),
                "status": status_code(e),
                "attempt": attempt,
                "wait_s": round(wait, 2),
                "latency_s": round(latency, 3),
                "prompt_chars": prompt_chars,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )

    def items(self, out: Any, keys: tuple[str, ...], prompt: str) -> list[dict[str, Any]]:
        """The objects an answer lists: under the first of ``keys`` it has, the answer itself when it is a
        list, or its only list of objects. An answer with none of these (a scalar, an object of other fields)
        is unusable: ``LLMFormatError``, marked like an unparseable answer."""
        if isinstance(out, list):
            return json_items(out)
        if isinstance(out, dict):
            for k in keys:
                if k in out:
                    return json_items(out[k])
            lists = [v for v in out.values() if isinstance(v, list) and any(isinstance(x, dict) for x in v)]
            if len(lists) == 1:
                return json_items(lists[0])
        error = LLMFormatError(f"{self.role}: the answer holds no {keys[0]} list: {str(out)[:200]!r}")
        raise self.unusable(error, [prompt, prompt + JSON_REPAIR])

    def ask_json(self, prompt: str) -> Any:
        """Call the LLM and parse its answer as JSON, re-asking up to ``json_retries`` times."""
        asked = [prompt]
        text = self.call(prompt)
        for attempt in range(self.json_retries + 1):
            try:
                return extract_json(text)
            except ValueError:
                if attempt == self.json_retries:
                    error = LLMFormatError(f"{self.role}: unparseable answer: {text[:300]!r}")
                    raise self.unusable(error, asked) from None
                asked.append(prompt + JSON_REPAIR)
                text = self.call(asked[-1])
        raise RuntimeError("unreachable")


def _empty(r: LLMResponse) -> bool:
    """An answer with no choices, or with nothing in it and no reason to be empty (not filtered, not cut)."""
    finish = str(r.finish_reason or "").lower()
    if finish == "empty":
        return True
    return not str(r.text or "").strip() and not is_content_filtered(finish) and not is_truncated(finish)


def _stored(r: LLMResponse, limit: int | None = None) -> dict[str, Any]:
    """What the run directory keeps of one answer (for one cut at the output limit, the limit it was asked
    with, so a rerun with a higher one asks again)."""
    out: dict[str, Any] = {
        "text": r.text,
        "model": r.model,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
    }
    if r.finish_reason and (is_truncated(r.finish_reason) or is_content_filtered(r.finish_reason)):
        out["finish_reason"] = r.finish_reason
    if is_truncated(r.finish_reason) and (r.max_tokens or limit):
        out["max_tokens"] = r.max_tokens or limit
    return out


def forget_cause(error: BaseException) -> int:
    """Drop from the run directory the answers marked as the cause of ``error`` (``Stage.unusable``), so a
    rerun asks for them again; returns how many. Each failed part of a stage marks its own answer by its own
    error: a filtered answer is never marked (the same prompt to the same model would be stopped again), while
    an unusable or truncated one is, whatever error ended the run."""
    n = 0
    for stage, prompts_ in getattr(error, "forget", None) or []:
        stage.forget(prompts_)
        n += len(prompts_)
    return n


def in_halves(
    stage: Stage, fn, items: list, depth: int = SPLIT_DEPTH, keep: tuple[type, ...] = ()
) -> list[tuple[list, Any, Any]]:
    """``fn(items)`` for one batch of a whole-run stage, as ``[(items, result, None)]``. When the answer
    cannot be used (unparseable, truncated, timed out, filtered, or its prompt too long), each half of the
    batch is asked again, down to ``depth`` halvings; a part that still fails comes back as ``(items, None,
    error)``. Errors of the types in ``keep`` are raised instead of split (halving cannot help them). The
    halvings and the failures are recorded in ``stage.events``. A spent budget or a provider that stays down
    is raised: those stop the run."""
    try:
        return [(items, fn(items), None)]
    except LLMFormatError as e:
        if keep and isinstance(e, keep):
            raise
        names = [_item_name(x) for x in items]
        if len(items) > 1 and depth > 0:
            stage.note("split", items=len(items), error=short(e))
            logger.warning(
                "TabSchema %s: an unusable answer for %d items (%s); asking for each half",
                stage.role,
                len(items),
                short(e, 120),
            )
            mid = (len(items) + 1) // 2
            parts = in_halves(stage, fn, items[:mid], depth - 1, keep) + in_halves(
                stage, fn, items[mid:], depth - 1, keep
            )
            for _, _, err in parts:  # a failure that stops the run drops this answer too (``forget_cause``)
                if err is not None:
                    err.forget = [*(getattr(err, "forget", None) or []), *(getattr(e, "forget", None) or [])]
            return parts
        stage.note("unanswered", items=len(items), features=names[:50], error=short(e))
        return [(items, None, e)]


def _item_name(x: Any) -> str:
    name = getattr(x, "feature_name", None)
    if name is None and isinstance(x, dict):
        name = x.get("feature_name")
    return str(name or "")


def all_failed(parts: list[tuple[list, Any, Any]]) -> BaseException | None:
    """The error to raise when every part of a whole-run stage failed (the first one, marked with every
    part's unusable answers), else None."""
    errors = [e for _, _, e in parts]
    if not parts or any(e is None for e in errors):
        return None
    first = errors[0]
    marks = [m for e in errors for m in (getattr(e, "forget", None) or [])]
    if marks:
        first.forget = marks
    return first


# --------------------------------------------------------------------------- analyzer
def _json_len(value: Any) -> int:
    return len(json.dumps(value, default=str, ensure_ascii=False))


def _longest_list(value: Any) -> list | None:
    """The list inside a JSON value whose JSON is longest (lists of fewer than 3 items excluded)."""
    best, best_len = None, -1
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, list):
            if len(v) > 2:
                n = _json_len(v)
                if n > best_len:
                    best, best_len = v, n
            stack.extend(v)
    return best


def fit_for_prompt(value: Any, budget: int) -> Any:
    """``value`` as a prompt shows it within ``budget`` characters of JSON: itself when it fits; else with its
    longest strings clipped (:func:`~tabagent.tabschema.trajectory.clip_for_prompt`); and when even that is
    too long (a history of many short steps), with the middle of its longest lists left out, keeping their
    first and latest items and saying how many were omitted."""
    out = clip_for_prompt(value, budget)
    if out is value or _json_len(out) <= budget:
        return out
    for _ in range(64):  # each pass halves one list
        lst = _longest_list(out)
        if lst is None:
            break
        n = len(lst)
        keep = max(2, n // 2)
        head, tail = lst[: keep // 2], lst[n - (keep - keep // 2) :]
        lst[:] = [*head, f"... {n - len(head) - len(tail)} items omitted to fit the prompt ...", *tail]
        if _json_len(out) <= budget:
            break
    return out


def analyzer_prompt(dp: DecisionPoint, target: str, budget: int = 400_000) -> str:
    """The FeatureAnalyzer prompt for ``dp``: its prefix fit to ``budget`` characters of JSON
    (:func:`fit_for_prompt`), the schema of the full prefix, the target step (fit to a quarter of the
    budget)."""
    return prompts.render(
        "feature_analyzer",
        trajectory_json=_dump(fit_for_prompt(dp.prefix, budget)),
        trajectory_schema=union_schema([dp.prefix]),
        target_name=target,
        target_json=_dump(fit_for_prompt(dp.target_step, budget // 4)),
        trajectory_file=dp.source or dp.trajectory_id,
    )


class FeatureAnalyzer(Stage):
    """Proposes features from one decision point: its prefix, schema and the target step. Prefixes whose JSON
    exceeds ``max_chars`` characters reach the prompt with their longest strings clipped
    (:func:`fit_for_prompt`); the schema always describes the full prefix. When the provider refuses the
    prompt as longer than the model's context, it is sent again with the prefix fit to half the budget (twice
    at most). At most ``MAX_PROPOSALS`` proposals of an answer are kept."""

    role = "analyzer"
    max_chars = 400_000

    def prompt(self, dp: DecisionPoint, target: str, budget: int | None = None) -> str:
        """The FeatureAnalyzer prompt for ``dp`` (its prefix fit to ``budget`` characters, default
        ``max_chars``)."""
        return analyzer_prompt(dp, target, budget or self.max_chars)

    def __call__(self, dp: DecisionPoint, target: str) -> list[FeatureSpec]:
        budget = self.max_chars
        for tries in range(3):
            try:
                prompt = self.prompt(dp, target, budget)
                out = self.ask_json(prompt)
                break
            except LLMPromptTooLong:
                if tries == 2:
                    raise
                budget //= 2
                self.note("clipped", decision_point=dp.id, max_chars=budget)
        specs = [
            FeatureSpec.from_dict(x) for x in self.items(out, ("potential_features", "features"), prompt)
        ]
        specs = [s for s in specs if s.key]
        if len(specs) > MAX_PROPOSALS:
            self.note("capped", decision_point=dp.id, proposed=len(specs), kept=MAX_PROPOSALS)
            specs = specs[:MAX_PROPOSALS]
        for s in specs:
            s.trajectory_file = s.trajectory_file or dp.source
        return specs


# --------------------------------------------------------------------------- extraction code
def generate_retry_prompt(retry_count: int, error: str, failed_code: str) -> str:
    """The retry block appended to the CodeExtractor prompt after a failed attempt (App. D.1): the error, the
    failed code and the instruction to fix the root cause."""
    if retry_count == 0:
        return ""
    return f"""
PREVIOUS ATTEMPT FAILED:
Error Message: {error}

Failed Code:
```python
{failed_code}
```

Please fix the issues identified above.

RETRY INSTRUCTIONS (Attempt #{retry_count + 1}):
- Analyze the previous error and fix the root cause
- Make sure all required variables are properly defined and named correctly
"""


#: Candidates of the decision an extractor or validator prompt shows, the characters each of their values
#: keeps there, and the characters of JSON a candidate's catalog entry (``spec``) keeps there.
EXAMPLE_CANDIDATES, CANDIDATE_VALUE_CHARS, SPEC_PROMPT_CHARS = 3, 300, 2_000


def shown_candidate(c: dict, value_chars: int = CANDIDATE_VALUE_CHARS) -> dict:
    """A candidate dict as a prompt shows it: long values clipped, and its catalog entry (``spec``) fit to
    ``SPEC_PROMPT_CHARS`` characters of JSON (:func:`fit_for_prompt`: long strings clipped, then the middle
    of long lists left out)."""
    out = {}
    for k, v in c.items():
        if k == "spec" and v is not None:
            out[k] = fit_for_prompt(copy.deepcopy(v), SPEC_PROMPT_CHARS)
        else:
            out[k] = clip_strings(v, value_chars)
    return out


def candidate_examples(candidates: list[dict] | None) -> list[dict] | None:
    """The decision's first ``EXAMPLE_CANDIDATES`` candidate dicts as the prompts show them
    (:func:`shown_candidate`; None without candidates)."""
    if not candidates:
        return None
    return [shown_candidate(c) for c in candidates[:EXAMPLE_CANDIDATES]]


class CodeExtractor(Stage):
    """Writes extraction code for a list of feature plans against a trajectory schema. The paper's prompt is
    followed by the release's additions (``prompts.ADDITIONS``): always the one-form-per-text-feature rule,
    and, for a decision with ``candidates`` (example candidate dicts), the per-candidate function."""

    role = "extractor"

    def __call__(
        self,
        specs: list[FeatureSpec],
        schema_example: dict,
        target: str,
        retry_block: str = "",
        candidates: list[dict] | None = None,
    ) -> str:
        return self.write(specs, schema_example, target, retry_block, candidates)[0]

    def prompt(
        self,
        specs: list[FeatureSpec],
        schema_example: dict,
        target: str,
        retry_block: str = "",
        candidates: list[dict] | None = None,
    ) -> str:
        """The CodeExtractor prompt for the feature plans ``specs`` (``candidates``: example candidates of a
        decision that has them)."""
        spec_dicts = [{k: v for k, v in s.to_dict().items() if k in SPEC_FIELDS} for s in specs]
        paper = prompts.render(
            "code_extractor",
            potential_features=_dump(spec_dicts),
            trajectory_schema=union_schema([schema_example]),
            features_schema=get_json_schema(spec_dicts),
            target_name=target,
            retry_block=retry_block,
        )
        examples = candidate_examples(candidates)
        return prompts.append(
            paper,
            prompts.render("tabagent_extractor"),
            prompts.render("tabagent_extractor_candidates", candidate_examples=_dump(examples))
            if examples
            else "",
        )

    def write(
        self,
        specs: list[FeatureSpec],
        schema_example: dict,
        target: str,
        retry_block: str = "",
        candidates: list[dict] | None = None,
    ) -> tuple[str, list[str]]:
        """The code, and the prompts asked for it (the prompt, and the request for tags when the first answer
        had none)."""
        prompt = self.prompt(specs, schema_example, target, retry_block, candidates)
        text = self.call(prompt)
        try:
            return extract_code(text), [prompt]
        except ValueError:
            pass
        retry = prompt + "\n\nReturn the code between <code> and </code> tags."
        try:
            return extract_code(self.call(retry)), [prompt, retry]
        except ValueError:
            error = LLMFormatError(f"{self.role}: no code in the answer: {text[:300]!r}")
            raise self.unusable(error, [prompt, retry]) from None


@dataclass
class ExtractionAttempt:
    """One generated extractor: its code, whether it passed, per-prefix results, issues and the error."""

    code: str
    ok: bool
    results: list[ExecResult] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    error: str = ""
    prompts: list[str] = field(default_factory=list, repr=False)  # the CodeExtractor prompts of this attempt
    #: Why the per-candidate function failed too often (the attempt is ``ok`` when the code ran on enough
    #: prefixes: its prefix features stand, and a retry asks for a fix of the per-candidate function).
    candidate_error: str = ""


class CodeExecutor:
    """Deterministic checks, then sandboxed execution over one or more trajectory prefixes."""

    def __init__(
        self,
        sandbox: SubprocessSandbox | None = None,
        known_paths: set | None = None,
        outcome_keys: tuple[str, ...] = (),
        max_failure_rate: float = 0.0,
    ):
        self.sandbox = sandbox or SubprocessSandbox()
        self.known_paths, self.outcome_keys, self.max_failure_rate = (
            known_paths,
            outcome_keys,
            max_failure_rate,
        )

    def run(
        self, code: str, prefixes: list[dict], candidates: list[list[dict] | None] | None = None
    ) -> ExtractionAttempt:
        """Check ``code`` statically, then run it on every prefix; fail above ``max_failure_rate``. With
        ``candidates`` (aligned with ``prefixes``) and code that defines the per-candidate function, it runs
        on each prefix's candidates too; when it fails on more than ``max_failure_rate`` of the (prefix,
        candidate) pairs, the attempt keeps its prefix features and says why in ``candidate_error``."""
        fatal = {"syntax", "import", "call", "dunder", "internals", "uses_get", "unknown_field"}
        fatal |= {"outcome_field", "clock"}
        issues = [i for i in checks.check_code(code, self.known_paths, self.outcome_keys) if i.kind in fatal]
        if issues:
            msg = "The code violates the extraction rules:\n" + "\n".join(f"- {i}" for i in issues)
            return ExtractionAttempt(code, False, issues=[str(i) for i in issues], error=msg)
        from tabagent.tabschema.compile import with_flatten  # compile imports this module

        if candidates is not None and not checks.defines_candidate_function(code):
            candidates = None
        if candidates is None:
            results = self.sandbox.run_batch(with_flatten(code), prefixes)
        else:
            results = self.sandbox.run_batch(with_flatten(code), prefixes, candidates=candidates)
        out_issues: list[str] = []
        for n, r in enumerate(results):  # an output that is not a flat dict of JSON scalars is a failure
            bad = [str(i) for i in checks.check_output(r.features)] if r.ok else []
            if bad:
                out_issues += bad
                results[n] = ExecResult(False, error="\n".join(bad))
            elif r.ok and r.candidates is not None:
                for k, v in enumerate(r.candidates):
                    wrong = [str(i) for i in checks.check_output(v)] if v is not None else []
                    if wrong:
                        out_issues += [f"extract_candidate_features: {w}" for w in wrong]
                        r.candidates[k] = None
                        if len(r.candidate_errors) < MAX_CANDIDATE_ERRORS:
                            r.candidate_errors.append("\n".join(wrong))
        failed = [r for r in results if not r.ok]
        rate = len(failed) / max(1, len(results))
        if rate > self.max_failure_rate:
            error = str(failed[0].error)
            if len(prefixes) > 1:
                error = f"{len(failed)}/{len(results)} trajectories failed. First errors:\n" + "\n---\n".join(
                    str(r.error) for r in failed[:3]
                )
            return ExtractionAttempt(code, False, results, sorted(set(out_issues)), error)
        attempt = ExtractionAttempt(code, True, results, sorted(set(out_issues)))
        if candidates is not None:
            attempt.candidate_error = candidate_failures(results, candidates, self.max_failure_rate)
        return attempt


def candidate_failures(
    results: list[ExecResult], candidates: list[list[dict] | None], max_failure_rate: float
) -> str:
    """Why the per-candidate function failed on more than ``max_failure_rate`` of the (prefix, candidate)
    pairs of the prefixes it ran on, with its first distinct errors, for the retry prompt; "" otherwise."""
    pairs = failed = 0
    errors: list[str] = []
    for r, cands in zip(results, candidates, strict=True):
        if not r.ok or not cands:
            continue
        pairs += len(cands)
        per = r.candidates if r.candidates is not None else [None] * len(cands)
        failed += sum(v is None for v in per)
        errors += [e for e in r.candidate_errors if e not in errors]
    if not pairs or failed / pairs <= max_failure_rate:
        return ""
    first = "\n---\n".join(errors[:3]) or "no features were returned"
    return (
        f"extract_candidate_features failed on {failed} of {pairs} candidates (the prefix features are "
        f"fine). First errors:\n{first}"
    )


def extract_with_retries(
    extractor: CodeExtractor,
    executor: CodeExecutor,
    specs: list[FeatureSpec],
    prefixes: list[dict],
    schema_example: dict,
    target: str,
    max_retries: int = 3,
    candidates: list[list[dict] | None] | None = None,
) -> tuple[ExtractionAttempt, list[ExtractionAttempt]]:
    """Generate, check and execute extraction code; on failure retry up to ``max_retries`` times with the
    error and the failed code in the prompt (App. D.1). Returns the final attempt and the history.

    ``candidates`` (aligned with ``prefixes``) are the decision's candidate dicts: the prompt shows three of
    them, and the per-candidate function runs on them. An attempt whose per-candidate function fails too often
    (``candidate_error``) is retried like a failed one; when no later attempt does better, the last such
    attempt is returned, its prefix features intact."""
    history: list[ExtractionAttempt] = []
    errors: list[str] = []
    retry_block = ""
    write = getattr(extractor, "write", None)
    examples = None
    if candidates is not None:
        examples = next((c for c in candidates if c), None)
    more = {} if examples is None else {"candidates": examples}
    fallback: ExtractionAttempt | None = None
    for n in range(max_retries + 1):
        try:
            if write is not None:
                code, asked = write(specs, schema_example, target, retry_block, **more)
            else:
                code, asked = extractor(specs, schema_example, target, retry_block, **more), []
        except LLMContentFiltered:  # the same prompt to the same model would be stopped again
            raise
        except (
            LLMFormatError
        ) as e:  # an answer without code (or cut short) is a failed attempt like any other
            asked = [p for _, ps in getattr(e, "forget", None) or [] for p in ps]
            attempt = ExtractionAttempt("", False, error=str(e), prompts=asked)
        else:
            attempt = (
                executor.run(code, prefixes)
                if candidates is None
                else executor.run(code, prefixes, candidates)
            )
            attempt.prompts = asked
        history.append(attempt)
        if attempt.ok and not attempt.candidate_error:
            if fallback is not None:
                fallback.results = []
            return attempt, history
        if attempt.ok:  # the prefix features stand; the per-candidate function is asked to be fixed
            if fallback is not None:
                fallback.results = []
            fallback = attempt
        else:
            attempt.results = []  # a failed attempt is kept for its error; its per-row results are not needed
        errors.append(attempt.error or attempt.candidate_error)
        retry_block = generate_retry_prompt(n + 1, previous_errors(errors), attempt.code)
    return (fallback or history[-1]), history


def previous_errors(errors: list[str], limit: int = 6000) -> str:
    """The errors of the failed attempts so far, for the retry prompt: the paper concatenates the prior traces
    (App. D.1). One error is passed as is; several are labeled by attempt, each clipped, newest kept."""
    if len(errors) == 1:
        return errors[0]
    text = "\n\n".join(f"[attempt {i}] {e[:2000]}" for i, e in enumerate(errors, 1))
    return text[-limit:]


# --------------------------------------------------------------------------- validator and aggregator
VALID = "valid"
VERDICTS = (VALID, "hallucinated_field", "plan_mismatch", "missing")
#: Verdicts that reject a proposal at analysis (an unrecognized verdict leaves the proposal to the later
#: stages).
REJECTING = frozenset(VERDICTS) - {VALID}


def normalize_verdict(value: Any) -> str:
    """Map a CodeValidator verdict onto ``valid``, ``hallucinated_field``, ``plan_mismatch`` or ``missing``,
    tolerating case, punctuation and trailing explanations (``"Valid."``, ``"hallucinated field"``)."""
    v = normalize_name(value)
    return next((x for x in VERDICTS if v.startswith(x)), v)


#: Keys and characters per value of each sample output the CodeValidator sees. Code that returns a dict per
#: row (word counts) can produce thousands of keys, which would fill the prompt.
SAMPLE_OUTPUT_KEYS, SAMPLE_VALUE_CHARS = 150, 300


def sample_output(features: Any) -> Any:
    """One extractor output as the CodeValidator sees it: at most ``SAMPLE_OUTPUT_KEYS`` keys (with a count of
    the rest) and long values clipped."""
    if not isinstance(features, dict):
        return features
    keys = list(features)
    out = {
        k: (v[:SAMPLE_VALUE_CHARS] + "..." if isinstance(v, str) and len(v) > SAMPLE_VALUE_CHARS else v)
        for k, v in ((k, features[k]) for k in keys[:SAMPLE_OUTPUT_KEYS])
    }
    if len(keys) > SAMPLE_OUTPUT_KEYS:
        out["..."] = f"{len(keys) - SAMPLE_OUTPUT_KEYS} more keys not shown"
    return out


#: Per-candidate outputs a CodeValidator prompt shows at most.
SAMPLE_CANDIDATE_OUTPUTS = 6


def candidate_samples(results: list[ExecResult], candidates: list[list[dict] | None]) -> list[dict]:
    """Per-candidate outputs for the CodeValidator: up to two candidates of each of the first prefixes the
    function ran on, each output with the candidate it was computed for (``SAMPLE_CANDIDATE_OUTPUTS`` in
    all)."""
    out: list[dict] = []
    for r, cands in zip(results, candidates, strict=False):
        if not r.ok or not r.candidates or not cands:
            continue
        done = [(c, v) for c, v in zip(cands, r.candidates, strict=False) if v is not None][:2]
        out += [{"candidate": shown_candidate(c), "features": sample_output(v)} for c, v in done]
        if len(out) >= SAMPLE_CANDIDATE_OUTPUTS:
            break
    return out[:SAMPLE_CANDIDATE_OUTPUTS]


class CodeValidator(Stage):
    """Checks that executed extraction code reads only real fields and follows each feature's plan; returns
    a verdict per feature (``valid``, ``hallucinated_field``, ``plan_mismatch``). For a decision with
    ``candidates`` (example candidate dicts), the release's addition (``tabagent_validator_candidates``) shows
    them and the per-candidate function's ``candidate_outputs``."""

    role = "validator"

    def __call__(
        self,
        specs: list[FeatureSpec],
        code: str,
        schema_example: dict,
        outputs: list[dict],
        target: str,
        candidates: list[dict] | None = None,
        candidate_outputs: list[dict] | None = None,
    ) -> dict[str, str]:
        return self.judge(specs, code, schema_example, outputs, target, candidates, candidate_outputs)[0]

    def judge(
        self,
        specs: list[FeatureSpec],
        code: str,
        schema_example: dict,
        outputs: list[dict],
        target: str,
        candidates: list[dict] | None = None,
        candidate_outputs: list[dict] | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """The verdict per feature and the validator's evidence for it (the code or schema detail)."""
        spec_dicts = [{k: v for k, v in s.to_dict().items() if k in SPEC_FIELDS} for s in specs]
        prompt = prompts.render(
            "code_validator",
            target_name=target,
            feature_specs=_dump(spec_dicts),
            trajectory_schema=union_schema([schema_example]),
            code=code,
            sample_output=_dump([sample_output(o) for o in outputs[:3]]),
        )
        examples = candidate_examples(candidates)
        if examples:
            prompt = prompts.append(
                prompt,
                prompts.render(
                    "tabagent_validator_candidates",
                    candidate_examples=_dump(examples),
                    candidate_outputs=_dump(candidate_outputs or []),
                ),
            )
        out = self.ask_json(prompt)
        items = self.items(out, ("features", "verdicts"), prompt)
        verdicts: dict[str, str] = {}
        evidence: dict[str, str] = {}
        for x in items:  # the first verdict on each feature
            key = normalize_name(x.get("feature_name") or "")
            if key and key not in verdicts:
                verdicts[key] = normalize_verdict(x.get("verdict") or "")
                evidence[key] = str(x.get("evidence") or "")[:600]
        return verdicts, evidence


class FeatureAggregator(Stage):
    """Merges the proposals of all analyzed decision points; returns (kept, rejected) features, one per
    normalized name. More than ``batch`` proposals are aggregated ``batch`` at a time (in the given order,
    most supported first), so that no single answer has to restate every proposal.

    A batch whose answer cannot be used is asked again in halves (:func:`in_halves`); proposals that still get
    no usable answer are kept as proposed, unaggregated (listed in ``unaggregated`` and in ``events``), unless
    no batch at all could be aggregated, which raises. An answer keeps at most as many features as its batch
    proposed (an aggregator merges; it does not invent)."""

    role = "aggregator"
    batch = 40

    def __call__(self, specs: list[FeatureSpec]) -> tuple[list[FeatureSpec], list[FeatureSpec]]:
        support = {s.key: s.support for s in specs}
        kept: dict[str, FeatureSpec] = {}
        rejected: dict[str, FeatureSpec] = {}
        self.unaggregated: list[str] = []
        batches = similar_together(specs, self.batch)
        parts = [
            p
            for ps in in_parallel(lambda b: in_halves(self, self._aggregate, b), batches, self.workers)
            for p in ps
        ]
        error = all_failed(parts)
        if error is not None:
            raise error
        for items, result, failed in parts:  # merged in batch order
            if failed is not None:
                for s in items:
                    kept.setdefault(s.key, s)
                    self.unaggregated.append(s.feature_name)
                continue
            k, r = result
            first: dict[str, FeatureSpec] = {}
            for s in k:  # one per name, the first one
                first.setdefault(s.key, s)
            k = list(first.values())
            if len(k) > len(items):
                self.note("trimmed", proposed=len(items), kept=len(k))
                k = k[: len(items)]
            for s in k:
                kept.setdefault(s.key, s)
            for s in r:
                rejected.setdefault(s.key, s)
        for s in [*kept.values(), *rejected.values()]:
            s.support = support.get(s.key, s.support)
        return list(kept.values()), [s for k, s in rejected.items() if k not in kept]

    def _aggregate(self, specs: list[FeatureSpec]) -> tuple[list[FeatureSpec], list[FeatureSpec]]:
        dicts = [s.to_dict() for s in specs]
        prompt = prompts.render(
            "feature_aggregator", potential_features=_dump(dicts), features_schema=get_json_schema(dicts)
        )
        out = self.ask_json(prompt)
        if isinstance(out, list):  # a bare list: the features it keeps
            out = {"filtered_features": out}
        if not (isinstance(out, dict) and {"filtered_features", "rejected_features"} & set(out)):
            out = {"filtered_features": self.items(out, ("filtered_features",), prompt)}

        def parse(items: Any) -> list[FeatureSpec]:
            return [s for s in map(FeatureSpec.from_dict, json_items(items)) if s.key]

        return parse(out.get("filtered_features")), parse(out.get("rejected_features"))


#: Name words that say how a feature is encoded, not what it is about (``has_user_id``, ``user_id_present``).
_ENCODING_WORDS = frozenset(
    [
        "has",
        "is",
        "flag",
        "present",
        "provided",
        "count",
        "num",
        "number",
        "of",
        "the",
        "in",
        "any",
        "bool",
        "indicator",
        "mentioned",
        "mention",
        "mentions",
        "keyword",
        "keywords",
        "n",
        "total",
    ]
)


def _topic(spec: FeatureSpec) -> tuple[str, ...]:
    """The words of a feature's name that say what it is about, for placing likely duplicates together."""
    words = [w for w in re.split(r"[^a-z0-9]+", spec.feature_name.lower()) if w]
    return tuple(sorted(set(words) - _ENCODING_WORDS)) or tuple(words)


def similar_together(specs: list[FeatureSpec], batch: int) -> list[list[FeatureSpec]]:
    """Split ``specs`` into batches of at most ``batch``, with features about the same thing (by the words of
    their names) in the same batch, so the FeatureAggregator, which merges only within a call, sees likely
    duplicates side by side and decides on them itself. Groups keep the given order (most supported first);
    a group is split only when it is larger than a batch."""
    groups: dict[tuple[str, ...], list[FeatureSpec]] = {}
    for s in specs:
        groups.setdefault(_topic(s), []).append(s)
    batches: list[list[FeatureSpec]] = [[]]
    for group in groups.values():
        if len(batches[-1]) + len(group) > batch and batches[-1]:
            batches.append([])
        for s in group:
            if len(batches[-1]) == batch:
                batches.append([])
            batches[-1].append(s)
    return [b for b in batches if b]


def premerge(specs: list[FeatureSpec]) -> list[FeatureSpec]:
    """Deduplicate proposals by normalized name, keeping the first description and counting support."""
    merged: dict[str, FeatureSpec] = {}
    for s in specs:
        if s.key in merged:
            merged[s.key].support += 1
        else:
            merged[s.key] = FeatureSpec(**{**s.to_dict(), "support": 1})
    return sorted(merged.values(), key=lambda s: (-s.support, s.key))
