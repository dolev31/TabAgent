"""TabSchema's feature-extraction agents (paper Sec. 3.1; App. D).

FeatureAnalyzer   proposes features from one decision point (App. D.2, FeatureAnalyzer prompt)
CodeExtractor     writes the extraction code for them (App. D.2, CodeExtractor prompt)
CodeExecutor      runs it in the sandbox, with up to three retries that feed errors back (App. D.1)
CodeValidator     an LLM check that the executed code reads only real fields and follows the plan (App. D.1)
FeatureAggregator merges the proposals of many trajectories and drops unobtainable ones (App. D.2)
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from tabagent.tabschema import checks, prompts
from tabagent.tabschema.llm import (
    LLM,
    CostMeter,
    LLMContentFiltered,
    LLMFormatError,
    LLMResponse,
    LLMTruncated,
    extract_code,
    extract_json,
    is_content_filtered,
    is_truncated,
)
from tabagent.tabschema.runstore import RunStore
from tabagent.tabschema.sandbox import ExecResult, SubprocessSandbox
from tabagent.tabschema.trajectory import DecisionPoint, clip_for_prompt, get_json_schema, union_schema

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
        except (TypeError, ValueError):
            support = 1
        return cls(**{k: str(v) for k, v in known.items()}, support=support)

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
    """``[fn(x) for x in items]``, run on up to ``workers`` threads; results keep the order of ``items``."""
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(fn, items))


class Stage:
    """Base of the LLM-backed stages: one prompt in, one answer out, metered and optionally cached.
    ``workers`` bounds the concurrent calls of a stage that splits its work into batches."""

    role = "stage"
    workers = 1

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

    def _key(self, prompt: str) -> dict[str, str]:
        return {
            "role": self.role,
            "model": self.llm.model,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }

    def call(self, prompt: str) -> str:
        """One LLM call. With a run store, calls are content-addressed: a resumed or repeated run reuses
        stored answers instead of paying for them again (the meter counts them as cached, at the cost they
        had when first paid)."""
        if self.store is None:
            return self._live(prompt).text
        paid: list[LLMResponse] = []

        def fetch() -> dict[str, Any]:
            r = self._live(prompt)
            paid.append(r)
            return {
                "text": r.text,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }

        value = self.store.cached("llm", self._key(prompt), fetch)
        if not isinstance(value, dict):  # a cache entry written before usage was stored with the answer
            value = {"text": value}
        if not paid:
            self.meter.record_cached(
                self.role,
                LLMResponse(
                    value["text"],
                    value.get("model", self.llm.model),
                    value.get("input_tokens", 0),
                    value.get("output_tokens", 0),
                ),
            )
        return value["text"]

    def forget(self, prompts: list[str]) -> None:
        """Drop the stored answers to ``prompts``, so a rerun asks again instead of replaying an answer that
        could not be used."""
        if self.store is not None:
            for p in prompts:
                self.store.forget("llm", self._key(p))

    def _live(self, prompt: str) -> LLMResponse:
        r = self._ask(self.llm, prompt)
        if is_content_filtered(r.finish_reason):
            if self.fallback is not None and self.fallback.model != self.llm.model:
                r = self._ask(self.fallback, prompt, fallback_from=self.llm.model)
            if is_content_filtered(r.finish_reason):
                stage = "judges" if self.role.startswith("judge_") else self.role
                raise LLMContentFiltered(
                    f"{self.role}: the provider's content filter stopped {r.model}'s answer "
                    f"(finish_reason={r.finish_reason!r}, {r.output_tokens} tokens). The same prompt to the "
                    "same model would be stopped again: give this stage another model with "
                    f"TabSchemaConfig(stage_models={{{stage!r}: ...}}) (CLI --stage-model {stage}=MODEL) "
                    "or a fallback_model (--fallback-model MODEL)"
                )
        return r

    def _ask(self, llm: LLM, prompt: str, fallback_from: str = "") -> LLMResponse:
        """One paid call to ``llm``: budgeted, metered, logged, and checked for truncation."""
        self.meter.check()  # a spent budget stops every stage before its next paid call
        est = self.meter.reserve(self.role, llm.model, len(prompt))  # and so does one about to be
        try:
            r = llm.complete(prompt, role=self.role)
        finally:
            self.meter.release(est)
        # A paid answer is kept (and cached) even when it crosses the budget; the check before the next call
        # stops the run, and a resumed run does not pay for this one again.
        self.meter.record(self.role, r, check=False)
        if self.store is not None:
            self.store.log_call(
                {
                    "stage": self.role,
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "latency_s": round(r.latency_s, 3),
                    "finish_reason": r.finish_reason,
                    **({"fallback_from": fallback_from} if fallback_from else {}),
                }
            )
        if is_truncated(r.finish_reason):
            raise LLMTruncated(
                f"{self.role}: the answer stopped at the model's output limit ({r.output_tokens} tokens), so "
                "it is incomplete; raise the backend's max_tokens (whole_run_max_tokens for the aggregator, "
                "the judges and the MetaJudge)"
            )
        return r

    def ask_json(self, prompt: str) -> Any:
        """Call the LLM and parse its answer as JSON, re-asking up to ``json_retries`` times."""
        asked = [prompt]
        text = self.call(prompt)
        for attempt in range(self.json_retries + 1):
            try:
                return extract_json(text)
            except ValueError:
                if attempt == self.json_retries:
                    self.forget(asked)
                    raise LLMFormatError(f"{self.role}: unparseable answer: {text[:300]!r}") from None
                asked.append(prompt + JSON_REPAIR)
                text = self.call(asked[-1])
        raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- analyzer
class FeatureAnalyzer(Stage):
    """Proposes features from one decision point: its prefix, schema and the target step. Prefixes whose JSON
    exceeds ``max_chars`` characters reach the prompt with their longest strings clipped
    (:func:`~tabagent.tabschema.trajectory.clip_for_prompt`); the schema always describes the full prefix."""

    role = "analyzer"
    max_chars = 400_000

    def __call__(self, dp: DecisionPoint, target: str) -> list[FeatureSpec]:
        prompt = prompts.render(
            "feature_analyzer",
            trajectory_json=_dump(clip_for_prompt(dp.prefix, self.max_chars)),
            trajectory_schema=union_schema([dp.prefix]),
            target_name=target,
            target_json=_dump(clip_for_prompt(dp.target_step, self.max_chars // 4)),
            trajectory_file=dp.source or dp.trajectory_id,
        )
        out = self.ask_json(prompt)
        items = out.get("potential_features") if isinstance(out, dict) else out
        specs = [FeatureSpec.from_dict(x) for x in json_items(items)]
        for s in specs:
            s.trajectory_file = s.trajectory_file or dp.source
        return [s for s in specs if s.key]


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


class CodeExtractor(Stage):
    """Writes extraction code for a list of feature plans against a trajectory schema."""

    role = "extractor"

    def __call__(
        self, specs: list[FeatureSpec], schema_example: dict, target: str, retry_block: str = ""
    ) -> str:
        spec_dicts = [{k: v for k, v in s.to_dict().items() if k in SPEC_FIELDS} for s in specs]
        prompt = prompts.render(
            "code_extractor",
            potential_features=_dump(spec_dicts),
            trajectory_schema=union_schema([schema_example]),
            features_schema=get_json_schema(spec_dicts),
            target_name=target,
            retry_block=retry_block,
        )
        text = self.call(prompt)
        try:
            return extract_code(text)
        except ValueError:
            pass
        retry = prompt + "\n\nReturn the code between <code> and </code> tags."
        try:
            return extract_code(self.call(retry))
        except ValueError:
            self.forget([prompt, retry])
            raise LLMFormatError(f"{self.role}: no code in the answer: {text[:300]!r}") from None


@dataclass
class ExtractionAttempt:
    """One generated extractor: its code, whether it passed, per-prefix results, issues and the error."""

    code: str
    ok: bool
    results: list[ExecResult] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    error: str = ""


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

    def run(self, code: str, prefixes: list[dict]) -> ExtractionAttempt:
        """Check ``code`` statically, then run it on every prefix; fail above ``max_failure_rate``."""
        fatal = {"syntax", "import", "call", "dunder", "internals", "uses_get", "unknown_field"}
        fatal |= {"outcome_field", "clock"}
        issues = [i for i in checks.check_code(code, self.known_paths, self.outcome_keys) if i.kind in fatal]
        if issues:
            msg = "The code violates the extraction rules:\n" + "\n".join(f"- {i}" for i in issues)
            return ExtractionAttempt(code, False, issues=[str(i) for i in issues], error=msg)
        results = self.sandbox.run_batch(code, prefixes)
        out_issues: list[str] = []
        for n, r in enumerate(results):  # an output that is not a flat dict of JSON scalars is a failure
            bad = [str(i) for i in checks.check_output(r.features)] if r.ok else []
            if bad:
                out_issues += bad
                results[n] = ExecResult(False, error="\n".join(bad))
        failed = [r for r in results if not r.ok]
        rate = len(failed) / max(1, len(results))
        if rate > self.max_failure_rate:
            error = str(failed[0].error)
            if len(prefixes) > 1:
                error = f"{len(failed)}/{len(results)} trajectories failed. First errors:\n" + "\n---\n".join(
                    str(r.error) for r in failed[:3]
                )
            return ExtractionAttempt(code, False, results, sorted(set(out_issues)), error)
        return ExtractionAttempt(code, True, results, sorted(set(out_issues)))


def extract_with_retries(
    extractor: CodeExtractor,
    executor: CodeExecutor,
    specs: list[FeatureSpec],
    prefixes: list[dict],
    schema_example: dict,
    target: str,
    max_retries: int = 3,
) -> tuple[ExtractionAttempt, list[ExtractionAttempt]]:
    """Generate, check and execute extraction code; on failure retry up to ``max_retries`` times with the
    error and the failed code in the prompt (App. D.1). Returns the final attempt and the history."""
    history: list[ExtractionAttempt] = []
    errors: list[str] = []
    retry_block = ""
    for n in range(max_retries + 1):
        try:
            code = extractor(specs, schema_example, target, retry_block)
        except LLMContentFiltered:  # the same prompt to the same model would be stopped again
            raise
        except LLMFormatError as e:  # an answer without code is a failed attempt like any other
            attempt = ExtractionAttempt("", False, error=str(e))
        else:
            attempt = executor.run(code, prefixes)
        history.append(attempt)
        if attempt.ok:
            return attempt, history
        errors.append(attempt.error)
        retry_block = generate_retry_prompt(n + 1, previous_errors(errors), attempt.code)
    return history[-1], history


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


def normalize_verdict(value: Any) -> str:
    """Map a CodeValidator verdict onto ``valid``, ``hallucinated_field``, ``plan_mismatch`` or ``missing``,
    tolerating case, punctuation and trailing explanations (``"Valid."``, ``"hallucinated field"``)."""
    v = normalize_name(value)
    return next((x for x in VERDICTS if v.startswith(x)), v)


class CodeValidator(Stage):
    """Checks that executed extraction code reads only real fields and follows each feature's plan; returns
    a verdict per feature (``valid``, ``hallucinated_field``, ``plan_mismatch``)."""

    role = "validator"

    def __call__(
        self, specs: list[FeatureSpec], code: str, schema_example: dict, outputs: list[dict], target: str
    ) -> dict[str, str]:
        spec_dicts = [{k: v for k, v in s.to_dict().items() if k in SPEC_FIELDS} for s in specs]
        prompt = prompts.render(
            "code_validator",
            target_name=target,
            feature_specs=_dump(spec_dicts),
            trajectory_schema=union_schema([schema_example]),
            code=code,
            sample_output=_dump(outputs[:3]),
        )
        out = self.ask_json(prompt)
        items = out.get("features") if isinstance(out, dict) else out
        return {
            normalize_name(x.get("feature_name") or ""): normalize_verdict(x.get("verdict") or "")
            for x in json_items(items)
        }


class FeatureAggregator(Stage):
    """Merges the proposals of all analyzed decision points; returns (kept, rejected) features, one per
    normalized name. More than ``batch`` proposals are aggregated ``batch`` at a time (in the given order,
    most supported first), so that no single answer has to restate every proposal."""

    role = "aggregator"
    batch = 40

    def __call__(self, specs: list[FeatureSpec]) -> tuple[list[FeatureSpec], list[FeatureSpec]]:
        support = {s.key: s.support for s in specs}
        kept: dict[str, FeatureSpec] = {}
        rejected: dict[str, FeatureSpec] = {}
        batches = similar_together(specs, self.batch)
        for k, r in in_parallel(self._aggregate, batches, self.workers):  # merged in batch order
            for s in k:
                kept.setdefault(s.key, s)
            for s in r:
                rejected.setdefault(s.key, s)
        for s in [*kept.values(), *rejected.values()]:
            s.support = support.get(s.key, s.support)
        return list(kept.values()), [s for k, s in rejected.items() if k not in kept]

    def _aggregate(self, specs: list[FeatureSpec]) -> tuple[list[FeatureSpec], list[FeatureSpec]]:
        dicts = [s.to_dict() for s in specs]
        out = self.ask_json(
            prompts.render(
                "feature_aggregator", potential_features=_dump(dicts), features_schema=get_json_schema(dicts)
            )
        )
        if isinstance(out, list):  # a bare list: the features it keeps
            out = {"filtered_features": out}
        out = out if isinstance(out, dict) else {}

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
