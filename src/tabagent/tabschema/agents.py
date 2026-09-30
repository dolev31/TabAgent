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
from dataclasses import asdict, dataclass, field
from typing import Any

from tabagent.tabschema import checks, prompts
from tabagent.tabschema.llm import LLM, CostMeter, extract_code, extract_json
from tabagent.tabschema.runstore import RunStore
from tabagent.tabschema.sandbox import ExecResult, SubprocessSandbox
from tabagent.tabschema.trajectory import DecisionPoint, get_json_schema

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


class LLMFormatError(ValueError):
    """The model's answer could not be parsed after the allowed repair attempts."""


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
        known = {
            k: (str(v) if k != "support" else int(v)) for k, v in d.items() if k in cls.__dataclass_fields__
        }
        known.setdefault("feature_name", str(d.get("name", "")))
        return cls(**known)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        return normalize_name(self.feature_name)


def normalize_name(name: str) -> str:
    """Lowercase ``name`` and collapse each run of non-alphanumerics to ``_`` (the key features match on)."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


class Stage:
    """Base of the LLM-backed stages: one prompt in, one answer out, metered and optionally cached."""

    role = "stage"

    def __init__(
        self, llm: LLM, meter: CostMeter | None = None, store: RunStore | None = None, json_retries: int = 1
    ):
        self.llm, self.meter, self.store, self.json_retries = llm, meter or CostMeter(), store, json_retries

    def call(self, prompt: str) -> str:
        """One LLM call. With a run store, calls are content-addressed: a resumed or repeated run reuses
        stored answers instead of paying for them again."""
        if self.store is None:
            return self._live(prompt).text
        key = {
            "role": self.role,
            "model": self.llm.model,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        return self.store.cached("llm", key, lambda: self._live(prompt).text)

    def _live(self, prompt: str):
        self.meter.check()  # a spent budget stops every stage before its next paid call
        r = self.llm.complete(prompt, role=self.role)
        self.meter.record(self.role, r)
        if self.store is not None:
            self.store.log_call(
                {
                    "stage": self.role,
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "latency_s": round(r.latency_s, 3),
                }
            )
        return r

    def ask_json(self, prompt: str) -> Any:
        """Call the LLM and parse its answer as JSON, re-asking up to ``json_retries`` times."""
        text = self.call(prompt)
        for attempt in range(self.json_retries + 1):
            try:
                return extract_json(text)
            except ValueError:
                if attempt == self.json_retries:
                    raise LLMFormatError(f"{self.role}: unparseable answer: {text[:300]!r}") from None
                text = self.call(prompt + JSON_REPAIR)
        raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- analyzer
class FeatureAnalyzer(Stage):
    """Proposes features from one decision point: its prefix, schema and the target step."""

    role = "analyzer"

    def __call__(self, dp: DecisionPoint, target: str) -> list[FeatureSpec]:
        prompt = prompts.render(
            "feature_analyzer",
            trajectory_json=_dump(dp.prefix),
            trajectory_schema=get_json_schema(dp.prefix),
            target_name=target,
            target_json=_dump(dp.target_step),
            trajectory_file=dp.source or dp.trajectory_id,
        )
        out = self.ask_json(prompt)
        items = out.get("potential_features", []) if isinstance(out, dict) else out
        specs = [FeatureSpec.from_dict(x) for x in items if isinstance(x, dict)]
        for s in specs:
            s.trajectory_file = s.trajectory_file or dp.source
        return [s for s in specs if s.feature_name]


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
            trajectory_schema=get_json_schema(schema_example),
            features_schema=get_json_schema(spec_dicts),
            target_name=target,
            retry_block=retry_block,
        )
        text = self.call(prompt)
        try:
            return extract_code(text)
        except ValueError:
            return extract_code(self.call(prompt + "\n\nReturn the code between <code> and </code> tags."))


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
        fatal = {"syntax", "import", "call", "uses_get", "unknown_field", "outcome_field"}
        issues = [i for i in checks.check_code(code, self.known_paths, self.outcome_keys) if i.kind in fatal]
        if issues:
            msg = "The code violates the extraction rules:\n" + "\n".join(f"- {i}" for i in issues)
            return ExtractionAttempt(code, False, issues=[str(i) for i in issues], error=msg)
        results = self.sandbox.run_batch(code, prefixes)
        failed = [r for r in results if not r.ok]
        out_issues = [str(i) for r in results if r.ok for i in checks.check_output(r.features)]
        rate = len(failed) / max(1, len(results))
        if rate > self.max_failure_rate or (out_issues and len(prefixes) == 1):
            error = failed[0].error if failed else "\n".join(sorted(set(out_issues))[:10])
            if len(prefixes) > 1 and failed:
                error = f"{len(failed)}/{len(results)} trajectories failed. First errors:\n" + "\n---\n".join(
                    r.error for r in failed[:3]
                )
            return ExtractionAttempt(code, False, results, out_issues, error)
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
    retry_block = ""
    for n in range(max_retries + 1):
        code = extractor(specs, schema_example, target, retry_block)
        attempt = executor.run(code, prefixes)
        history.append(attempt)
        if attempt.ok:
            return attempt, history
        retry_block = generate_retry_prompt(n + 1, attempt.error, code)
    return history[-1], history


# --------------------------------------------------------------------------- validator and aggregator
VALID = "valid"


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
            trajectory_schema=get_json_schema(schema_example),
            code=code,
            sample_output=_dump(outputs[:3]),
        )
        out = self.ask_json(prompt)
        items = out.get("features", []) if isinstance(out, dict) else out
        return {
            normalize_name(x.get("feature_name", "")): str(x.get("verdict", "")).strip().lower()
            for x in items
            if isinstance(x, dict)
        }


class FeatureAggregator(Stage):
    """Merges the proposals of all analyzed decision points; returns (kept, rejected) features."""

    role = "aggregator"

    def __call__(self, specs: list[FeatureSpec]) -> tuple[list[FeatureSpec], list[FeatureSpec]]:
        dicts = [s.to_dict() for s in specs]
        out = self.ask_json(
            prompts.render(
                "feature_aggregator", potential_features=_dump(dicts), features_schema=get_json_schema(dicts)
            )
        )
        support = {s.key: s.support for s in specs}

        def parse(items):
            res = [FeatureSpec.from_dict(x) for x in items or [] if isinstance(x, dict)]
            for s in res:
                s.support = support.get(s.key, s.support)
            return [s for s in res if s.feature_name]

        return parse(out.get("filtered_features")), parse(out.get("rejected_features"))


def premerge(specs: list[FeatureSpec]) -> list[FeatureSpec]:
    """Deduplicate proposals by normalized name, keeping the first description and counting support."""
    merged: dict[str, FeatureSpec] = {}
    for s in specs:
        if s.key in merged:
            merged[s.key].support += 1
        else:
            merged[s.key] = FeatureSpec(**{**s.to_dict(), "support": 1})
    return sorted(merged.values(), key=lambda s: (-s.support, s.key))
