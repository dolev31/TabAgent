"""What a TabSchema discovery will cost, before it runs (``onboard(dry_run=True)``,
``tabagent onboard --dry-run``).

The estimate follows the pipeline stage by stage and builds what it can from the data itself:

* the **FeatureAnalyzer** prompts are rendered exactly, for the very decision points the run will analyze (the
  same sample, the same clipping, and, for a decision with candidates, the same section on features of a
  candidate with the point's own candidates);
* the **CodeExtractor** and **CodeValidator** prompts are the templates plus each analyzed point's own schema
  (exact) plus the feature plans and code the answers will hold (estimated); the retries carry the failed code
  and its error;
* the **FeatureAggregator**, the three **judges** and the **MetaJudge** see the distinct proposals, in batches
  of 40 and 25, as the run sends them;
* the **compile** step writes one extractor against the schema merged over the onboarding decision points
  (exact) for the compiled features, and the CodeValidator reviews it with sample outputs; one **repair**
  round (one more CodeExtractor and CodeValidator call) is counted, since it is asked whenever an accepted
  feature has no column, is flagged or indexes candidates by position, which most real runs have;
* for a decision with candidates, the CodeExtractor and CodeValidator prompts carry the release's section on
  the per-candidate function, with three of the decision's candidates (measured on the data), and the
  FeatureAggregator, judge and MetaJudge prompts its note on features of a candidate;
* the **OutputAuditor** prompts hold the audited points' prefixes, clipped as the run clips them (exact),
  and up to 48 features each.

How the LLM behaves (how many features it proposes, how many survive each stage, how long its answers are, how
often code needs a retry) cannot be read from the data: ``Profile`` holds those numbers, measured on real
discovery runs (Claude Opus 5.5 with GPT-6 Sol writing the code; the public τ²-bench GPT-4.1 logs, next tool
call and tool shortlist, and two private agent-log sets with prefixes of up to 160,000 tokens). Characters are
turned into tokens with the model's measured ratio (``llm.chars_per_token``: 2.3 characters per token for
Claude models, 3.9 for others), the main source of the earlier estimate's error on long prompts, where 4 was
assumed for every model.

On those runs the estimate lands within 30% of the actual cost (``tests/tabschema/test_estimate.py`` checks
this on the public runs' recorded sizes, and on scripted runs). A model that answers much more or less
verbosely than the profile, or data whose analyzed points propose unusually few distinct features, moves the
actual cost away from it; ``tabagent runs show`` gives the actual cost of each stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from tabagent.tabschema import prompts
from tabagent.tabschema.agents import (
    MAX_PROPOSALS,
    _dump,
    analyzer_prompt,
    candidate_examples,
    fit_for_prompt,
    pick_examples,
)
from tabagent.tabschema.audit import AUDIT_MAX_CHARS, AUDIT_MAX_COLUMNS
from tabagent.tabschema.judges import BATCH as JUDGE_BATCH
from tabagent.tabschema.llm import chars_per_token
from tabagent.tabschema.trajectory import candidate_dict, merge_examples, union_schema

AGGREGATOR_BATCH = 40
#: Decision points the compile step's merged schema is estimated on at most (it is a union: more add little).
SCHEMA_SAMPLE = 400


@dataclass(frozen=True)
class Profile:
    """How the LLMs behave in a discovery run, as measured on real runs (see the module docstring). Sizes are
    characters of prompt text or tokens of answers; shares are of the previous stage's output."""

    proposals_per_point: float = 15.0  # features one FeatureAnalyzer answer proposes
    distinct_share: float = 0.65  # of all proposals, the share left after merging identical names
    kept_share: float = 0.62  # of the distinct proposals, the share the FeatureAggregator keeps
    decision_share: tuple[tuple[str, float], ...] = (
        ("accept", 0.37),
        ("conditional", 0.37),
        ("reject", 0.26),
    )
    extractor_calls_per_point: float = 2.3  # code attempts per analyzed point, retries included
    compile_extractor_calls: float = 2.0  # code attempts of the compile step, its retries included
    compile_validator_calls: float = 1.0  # CodeValidator reviews of the compile step's code
    compile_repair_calls: float = 1.0  # the repair round: one more CodeExtractor and CodeValidator call
    candidate_outputs_chars: int = 2_400  # the per-candidate outputs a CodeValidator prompt shows
    spec_chars: int = 650  # one feature plan as the prompts list it
    meta_row_chars: int = 2_900  # one MetaJudge table row
    code_chars: int = 10_000  # the code of one analyzed point's extractor
    compiled_code_chars: int = 30_000  # the compiled extractor's code
    error_chars: int = 2_500  # the error fed back with a retry
    columns_per_feature: float = 1.6  # compiled columns per compiled feature (sub-keys, expanded dicts)
    audit_item_chars: int = 500  # one feature in an auditor prompt: its plan and value
    output_value_chars: int = 60  # one value of a sample output in a CodeValidator prompt
    # answer tokens
    analyzer_out: int = 4_500
    extractor_out: int = 3_900
    validator_out: int = 2_900
    aggregator_out_per_proposal: int = 240
    judge_out_per_feature: int = 350
    meta_out_per_feature: int = 400
    auditor_out_per_column: int = 70


@dataclass
class DataSummary:
    """The sizes of the run's prompts that the data decides, measured on the data (see :func:`summarize`)."""

    analyzer_chars: list[int]  # the rendered FeatureAnalyzer prompt of each analyzed point
    schema_chars: list[int]  # each analyzed point's schema as the CodeExtractor and CodeValidator see it
    merged_schema_chars: int  # the schema merged over the decision points (the compile step's)
    audit_prefix_chars: list[int] = field(default_factory=list)  # each audited point's clipped prefix
    decision_points: int = 0
    candidate_chars: int = 0  # the decision's example candidates as the prompts show them (0: no candidates)


@dataclass
class StageEstimate:
    """One stage's estimated calls, tokens and cost (None without a price for its model)."""

    calls: float
    input_tokens: int
    output_tokens: int
    model: str
    usd: float | None = None


@dataclass
class Estimate:
    """The estimate of a discovery run, per stage and in total."""

    stages: dict[str, StageEstimate]
    proposals: float = 0.0
    features: float = 0.0
    compiled: float = 0.0

    @property
    def input_tokens(self) -> int:
        return int(sum(s.input_tokens for s in self.stages.values()))

    @property
    def output_tokens(self) -> int:
        return int(sum(s.output_tokens for s in self.stages.values()))

    @property
    def usd(self) -> float | None:
        costs = [s.usd for s in self.stages.values()]
        return None if any(c is None for c in costs) else float(sum(costs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": {k: vars(v) for k, v in self.stages.items()},
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd": self.usd,
            "proposals": round(self.proposals, 1),
            "features": round(self.features, 1),
            "compiled": round(self.compiled, 1),
        }


def _schema_len(value: Any) -> int:
    return len(str(union_schema([value])))


def summarize(
    sample: list,
    decision_points: list,
    target: str,
    max_chars: int = 400_000,
    audit_sample: list | None = None,
    candidates_of: Any = None,
    seed: int = 0,
) -> DataSummary:
    """Measure the data-dependent prompt sizes: the FeatureAnalyzer prompts of the analyzed ``sample``
    (rendered as the run renders them), each analyzed point's schema, the schema merged over (a sample of) the
    ``decision_points``, the clipped prefixes of the ``audit_sample``, and the example candidates the
    CodeExtractor and CodeValidator prompts show (``candidates_of``: the adapter's candidates, for a decision
    that has them; the FeatureAnalyzer prompts show each analyzed point's own, picked with the run's
    ``seed`` as the run picks them)."""
    step = max(1, len(decision_points) // SCHEMA_SAMPLE)
    merged = merge_examples([dp.prefix for dp in decision_points[::step]])

    def cands(dp) -> list[dict] | None:
        return None if candidates_of is None else [candidate_dict(c) for c in candidates_of(dp) or ()]

    shown = 0
    for dp in decision_points[:50] if candidates_of is not None else []:
        if found := cands(dp):
            shown = len(_dump(candidate_examples(pick_examples(found, seed, dp.id))))
            break
    return DataSummary(
        analyzer_chars=[len(analyzer_prompt(dp, target, max_chars, cands(dp), seed)) for dp in sample],
        schema_chars=[_schema_len(dp.prefix) for dp in sample],
        merged_schema_chars=len(str(union_schema([merged]))),
        audit_prefix_chars=[
            len(_dump(fit_for_prompt(dp.prefix, AUDIT_MAX_CHARS))) for dp in audit_sample or []
        ],
        decision_points=len(decision_points),
        candidate_chars=shown,
    )


def estimate(
    data: DataSummary,
    *,
    model: str,
    prices: dict[str, tuple[float, float]] | None = None,
    stage_models: dict[str, str] | None = None,
    accept: tuple[str, ...] = ("accept", "conditional"),
    audit: bool = True,
    compile_repair: bool = True,
    max_code_retries: int = 3,
    profile: Profile | None = None,
    chars_per_token_of: Any = None,
) -> Estimate:
    """The estimated calls, tokens and cost of each stage of a discovery run on ``data`` (see the module
    docstring). ``model`` is the run's model and ``stage_models`` its per-stage models (priced from
    ``prices``, USD per 1M input and output tokens); ``chars_per_token_of(model)`` overrides the tokenizer
    ratio."""
    pr = profile or Profile()
    ratio = chars_per_token_of or chars_per_token
    models = stage_models or {}
    text = prompts.text
    n = len(data.analyzer_chars)
    proposals = n * min(pr.proposals_per_point, MAX_PROPOSALS)
    distinct = pr.distinct_share * proposals if n > 1 else proposals
    kept = pr.kept_share * distinct
    shares = dict(pr.decision_share)
    compiled = kept * sum(shares.get(d, 0.0) for d in set(accept))
    columns = compiled * pr.columns_per_feature
    per_point_specs = pr.proposals_per_point * pr.spec_chars

    def model_of(stage: str) -> str:
        if stage.startswith("judge_"):
            return models.get(stage) or models.get("judges") or model
        return models.get(stage) or model

    stages: dict[str, tuple[float, float, float]] = {}  # stage -> (calls, input chars, output tokens)
    stages["analyzer"] = (n, sum(data.analyzer_chars), n * pr.analyzer_out)

    # per analyzed point: the first attempt, then retries carrying the failed code and its error
    calls_per_point = min(pr.extractor_calls_per_point, max_code_retries + 1)
    retries = max(0.0, calls_per_point - 1)
    extractor_prompt = len(text("code_extractor")) + len(text("tabagent_extractor"))  # and its addition
    validator_prompt = len(text("code_validator"))
    if data.candidate_chars:  # the sections on the per-candidate function, with the example candidates
        extractor_prompt += len(text("tabagent_extractor_candidates")) + data.candidate_chars
        validator_prompt += (
            len(text("tabagent_validator_candidates")) + data.candidate_chars + pr.candidate_outputs_chars
        )
    ext_base = extractor_prompt + per_point_specs + pr.proposals_per_point * 60
    ext_in = sum(
        calls_per_point * (ext_base + s) + retries * (pr.code_chars + pr.error_chars)
        for s in data.schema_chars
    )
    val_in = sum(
        validator_prompt
        + per_point_specs
        + s
        + pr.code_chars
        + pr.proposals_per_point * pr.output_value_chars
        for s in data.schema_chars
    )
    # the compile step: one extractor for the compiled features, against the merged schema
    comp_specs = compiled * pr.spec_chars
    repair_calls = pr.compile_repair_calls if compile_repair else 0.0
    comp_ext_calls = pr.compile_extractor_calls + repair_calls
    comp_retries = max(
        0.0, comp_ext_calls - 1
    )  # each later call carries the code and the error or objections
    comp_ext_in = comp_ext_calls * (
        extractor_prompt + comp_specs + compiled * 60 + data.merged_schema_chars
    ) + comp_retries * (pr.compiled_code_chars + pr.error_chars)
    comp_val_calls = pr.compile_validator_calls + repair_calls
    comp_val_in = comp_val_calls * (
        validator_prompt
        + comp_specs
        + data.merged_schema_chars
        + pr.compiled_code_chars
        + 3 * min(columns, 150) * pr.output_value_chars
    )
    stages["extractor"] = (
        n * calls_per_point + comp_ext_calls,
        ext_in + comp_ext_in,
        (n * calls_per_point + comp_ext_calls) * pr.extractor_out,
    )
    stages["validator"] = (
        n + comp_val_calls,
        val_in + comp_val_in,
        (n + comp_val_calls) * pr.validator_out,
    )

    note = len(text("tabagent_candidate_note")) if data.candidate_chars else 0  # with candidates
    agg_calls = math.ceil(distinct / AGGREGATOR_BATCH) if distinct else 0
    stages["aggregator"] = (
        agg_calls,
        agg_calls * (len(text("feature_aggregator")) + note + 60 * AGGREGATOR_BATCH)
        + distinct * pr.spec_chars,
        distinct * pr.aggregator_out_per_proposal,
    )
    judge_calls = math.ceil(kept / JUDGE_BATCH) if kept else 0
    for dim in ("relevance", "generality", "impact"):
        stages[f"judge_{dim}"] = (
            judge_calls,
            judge_calls * (len(text(f"judge_{dim}")) + note + 60 * JUDGE_BATCH) + kept * pr.spec_chars,
            kept * pr.judge_out_per_feature,
        )
    stages["meta_judge"] = (
        judge_calls,
        judge_calls * (len(text("meta_judge")) + note + 3 * 300) + kept * pr.meta_row_chars,
        kept * pr.meta_out_per_feature,
    )
    if audit and data.audit_prefix_chars:
        items = min(AUDIT_MAX_COLUMNS, columns)
        k = len(data.audit_prefix_chars)
        stages["auditor"] = (
            k,
            sum(
                len(text("output_auditor")) + c + items * pr.audit_item_chars for c in data.audit_prefix_chars
            ),
            k * items * pr.auditor_out_per_column,
        )

    out: dict[str, StageEstimate] = {}
    for stage, (calls, chars, tout) in stages.items():
        name = model_of(stage)
        tin = int(chars / ratio(name))
        price = (prices or {}).get(name)
        usd = None if price is None else (tin * price[0] + tout * price[1]) / 1e6
        out[stage] = StageEstimate(round(calls, 1), tin, int(tout), name, usd)
    return Estimate(out, proposals=proposals, features=kept, compiled=compiled)
