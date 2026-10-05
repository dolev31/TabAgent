"""The OutputAuditor: an LLM judge of the values the compiled extractor computes. This stage is the release's
addition to TabSchema; the paper has no such stage.

The CodeValidator (paper App. D.1) reads the extraction *code* and a few of its outputs, and the
compile-time checks (``verify``) test the code's behavior without knowing what a value should be. A
regex for a reservation code that also matches any six-letter uppercase word passes both. The
OutputAuditor reads the decision point the way the FeatureAnalyzer did and checks the values themselves:

1. sample ``audit_samples`` decision points with the coverage sampler (new tasks first, then uncovered
   positions in the run, prefix lengths and labels), among the points the FeatureAnalyzer did not see (the
   extractor was written with those in view). When fewer such points exist, every one of them is audited
   and analyzed points fill the rest; the record and the report count them (``analyzed_points``);
2. per point, one call: the clipped prefix, inside data delimiters the prompt tells the model never to
   follow as instructions, and, for every LLM-written extractor feature, its plan (name, description,
   extraction source, computation) and the value the extractor produced. The answer is ``correct``,
   ``incorrect`` or ``cannot_tell`` per feature, with a one-line reason (``prompts/output_auditor.txt``,
   written for the release). An answer that repeats the template or hedges counts as ``cannot_tell``;
3. per feature, the incorrect rate over the points where it was judged (``cannot_tell`` does not count).
   A feature is dropped only when at least ``MIN_INCORRECT`` of at least ``MIN_JUDGED`` judged points are
   incorrect **and** the one-sided 95% Wilson lower bound of its incorrect rate exceeds
   ``audit_max_incorrect``, so a few judge errors do not drop a correct feature. The auditor's reasons are
   recorded. The audit never empties a card: when it would drop every LLM-written feature, it drops none and
   says so prominently (the paid run is kept).

The auditor is a stage like the others (role ``auditor``): ``TabSchemaConfig.stage_models`` can route it,
and it is metered, budgeted and cached in the run directory. The audit only removes features, so an
unusable answer, a content filter or a provider error on one point loses only that point. When no point
gets a usable answer, or the budget runs out first, the audit is skipped with a note and the extractor is
kept as compiled.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from typing import Any

from tabagent.tabschema import prompts
from tabagent.tabschema.agents import Stage, _dump, in_parallel, json_items, normalize_name
from tabagent.tabschema.llm import BudgetExceeded
from tabagent.tabschema.sampling import sample_decision_points
from tabagent.tabschema.trajectory import DecisionPoint, clip_for_prompt, clip_strings

#: The prefix's JSON budget in an auditor prompt (longer prefixes have their longest strings clipped).
AUDIT_MAX_CHARS = 60_000
#: Feature values longer than this reach the auditor clipped.
VALUE_CHARS = 500
#: Columns one auditor call judges at most. A wide extractor (one column per tool or word) would otherwise put
#: hundreds of plans in one prompt and truncate the answer; each point audits a share of the columns instead,
#: spread over the planned features and preferring the values that are set at that point.
AUDIT_MAX_COLUMNS = 48
#: A feature is dropped only when it was judged (correct or incorrect) on at least ``MIN_JUDGED`` points, at
#: least ``MIN_INCORRECT`` of them incorrect, and the one-sided Wilson lower bound (``WILSON_Z``) of its
#: incorrect rate exceeds the threshold.
MIN_JUDGED, MIN_INCORRECT, WILSON_Z = 3, 2, 1.645
VERDICTS = ("correct", "incorrect", "cannot_tell")


def normalize_audit_verdict(value: Any) -> str:
    """Map an auditor verdict onto ``correct``, ``incorrect`` or ``cannot_tell`` by its leading word
    (``"incorrect, it cannot be 3"`` is incorrect). An answer that lists several verdicts (an echoed
    ``correct/incorrect/cannot_tell``) or hedges (``partially correct``) is ``cannot_tell``."""
    text = str(value)
    v = normalize_name(text)
    words = v.split("_")
    listed = sum(w in ("correct", "incorrect", "cannot") for w in words)
    if ("/" in text or "|" in text) and listed > 1:
        return "cannot_tell"
    if words[0] in ("incorrect", "wrong", "false") or v.startswith("not_correct"):
        return "incorrect"
    if words[0] == "correct":
        return "correct"
    return "cannot_tell"


def wilson_lower(k: int, n: int, z: float = WILSON_Z) -> float:
    """The Wilson score lower bound of a rate of ``k`` in ``n`` (0 when ``n`` is 0)."""
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin)


class OutputAuditor(Stage):
    """Judges, for one decision point, whether each feature value is what its plan describes; returns
    ``{normalized feature name: (verdict, reason)}``."""

    role = "auditor"
    max_chars = AUDIT_MAX_CHARS

    def __call__(self, prefix: dict, items: list[dict], target: str) -> dict[str, tuple[str, str]]:
        prompt = prompts.render(
            "output_auditor",
            target_name=target,
            trajectory_json=_dump(clip_for_prompt(prefix, self.max_chars)),
            features=_dump(items),
        )
        out = self.ask_json(prompt)
        rows = out.get("features") if isinstance(out, dict) else out
        return {
            normalize_name(x.get("feature_name") or ""): (
                normalize_audit_verdict(x.get("verdict") or ""),
                str(x.get("reason") or "")[:300],
            )
            for x in json_items(rows)
        }


def _is_set(value: Any) -> bool:
    return value not in (None, 0, False, "", "none", "None") and value == value  # NaN is not set


def audit_columns(
    columns: dict[str, dict], features: dict[str, Any], limit: int = AUDIT_MAX_COLUMNS
) -> list[str]:
    """At most ``limit`` columns to audit at one point: round-robin over the planned features (each column's
    ``feature_name``), and within a feature the columns whose value is set at this point first, then by name.
    Deterministic, so a rerun asks the same questions (and reuses the cached answers)."""
    groups: dict[str, list[str]] = {}
    for col, plan in columns.items():
        groups.setdefault(str(plan.get("feature_name") or col), []).append(col)
    queues = [
        sorted(cols, key=lambda c: (not _is_set(features.get(c)), c)) for _, cols in sorted(groups.items())
    ]
    out: list[str] = []
    while len(out) < limit and any(queues):
        for q in queues:
            if q and len(out) < limit:
                out.append(q.pop(0))
    return out


def audit_items(
    columns: dict[str, dict], features: dict[str, Any], limit: int = AUDIT_MAX_COLUMNS
) -> list[dict[str, Any]]:
    """The auditor's view of one decision point: at most ``limit`` columns (see :func:`audit_columns`), each
    with its plan and the value the extractor produced (null when it produced none; long values clipped)."""
    items = []
    for col in audit_columns(columns, features, limit):
        plan = columns[col]
        item = {
            "feature_name": col,
            "feature_type": plan.get("feature_type", ""),
            "description": plan.get("description", ""),
            "extraction_source": plan.get("extraction_source", ""),
            "computation": plan.get("computation", ""),
            "value": clip_strings(features.get(col), VALUE_CHARS),
        }
        if normalize_name(plan.get("feature_name", col)) != normalize_name(col):
            item["part_of_feature"] = plan.get("feature_name", "")
        items.append(item)
    return items


def run_audit(
    auditor: OutputAuditor,
    columns: dict[str, dict],
    decision_points: list[DecisionPoint],
    outputs: list[dict | None],
    target: str,
    *,
    samples: int,
    max_incorrect: float,
    seed: int,
    exclude: frozenset[str] | set[str] = frozenset(),
    workers: int = 1,
    label_of: Callable[[DecisionPoint], Iterable[Any]] | None = None,
) -> dict[str, Any]:
    """Audit the extractor's values for ``columns`` (``{column: plan}``) on a sample of the decision points it
    ran on (``outputs`` aligned with ``decision_points``; ``exclude``: the analyzed points; ``label_of``: the
    adapter's labels, for the coverage sample; see the module docstring). Returns the JSON-able audit record,
    masked; its ``dropped`` maps each feature to drop to the reason."""
    from tabagent.tabschema.verify import mask_secrets

    ran = [dp for dp, out in zip(decision_points, outputs, strict=True) if out is not None]
    values = {dp.id: out for dp, out in zip(decision_points, outputs, strict=True) if out is not None}
    unseen = [dp for dp in ran if dp.id not in exclude]
    sample = sample_decision_points(unseen, samples, seed, label_of, "coverage")
    if len(sample) < samples:  # every unanalyzed point, and analyzed ones for the rest (counted)
        seen = [dp for dp in ran if dp.id in exclude]
        sample += sample_decision_points(seen, samples - len(sample), seed, label_of, "coverage")
    rec: dict[str, Any] = {
        "samples": len(sample),
        "analyzed_points": sum(dp.id in exclude for dp in sample),
        "max_incorrect": max_incorrect,
        "min_judged": MIN_JUDGED,
        "min_incorrect": MIN_INCORRECT,
        "answered": 0,
        "errors": [],
        "features": {},
        "points": [],
        "dropped": {},
        "skipped": None,
    }
    if not sample or not columns:
        rec["skipped"] = "nothing to audit (no decision point with features, or no LLM-written feature)"
        return mask_secrets(rec, by_key=False)

    def one(dp: DecisionPoint) -> dict[str, Any]:
        try:
            verdicts = auditor(dp.prefix, audit_items(columns, values[dp.id]), target)
        except BudgetExceeded as e:
            return {"decision_point": dp.id, "error": f"BudgetExceeded: {e}", "budget": True}
        except Exception as e:  # an unusable answer, a content filter, a provider error: this point only
            return {"decision_point": dp.id, "error": f"{type(e).__name__}: {e}"[:300]}
        if not verdicts:
            return {"decision_point": dp.id, "error": "the answer judged no feature"}
        return {"decision_point": dp.id, "verdicts": verdicts}

    results = in_parallel(one, sample, workers)
    rec["errors"] = [r["error"] for r in results if "error" in r]
    answered = [r for r in results if "verdicts" in r]
    rec["answered"] = len(answered)
    if not answered:
        budget = any(r.get("budget") for r in results)
        first = rec["errors"][0] if rec["errors"] else "no answer"
        rec["skipped"] = (
            "the LLM budget ran out before the audit" if budget else f"no usable auditor answer ({first})"
        )
        return mask_secrets(rec, by_key=False)
    for col in columns:
        counts = dict.fromkeys(VERDICTS, 0)
        counts["not_judged"] = 0
        examples = []
        for r in answered:
            verdict, reason = r["verdicts"].get(normalize_name(col), (None, ""))
            if verdict is None:
                counts["not_judged"] += 1
                continue
            counts[verdict] += 1
            if verdict == "incorrect" and len(examples) < 3:
                value = clip_strings(values[r["decision_point"]].get(col), 120)
                examples.append({"decision_point": r["decision_point"], "value": value, "reason": reason})
        judged = counts["correct"] + counts["incorrect"]
        rate = counts["incorrect"] / judged if judged else None
        lower = wilson_lower(counts["incorrect"], judged)
        rec["features"][col] = {
            **counts,
            "incorrect_rate": None if rate is None else round(rate, 3),
            "incorrect_lower_bound": round(lower, 3),
            "examples": examples,
        }
        drop = judged >= MIN_JUDGED and counts["incorrect"] >= MIN_INCORRECT and lower > max_incorrect
        if drop:
            why = (
                f"; e.g. {json.dumps(examples[0]['value'], default=str)}: {examples[0]['reason']}"
                if examples
                else ""
            )
            rec["dropped"][col] = (
                f"OutputAuditor: incorrect on {counts['incorrect']} of {judged} judged decision points "
                f"({rate:.0%}; its 95% lower bound {lower:.0%} > {max_incorrect:.0%}){why}"
            )
    rec["points"] = [
        {"decision_point": r["decision_point"], "verdicts": {k: list(v) for k, v in r["verdicts"].items()}}
        for r in answered
    ]
    return mask_secrets(rec, by_key=False)


def estimate(prefixes: list[dict], n_features: int = 20) -> tuple[int, int]:
    """Estimated (input, output) tokens of auditing ``prefixes`` with ``n_features`` features each, at most
    ``AUDIT_MAX_COLUMNS`` per point (4 characters per token): the template, the clipped prefix, the features'
    plans and values; about 60 output tokens per feature."""
    n_features = min(n_features, AUDIT_MAX_COLUMNS)
    template = len(prompts.text("output_auditor"))
    chars = sum(
        template + len(_dump(clip_for_prompt(p, AUDIT_MAX_CHARS))) + n_features * 450 for p in prefixes
    )
    return chars // 4, len(prefixes) * n_features * 60
