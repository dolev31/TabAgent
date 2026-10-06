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

Per-candidate features (``extract_candidate_features``) are judged on up to ``AUDIT_CANDIDATES`` candidates
of each sampled point, chosen from the seed and the decision point alone (never by label): each value is an
item named ``<column> (candidate <n>)``, and the prompt's addition (``tabagent_auditor_candidates``)
lists the candidates as the code received them. A point's items stay within ``AUDIT_MAX_COLUMNS``, the
per-candidate values of a column counting one item each; a per-candidate column's judged (point, candidate)
pairs decide on it as a prefix column's points do.

The auditor is a stage like the others (role ``auditor``): ``TabSchemaConfig.stage_models`` can route it, and
it is metered, budgeted and cached in the run directory. The audit only removes features, so an unusable
answer on one point loses only that point (an answer cut at the output limit is asked again for each half of
the point's features first), and so does a content filter or a prompt too long once the point's prefix has
been clipped to half its budget and asked again. When no point gets a usable answer, the audit is skipped with
a note and the extractor is kept as compiled. A spent budget, a provider that stays down, or any other
provider error (an invalid key, a model that does not exist, a bad request) stops the run, as at every other
stage: a partial audit would decide on fewer points than the configured one (rerun with the same run
directory, or ``audit=False``).
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Iterable
from typing import Any

from tabagent.tabschema import prompts
from tabagent.tabschema.agents import (
    Stage,
    _dump,
    all_failed,
    fit_for_prompt,
    in_halves,
    in_parallel,
    normalize_name,
    short,
    shown_candidate,
)
from tabagent.tabschema.llm import (
    BudgetExceeded,
    LLMContentFiltered,
    LLMPromptTooLong,
    LLMUnavailable,
    status_code,
)
from tabagent.tabschema.sampling import sample_decision_points
from tabagent.tabschema.trajectory import DecisionPoint, clip_strings

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
#: Candidates of a sampled point whose per-candidate values are judged.
AUDIT_CANDIDATES = 3
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

    def __call__(
        self,
        prefix: dict,
        items: list[dict],
        target: str,
        max_chars: int | None = None,
        candidates: list[dict] | None = None,
    ) -> dict[str, tuple[str, str]]:
        prompt = prompts.render(
            "output_auditor",
            target_name=target,
            trajectory_json=_dump(fit_for_prompt(prefix, max_chars or self.max_chars)),
            features=_dump(items),
        )
        if candidates:  # per-candidate values: the candidates they were computed for
            shown = [{"candidate": n, **shown_candidate(c, VALUE_CHARS)} for n, c in enumerate(candidates, 1)]
            prompt = prompts.append(
                prompt, prompts.render("tabagent_auditor_candidates", candidates=_dump(shown))
            )
        out = self.ask_json(prompt)
        verdicts: dict[str, tuple[str, str]] = {}
        for x in self.items(out, ("features",), prompt):  # the first verdict on each feature
            key = normalize_name(x.get("feature_name") or "")
            if key and key not in verdicts:
                verdicts[key] = (
                    normalize_audit_verdict(x.get("verdict") or ""),
                    str(x.get("reason") or "")[:300],
                )
        return verdicts


def _is_set(value: Any) -> bool:
    return value not in (None, 0, False, "", "none", "None") and value == value  # NaN is not set


def audit_columns(
    columns: dict[str, dict],
    features: dict[str, Any],
    limit: int = AUDIT_MAX_COLUMNS,
    cost: dict[str, int] | None = None,
) -> list[str]:
    """At most ``limit`` columns to audit at one point: round-robin over the planned features (each column's
    ``feature_name``), and within a feature the columns whose value is set at this point first, then by name.
    Deterministic, so a rerun asks the same questions (and reuses the cached answers). ``cost``: the items a
    column takes (a per-candidate column, one per audited candidate; 1 otherwise), all within ``limit``."""
    cost = cost or {}
    groups: dict[str, list[str]] = {}
    for col, plan in columns.items():
        groups.setdefault(str(plan.get("feature_name") or col), []).append(col)
    queues = [
        sorted(cols, key=lambda c: (not _is_set(features.get(c)), c)) for _, cols in sorted(groups.items())
    ]
    out: list[str] = []
    used = 0
    while used < limit and any(queues):
        for q in queues:
            if q and used + cost.get(q[0], 1) <= limit:
                used += cost.get(q[0], 1)
                out.append(q.pop(0))
            elif q:  # it does not fit: nor does the rest of this feature
                q.clear()
    return out


def candidate_item(col: str, n: int) -> str:
    """The name of a per-candidate value's item: ``<column> (candidate <n>)``."""
    return f"{col} (candidate {n})"


def audit_items(
    columns: dict[str, dict],
    features: dict[str, Any],
    limit: int = AUDIT_MAX_COLUMNS,
    candidate_columns: dict[str, dict] | None = None,
    candidate_features: list[dict | None] | None = None,
) -> list[dict[str, Any]]:
    """The auditor's view of one decision point: at most ``limit`` items (see :func:`audit_columns`), each a
    column with its plan and the value the extractor produced (null when it produced none; long values
    clipped). ``candidate_features`` are the per-candidate function's values on the audited candidates: each
    chosen per-candidate column gives one item per candidate (:func:`candidate_item`)."""
    per = candidate_columns or {}
    cands = candidate_features or []
    shown = dict(features)
    for col in per:  # set at this point when set for any audited candidate
        shown[col] = next((v[col] for v in cands if v is not None and _is_set(v.get(col))), None)
    cost = {col: len(cands) for col in per}
    items = []
    for col in audit_columns({**columns, **per}, shown, limit, cost):
        plan = per.get(col) or columns[col]
        values = (
            [(candidate_item(col, n), (v or {}).get(col)) for n, v in enumerate(cands, 1)]
            if col in per
            else [(col, features.get(col))]
        )
        for name, value in values:
            item = {
                "feature_name": name,
                "feature_type": plan.get("feature_type", ""),
                "description": plan.get("description", ""),
                "extraction_source": plan.get("extraction_source", ""),
                "computation": plan.get("computation", ""),
                "value": clip_strings(value, VALUE_CHARS),
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
    progress: Callable[[int, int, int], None] | None = None,
    candidate_columns: dict[str, dict] | None = None,
    candidate_outputs: list[list[dict | None] | None] | None = None,
    candidates_of: Callable[[DecisionPoint], list[dict]] | None = None,
) -> dict[str, Any]:
    """Audit the extractor's values for ``columns`` (``{column: plan}``) on a sample of the decision points it
    ran on (``outputs`` aligned with ``decision_points``; ``exclude``: the analyzed points; ``label_of``: the
    adapter's labels, for the coverage sample; see the module docstring). Returns the JSON-able audit record,
    masked; its ``dropped`` maps each feature to drop to the reason.

    ``candidate_columns`` (``{column: plan}``) are the per-candidate function's columns, with their values
    ``candidate_outputs`` (aligned with ``decision_points``, then with each one's candidates as
    ``candidates_of`` gives them, candidate dicts)."""
    from tabagent.tabschema.verify import mask_secrets, pick_candidates

    per_cols = candidate_columns or {}
    index = {dp.id: i for i, dp in enumerate(decision_points)}

    def audited(dp: DecisionPoint) -> tuple[list[dict], list[dict | None]]:
        """The candidates of ``dp`` whose per-candidate values are judged, and those values."""
        if not per_cols or candidates_of is None or candidate_outputs is None:
            return [], []
        cands = candidates_of(dp)
        values = candidate_outputs[index[dp.id]] or [None] * len(cands)
        chosen = pick_candidates(cands, seed, dp.id, AUDIT_CANDIDATES)
        ids = [id(c) for c in chosen]
        return chosen, [values[k] for k, c in enumerate(cands) if id(c) in ids]

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
    if not sample or not (columns or per_cols):
        rec["skipped"] = "nothing to audit (no decision point with features, or no LLM-written feature)"
        return mask_secrets(rec, by_key=False)
    shown: dict[str, list[dict | None]] = {}  # the per-candidate values each point's verdicts judge

    done = [0, 0]  # points audited, of which failed
    lock = threading.Lock()

    def one(dp: DecisionPoint) -> dict[str, Any]:
        out = judge(dp)
        with lock:
            done[0] += 1
            done[1] += "error" in out
            if progress is not None:
                progress(done[0], len(sample), done[1])
        return out

    def judge(dp: DecisionPoint) -> dict[str, Any]:
        chosen, cand_values = audited(dp)
        shown[dp.id] = cand_values
        items = audit_items(columns, values[dp.id], AUDIT_MAX_COLUMNS, per_cols, cand_values)
        more = {"candidates": chosen} if chosen else {}
        budget = auditor.max_chars
        for tries in range(2):
            try:
                # an answer cut at the output limit, or unusable, is asked again for each half of the
                # features; a filter or a prompt too long is about the prefix: it is clipped harder, once
                parts = in_halves(
                    auditor,
                    lambda its, b=budget: auditor(dp.prefix, its, target, b, **more),
                    items,
                    keep=(LLMContentFiltered, LLMPromptTooLong),
                )
                error = all_failed(parts)
                if error is not None:
                    raise error
                break
            except (LLMContentFiltered, LLMPromptTooLong) as e:
                if tries == 1:
                    return {"decision_point": dp.id, "error": f"{type(e).__name__}: {e}"[:300]}
                budget //= 2
                auditor.note("clipped", decision_point=dp.id, max_chars=budget, error=short(e, 120))
            except (BudgetExceeded, LLMUnavailable):  # the run stops, resumable: no partial audit
                raise
            except Exception as e:  # an unusable answer, or any failure of this point's own: it is skipped
                if (
                    status_code(e) is not None
                ):  # a provider's error (an invalid key, a bad request): the run's
                    raise
                return {"decision_point": dp.id, "error": f"{type(e).__name__}: {e}"[:300]}
        verdicts = {k: v for _, got, _ in parts for k, v in (got or {}).items()}
        if not verdicts:
            return {"decision_point": dp.id, "error": "the answer judged no feature"}
        return {"decision_point": dp.id, "verdicts": verdicts}

    results = in_parallel(one, sample, workers)
    rec["errors"] = [r["error"] for r in results if "error" in r]
    answered = [r for r in results if "verdicts" in r]
    rec["answered"] = len(answered)
    if not answered:
        first = rec["errors"][0] if rec["errors"] else "no answer"
        rec["skipped"] = f"no usable auditor answer ({first})"
        return mask_secrets(rec, by_key=False)
    for col in [*columns, *per_cols]:
        counts = dict.fromkeys(VERDICTS, 0)
        counts["not_judged"] = 0
        examples = []
        for r in answered:
            point = r["decision_point"]
            if col in per_cols:  # one judgment per audited candidate
                pairs = [
                    (
                        r["verdicts"].get(normalize_name(candidate_item(col, n)), (None, "")),
                        (v or {}).get(col),
                    )
                    for n, v in enumerate(shown.get(point) or [], 1)
                ]
            else:
                pairs = [(r["verdicts"].get(normalize_name(col), (None, "")), values[point].get(col))]
            for (verdict, reason), value in pairs:
                if verdict is None:
                    counts["not_judged"] += 1
                    continue
                counts[verdict] += 1
                if verdict == "incorrect" and len(examples) < 3:
                    shown_value = clip_strings(value, 120)
                    examples.append({"decision_point": point, "value": shown_value, "reason": reason})
        judged = counts["correct"] + counts["incorrect"]
        rate = counts["incorrect"] / judged if judged else None
        lower = wilson_lower(counts["incorrect"], judged)
        rec["features"][col] = {
            **counts,
            "incorrect_rate": None if rate is None else round(rate, 3),
            "incorrect_lower_bound": round(lower, 3),
            "examples": examples,
            **({"per_candidate": True} if col in per_cols else {}),
        }
        drop = judged >= MIN_JUDGED and counts["incorrect"] >= MIN_INCORRECT and lower > max_incorrect
        if drop:
            why = (
                f"; e.g. {json.dumps(examples[0]['value'], default=str)}: {examples[0]['reason']}"
                if examples
                else ""
            )
            where = "(decision, candidate) pairs" if col in per_cols else "decision points"
            rec["dropped"][col] = (
                f"OutputAuditor: incorrect on {counts['incorrect']} of {judged} judged {where} "
                f"({rate:.0%}; its 95% lower bound {lower:.0%} > {max_incorrect:.0%}){why}"
            )
    rec["points"] = [
        {"decision_point": r["decision_point"], "verdicts": {k: list(v) for k, v in r["verdicts"].items()}}
        for r in answered
    ]
    return mask_secrets(rec, by_key=False)
