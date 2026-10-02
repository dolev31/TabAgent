"""TabSchema's judge ensemble (paper Sec. 3.1; App. D.1; judge and MetaJudge prompts in App. D.3).

Three judges score every aggregated feature on a 1-5 rubric along distinct quality dimensions (relevance,
generality, impact and uniqueness) with a confidence and a justification. They are independent, so they run
concurrently. A deterministic join builds one row per feature with the three scores and justifications plus
the analyzer's original description, relevance rationale and support; the MetaJudge reads that table and
issues one explained decision per feature (accept / conditional / reject).

The judges and the MetaJudge see at most ``batch`` (25) features per call, so that no answer has to cover
an arbitrarily long feature list; up to that size every feature is judged in one call, as in the paper.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tabagent.tabschema import prompts
from tabagent.tabschema.agents import FeatureSpec, Stage, in_parallel, json_items, normalize_name
from tabagent.tabschema.trajectory import get_json_schema

JUDGES = {"relevance": "judge_relevance", "generality": "judge_generality", "impact": "judge_impact"}
DECISIONS = ("accept", "conditional", "reject")
BATCH = 25  # features per judge or MetaJudge call


class Judge(Stage):
    """One judge (``relevance``, ``generality`` or ``impact``): scores every feature on a 1-5 rubric."""

    def __init__(self, dimension: str, *args, **kw):
        super().__init__(*args, **kw)
        self.dimension, self.role = dimension, f"judge_{dimension}"

    def __call__(self, features: list[FeatureSpec]) -> list[dict[str, Any]]:
        dicts = [f.to_dict() for f in features]
        out = self.ask_json(
            prompts.render(
                JUDGES[self.dimension],
                filtered_features=json.dumps(dicts, indent=2, ensure_ascii=False),
                features_schema=get_json_schema(dicts),
            )
        )
        items = out.get("features") if isinstance(out, dict) else out
        return [x for x in json_items(items) if normalize_name(x.get("feature_name") or "")]


def run_judges(
    features: list[FeatureSpec], make_stage, workers: int = 3, batch: int = BATCH
) -> dict[str, list[dict]]:
    """Run the three judges concurrently, on at most ``batch`` features per call; ``make_stage(dimension) ->
    Judge``."""
    chunks = [features[i : i + batch] for i in range(0, len(features), batch)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {d: [pool.submit(make_stage(d), c) for c in chunks] for d in JUDGES}
        return {d: [x for f in fs for x in f.result()] for d, fs in futures.items()}


def judge_table(verdicts: dict[str, list[dict]], analyzed: list[FeatureSpec]) -> list[dict[str, Any]]:
    """One row per feature with each judge's score, confidence, assessment, key factors, source,
    computation and rationale, joined with the analyzer's description, classification relevance and support,
    so the MetaJudge also sees the analyzer's view of each feature (App. D.1)."""
    rows: dict[str, dict[str, Any]] = {}
    for dimension, items in verdicts.items():
        for feat in items:
            key = normalize_name(feat["feature_name"])
            row = rows.setdefault(key, {"feature_name": feat["feature_name"]})
            prefix = f"judge_{dimension}"
            for fld in (
                "score",
                "confidence",
                "assessment",
                "key_factors",
                "extraction_source",
                "computation",
                "rationale",
            ):
                row[f"{prefix}_{fld}"] = feat.get(fld)
    by_key = {f.key: f for f in analyzed}
    for key, row in rows.items():
        spec = by_key.get(key)
        if spec is not None:
            row["analyzer_description"] = spec.description
            row["analyzer_classification_relevance"] = spec.classification_relevance
            row["analyzer_feature_type"] = spec.feature_type
            row["support"] = spec.support
    return list(rows.values())


def normalize_decision(value: Any) -> str:
    """Map a MetaJudge outcome onto ``accept``, ``conditional`` or ``reject`` (App. D.1: accept, conditionally
    accept or reject): any mention of a condition or a revision (``"Accept (conditional)"``,
    ``"conditionally accept"``) is ``conditional``. Anything unrecognized counts as a rejection."""
    v = str(value or "").strip().lower()
    if v.startswith("reject"):
        return "reject"
    if re.search(r"\bconditional|\brevise", v):
        return "conditional"
    if v.startswith("accept"):
        return "accept"
    return "reject"


class MetaJudge(Stage):
    """Reconciles the three judges' rows into one explained decision per feature (the first one, if it
    decides on a feature twice), ``batch`` table rows per call."""

    role = "meta_judge"
    batch = BATCH

    def __call__(self, verdicts: dict[str, list[dict]], table: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def decide(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            keys = {normalize_name(r["feature_name"]) for r in rows}
            part = {
                d: [x for x in v if normalize_name(x["feature_name"]) in keys] for d, v in verdicts.items()
            }
            return self._decide(part, rows)

        batches = [table[i : i + self.batch] for i in range(0, len(table), self.batch)]
        decisions: dict[str, dict[str, Any]] = {}
        for out in in_parallel(decide, batches, self.workers):  # merged in batch order
            for d in out:
                decisions.setdefault(normalize_name(d["feature_name"]), d)
        return list(decisions.values())

    def _decide(self, verdicts: dict[str, list[dict]], table: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompt = prompts.render(
            "meta_judge",
            relevance_schema=get_json_schema(verdicts.get("relevance", [])),
            generality_schema=get_json_schema(verdicts.get("generality", [])),
            impact_schema=get_json_schema(verdicts.get("impact", [])),
            all_judge_features=json.dumps(table, indent=2, default=str, ensure_ascii=False),
        )
        out = self.ask_json(prompt)
        if isinstance(out, dict):
            out = next((out[k] for k in ("features", "decisions") if isinstance(out.get(k), list)), [])
        decisions: dict[str, dict[str, Any]] = {}  # the first decision on each feature
        for x in json_items(out):
            key = normalize_name(x.get("feature_name") or "")
            if key and key not in decisions:
                decisions[key] = {**x, "final_decision": normalize_decision(x.get("final_decision"))}
        return list(decisions.values())
