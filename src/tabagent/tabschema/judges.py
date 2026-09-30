"""TabSchema's judge ensemble (paper Sec. 3.1; App. D.1; judge and MetaJudge prompts in App. D.3).

Three judges score every aggregated feature on a 1-5 rubric along distinct quality dimensions (relevance,
generality, impact and uniqueness) with a confidence and a justification. They are independent, so they run
concurrently. A deterministic join builds one row per feature with the three scores and justifications plus
the analyzer's original description, relevance rationale and support; the MetaJudge reads that table and
issues one explained decision per feature (accept / conditional / reject).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tabagent.tabschema import prompts
from tabagent.tabschema.agents import FeatureSpec, Stage, normalize_name
from tabagent.tabschema.trajectory import get_json_schema

JUDGES = {"relevance": "judge_relevance", "generality": "judge_generality", "impact": "judge_impact"}
DECISIONS = ("accept", "conditional", "reject")


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
                filtered_features=json.dumps(dicts, indent=2),
                features_schema=get_json_schema(dicts),
            )
        )
        items = out.get("features", []) if isinstance(out, dict) else out
        return [x for x in items if isinstance(x, dict) and x.get("feature_name")]


def run_judges(features: list[FeatureSpec], make_stage, workers: int = 3) -> dict[str, list[dict]]:
    """Run the three judges concurrently; ``make_stage(dimension) -> Judge``."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {d: pool.submit(make_stage(d), features) for d in JUDGES}
        return {d: f.result() for d, f in futures.items()}


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
    accept or reject). Anything unrecognized counts as a rejection."""
    v = str(value or "").strip().lower()
    if v.startswith("accept"):
        return "accept"
    if v.startswith(("conditional", "revise")):
        return "conditional"
    return "reject"


class MetaJudge(Stage):
    """Reconciles the three judges' rows into one explained decision per feature."""

    role = "meta_judge"

    def __call__(self, verdicts: dict[str, list[dict]], table: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompt = prompts.render(
            "meta_judge",
            relevance_schema=get_json_schema(verdicts.get("relevance", [])),
            generality_schema=get_json_schema(verdicts.get("generality", [])),
            impact_schema=get_json_schema(verdicts.get("impact", [])),
            all_judge_features=json.dumps(table, indent=2, default=str),
        )
        out = self.ask_json(prompt)
        items = out if isinstance(out, list) else out.get("features", out.get("decisions", []))
        decisions = []
        for x in items:
            if isinstance(x, dict) and x.get("feature_name"):
                decisions.append({**x, "final_decision": normalize_decision(x.get("final_decision"))})
        return decisions
