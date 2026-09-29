"""TabSchema prompts. The FeatureAnalyzer, CodeExtractor, FeatureAggregator, three judge and MetaJudge prompts
reproduce paper App. C (L1222-1684) verbatim; ``code_validator.txt`` is written from the CodeValidator's
description (L1216), since the paper prints no prompt for it. Prompts are ``string.Template`` files
(``${name}`` placeholders), so their JSON examples need no brace escaping."""

from __future__ import annotations

import hashlib
from importlib import resources
from string import Template

PAPER_PROMPTS = (
    "feature_analyzer",
    "code_extractor",
    "feature_aggregator",
    "judge_relevance",
    "judge_generality",
    "judge_impact",
    "meta_judge",
)
PROMPTS = (*PAPER_PROMPTS, "code_validator")


def text(name: str) -> str:
    return (resources.files(__package__) / f"{name}.txt").read_text()


def render(name: str, **fields: object) -> str:
    return Template(text(name)).substitute({k: str(v) for k, v in fields.items()})


def sha256(name: str) -> str:
    return hashlib.sha256(text(name).encode()).hexdigest()
