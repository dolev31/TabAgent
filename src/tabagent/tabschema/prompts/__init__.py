"""TabSchema prompts. The FeatureAnalyzer, CodeExtractor, FeatureAggregator, three judge and MetaJudge prompts
reproduce paper App. D.2-D.3 verbatim (``PAPER_PROMPTS``). The others are the release's (``RELEASE_PROMPTS``):
``code_validator.txt`` is written from the CodeValidator's description (App. D.1), since the paper prints no
prompt for it, and ``output_auditor.txt`` is the prompt of the release's OutputAuditor, a stage the paper does
not have, which judges the values the compiled extractor computes (``tabagent.tabschema.audit``). Prompts are
``string.Template`` files (``${name}`` placeholders), so their JSON examples need no brace escaping."""

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
RELEASE_PROMPTS = ("code_validator", "output_auditor")
PROMPTS = (*PAPER_PROMPTS, *RELEASE_PROMPTS)


def text(name: str) -> str:
    """The raw template of prompt ``name``."""
    return (resources.files(__package__) / f"{name}.txt").read_text()


def render(name: str, **fields: object) -> str:
    """Prompt ``name`` with every ``${field}`` substituted (a missing field raises ``KeyError``)."""
    return Template(text(name)).substitute({k: str(v) for k, v in fields.items()})


def sha256(name: str) -> str:
    """SHA-256 of prompt ``name``'s template (recorded in every card TabSchema writes)."""
    return hashlib.sha256(text(name).encode()).hexdigest()
