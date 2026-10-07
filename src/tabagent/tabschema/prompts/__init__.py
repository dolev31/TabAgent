"""TabSchema prompts. The FeatureAnalyzer, CodeExtractor, FeatureAggregator, three judge and MetaJudge prompts
reproduce paper App. D.2-D.3 verbatim (``PAPER_PROMPTS``). The others are the release's (``RELEASE_PROMPTS``):
``code_validator.txt`` is written from the CodeValidator's description (App. D.1), since the paper prints no
prompt for it, and ``output_auditor.txt`` is the prompt of the release's OutputAuditor, a stage the paper does
not have, which judges the values the compiled extractor computes (``tabagent.tabschema.audit``). Prompts are
``string.Template`` files (``${name}`` placeholders), so their JSON examples need no brace escaping.

``ADDITIONS`` are the release's sections appended after a rendered prompt, as the release's other additions
are (the retry block, the request for a parseable answer); the prompts themselves stay verbatim, and each
addition sits under the heading ``# Additional rules``:

* ``tabagent_analyzer_candidates``: appended to the FeatureAnalyzer prompt when the decision has candidates
  (a ranking or multi-label decision): each row is one candidate of one decision; besides features of the
  trajectory, features of a candidate or of how the trajectory relates to one may be proposed, computed per
  candidate from its fields; up to three of the decision's candidates; neutral examples; and that every
  feature is computable before the decision and never encodes which candidate was chosen;
* ``tabagent_candidate_note``: appended to the FeatureAggregator, judge and MetaJudge prompts when the
  decision has candidates: a feature of a candidate is obtainable (its fields are known before the decision,
  for every candidate of every decision), and its generality is judged as any other feature's;
* ``tabagent_extractor``: appended to every CodeExtractor prompt, under the heading ``# Additional rules``:
  a planned feature that is a tokenized, bag-of-words or n-gram form of a text is emitted once, as the plain
  text, under its own name, unless another feature already emits that text;
* ``tabagent_extractor_candidates``: appended after it when the decision has candidates (a ranking or
  multi-label decision): the code may define the per-candidate function
  ``extract_candidate_features(trajectory_data, candidate)``; up to three of the decision's candidates as it
  receives them, catalog entry included; that a feature about a candidate belongs there (with examples for
  model, skill, application and tool decisions, and for HTTP APIs when the catalog describes them), that a
  feature about the history of earlier calls belongs in the main function, and that no column may index the
  candidates by position;
* ``tabagent_validator_candidates``: appended to the CodeValidator prompt when the decision has
  candidates: a candidate's fields (its catalog entry's included) are real data, and each feature is judged
  against the function that implements it;
* ``tabagent_auditor_candidates``: appended to the OutputAuditor prompt, under the same heading, when it
  judges per-candidate values: the candidates they were computed for."""

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
ADDITIONS = (
    "tabagent_analyzer_candidates",
    "tabagent_candidate_note",
    "tabagent_extractor",
    "tabagent_extractor_candidates",
    "tabagent_validator_candidates",
    "tabagent_auditor_candidates",
)


def text(name: str) -> str:
    """The raw template of prompt ``name``."""
    return (resources.files(__package__) / f"{name}.txt").read_text()


def render(name: str, **fields: object) -> str:
    """Prompt ``name`` with every ``${field}`` substituted (a missing field raises ``KeyError``)."""
    return Template(text(name)).substitute({k: str(v) for k, v in fields.items()})


def sha256(name: str) -> str:
    """SHA-256 of prompt ``name``'s template (recorded in every card TabSchema writes)."""
    return hashlib.sha256(text(name).encode()).hexdigest()


def append(prompt: str, *sections: str) -> str:
    """``prompt`` with the rendered ``ADDITIONS`` ``sections`` after it, each after a blank line."""
    return "\n\n".join([prompt.rstrip("\n"), *(s.strip("\n") for s in sections if s)]) + "\n"
