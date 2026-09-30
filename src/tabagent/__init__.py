"""TabAgent: replace an LLM agent's closed-set decision heads with a cheap tabular classifier.

The framework has three parts, named as in the paper:

* **TabSchema** extracts schema, state and dependency features from an agent's successful trajectories.
* **TabSynth** adds schema-aligned, label-preserving paraphrase rows where trajectory coverage is thin.
* **TabHead** is a tabular classifier that scores every candidate of a decision in a single pass.

The library logs progress (onboarding stages, TabSchema discovery, cross-validation folds, downloads) to the
``tabagent`` logger at INFO level; enable it with ``logging.basicConfig(level=logging.INFO)`` or the CLI's
``-v``.
"""

import logging

from tabagent.agent import TabAgent
from tabagent.api import TabAgentHead, load, onboard
from tabagent.evaluation import cross_validate
from tabagent.sklearn import TabAgentClassifier, TabAgentRanker, TabAgentRouter, pointwise, scorer
from tabagent.tabhead import TabHead
from tabagent.table import DecisionSpec, DecisionTable, LeakageError
from tabagent.tabschema import Feature, FeatureCard, TabSchema, TabSchemaConfig
from tabagent.tabsynth import TabSynth

__version__ = "0.1.0.dev0"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "DecisionSpec",
    "DecisionTable",
    "Feature",
    "FeatureCard",
    "LeakageError",
    "TabAgent",
    "TabAgentClassifier",
    "TabAgentHead",
    "TabAgentRanker",
    "TabAgentRouter",
    "TabHead",
    "TabSchema",
    "TabSchemaConfig",
    "TabSynth",
    "cross_validate",
    "load",
    "onboard",
    "pointwise",
    "scorer",
]
