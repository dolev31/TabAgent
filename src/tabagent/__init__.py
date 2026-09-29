"""TabAgent: replace an LLM agent's closed-set decision heads with a cheap tabular classifier.

The framework has three parts, named as in the paper:

* **TabSchema** extracts schema, state and dependency features from an agent's successful trajectories.
* **TabSynth** adds schema-aligned, label-preserving paraphrase rows where trajectory coverage is thin.
* **TabHead** is a tabular classifier that scores every candidate of a decision in a single pass.
"""

from tabagent.agent import TabAgent
from tabagent.artifact import load
from tabagent.evaluation import cross_validate
from tabagent.sklearn import TabAgentClassifier, TabAgentRanker, TabAgentRouter, pointwise, scorer
from tabagent.tabhead import TabHead
from tabagent.table import DecisionSpec, DecisionTable, LeakageError
from tabagent.tabschema import Feature, FeatureCard, TabSchema, TabSchemaConfig
from tabagent.tabsynth import TabSynth

__version__ = "0.1.0.dev0"

__all__ = [
    "DecisionSpec",
    "DecisionTable",
    "Feature",
    "FeatureCard",
    "LeakageError",
    "TabAgent",
    "TabAgentClassifier",
    "TabAgentRanker",
    "TabAgentRouter",
    "TabHead",
    "TabSchema",
    "TabSchemaConfig",
    "TabSynth",
    "cross_validate",
    "load",
    "pointwise",
    "scorer",
]
