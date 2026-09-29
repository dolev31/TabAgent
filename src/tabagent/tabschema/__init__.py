"""TabSchema: behavior-grounded feature cards for an agent's closed-set decisions (paper Sec. 3.1, App. C).

``TabSchema(llm).discover(trajectories, target)`` runs the paper's FeatureOrchestrator: FeatureAnalyzer,
CodeExtractor/CodeExecutor (three retries), CodeValidator, FeatureAggregator, three judges and the MetaJudge,
then compiles one reusable extractor and returns a ``FeatureCard``.
"""

from tabagent.tabschema.card import Feature, FeatureCard
from tabagent.tabschema.orchestrator import DiscoveryResult, TabSchema, TabSchemaConfig
from tabagent.tabschema.trajectory import Candidate, DecisionPoint, GenericJsonAdapter, Trajectory

__all__ = [
    "Candidate",
    "DecisionPoint",
    "DiscoveryResult",
    "Feature",
    "FeatureCard",
    "GenericJsonAdapter",
    "TabSchema",
    "TabSchemaConfig",
    "Trajectory",
]
