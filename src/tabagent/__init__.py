"""TabAgent: replace an LLM agent's closed-set decision heads with a cheap tabular classifier.

The framework has three parts, named as in the paper:

* **TabSchema** extracts schema, state and dependency features from an agent's successful trajectories.
* **TabSynth** adds schema-aligned, label-preserving paraphrase rows where trajectory coverage is thin.
* **TabHead** is a tabular classifier that scores every candidate of a decision in a single pass.
"""

__version__ = "0.1.0.dev0"
